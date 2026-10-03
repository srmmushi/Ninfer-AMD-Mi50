#include "ops/ops.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr int kQuantBlock = 256;  // 8 warps -> 8 groups per block
constexpr int kQ4Tile = 8192;     // LDS staging chunk (256 groups of 32)

__device__ inline float q4_warp_reduce(float v) {
  for (int off = 64; off > 0; off >>= 1) {
    v += __shfl_down(v, off, 64);
  }
  return v;
}

// ---------------------------------------------------------------------------
// Quantization: one warp per group of 32 weights.
//   scale = max|x| / 7 ; q = clamp(round(x / scale) + 8, 0, 15)
//   dequant(q) = (q - 8) * scale
// ---------------------------------------------------------------------------
__global__ void quantize_q4_kernel(const __half* w, uint8_t* packed,
                                   __half* scales, int cols, int groups_per_row,
                                   int total_groups) {
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int g = blockIdx.x * (kQuantBlock / 32) + warp;
  if (g >= total_groups) return;
  int n = g / groups_per_row;
  int gi = g % groups_per_row;
  int k0 = gi * kQ4Group;
  const __half* row = w + static_cast<int64_t>(n) * cols;

  float v = fabsf(__half2float(row[k0 + lane]));
  float mx = q4_warp_reduce(v);
  float scale = mx / 7.0f;
  if (scale == 0.0f) scale = 1.0f;
  scales[static_cast<int64_t>(n) * groups_per_row + gi] = __float2half(scale);

  float x = __half2float(row[k0 + lane]);
  int q = static_cast<int>(roundf(x / scale)) + 8;
  if (q < 0) q = 0;
  if (q > 15) q = 15;
  int q_hi = __shfl_xor(q, 1);  // partner lane packs the high nibble
  if (lane % 2 == 0) {
    packed[static_cast<int64_t>(n) * (cols / 2) + (k0 + lane) / 2] =
        static_cast<uint8_t>(q | (q_hi << 4));
  }
}

// ---------------------------------------------------------------------------
// Q4 GEMV (decode): one block per output row. x is staged through LDS in
// kQ4Tile-element chunks; each thread owns one group (32 weights = one uint4
// = 16 bytes of packed data) and accumulates dequant * x.
// ---------------------------------------------------------------------------
template <bool F32OUT>
__global__ void gemv_q4_kernel(__half* y16, float* y32, const uint8_t* packed,
                               const __half* scales, const __half* x, int cols,
                               int groups_per_row) {
  __shared__ __half x_s[kQ4Tile];
  __shared__ float red[kQuantBlock / 64];
  int64_t n = blockIdx.x;
  const uint4* qrow = reinterpret_cast<const uint4*>(
      packed + n * static_cast<int64_t>(cols / 2));
  const __half* srow = scales + n * static_cast<int64_t>(groups_per_row);
  float acc = 0.0f;

  for (int k0 = 0; k0 < cols; k0 += kQ4Tile) {
    int chunk = kQ4Tile < (cols - k0) ? kQ4Tile : (cols - k0);
    for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
      x_s[i] = x[k0 + i];
    }
    __syncthreads();
    int g = k0 / kQ4Group + threadIdx.x;
    if (g < groups_per_row && (g - k0 / kQ4Group) * kQ4Group < chunk) {
      uint4 p = qrow[g];
      float sc = __half2float(srow[g]);
      uint32_t words[4] = {p.x, p.y, p.z, p.w};
      int base = (g - k0 / kQ4Group) * kQ4Group;
      for (int w = 0; w < 4; ++w) {
        for (int j = 0; j < 8; ++j) {
          int nib = static_cast<int>((words[w] >> (4 * j)) & 0xFu);
          acc += (static_cast<float>(nib) - 8.0f) * sc *
                 __half2float(x_s[base + w * 8 + j]);
        }
      }
    }
    __syncthreads();
  }

  acc = q4_warp_reduce(acc);
  if (threadIdx.x % 64 == 0) red[threadIdx.x / 64] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < blockDim.x / 64; ++i) total += red[i];
    if (F32OUT) {
      y32[n] = total;
    } else {
      y16[n] = __float2half(total);
    }
  }
}

// ---------------------------------------------------------------------------
// Q4 GEMM (prefill): one block per (output column n, M-tile). The weight row
// is dequantized into LDS in chunks; threads sweep k and accumulate MT rows.
// Weights are read once (INT4), activations coalesced — no FP16
// materialization of the weight matrix.
// ---------------------------------------------------------------------------
template <int MT>
__global__ void gemm_q4_kernel(__half* out, const __half* x,
                               const uint8_t* packed, const __half* scales,
                               int M, int N, int cols, int groups_per_row) {
  __shared__ __half w_s[kQ4Tile];
  __shared__ float part[MT][kQuantBlock / 64];  // per-warp partials
  __shared__ float acc_s[MT];
  int n = blockIdx.x;
  int m0 = blockIdx.y * MT;
  const uint4* qrow = reinterpret_cast<const uint4*>(
      packed + static_cast<int64_t>(n) * (cols / 2));
  const __half* srow = scales + static_cast<int64_t>(n) * groups_per_row;
  float acc[MT];
  for (int i = 0; i < MT; ++i) acc[i] = 0.0f;

  for (int k0 = 0; k0 < cols; k0 += kQ4Tile) {
    int chunk = kQ4Tile < (cols - k0) ? kQ4Tile : (cols - k0);
    // Dequantize this chunk into LDS: each thread handles one group.
    int g = k0 / kQ4Group + threadIdx.x;
    int local = g - k0 / kQ4Group;
    if (local * kQ4Group < chunk && g < groups_per_row) {
      uint4 p = qrow[g];
      float sc = __half2float(srow[g]);
      uint32_t words[4] = {p.x, p.y, p.z, p.w};
      int base = local * kQ4Group;
      for (int w = 0; w < 4; ++w) {
        for (int j = 0; j < 8; ++j) {
          int nib = static_cast<int>((words[w] >> (4 * j)) & 0xFu);
          w_s[base + w * 8 + j] =
              __float2half((static_cast<float>(nib) - 8.0f) * sc);
        }
      }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < chunk; k += blockDim.x) {
      float wv = __half2float(w_s[k]);
      int64_t off = static_cast<int64_t>(k0) + k;
      for (int i = 0; i < MT; ++i) {
        int m = m0 + i;
        if (m < M) {
          acc[i] += __half2float(x[static_cast<int64_t>(m) * cols + off]) * wv;
        }
      }
    }
    __syncthreads();
  }

  // Reduce each of the MT accumulators across the block: warp shuffle first,
  // then one barrier, then a single-threaded cross-warp sum (uniform control
  // flow — no barrier inside a conditional).
  for (int i = 0; i < MT; ++i) {
    float v = q4_warp_reduce(acc[i]);
    if (threadIdx.x % 64 == 0) {
      part[i][threadIdx.x / 64] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < MT; ++i) {
      float total = 0.0f;
      for (int w = 0; w < blockDim.x / 64; ++w) total += part[i][w];
      acc_s[i] = total;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < MT; ++i) {
      int m = m0 + i;
      if (m < M) out[static_cast<int64_t>(m) * N + n] = __float2half(acc_s[i]);
    }
  }
}

}  // namespace

void quantize_q4_fp16(const void* w_fp16, void* packed, void* scales, int rows,
                      int cols, hipStream_t stream) {
  if (cols % kQ4Group != 0) {
    throw std::runtime_error("Q4 requires cols divisible by " +
                             std::to_string(kQ4Group) + "; got " +
                             std::to_string(cols));
  }
  int groups_per_row = cols / kQ4Group;
  int total_groups = rows * groups_per_row;
  int blocks = (total_groups + (kQuantBlock / 32) - 1) / (kQuantBlock / 32);
  hipLaunchKernelGGL(quantize_q4_kernel, dim3(blocks), dim3(kQuantBlock), 0,
                     stream, static_cast<const __half*>(w_fp16),
                     static_cast<uint8_t*>(packed),
                     static_cast<__half*>(scales), cols, groups_per_row,
                     total_groups);
}

void gemv_q4_fp16(void* y, const void* packed, const void* scales,
                  const void* x, int n_rows, int k_cols, hipStream_t stream) {
  if (n_rows <= 0) return;
  hipLaunchKernelGGL((gemv_q4_kernel<false>), dim3(n_rows), dim3(kQuantBlock), 0,
                     stream, static_cast<__half*>(y),
                     static_cast<float*>(nullptr),
                     static_cast<const uint8_t*>(packed),
                     static_cast<const __half*>(scales),
                     static_cast<const __half*>(x), k_cols,
                     k_cols / kQ4Group);
}

void gemv_q4_fp32out(void* y, const void* packed, const void* scales,
                     const void* x, int n_rows, int k_cols,
                     hipStream_t stream) {
  if (n_rows <= 0) return;
  hipLaunchKernelGGL((gemv_q4_kernel<true>), dim3(n_rows), dim3(kQuantBlock), 0,
                     stream, static_cast<__half*>(nullptr),
                     static_cast<float*>(y),
                     static_cast<const uint8_t*>(packed),
                     static_cast<const __half*>(scales),
                     static_cast<const __half*>(x), k_cols,
                     k_cols / kQ4Group);
}

void gemm_q4_fp16(void* out, const void* x, const void* packed,
                  const void* scales, int M, int N, int K, hipStream_t stream) {
  if (M <= 0 || N <= 0) return;
  constexpr int MT = 32;
  dim3 grid(N, (M + MT - 1) / MT);
  hipLaunchKernelGGL((gemm_q4_kernel<MT>), grid, dim3(kQuantBlock), 0, stream,
                     static_cast<__half*>(out), static_cast<const __half*>(x),
                     static_cast<const uint8_t*>(packed),
                     static_cast<const __half*>(scales), M, N, K,
                     K / kQ4Group);
}

}  // namespace ninfer
