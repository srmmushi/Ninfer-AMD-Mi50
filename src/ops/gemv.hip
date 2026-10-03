#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr int kGemvBlock = 256;
constexpr int kGemvChunk = 1024;

__device__ inline float gemv_warp_reduce(float v) {
  for (int off = 64; off > 0; off >>= 1) {
    v += __shfl_down(v, off, 64);  // wave64 on gfx906
  }
  return v;
}

// One block computes one output row: y[n] = sum_k W[n][k] * x[k].
// The activation is staged through LDS in 1024-element chunks so each weight
// element is streamed exactly once and fully coalesced (consecutive threads
// read consecutive k). FP32 accumulation throughout.
//   mode 0: y16[n] = acc            (fp16 out)
//   mode 1: y16[n] += acc           (unused; kept for symmetry)
//   mode 2: y32[n] = acc            (fp32 out: router, LM head)
__device__ inline void gemv_unpack8(const uint4& v, float out[8]) {
  uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] =
        __half2float(__ushort_as_half(static_cast<uint16_t>(words[i] & 0xFFFFu)));
    out[2 * i + 1] =
        __half2float(__ushort_as_half(static_cast<uint16_t>(words[i] >> 16)));
  }
}

// ---------------------------------------------------------------------------
// Vectorized wave64 GEMV (decode fast path).
// One block per output row; every thread consumes 8 halves (one uint4 = 16 B)
// of both the weight row and the activation, unpacks them in registers and
// accumulates in FP32.
//   * 8x fewer memory instructions than the scalar version (16 B per load
//     instead of 2 B), and 32 lanes x 16 B covers full cachelines;
//   * no LDS staging: x is not shared between threads here, so the previous
//     global -> LDS -> register path was pure overhead plus two barriers per
//     chunk; the only shared memory left is the final warp reduction.
// Used when K % 8 == 0 (all real Qwen shapes); the scalar kernel below stays
// as the fallback.
// ---------------------------------------------------------------------------
template <bool F32OUT>
__global__ void gemv_kernel_vec8(__half* __restrict__ y16,
                                 float* __restrict__ y32,
                                 const __half* __restrict__ w,
                                 const __half* __restrict__ x, int vec8_count,
                                 int K) {
  int64_t n = blockIdx.x;
  const uint4* __restrict__ wrow =
      reinterpret_cast<const uint4*>(w + n * static_cast<int64_t>(K));
  const uint4* __restrict__ xrow = reinterpret_cast<const uint4*>(x);
  __shared__ float red[kGemvBlock / 64];
  float acc = 0.0f;

  for (int i = threadIdx.x; i < vec8_count; i += blockDim.x) {
    uint4 wv = wrow[i];
    uint4 xv = xrow[i];
    float wf[8], xf[8];
    gemv_unpack8(wv, wf);
    gemv_unpack8(xv, xf);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += wf[j] * xf[j];
  }
  // Scalar tail for K not divisible by 8 (vectorized path already covers
  // the aligned bulk).
  for (int k = vec8_count * 8 + threadIdx.x; k < K; k += blockDim.x) {
    acc += __half2float(w[n * static_cast<int64_t>(K) + k]) *
           __half2float(x[k]);
  }

  acc = gemv_warp_reduce(acc);
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

// Scalar fallback (K % 8 != 0): LDS-staged activation, one element per load.
__global__ void gemv_kernel(__half* y16, float* y32, const __half* w,
                            const __half* x, int K, int mode) {
  int64_t n = blockIdx.x;
  const __half* wrow = w + n * K;
  __shared__ __half x_s[kGemvChunk];
  __shared__ float red[kGemvBlock / 64];
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += kGemvChunk) {
    int chunk = kGemvChunk < (K - k0) ? kGemvChunk : (K - k0);
    for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
      x_s[i] = x[k0 + i];
    }
    __syncthreads();
    for (int k = threadIdx.x; k < chunk; k += blockDim.x) {
      acc += __half2float(wrow[k0 + k]) * __half2float(x_s[k]);
    }
    __syncthreads();
  }
  acc = gemv_warp_reduce(acc);
  if (threadIdx.x % 64 == 0) red[threadIdx.x / 64] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < blockDim.x / 64; ++i) total += red[i];
    if (mode == 2) {
      y32[n] = total;
    } else {
      y16[n] = __float2half(total);
    }
  }
}

// ---------------------------------------------------------------------------
// Batched decode post-QKV fusion: grid.y walks the in-flight rows
// (one row per active sequence), grid.x walks (q | k | v) heads.
// ---------------------------------------------------------------------------
__global__ void post_qkv_batch_kernel(__half* qkv, __half* kc, __half* vc,
                                      const int* row_pos, const int* seq_id,
                                      int Hq, int Hkv, int D, int qkv_row_stride,
                                      int64_t seq_stride, const float* inv_freq,
                                      int has_qknorm, const __half* qn,
                                      const __half* kn, float eps) {
  int hb = blockIdx.x;
  int r = blockIdx.y;
  int d = threadIdx.x;
  __shared__ float red[2];
  int half = D / 2;
  bool is_q = hb < Hq;
  bool is_k = !is_q && hb < Hq + Hkv;
  int kvh = is_q ? 0 : (is_k ? hb - Hq : hb - Hq - Hkv);
  int64_t row_base = static_cast<int64_t>(r) * qkv_row_stride;
  __half* src = is_q ? qkv + row_base + static_cast<int64_t>(hb) * D
              : is_k ? qkv + row_base + static_cast<int64_t>(Hq + kvh) * D
                     : qkv + row_base + static_cast<int64_t>(Hq + Hkv + kvh) * D;

  float v = __half2float(src[d]);
  if (has_qknorm && hb < Hq + Hkv) {
    float sumsq = gemv_warp_reduce(v * v);
    if (threadIdx.x % 64 == 0) red[threadIdx.x / 64] = sumsq;
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
      for (int i = 0; i < blockDim.x / 64; ++i) total += red[i];
      red[1] = rsqrtf(total / D + eps);
    }
    __syncthreads();
    v *= red[1] * __half2float(is_q ? qn[d] : kn[d]);
    src[d] = __float2half(v);
    __syncthreads();
  }

  __half* dst;
  if (is_q) {
    dst = qkv + row_base + static_cast<int64_t>(hb) * D;
  } else {
    int64_t cache_off = static_cast<int64_t>(seq_id[r]) * seq_stride +
                        static_cast<int64_t>(row_pos[r]) * (Hkv * D) +
                        static_cast<int64_t>(kvh) * D;
    dst = (is_k ? kc : vc) + cache_off;
  }

  if (hb < Hq + Hkv) {
    if (d < half) {
      float x1 = __half2float(src[d]);
      float x2 = __half2float(src[d + half]);
      float angle = static_cast<float>(row_pos[r]) * inv_freq[d];
      float c = cosf(angle), s = sinf(angle);
      dst[d] = __float2half(x1 * c - x2 * s);
      dst[d + half] = __float2half(x1 * s + x2 * c);
    }
  } else {
    dst[d] = src[d];
  }
}

// ---------------------------------------------------------------------------
// Decode post-QKV fusion. One block per (q head | k head | v head):
//   q heads:   optional per-head RMSNorm, RoPE, write back to qkv buffer
//   k heads:   optional RMSNorm, RoPE, write straight into the KV cache
//   v heads:   straight copy into the cache
// Replaces 6 separate launches with 1 on the decode path.
// ---------------------------------------------------------------------------
__global__ void post_qkv_kernel(__half* qkv, __half* kc, __half* vc,
                                const float* inv_freq, int pos0, int Hq,
                                int Hkv, int D, int has_qknorm,
                                const __half* qn, const __half* kn, float eps) {
  int hb = blockIdx.x;   // 0 .. Hq + 2*Hkv - 1
  int d = threadIdx.x;   // blockDim == D
  __shared__ float red[2];
  int half = D / 2;
  bool is_q = hb < Hq;
  bool is_k = !is_q && hb < Hq + Hkv;
  int kvh = is_q ? 0 : (is_k ? hb - Hq : hb - Hq - Hkv);
  __half* src = is_q ? qkv + static_cast<int64_t>(hb) * D
              : is_k ? qkv + static_cast<int64_t>(Hq + kvh) * D
                     : qkv + static_cast<int64_t>(Hq + Hkv + kvh) * D;

  float v = __half2float(src[d]);

  if (has_qknorm && hb < Hq + Hkv) {
    float sumsq = gemv_warp_reduce(v * v);
    if (threadIdx.x % 64 == 0) red[threadIdx.x / 64] = sumsq;
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
      for (int i = 0; i < blockDim.x / 64; ++i) total += red[i];
      red[1] = rsqrtf(total / D + eps);
    }
    __syncthreads();
    v *= red[1] * __half2float(is_q ? qn[d] : kn[d]);
    // Stage the normed value so the rope stage reads normed pairs.
    src[d] = __float2half(v);
    __syncthreads();
  }

  __half* dst;
  if (is_q) {
    dst = qkv + static_cast<int64_t>(hb) * D;
  } else {
    dst = (is_k ? kc : vc) + static_cast<int64_t>(pos0) * (Hkv * D) +
          static_cast<int64_t>(kvh) * D;
  }

  if (hb < Hq + Hkv) {
    // RoPE half-split: thread d < half owns the (d, d + half) pair.
    if (d < half) {
      float x1 = __half2float(src[d]);
      float x2 = __half2float(src[d + half]);
      float angle = static_cast<float>(pos0) * inv_freq[d];
      float c = cosf(angle), s = sinf(angle);
      dst[d] = __float2half(x1 * c - x2 * s);
      dst[d + half] = __float2half(x1 * s + x2 * c);
    }
  } else {
    dst[d] = src[d];
  }
}

}  // namespace

void gemv_fp16(void* y, const void* w, const void* x, int n_rows, int k_cols,
               hipStream_t stream) {
  if (n_rows <= 0) return;
  if (k_cols % 8 == 0) {
    int vec8 = k_cols / 8;
    // Size the block to the actual vector count so no wave idles.
    int block = std::min(kGemvBlock, (vec8 + 63) / 64 * 64);
    hipLaunchKernelGGL((gemv_kernel_vec8<false>), dim3(n_rows), dim3(block), 0,
                       stream, static_cast<__half*>(y),
                       static_cast<float*>(nullptr),
                       static_cast<const __half*>(w),
                       static_cast<const __half*>(x), vec8, k_cols);
    return;
  }
  hipLaunchKernelGGL(gemv_kernel, dim3(n_rows), dim3(kGemvBlock), 0, stream,
                     static_cast<__half*>(y), static_cast<float*>(nullptr),
                     static_cast<const __half*>(w),
                     static_cast<const __half*>(x), k_cols, /*mode=*/0);
}

void gemv_fp32out(void* y, const void* w, const void* x, int n_rows, int k_cols,
                  hipStream_t stream) {
  if (n_rows <= 0) return;
  if (k_cols % 8 == 0) {
    int vec8 = k_cols / 8;
    int block = std::min(kGemvBlock, (vec8 + 63) / 64 * 64);
    hipLaunchKernelGGL((gemv_kernel_vec8<true>), dim3(n_rows), dim3(block), 0,
                       stream, static_cast<__half*>(nullptr),
                       static_cast<float*>(y), static_cast<const __half*>(w),
                       static_cast<const __half*>(x), vec8, k_cols);
    return;
  }
  hipLaunchKernelGGL(gemv_kernel, dim3(n_rows), dim3(kGemvBlock), 0, stream,
                     static_cast<__half*>(nullptr), static_cast<float*>(y),
                     static_cast<const __half*>(w),
                     static_cast<const __half*>(x), k_cols, /*mode=*/2);
}

void decode_post_qkv_batch_fp16(void* qkv, void* k_base, void* v_base,
                                const void* row_pos, const void* seq_id,
                                int rows, int heads, int kv_heads, int head_dim,
                                int qkv_row_stride, int64_t seq_stride_elems,
                                const void* inv_freq, bool has_qknorm,
                                const void* qn_weight, const void* kn_weight,
                                float eps, hipStream_t stream) {
  if (rows <= 0) return;
  dim3 grid(heads + 2 * kv_heads, rows);
  if (head_dim == 128) {
    hipLaunchKernelGGL(post_qkv_batch_kernel, grid, dim3(128), 0, stream,
                       static_cast<__half*>(qkv), static_cast<__half*>(k_base),
                       static_cast<__half*>(v_base),
                       static_cast<const int*>(row_pos),
                       static_cast<const int*>(seq_id), heads, kv_heads,
                       head_dim, qkv_row_stride, seq_stride_elems,
                       static_cast<const float*>(inv_freq),
                       has_qknorm ? 1 : 0,
                       static_cast<const __half*>(qn_weight),
                       static_cast<const __half*>(kn_weight), eps);
  } else if (head_dim == 64) {
    hipLaunchKernelGGL(post_qkv_batch_kernel, grid, dim3(64), 0, stream,
                       static_cast<__half*>(qkv), static_cast<__half*>(k_base),
                       static_cast<__half*>(v_base),
                       static_cast<const int*>(row_pos),
                       static_cast<const int*>(seq_id), heads, kv_heads,
                       head_dim, qkv_row_stride, seq_stride_elems,
                       static_cast<const float*>(inv_freq),
                       has_qknorm ? 1 : 0,
                       static_cast<const __half*>(qn_weight),
                       static_cast<const __half*>(kn_weight), eps);
  } else {
    throw std::runtime_error("unsupported head_dim " +
                             std::to_string(head_dim) +
                             " for batched decode post-qkv fusion");
  }
}

void decode_post_qkv_fp16(void* qkv, void* k_cache, void* v_cache,
                          const void* inv_freq, int pos0, int heads,
                          int kv_heads, int head_dim, bool has_qknorm,
                          const void* qn_weight, const void* kn_weight,
                          float eps, hipStream_t stream) {
  int blocks = heads + 2 * kv_heads;
  if (head_dim == 128) {
    hipLaunchKernelGGL(post_qkv_kernel, dim3(blocks), dim3(128), 0, stream,
                       static_cast<__half*>(qkv), static_cast<__half*>(k_cache),
                       static_cast<__half*>(v_cache),
                       static_cast<const float*>(inv_freq), pos0, heads,
                       kv_heads, head_dim, has_qknorm ? 1 : 0,
                       static_cast<const __half*>(qn_weight),
                       static_cast<const __half*>(kn_weight), eps);
  } else if (head_dim == 64) {
    hipLaunchKernelGGL(post_qkv_kernel, dim3(blocks), dim3(64), 0, stream,
                       static_cast<__half*>(qkv), static_cast<__half*>(k_cache),
                       static_cast<__half*>(v_cache),
                       static_cast<const float*>(inv_freq), pos0, heads,
                       kv_heads, head_dim, has_qknorm ? 1 : 0,
                       static_cast<const __half*>(qn_weight),
                       static_cast<const __half*>(kn_weight), eps);
  } else {
    throw std::runtime_error("unsupported head_dim " +
                             std::to_string(head_dim) +
                             " for decode post-qkv fusion");
  }
}

}  // namespace ninfer
