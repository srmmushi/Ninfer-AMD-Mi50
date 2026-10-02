#include "ops/ops.h"

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
  hipLaunchKernelGGL(gemv_kernel, dim3(n_rows), dim3(kGemvBlock), 0, stream,
                     static_cast<__half*>(y), static_cast<float*>(nullptr),
                     static_cast<const __half*>(w),
                     static_cast<const __half*>(x), k_cols, /*mode=*/0);
}

void gemv_fp32out(void* y, const void* w, const void* x, int n_rows, int k_cols,
                  hipStream_t stream) {
  if (n_rows <= 0) return;
  hipLaunchKernelGGL(gemv_kernel, dim3(n_rows), dim3(kGemvBlock), 0, stream,
                     static_cast<__half*>(nullptr), static_cast<float*>(y),
                     static_cast<const __half*>(w),
                     static_cast<const __half*>(x), k_cols, /*mode=*/2);
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
