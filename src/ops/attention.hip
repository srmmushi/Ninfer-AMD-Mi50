#include "ops/ops.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

// ---------------------------------------------------------------------------
// Fused GQA flash-decode: one block per head, streaming (online) softmax over
// the KV cache. gfx906 wavefront = 64 lanes.
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_attention_kernel(__half* out, const __half* q,
                                        const __half* kc, const __half* vc,
                                        int kv_len, int heads, int kv_heads,
                                        float scale) {
  constexpr int kTile = 256;
  __shared__ float q_s[D];
  __shared__ float acc[D];
  __shared__ float s_s[kTile];
  __shared__ float red[kTile];
  __shared__ float m_run, l_run, factor;

  int h = blockIdx.x;
  int group = heads / kv_heads;
  int kvh = h / group;
  const __half* kbase = kc + static_cast<int64_t>(kvh) * D;
  const __half* vbase = vc + static_cast<int64_t>(kvh) * D;

  if (threadIdx.x < D) {
    q_s[threadIdx.x] = __half2float(q[static_cast<int64_t>(h) * D + threadIdx.x]);
    acc[threadIdx.x] = 0.0f;
  }
  if (threadIdx.x == 0) {
    m_run = -INFINITY;
    l_run = 0.0f;
  }
  __syncthreads();

  for (int tile = 0; tile < kv_len; tile += kTile) {
    int tile_n = kTile < (kv_len - tile) ? kTile : (kv_len - tile);
    int j = threadIdx.x;
    int kidx = tile + j;
    if (kidx < kv_len) {
      const __half* krow = kbase + static_cast<int64_t>(kidx) * kv_heads * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        dot += q_s[d] * __half2float(krow[d]);
      }
      s_s[j] = dot * scale;
    } else {
      s_s[j] = -INFINITY;
    }
    __syncthreads();

    // Tile max -> online softmax update (single-thread serial tail).
    red[threadIdx.x] = s_s[j];
    __syncthreads();
    if (threadIdx.x == 0) {
      float m_tile = red[0];
      for (int k = 1; k < kTile; ++k) m_tile = fmaxf(m_tile, red[k]);
      float m_new = fmaxf(m_run, m_tile);
      float l_add = 0.0f;
      for (int k = 0; k < tile_n; ++k) {
        s_s[k] = expf(s_s[k] - m_new);
        l_add += s_s[k];
      }
      factor = expf(m_run - m_new);
      l_run = l_run * factor + l_add;
      m_run = m_new;
    }
    __syncthreads();

    // Rescale the running output by the max change, then accumulate V.
    if (threadIdx.x < D) {
      float a = acc[threadIdx.x] * factor;
      for (int k = 0; k < tile_n; ++k) {
        const __half* vrow =
            vbase + static_cast<int64_t>(tile + k) * kv_heads * D;
        a += s_s[k] * __half2float(vrow[threadIdx.x]);
      }
      acc[threadIdx.x] = a;
    }
    __syncthreads();
  }

  if (threadIdx.x < D) {
    out[static_cast<int64_t>(h) * D + threadIdx.x] =
        __float2half(acc[threadIdx.x] / l_run);
  }
}

__global__ void cast_fp32_fp16_kernel(__half* dst, const float* src,
                                      int64_t count) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) dst[i] = __float2half(src[i]);
}

}  // namespace

void attention_decode_fp16(void* out, const void* q, const void* k_cache,
                           const void* v_cache, int kv_len, int heads,
                           int kv_heads, int head_dim, float scale,
                           hipStream_t stream) {
  if (head_dim == 128) {
    hipLaunchKernelGGL((decode_attention_kernel<128>), dim3(heads), dim3(256), 0,
                       stream, static_cast<__half*>(out),
                       static_cast<const __half*>(q),
                       static_cast<const __half*>(k_cache),
                       static_cast<const __half*>(v_cache), kv_len, heads,
                       kv_heads, scale);
  } else if (head_dim == 64) {
    hipLaunchKernelGGL((decode_attention_kernel<64>), dim3(heads), dim3(256), 0,
                       stream, static_cast<__half*>(out),
                       static_cast<const __half*>(q),
                       static_cast<const __half*>(k_cache),
                       static_cast<const __half*>(v_cache), kv_len, heads,
                       kv_heads, scale);
  } else {
    throw std::runtime_error(
        "unsupported head_dim for gfx906 decode attention: " +
        std::to_string(head_dim));
  }
}

void attention_prefill_head_fp16(void* out_head, const void* q_head,
                                 const void* k_head, const void* v_head,
                                 float* scores_scratch, int tq, int tk,
                                 int q_offset, int head_dim, int q_row_stride,
                                 int out_row_stride, float scale,
                                 int kv_stride_token, Blas& blas,
                                 hipStream_t stream) {
  // Layout (row-major):
  //   q_head: [tq, D] with row stride q_row_stride
  //   k_head / v_head: rows of a [kv_len, kv_heads, D] cache, token stride
  //                    kv_stride_token
  //   out_head: [tq, D] with row stride out_row_stride
  //   scores_scratch: fp32 [tq, tk], followed by an fp16 [tq, tk] region.

  // S_cm [tk, tq] = K_cm^T [tk, D] * Q_cm [D, tq]
  blas.gemm_view(k_head, kv_stride_token, /*trans_a=*/true, q_head,
                 q_row_stride, /*trans_b=*/false, scores_scratch, /*ldc=*/tk,
                 tk, tq, head_dim, /*c_fp32=*/true, stream);

  softmax_causal_rows_fp32(scores_scratch, tq, tk, q_offset, scale, stream);

  // fp16 copy of S placed after the fp32 region (scratch holds 6 bytes/elt).
  __half* s16 =
      reinterpret_cast<__half*>(scores_scratch) + 2 * static_cast<int64_t>(tq) * tk;
  int64_t count = static_cast<int64_t>(tq) * tk;
  int blocks = static_cast<int>((count + 255) / 256);
  hipLaunchKernelGGL(cast_fp32_fp16_kernel, dim3(blocks), dim3(256), 0, stream,
                     s16, scores_scratch, count);

  // O_cm [D, tq] = V_cm [D, tk] * S_cm [tk, tq]  (s16 is S_cm in memory)
  blas.gemm_view(v_head, kv_stride_token, /*trans_a=*/false, s16, /*ldb=*/tk,
                 /*trans_b=*/false, out_head, out_row_stride, head_dim, tq, tk,
                 /*c_fp32=*/false, stream);
}

}  // namespace ninfer
