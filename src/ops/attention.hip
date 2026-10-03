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
                                        const int* pos_ptr, int heads,
                                        int kv_heads, float scale) {
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
  int kv_len = *pos_ptr + 1;

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
                           const void* v_cache, const void* pos_ptr, int heads,
                           int kv_heads, int head_dim, float scale,
                           hipStream_t stream) {
  const int* pp = static_cast<const int*>(pos_ptr);
  if (head_dim == 128) {
    hipLaunchKernelGGL((decode_attention_kernel<128>), dim3(heads), dim3(256), 0,
                       stream, static_cast<__half*>(out),
                       static_cast<const __half*>(q),
                       static_cast<const __half*>(k_cache),
                       static_cast<const __half*>(v_cache), pp, heads, kv_heads,
                       scale);
  } else if (head_dim == 64) {
    hipLaunchKernelGGL((decode_attention_kernel<64>), dim3(heads), dim3(256), 0,
                       stream, static_cast<__half*>(out),
                       static_cast<const __half*>(q),
                       static_cast<const __half*>(k_cache),
                       static_cast<const __half*>(v_cache), pp, heads, kv_heads,
                       scale);
  } else {
    throw std::runtime_error(
        "unsupported head_dim for gfx906 decode attention: " +
        std::to_string(head_dim));
  }
}

// ---------------------------------------------------------------------------
// Fused GQA prefill, ALL heads in one launch, flash-style online softmax.
// Grid: (heads, ceil(q_len / BQ)); each block owns a 16-query tile of one
// head and streams K/V tiles of 32 keys through LDS (gfx906 has no TMA or
// cp.async — the LDS staging here is the latency-hiding equivalent).
// ---------------------------------------------------------------------------
template <int D>
__global__ void prefill_attention_kernel(__half* out, const __half* q,
                                         const __half* kc, const __half* vc,
                                         int q_len, int kv_len, int pos0,
                                         int heads, int kv_heads, float scale,
                                         int q_row_stride, int out_row_stride) {
  constexpr int BQ = 16;
  constexpr int BK = 32;
  constexpr int EPT = BK * D / 256;  // K/V elements staged per thread
  static_assert(BK * D % 256 == 0, "tile must split evenly across 256 threads");
  // Double-buffered K/V tiles: tile t+1 is fetched into registers while tile
  // t is consumed from LDS — the gfx906 substitute for cp.async pipelining.
  __shared__ float q_s[BQ][D];
  __shared__ float acc[BQ][D];
  __shared__ __half k_s[2][BK][D];
  __shared__ __half v_s[2][BK][D];
  __shared__ float s_s[BQ][BK];
  __shared__ float m_s[BQ], l_s[BQ], factor_s[BQ];

  int h = blockIdx.x;
  int t0 = blockIdx.y * BQ;
  int group = heads / kv_heads;
  int kvh = h / group;
  int kv_stride = kv_heads * D;
  const __half* kbase = kc + static_cast<int64_t>(kvh) * D;
  const __half* vbase = vc + static_cast<int64_t>(kvh) * D;

  for (int e = threadIdx.x; e < BQ * D; e += blockDim.x) {
    int qi = e / D, d = e % D;
    int t = t0 + qi;
    q_s[qi][d] = (t < q_len)
                     ? __half2float(q[static_cast<int64_t>(t) * q_row_stride +
                                      h * D + d])
                     : 0.0f;
    acc[qi][d] = 0.0f;
  }
  if (threadIdx.x < BQ) {
    m_s[threadIdx.x] = -INFINITY;
    l_s[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  // Software pipeline: global -> registers -> LDS.
  __half rk[EPT], rv[EPT];
  auto load_regs = [&](int k0) {
    for (int j = 0; j < EPT; ++j) {
      int e = threadIdx.x + j * 256;
      int ki = e / D, d = e % D;
      int t = k0 + ki;
      rk[j] = (t < kv_len) ? kbase[static_cast<int64_t>(t) * kv_stride + d]
                           : __float2half(0.0f);
      rv[j] = (t < kv_len) ? vbase[static_cast<int64_t>(t) * kv_stride + d]
                           : __float2half(0.0f);
    }
  };
  auto store_regs = [&](int buf) {
    for (int j = 0; j < EPT; ++j) {
      int e = threadIdx.x + j * 256;
      k_s[buf][e / D][e % D] = rk[j];
      v_s[buf][e / D][e % D] = rv[j];
    }
  };

  load_regs(0);
  store_regs(0);
  __syncthreads();
  int cur = 0;

  for (int k0 = 0; k0 < kv_len; k0 += BK) {
    int kn = BK < (kv_len - k0) ? BK : (kv_len - k0);
    int nxt = k0 + BK;
    // Issue the next tile's global loads now; they overlap the LDS-bound
    // score/accumulate work below (registers are only waited on at store).
    if (nxt < kv_len) load_regs(nxt);

    // Scores for the BQ x BK sub-block; causal mask on absolute positions.
    for (int p = threadIdx.x; p < BQ * BK; p += blockDim.x) {
      int qi = p / BK, ki = p % BK;
      int t = t0 + qi;
      int k_abs = k0 + ki;
      bool valid = (t < q_len) && (ki < kn) && (k_abs <= pos0 + t);
      float dot = 0.0f;
      if (valid) {
        for (int d = 0; d < D; ++d) {
          dot += q_s[qi][d] * __half2float(k_s[cur][ki][d]);
        }
      }
      s_s[qi][ki] = valid ? dot * scale : -INFINITY;
    }
    __syncthreads();

    // Per-row online softmax update (serial tail over BQ threads).
    if (threadIdx.x < BQ) {
      int qi = threadIdx.x;
      float m_tile = -INFINITY;
      for (int ki = 0; ki < BK; ++ki) m_tile = fmaxf(m_tile, s_s[qi][ki]);
      float m_new = fmaxf(m_s[qi], m_tile);
      float l_add = 0.0f;
      for (int ki = 0; ki < BK; ++ki) {
        s_s[qi][ki] = expf(s_s[qi][ki] - m_new);
        l_add += s_s[qi][ki];
      }
      factor_s[qi] = expf(m_s[qi] - m_new);
      l_s[qi] = l_s[qi] * factor_s[qi] + l_add;
      m_s[qi] = m_new;
    }
    __syncthreads();

    // Rescale the running output and accumulate the weighted V tile (LDS:
    // each v element is reused across all BQ query rows without DRAM refetch).
    for (int e = threadIdx.x; e < BQ * D; e += blockDim.x) {
      int qi = e / D, d = e % D;
      float a = acc[qi][d] * factor_s[qi];
      for (int ki = 0; ki < kn; ++ki) {
        a += s_s[qi][ki] * __half2float(v_s[cur][ki][d]);
      }
      acc[qi][d] = a;
    }
    __syncthreads();
    if (nxt < kv_len) store_regs(cur ^ 1);
    __syncthreads();
    cur ^= 1;
  }

  for (int e = threadIdx.x; e < BQ * D; e += blockDim.x) {
    int qi = e / D, d = e % D;
    int t = t0 + qi;
    if (t < q_len) {
      out[static_cast<int64_t>(t) * out_row_stride + h * D + d] =
          __float2half(acc[qi][d] / l_s[qi]);
    }
  }
}

void attention_prefill_fused_fp16(void* out, const void* q, const void* k_cache,
                                  const void* v_cache, int q_len, int kv_len,
                                  int pos0, int heads, int kv_heads,
                                  int head_dim, int q_row_stride,
                                  int out_row_stride, float scale,
                                  hipStream_t stream) {
  dim3 grid(heads, (q_len + 15) / 16);
  if (head_dim == 128) {
    hipLaunchKernelGGL((prefill_attention_kernel<128>), grid, dim3(256), 0,
                       stream, static_cast<__half*>(out),
                       static_cast<const __half*>(q),
                       static_cast<const __half*>(k_cache),
                       static_cast<const __half*>(v_cache), q_len, kv_len,
                       pos0, heads, kv_heads, scale, q_row_stride,
                       out_row_stride);
  } else if (head_dim == 64) {
    hipLaunchKernelGGL((prefill_attention_kernel<64>), grid, dim3(256), 0,
                       stream, static_cast<__half*>(out),
                       static_cast<const __half*>(q),
                       static_cast<const __half*>(k_cache),
                       static_cast<const __half*>(v_cache), q_len, kv_len,
                       pos0, heads, kv_heads, scale, q_row_stride,
                       out_row_stride);
  } else {
    throw std::runtime_error(
        "unsupported head_dim for gfx906 prefill attention: " +
        std::to_string(head_dim));
  }
}

}  // namespace ninfer
