#include "ops/ops.h"

#include <cmath>

namespace ninfer {
namespace {

constexpr int kBlock = 256;

__device__ inline float warp_reduce_sum(float v) {
  for (int off = 64; off > 0; off >>= 1) {
    v += __shfl_down(v, off, 64);  // wave64 on gfx906
  }
  return v;
}

__global__ void rmsnorm_kernel(__half* out, const __half* x, const __half* w,
                               int size, float eps) {
  int row = blockIdx.x;
  const __half* xr = x + static_cast<int64_t>(row) * size;
  __half* orow = out + static_cast<int64_t>(row) * size;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float v = __half2float(xr[i]);
    acc += v * v;
  }
  acc = warp_reduce_sum(acc);
  __shared__ float warp_sums[kBlock / 64];
  if (threadIdx.x % 64 == 0) warp_sums[threadIdx.x / 64] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < blockDim.x / 64; ++i) total += warp_sums[i];
    warp_sums[0] = rsqrtf(total / size + eps);
  }
  __syncthreads();
  float inv = warp_sums[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    orow[i] = __float2half(__half2float(xr[i]) * inv * __half2float(w[i]));
  }
}

__global__ void rmsnorm_heads_kernel(__half* out, const __half* x,
                                     const __half* w, int heads, int head_dim,
                                     int row_stride, float eps) {
  // One block per (token, head): normalize head_dim contiguous values.
  int64_t idx = static_cast<int64_t>(blockIdx.x);
  int64_t t = idx / heads;
  int64_t h = idx % heads;
  const __half* xr = x + t * row_stride + h * head_dim;
  __half* orow = out + t * row_stride + h * head_dim;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float v = __half2float(xr[i]);
    acc += v * v;
  }
  acc = warp_reduce_sum(acc);
  __shared__ float warp_sums[kBlock / 64];
  if (threadIdx.x % 64 == 0) warp_sums[threadIdx.x / 64] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < blockDim.x / 64; ++i) total += warp_sums[i];
    warp_sums[0] = rsqrtf(total / head_dim + eps);
  }
  __syncthreads();
  float inv = warp_sums[0];
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    orow[i] = __float2half(__half2float(xr[i]) * inv * __half2float(w[i]));
  }
}

__global__ void rope_kernel(__half* x, int tokens, int head_dim, int heads,
                            int row_stride, const int* row_pos,
                            const float* inv_freq) {
  // x: rows of [heads, head_dim] with `row_stride` elements between tokens;
  // one block per token. Each row carries its own absolute position, which
  // makes the kernel usable for batched multi-sequence decode.
  int t = blockIdx.x;
  int half = head_dim / 2;
  int64_t base = static_cast<int64_t>(t) * row_stride;
  float pos = static_cast<float>(row_pos[t]);
  for (int h = 0; h < heads; ++h) {
    __half* xr = x + base + static_cast<int64_t>(h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      float angle = pos * inv_freq[i];
      float c = cosf(angle), s = sinf(angle);
      float x1 = __half2float(xr[i]);
      float x2 = __half2float(xr[i + half]);
      xr[i] = __float2half(x1 * c - x2 * s);
      xr[i + half] = __float2half(x1 * s + x2 * c);
    }
  }
}

// Fused residual-add + RMSNorm. Pass 1 accumulates the squared sum of
// (hidden + residual); pass 2 writes both outputs. The second read of the
// inputs hits L2, so the fusion trades one full DRAM round-trip for cheap
// cache hits.
__global__ void fused_residual_rmsnorm_kernel(__half* hidden,
                                              const __half* residual,
                                              __half* normed, const __half* w,
                                              int size, float eps) {
  int64_t base = static_cast<int64_t>(blockIdx.x) * size;
  __half* h = hidden + base;
  const __half* r = residual + base;
  __half* n = normed + base;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float v = __half2float(h[i]) + __half2float(r[i]);
    acc += v * v;
  }
  acc = warp_reduce_sum(acc);
  __shared__ float warp_sums[kBlock / 64];
  if (threadIdx.x % 64 == 0) warp_sums[threadIdx.x / 64] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < blockDim.x / 64; ++i) total += warp_sums[i];
    warp_sums[0] = rsqrtf(total / size + eps);
  }
  __syncthreads();
  float inv = warp_sums[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float v = __half2float(h[i]) + __half2float(r[i]);
    h[i] = __float2half(v);
    n[i] = __float2half(v * inv * __half2float(w[i]));
  }
}

__global__ void silu_and_mul_kernel(__half* out, const __half* x, int inner) {
  int64_t row = blockIdx.x;
  const __half* xr = x + row * 2 * inner;
  __half* orow = out + row * inner;
  for (int i = threadIdx.x; i < inner; i += blockDim.x) {
    float g = __half2float(xr[i]);
    float u = __half2float(xr[i + inner]);
    orow[i] = __float2half(g / (1.0f + expf(-g)) * u);
  }
}

__global__ void residual_add_kernel(__half* x, const __half* residual,
                                    int64_t count) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    x[i] = __float2half(__half2float(x[i]) + __half2float(residual[i]));
  }
}

__global__ void embed_gather_kernel(__half* out, const __half* table,
                                    const int64_t* ids, int dim) {
  int t = blockIdx.y;
  int64_t id = ids[t];
  const __half* src = table + id * dim;
  __half* dst = out + static_cast<int64_t>(t) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    dst[i] = src[i];
  }
}

__global__ void softmax_causal_rows_kernel(float* s, int tq, int tk_total,
                                           int q_offset, float scale) {
  // One block per q row.
  int i = blockIdx.x;
  if (i >= tq) return;
  int limit = q_offset + i + 1;
  if (limit > tk_total) limit = tk_total;
  float* row = s + static_cast<int64_t>(i) * tk_total;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < limit; j += blockDim.x) {
    row[j] *= scale;
    m = fmaxf(m, row[j]);
  }
  __shared__ float red[kBlock];
  red[threadIdx.x] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < blockDim.x; ++k) red[0] = fmaxf(red[0], red[k]);
    red[1] = red[0];  // broadcast max
  }
  __syncthreads();
  m = red[1];
  float sum = 0.0f;
  for (int j = threadIdx.x; j < tk_total; j += blockDim.x) {
    float v = (j < limit) ? expf(row[j] - m) : 0.0f;
    row[j] = v;
    sum += v;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int k = 0; k < blockDim.x; ++k) total += red[k];
    red[1] = 1.0f / total;
  }
  __syncthreads();
  float inv = red[1];
  for (int j = threadIdx.x; j < tk_total; j += blockDim.x) row[j] *= inv;
}

__global__ void slice_cols_kernel(__half* dst, const __half* src, int dst_cols,
                                  int src_cols, int col_off, int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int r = static_cast<int>(i / dst_cols);
  int c = static_cast<int>(i % dst_cols);
  dst[i] = src[static_cast<int64_t>(r) * src_cols + col_off + c];
}

__global__ void add_kernel(__half* dst, const __half* src, int64_t count) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

__global__ void scale_rows_kernel(__half* x, const float* w, int dim) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  float s = w[row];
  __half* xr = x + row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    xr[i] = __float2half(__half2float(xr[i]) * s);
  }
}

__global__ void copy_strided_kernel(__half* dst, const __half* src,
                                    int64_t src_elem_off,
                                    int64_t src_row_stride, int64_t count,
                                    int dim) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = count * dim;
  if (i >= total) return;
  int64_t row = i / dim;
  int64_t col = i % dim;
  dst[i] = src[src_elem_off + row * src_row_stride + col];
}

}  // namespace

void rmsnorm_fp16(void* out, const void* x, const void* weight, int rows,
                  int size, float eps, hipStream_t stream) {
  int block = size < kBlock ? (size + 63) / 64 * 64 : kBlock;
  hipLaunchKernelGGL(rmsnorm_kernel, dim3(rows), dim3(block), 0, stream,
                     static_cast<__half*>(out), static_cast<const __half*>(x),
                     static_cast<const __half*>(weight), size, eps);
}

void rmsnorm_heads_fp16(void* out, const void* x, const void* weight, int rows,
                        int heads, int head_dim, int row_stride, float eps,
                        hipStream_t stream) {
  int block = head_dim < kBlock ? (head_dim + 63) / 64 * 64 : kBlock;
  int64_t grid = static_cast<int64_t>(rows) * heads;
  hipLaunchKernelGGL(rmsnorm_heads_kernel, dim3(grid), dim3(block), 0, stream,
                     static_cast<__half*>(out), static_cast<const __half*>(x),
                     static_cast<const __half*>(weight), heads, head_dim,
                     row_stride, eps);
}

void rope_fp16(void* x, int tokens, int heads, int head_dim, int row_stride,
               const void* row_pos, const void* inv_freq, hipStream_t stream) {
  int half = head_dim / 2;
  int block = half < kBlock ? (half + 63) / 64 * 64 : kBlock;
  hipLaunchKernelGGL(rope_kernel, dim3(tokens), dim3(block), 0, stream,
                     static_cast<__half*>(x), tokens, head_dim, heads,
                     row_stride, static_cast<const int*>(row_pos),
                     static_cast<const float*>(inv_freq));
}

void fused_residual_rmsnorm_fp16(void* hidden_io, const void* residual,
                                 void* normed, const void* weight, int rows,
                                 int size, float eps, hipStream_t stream) {
  int block = size < kBlock ? (size + 63) / 64 * 64 : kBlock;
  hipLaunchKernelGGL(fused_residual_rmsnorm_kernel, dim3(rows), dim3(block), 0,
                     stream, static_cast<__half*>(hidden_io),
                     static_cast<const __half*>(residual),
                     static_cast<__half*>(normed),
                     static_cast<const __half*>(weight), size, eps);
}

void silu_and_mul_fp16(void* out, const void* x, int rows, int inner,
                       hipStream_t stream) {
  hipLaunchKernelGGL(silu_and_mul_kernel, dim3(rows), dim3(kBlock), 0, stream,
                     static_cast<__half*>(out), static_cast<const __half*>(x),
                     inner);
}

void residual_add_fp16(void* x_and_out, const void* residual, int rows,
                       int size, hipStream_t stream) {
  int64_t count = static_cast<int64_t>(rows) * size;
  int blocks = static_cast<int>((count + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(residual_add_kernel, dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<__half*>(x_and_out),
                     static_cast<const __half*>(residual), count);
}

void embed_gather_fp16(void* out, const void* table, const int64_t* ids,
                       int tokens, int dim, hipStream_t stream) {
  hipLaunchKernelGGL(embed_gather_kernel, dim3(1, tokens), dim3(kBlock), 0,
                     stream, static_cast<__half*>(out),
                     static_cast<const __half*>(table), ids, dim);
}

void softmax_causal_rows_fp32(float* s, int tq, int tk_total, int q_offset,
                              float scale, hipStream_t stream) {
  hipLaunchKernelGGL(softmax_causal_rows_kernel, dim3(tq), dim3(kBlock), 0,
                     stream, s, tq, tk_total, q_offset, scale);
}

void slice_cols_fp16(void* dst, const void* src, int rows, int dst_cols,
                     int src_cols, int col_off, hipStream_t stream) {
  int64_t total = static_cast<int64_t>(rows) * dst_cols;
  int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(slice_cols_kernel, dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<__half*>(dst), static_cast<const __half*>(src),
                     dst_cols, src_cols, col_off, total);
}

void add_fp16(void* dst, const void* src, int64_t count, hipStream_t stream) {
  int blocks = static_cast<int>((count + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(add_kernel, dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<__half*>(dst), static_cast<const __half*>(src),
                     count);
}

void scale_rows_fp16(void* x, const float* row_weights, int n, int dim,
                     hipStream_t stream) {
  hipLaunchKernelGGL(scale_rows_kernel, dim3(n), dim3(kBlock), 0, stream,
                     static_cast<__half*>(x), row_weights, dim);
}

void copy_strided_fp16(void* dst, const void* src, int64_t src_elem_off,
                       int64_t src_row_stride, int count, int dim,
                       hipStream_t stream) {
  int64_t total = static_cast<int64_t>(count) * dim;
  int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(copy_strided_kernel, dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<__half*>(dst), static_cast<const __half*>(src),
                     src_elem_off, src_row_stride, total, dim);
}

}  // namespace ninfer
