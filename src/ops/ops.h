#pragma once
// GPU operator surface (mirrors ninfer include/ninfer/ops/*): every device
// entry point the model forward pass may call. Storage is FP16, accumulation
// FP32 (gfx906 has no BF16 units).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#include <cstdint>
#include <vector>

namespace ninfer {

// ---------------------------------------------------------------------------
// GEMM (rocBLAS backends), all row-major on the caller side:
//   out[M,N] = x[M,K] * W[N,K]^T   (weights stored [N,K] like torch Linear)
// ---------------------------------------------------------------------------
class Blas {
 public:
  Blas();
  ~Blas();
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;

  // out fp16, accumulate fp32.
  void gemm_fp16(const void* x, const void* w, void* out, int M, int N, int K,
                 hipStream_t stream = 0);
  // logits-style: out fp32, inputs fp16.
  void gemm_fp32out(const void* x, const void* w, void* out, int M, int N, int K,
                    hipStream_t stream = 0);

  // Generic column-major entry (used by attention views with arbitrary
  // strides): C[M,N] = op(A)[M,K] * op(B)[K,N]; C is fp16 or fp32.
  void gemm_view(const void* a, int lda, bool trans_a, const void* b, int ldb,
                 bool trans_b, void* c, int ldc, int M, int N, int K,
                 bool c_fp32, hipStream_t stream = 0);

  rocblas_handle handle() const { return handle_; }

 private:
  rocblas_handle handle_ = nullptr;
};

// ---------------------------------------------------------------------------
// Elementwise / norm / rope / activation kernels. All layout row-major.
// ---------------------------------------------------------------------------
void rmsnorm_fp16(void* out, const void* x, const void* weight, int rows,
                  int size, float eps, hipStream_t stream = 0);

// Per-head RMSNorm over head_dim (Qwen3 q_norm/k_norm): rows are [T, H, D]
// with explicit token stride `row_stride` (fused-qkv aware).
void rmsnorm_heads_fp16(void* out, const void* x, const void* weight, int rows,
                        int heads, int head_dim, int row_stride, float eps,
                        hipStream_t stream = 0);

// RoPE (half-split rotate_half style) on [T, H, D] with explicit token
// stride; positions start at `pos_offset`. `inv_freq` is a device table of
// head_dim/2 FP32 frequencies (precomputed once — avoids per-element powf).
void rope_fp16(void* x, int tokens, int heads, int head_dim, int row_stride,
               int pos_offset, const void* inv_freq, hipStream_t stream = 0);

// Fused residual add + RMSNorm (one pass over the activation instead of two):
//   hidden = hidden + residual        (written back, keeps the residual chain)
//   normed = rmsnorm(hidden) * weight
void fused_residual_rmsnorm_fp16(void* hidden_io, const void* residual,
                                 void* normed, const void* weight, int rows,
                                 int size, float eps, hipStream_t stream = 0);

// SwiGLU: x [T, 2*I] -> out [T, I]; out = silu(x[:, :I]) * x[:, I:].
void silu_and_mul_fp16(void* out, const void* x, int rows, int inner,
                       hipStream_t stream = 0);

void residual_add_fp16(void* x_and_out, const void* residual, int rows, int size,
                       hipStream_t stream = 0);

void embed_gather_fp16(void* out, const void* table, const int64_t* ids,
                       int tokens, int dim, hipStream_t stream = 0);

// Causal softmax over the prefill score scratch of one head (legacy path,
// retained for reference; the fused kernel below supersedes it).
void softmax_causal_rows_fp32(float* s, int tq, int tk_total, int q_offset,
                              float scale, hipStream_t stream = 0);

// ---------------------------------------------------------------------------
// MoE helpers: token gather/scatter driven by a host-computed routing plan.
// ---------------------------------------------------------------------------
void gather_rows_fp16(void* dst, const void* src, const int64_t* order, int n,
                      int dim, hipStream_t stream = 0);
void scatter_add_rows_fp16(void* dst, const void* src, const int64_t* order,
                           int n, int dim, hipStream_t stream = 0);
void scale_rows_fp16(void* x, const float* row_weights, int n, int dim,
                     hipStream_t stream = 0);

// Strided row copy, used to append K/V rows into the cache:
//   dst assumed packed [count, dim]; src rows `count x dim` with byte-row
//   stride `src_row_stride` (in elements) and element offset `src_elem_off`.
void copy_strided_fp16(void* dst, const void* src, int64_t src_elem_off,
                       int64_t src_row_stride, int count, int dim,
                       hipStream_t stream = 0);

// Host-side router: softmax over `experts` logits rows then top-k.
//   logits: [tokens, experts] fp32 (device memory, copied to host).
//   Returns per-token selected expert ids (row-major [tokens, top_k]) and
//   the execution order (expert-grouped token indices) with per-expert
//   offsets. `order`/`offsets` are host vectors; upload them yourself.
struct MoERoutingPlan {
  std::vector<int64_t> order;                  // token indices grouped by expert
  std::vector<float> order_weights;            // routing weight per order row
  std::vector<int64_t> expert_offsets;         // size experts+1
  std::vector<int32_t> selected;               // [tokens, top_k]
  std::vector<float> weights;                  // [tokens, top_k] (renormalized)
};
MoERoutingPlan moe_route_cpu(const float* logits_device, int tokens, int experts,
                             int top_k, Blas& blas, hipStream_t stream = 0);

// ---------------------------------------------------------------------------
// Decode-path GEMV (M == 1). rocBLAS is kept for prefill (M > 1); at M = 1 a
// custom wave64 GEMV with LDS-staged activations matches DRAM bandwidth and
// allows kernel-count fusion.
// ---------------------------------------------------------------------------
// y = W(N,K) · x(K): fp16 out.
void gemv_fp16(void* y, const void* w, const void* x, int n_rows, int k_cols,
               hipStream_t stream = 0);
// y = W(N,K) · x(K): fp32 out (router / LM head).
void gemv_fp32out(void* y, const void* w, const void* x, int n_rows, int k_cols,
                  hipStream_t stream = 0);

// Decode post-QKV fusion, ONE kernel replacing qk-norm ×2 + rope ×2 + KV
// append ×2 (six launches): per-head blocks normalize (optional Qwen3
// QK-norm), apply RoPE, and write K/V rows directly into the cache at pos0.
// q is written back to the qkv buffer (rotated).
void decode_post_qkv_fp16(void* qkv, void* k_cache, void* v_cache,
                          const void* inv_freq, int pos0, int heads,
                          int kv_heads, int head_dim, bool has_qknorm,
                          const void* qn_weight, const void* kn_weight,
                          float eps, hipStream_t stream = 0);

// ---------------------------------------------------------------------------
// Attention.
// ---------------------------------------------------------------------------
// Fused GQA decode for one sequence: out[H, D], q[H, D], cache [T, Hkv, D].
void attention_decode_fp16(void* out, const void* q, const void* k_cache,
                           const void* v_cache, int kv_len, int heads,
                           int kv_heads, int head_dim, float scale,
                           hipStream_t stream = 0);

// Fused GQA prefill for ALL heads in a single kernel (flash-style online
// softmax, K/V tiles staged through LDS): removes the per-head GEMM/softmax
// launch storm and the score scratch entirely.
//   out: [q_len, heads, D]; q: [q_len, heads, D] view with row stride
//   q_row_stride; caches: [kv_len, kv_heads, D]; causal mask on absolute
//   positions pos0..pos0+q_len-1.
void attention_prefill_fused_fp16(void* out, const void* q, const void* k_cache,
                                  const void* v_cache, int q_len, int kv_len,
                                  int pos0, int heads, int kv_heads,
                                  int head_dim, int q_row_stride,
                                  int out_row_stride, float scale,
                                  hipStream_t stream = 0);

}  // namespace ninfer
