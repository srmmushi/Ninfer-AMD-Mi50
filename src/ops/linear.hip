#include "ops/ops.h"

#include <stdexcept>

#include "common/log.h"

namespace ninfer {

Blas::Blas() {
  if (rocblas_create_handle(&handle_) != rocblas_status_success) {
    throw std::runtime_error("rocblas_create_handle failed");
  }
}

Blas::~Blas() {
  if (handle_) rocblas_destroy_handle(handle_);
}

// Row-major Y[M,N] = X[M,K] * W[N,K]^T, viewed column-major as
// Y_cm[N,M] = W_cm^T[N,K] * X_cm[K,M].
static void run_gemm(rocblas_handle h, rocblas_operation ta, rocblas_operation tb,
                     int m, int n, int k, const void* a, int lda, const void* b,
                     int ldb, void* c, int ldc, rocblas_datatype c_type,
                     hipStream_t stream) {
  float alpha = 1.0f, beta = 0.0f;
  rocblas_set_stream(h, stream);
  auto st = rocblas_gemm_ex(
      h, ta, tb, m, n, k, &alpha, a, rocblas_datatype_f16_r, lda, b,
      rocblas_datatype_f16_r, ldb, &beta, c, c_type, ldc, c, c_type, ldc,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
  if (st != rocblas_status_success) {
    throw std::runtime_error(std::string("rocblas_gemm_ex failed: ") +
                             rocblas_status_to_string(st));
  }
}

void Blas::gemm_fp16(const void* x, const void* w, void* out, int M, int N,
                     int K, hipStream_t stream) {
  // Y_cm[N,M] = W[N,K] (row-major == col-major [K,N], op T) * X[K,M]
  run_gemm(handle_, rocblas_operation_transpose, rocblas_operation_none, N, M, K,
           w, /*lda=*/K, x, /*ldb=*/K, out, /*ldc=*/N, rocblas_datatype_f16_r,
           stream);
}

void Blas::gemm_fp32out(const void* x, const void* w, void* out, int M, int N,
                        int K, hipStream_t stream) {
  run_gemm(handle_, rocblas_operation_transpose, rocblas_operation_none, N, M, K,
           w, K, x, K, out, N, rocblas_datatype_f32_r, stream);
}

void Blas::gemm_view(const void* a, int lda, bool trans_a, const void* b,
                     int ldb, bool trans_b, void* c, int ldc, int M, int N,
                     int K, bool c_fp32, hipStream_t stream) {
  run_gemm(handle_,
           trans_a ? rocblas_operation_transpose : rocblas_operation_none,
           trans_b ? rocblas_operation_transpose : rocblas_operation_none, M, N,
           K, a, lda, b, ldb, c, ldc,
           c_fp32 ? rocblas_datatype_f32_r : rocblas_datatype_f16_r, stream);
}

void Blas::gemm_batched_fp16(const void* a, const void* b, void* out, int M,
                             int N, int K, int64_t stride_a, int64_t stride_b,
                             int64_t stride_c, int batch, hipStream_t stream) {
  float alpha = 1.0f, beta = 0.0f;
  rocblas_set_stream(handle_, stream);
  auto st = rocblas_gemm_strided_batched_ex(
      handle_, rocblas_operation_transpose, rocblas_operation_none, N, M, K,
      &alpha, a, rocblas_datatype_f16_r, K, stride_a, b,
      rocblas_datatype_f16_r, K, stride_b, &beta, out, rocblas_datatype_f16_r,
      N, stride_c, out, rocblas_datatype_f16_r, N, stride_c,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0, batch);
  if (st != rocblas_status_success) {
    throw std::runtime_error(std::string("rocblas_gemm_strided_batched_ex "
                                         "failed: ") +
                             rocblas_status_to_string(st));
  }
}

}  // namespace ninfer
