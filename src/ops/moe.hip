#include "ops/ops.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "core/device.h"

namespace ninfer {
namespace {

__global__ void gather_rows_kernel(__half* dst, const __half* src,
                                   const int64_t* order, int dim) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  int64_t src_row = order[row];
  const __half* s = src + src_row * dim;
  __half* d = dst + row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) d[i] = s[i];
}

__global__ void scatter_add_rows_kernel(__half* dst, const __half* src,
                                        const int64_t* order, int dim) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  int64_t dst_row = order[row];
  const __half* s = src + row * dim;
  __half* d = dst + dst_row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    d[i] = __float2half(__half2float(d[i]) + __half2float(s[i]));
  }
}

}  // namespace

void gather_rows_fp16(void* dst, const void* src, const int64_t* order, int n,
                      int dim, hipStream_t stream) {
  hipLaunchKernelGGL(gather_rows_kernel, dim3(n), dim3(256), 0, stream,
                     static_cast<__half*>(dst), static_cast<const __half*>(src),
                     order, dim);
}

void scatter_add_rows_fp16(void* dst, const void* src, const int64_t* order,
                           int n, int dim, hipStream_t stream) {
  hipLaunchKernelGGL(scatter_add_rows_kernel, dim3(n), dim3(256), 0, stream,
                     static_cast<__half*>(dst), static_cast<const __half*>(src),
                     order, dim);
}

MoERoutingPlan moe_route_cpu(const float* logits_device, int tokens,
                             int experts, int top_k, Blas& /*blas*/,
                             hipStream_t stream) {
  std::vector<float> logits(static_cast<size_t>(tokens) * experts);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(logits.data(), logits_device,
                      logits.size() * sizeof(float), hipMemcpyDeviceToHost));

  MoERoutingPlan plan;
  plan.selected.assign(static_cast<size_t>(tokens) * top_k, 0);
  plan.weights.assign(static_cast<size_t>(tokens) * top_k, 0.0f);
  plan.expert_offsets.assign(experts + 1, 0);

  // softmax + top-k per token, then count expert loads.
  std::vector<int> counts(experts, 0);
  std::vector<float> prob(experts);
  for (int t = 0; t < tokens; ++t) {
    const float* row = logits.data() + static_cast<size_t>(t) * experts;
    float m = *std::max_element(row, row + experts);
    float sum = 0.0f;
    for (int e = 0; e < experts; ++e) {
      prob[e] = std::exp(row[e] - m);
      sum += prob[e];
    }
    std::vector<int> idx(experts);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return prob[a] > prob[b]; });
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) wsum += prob[idx[k]];
    for (int k = 0; k < top_k; ++k) {
      int e = idx[k];
      plan.selected[static_cast<size_t>(t) * top_k + k] = e;
      plan.weights[static_cast<size_t>(t) * top_k + k] = prob[e] / wsum;
      ++counts[e];
    }
  }

  // Execution order grouped by expert (stable: token order preserved).
  plan.expert_offsets[0] = 0;
  for (int e = 0; e < experts; ++e) {
    plan.expert_offsets[e + 1] = plan.expert_offsets[e] + counts[e];
  }
  std::vector<int64_t> slot(experts);
  for (int e = 0; e < experts; ++e) slot[e] = plan.expert_offsets[e];
  plan.order.resize(static_cast<size_t>(tokens) * top_k);
  plan.order_weights.resize(static_cast<size_t>(tokens) * top_k);
  for (int t = 0; t < tokens; ++t) {
    for (int k = 0; k < top_k; ++k) {
      int e = plan.selected[static_cast<size_t>(t) * top_k + k];
      plan.order[slot[e]] = t;
      plan.order_weights[slot[e]] =
          plan.weights[static_cast<size_t>(t) * top_k + k];
      ++slot[e];
    }
  }
  return plan;
}

}  // namespace ninfer
