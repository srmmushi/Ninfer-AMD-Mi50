#include "runtime/kv_cache.h"

#include <stdexcept>

#include "core/device.h"
#include "ops/ops.h"

namespace ninfer {

KVCache::KVCache(int layers, int max_context, int kv_heads, int head_dim,
                 int num_sequences)
    : layers_(layers), max_context_(max_context), kv_heads_(kv_heads),
      head_dim_(head_dim), num_sequences_(num_sequences) {
  size_t row_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(uint16_t);
  size_t bytes =
      row_bytes * max_context * static_cast<size_t>(num_sequences);
  k_.resize(layers);
  v_.resize(layers);
  for (int l = 0; l < layers; ++l) {
    HIP_CHECK(hipMalloc(&k_[l], bytes));
    HIP_CHECK(hipMalloc(&v_[l], bytes));
  }
  length_.assign(num_sequences, 0);
}

KVCache::~KVCache() {
  for (void* p : k_) (void)hipFree(p);
  for (void* p : v_) (void)hipFree(p);
}

void KVCache::append(int layer, int seq, int pos, const void* src_k,
                     const void* src_v, int64_t src_row_offset,
                     int64_t src_elem_off_k, int64_t src_elem_off_v,
                     int64_t src_row_stride, int count, hipStream_t stream) {
  if (pos + count > max_context_) {
    throw std::runtime_error("KV cache overflow: context exceeds capacity");
  }
  int dim = kv_heads_ * head_dim_;
  int64_t dst_off = static_cast<int64_t>(seq) * seq_stride_elems() +
                    static_cast<int64_t>(pos) * dim;
  copy_strided_fp16(static_cast<char*>(k_[layer]) + dst_off * sizeof(uint16_t),
                    src_k, src_row_offset * src_row_stride + src_elem_off_k,
                    src_row_stride, count, dim, stream);
  copy_strided_fp16(static_cast<char*>(v_[layer]) + dst_off * sizeof(uint16_t),
                    src_v, src_row_offset * src_row_stride + src_elem_off_v,
                    src_row_stride, count, dim, stream);
}

size_t KVCache::bytes() const {
  size_t row_bytes = static_cast<size_t>(kv_heads_) * head_dim_ * sizeof(uint16_t);
  return 2ull * layers_ * row_bytes * max_context_ * num_sequences_;
}

}  // namespace ninfer
