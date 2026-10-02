#pragma once
// Per-layer contiguous FP16 KV cache (simplified from ninfer's paged KV
// pool: single resident sequence, startup-fixed capacity).

#include <vector>

#include <hip/hip_runtime.h>

namespace ninfer {

class DeviceBuffer;

class KVCache {
 public:
  KVCache(int layers, int max_context, int kv_heads, int head_dim);
  ~KVCache();
  KVCache(const KVCache&) = delete;
  KVCache& operator=(const KVCache&) = delete;

  int max_context() const { return max_context_; }
  void reset() { length_ = 0; }

  void* k_data(int layer) { return k_[layer]; }
  void* v_data(int layer) { return v_[layer]; }
  const void* k_data(int layer) const { return k_[layer]; }
  const void* v_data(int layer) const { return v_[layer]; }

  // Appends rows [count, kv_heads*head_dim] taken from a strided source
  // buffer (e.g. the k/v sections of a fused qkv row) at position `pos`.
  void append(int layer, int pos, const void* src_k, const void* src_v,
              int64_t src_elem_off_k, int64_t src_elem_off_v,
              int64_t src_row_stride, int count, hipStream_t stream);

  int length() const { return length_; }
  void set_length(int n) { length_ = n; }

  size_t bytes() const;

 private:
  int layers_;
  int max_context_;
  int kv_heads_;
  int head_dim_;
  int length_ = 0;
  std::vector<void*> k_;
  std::vector<void*> v_;
};

}  // namespace ninfer
