#pragma once
// Multi-sequence contiguous FP16 KV cache.
// Per layer: one slab of [num_sequences, max_context, kv_heads * head_dim],
// so a batched kernel can address any sequence with a single base pointer
// plus seq id * seq_stride_elems (no device pointer arrays needed).

#include <vector>

#include <hip/hip_runtime.h>

namespace ninfer {

class KVCache {
 public:
  KVCache(int layers, int max_context, int kv_heads, int head_dim,
          int num_sequences = 1);
  ~KVCache();
  KVCache(const KVCache&) = delete;
  KVCache& operator=(const KVCache&) = delete;

  int max_context() const { return max_context_; }
  int num_sequences() const { return num_sequences_; }
  int layers() const { return layers_; }
  // Elements between consecutive sequences within one layer slab.
  int64_t seq_stride_elems() const {
    return static_cast<int64_t>(max_context_) * kv_heads_ * head_dim_;
  }
  void reset() { length_.assign(num_sequences_, 0); }

  void* k_base(int layer) { return k_[layer]; }
  void* v_base(int layer) { return v_[layer]; }
  const void* k_base(int layer) const { return k_[layer]; }
  const void* v_base(int layer) const { return v_[layer]; }

  void* k_data(int layer, int seq) {
    return static_cast<char*>(k_[layer]) +
           static_cast<int64_t>(seq) * seq_stride_elems() * sizeof(uint16_t);
  }
  void* v_data(int layer, int seq) {
    return static_cast<char*>(v_[layer]) +
           static_cast<int64_t>(seq) * seq_stride_elems() * sizeof(uint16_t);
  }

  // Appends rows [count, kv_heads*head_dim] from a strided source buffer
  // (the k/v sections of a fused qkv row) at position `pos` of sequence
  // `seq`. `src_row_offset` selects the first source row.
  void append(int layer, int seq, int pos, const void* src_k, const void* src_v,
              int64_t src_row_offset, int64_t src_elem_off_k,
              int64_t src_elem_off_v, int64_t src_row_stride, int count,
              hipStream_t stream);

  int length(int seq) const { return length_[seq]; }
  void set_length(int seq, int n) { length_[seq] = n; }

  size_t bytes() const;

 private:
  int layers_, max_context_, kv_heads_, head_dim_, num_sequences_;
  std::vector<int> length_;
  std::vector<void*> k_;  // one slab per layer, all sequences
  std::vector<void*> v_;
};

}  // namespace ninfer
