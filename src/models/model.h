#pragma once
// Qwen2/Qwen3 (dense + MoE) model: weight materialization to FP16 device
// buffers and the prefill/decode forward pass. Mirrors ninfer's
// models/qwen3_5 Model + execution layer for the supported subset.
//
// Multi-GPU: the transformer layers are split into contiguous shards, one
// per device (pipeline / layer parallelism). Shard 0 owns the embedding,
// the last shard owns the final norm + LM head. Activations cross the shard
// boundary via hipMemcpyPeerAsync (PCIe P2P when the platform allows it).
//
// Batching: forward_batch() accepts one row per input token, each row
// carrying (token, absolute position, sequence slot). Rows of the same
// sequence form a group and share that sequence's KV slab — this is the
// continuous-batching entry point used by the C API / vLLM integration.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "models/config.h"

namespace ninfer {

class Blas;
class KVCache;
class DeviceBuffer;

struct ModelOptions {
  int max_context = 32768;
  int prefill_chunk = 512;
  // Devices to spread transformer layers across (pipeline parallelism).
  // {0} = single GPU; {0,1} = two-way layer split.
  std::vector<int> gpu_ids = {0};
  // Concurrent sequences (KV slabs) reserved at startup.
  int num_sequences = 1;
};

struct ForwardOutput {
  const float* logits = nullptr;  // device fp32 [rows, vocab_size]
  int rows = 0;                   // rows == number of row groups
  // Sequence slot for each output row (order of first appearance).
  std::vector<int> group_seq;
};

class Model {
 public:
  Model(const std::string& model_dir, const ModelOptions& options);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const ModelConfig& config() const;
  int prefill_chunk() const { return prefill_chunk_; }
  int max_context() const { return max_context_; }
  int device_count() const;
  int num_sequences() const;

  // Resets all per-shard KV caches (start of a new generation, all slots).
  void reset_cache();
  // Resets one sequence slot only (request-level recycle).
  void reset_cache(int seq);

  // Batched forward: `tokens[i]` sits at absolute `positions[i]` inside KV
  // slot `seq_ids[i]`. Rows must be grouped by sequence. Returns logits for
  // the LAST row of each sequence group.
  ForwardOutput forward_batch(const std::vector<int64_t>& tokens,
                              const std::vector<int>& positions,
                              const std::vector<int>& seq_ids,
                              hipStream_t stream);

  // Single-sequence convenience wrapper (slot 0, consecutive positions).
  const float* forward(const std::vector<int64_t>& tokens, int pos0,
                       hipStream_t stream);

  int output_device() const;

  size_t weight_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int prefill_chunk_;
  int max_context_;
};

}  // namespace ninfer
