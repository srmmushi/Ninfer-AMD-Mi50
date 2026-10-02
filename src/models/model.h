#pragma once
// Qwen2/Qwen3 (dense + MoE) model: weight materialization to FP16 device
// buffers and the prefill/decode forward pass. Mirrors ninfer's
// models/qwen3_5 Model + execution layer for the supported subset.
//
// Multi-GPU: the transformer layers are split into contiguous shards, one
// per device (pipeline / layer parallelism). Shard 0 owns the embedding,
// the last shard owns the final norm + LM head. Activations cross the shard
// boundary via hipMemcpyPeerAsync (PCIe P2P when the platform allows it).

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

  // Resets all per-shard KV caches (start of a new generation).
  void reset_cache();

  // Runs `tokens` (host, count <= prefill_chunk) at absolute positions
  // [pos0, pos0+count). Returns device fp32 logits [vocab_size] residing on
  // output_device(); the caller must hipSetDevice(output_device()) before
  // touching them.
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
