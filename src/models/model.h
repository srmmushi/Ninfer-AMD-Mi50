#pragma once
// Qwen2/Qwen3 (dense + MoE) model: weight materialization to FP16 device
// buffers and the prefill/decode forward pass. Mirrors ninfer's
// models/qwen3_5 Model + execution layer for the supported subset.

#include <memory>
#include <string>
#include <vector>

#include "models/config.h"

namespace ninfer {

class Blas;
class KVCache;
class DeviceBuffer;

struct ModelOptions {
  int max_context = 32768;
  int prefill_chunk = 512;
};

class Model {
 public:
  Model(const std::string& model_dir, const ModelOptions& options);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const ModelConfig& config() const { return config_; }
  int prefill_chunk() const { return prefill_chunk_; }
  int max_context() const { return max_context_; }

  // Runs `tokens` (host, count <= prefill_chunk) at absolute positions
  // [pos0, pos0+count). Returns device fp32 logits [vocab_size] of the final
  // position. The stream is synchronized by the caller as needed.
  const float* forward(const std::vector<int64_t>& tokens, int pos0,
                       KVCache& kv, hipStream_t stream);

  size_t weight_bytes() const { return weight_bytes_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  ModelConfig config_;
  int prefill_chunk_;
  int max_context_;
  size_t weight_bytes_ = 0;
};

}  // namespace ninfer
