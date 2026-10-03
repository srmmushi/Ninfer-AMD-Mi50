#pragma once
// Qwen2/Qwen3 (dense + MoE) model: weight materialization to device buffers
// and the prefill/decode forward pass. Mirrors ninfer's models/qwen3_5 Model
// + execution layer for the supported subset.
//
// Execution modes:
//   pp (default): transformer layers split into contiguous shards, one per
//                 device (pipeline parallelism). Shard 0 owns the embedding,
//                 the last shard owns the final norm + LM head.
//   tp:           tensor parallelism — every device holds all layers with
//                 row/column-split weights (Megatron layout) and the partial
//                 o_proj/down outputs are all-reduced over PCIe.
//
// Weight format: fp16 (default) or groupwise INT4 ("q4"), quantized at load
// time (gfx906 has no FP4/FP8 hardware — INT4 is dequantized during GEMM).
//
// Batching: forward_batch() takes one row per input token, each row carrying
// (token, absolute position, sequence slot). Rows of the same sequence form a
// group sharing that sequence's KV slab — the continuous-batching entry point
// used by the C API / vLLM integration.

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
  std::vector<int> gpu_ids = {0};
  int num_sequences = 1;
  // "fp16" or "q4" (groupwise INT4, ~4x less weight traffic).
  std::string quant = "fp16";
  // "pp" = layer-split pipeline, "tp" = tensor parallel (dense models).
  std::string parallel = "pp";
};

struct ForwardOutput {
  const float* logits = nullptr;  // device fp32 [rows, vocab_size]
  int rows = 0;
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

  void reset_cache();
  void reset_cache(int seq);

  ForwardOutput forward_batch(const std::vector<int64_t>& tokens,
                              const std::vector<int>& positions,
                              const std::vector<int>& seq_ids,
                              hipStream_t stream);

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
