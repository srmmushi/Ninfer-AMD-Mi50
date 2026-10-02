#pragma once
// HuggingFace config.json parsing for the supported architectures
// (Qwen2/Qwen3 dense + Qwen3-MoE), mirroring ninfer's model Config.

#include <string>

#include "common/json.h"

namespace ninfer {

struct ModelConfig {
  std::string arch;          // e.g. "Qwen3ForCausalLM"
  std::string model_type;    // e.g. "qwen3", "qwen3_moe"
  int vocab_size = 0;
  int hidden_size = 0;
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int intermediate_size = 0;
  int head_dim = 0;          // defaults to hidden_size / heads when absent
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  bool tie_word_embeddings = false;
  bool use_qk_norm = false;  // Qwen3 applies RMSNorm to Q/K per head

  // MoE fields (Qwen3-MoE).
  bool is_moe = false;
  int num_experts = 0;
  int num_experts_per_tok = 0;
  int moe_intermediate_size = 0;

  int ffn_inner() const {
    return is_moe ? moe_intermediate_size : intermediate_size;
  }

  static ModelConfig parse(const Json& j);
  static ModelConfig from_file(const std::string& path);
};

}  // namespace ninfer
