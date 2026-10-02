#include "models/config.h"

#include <stdexcept>

#include "common/log.h"
#include "common/util.h"

namespace ninfer {

ModelConfig ModelConfig::parse(const Json& j) {
  ModelConfig c;
  c.arch = j.find("architectures") && j.at("architectures").items().size() > 0
               ? j.at("architectures").at(0).as_string()
               : std::string();
  c.model_type = j.find("model_type") ? j.at("model_type").as_string() : "";

  c.vocab_size = static_cast<int>(j.at("vocab_size").as_int());
  c.hidden_size = static_cast<int>(j.at("hidden_size").as_int());
  c.num_hidden_layers = static_cast<int>(j.at("num_hidden_layers").as_int());
  c.num_attention_heads = static_cast<int>(j.at("num_attention_heads").as_int());
  c.num_key_value_heads = j.find("num_key_value_heads")
                              ? static_cast<int>(j.at("num_key_value_heads").as_int())
                              : c.num_attention_heads;
  c.intermediate_size = static_cast<int>(j.at("intermediate_size").as_int());
  c.head_dim = j.find("head_dim")
                   ? static_cast<int>(j.at("head_dim").as_int())
                   : c.hidden_size / c.num_attention_heads;
  c.rms_norm_eps = static_cast<float>(j.at("rms_norm_eps").as_number(1e-6));
  c.rope_theta = static_cast<float>(
      j.find("rope_theta") ? j.at("rope_theta").as_number(1e6) : 1e6);
  c.tie_word_embeddings =
      j.find("tie_word_embeddings") ? j.at("tie_word_embeddings").as_bool() : false;

  const std::string& mt = c.model_type;
  c.use_qk_norm = (mt == "qwen3" || mt == "qwen3_moe");
  c.is_moe = (mt == "qwen3_moe" || mt == "qwen2_moe");
  if (j.find("num_experts")) {
    c.is_moe = true;
    c.num_experts = static_cast<int>(j.at("num_experts").as_int());
    c.num_experts_per_tok = static_cast<int>(
        j.find("num_experts_per_tok") ? j.at("num_experts_per_tok").as_int() : 8);
    c.moe_intermediate_size = static_cast<int>(j.at("moe_intermediate_size").as_int());
  }

  if (c.vocab_size <= 0 || c.hidden_size <= 0 || c.num_hidden_layers <= 0 ||
      c.head_dim <= 0) {
    throw std::runtime_error("incomplete/invalid config.json fields");
  }
  if (c.is_moe && (c.num_experts <= 0 || c.moe_intermediate_size <= 0)) {
    throw std::runtime_error("MoE config missing num_experts / "
                             "moe_intermediate_size");
  }
  LOG_DEBUG("config: %s %s layers=%d hidden=%d heads=%d kv=%d head_dim=%d moe=%d",
            c.arch.c_str(), c.model_type.c_str(), c.num_hidden_layers,
            c.hidden_size, c.num_attention_heads, c.num_key_value_heads,
            c.head_dim, c.is_moe);
  return c;
}

ModelConfig ModelConfig::from_file(const std::string& path) {
  return parse(Json::parse(read_file_text(path)));
}

}  // namespace ninfer
