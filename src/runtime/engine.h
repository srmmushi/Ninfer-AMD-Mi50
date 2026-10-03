#pragma once
// Generation engine: chunked prefill -> decode loop with CPU sampling.
// The facade mirrors ninfer's include/ninfer/engine.h Engine in miniature.

#include <functional>
#include <string>
#include <vector>

#include "frontend/chat.h"
#include "frontend/tokenizer.h"
#include "models/model.h"

namespace ninfer {

struct SamplingParams {
  float temperature = 0.7f;
  int top_k = 50;
  float top_p = 0.95f;
  float repetition_penalty = 1.05f;
  int64_t seed = 0;             // 0 = nondeterministic
  int max_new_tokens = 512;
};

using TokenCallback = std::function<void(const std::string& delta)>;

// Standalone sampler shared by the engine and the C API / vLLM integration:
// repetition penalty -> temperature -> top-k -> top-p -> multinomial.
// `recent` carries the tokens already produced (empty for stateless calls).
int64_t sample_logits(const float* logits, int vocab, const SamplingParams& sp,
                      const std::vector<int64_t>& recent, uint64_t seed);

class Engine {
 public:
  Engine(const std::string& model_dir, const ModelOptions& options,
         bool enable_thinking = true);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const ModelConfig& config() const { return model_->config(); }
  const Tokenizer& tokenizer() const { return *tokenizer_; }
  Model& model() { return *model_; }

  std::vector<int64_t> tokenize(const std::string& text) const;

  // Raw-prompt generation (already templated by the caller).
  std::string generate(const std::string& prompt, const SamplingParams& sp,
                       const TokenCallback& on_token = nullptr);

  // Chat-message generation through the ChatML template.
  std::string generate_chat(const std::vector<ChatMessage>& messages,
                            const SamplingParams& sp,
                            const TokenCallback& on_token = nullptr);

 private:
  std::string run(std::vector<int64_t> ids, const SamplingParams& sp,
                  const TokenCallback& on_token);

  std::unique_ptr<Model> model_;
  std::unique_ptr<Tokenizer> tokenizer_;
  ChatRenderer renderer_;
  ModelOptions opt_;                // stored so the decode loop can see use_graph
  void* pinned_logits_ = nullptr;  // page-locked logits staging buffer
};

}  // namespace ninfer
