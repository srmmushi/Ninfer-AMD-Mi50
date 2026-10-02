#include "runtime/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "common/log.h"
#include "common/util.h"
#include "core/device.h"

namespace ninfer {
namespace {

// CPU sampler: repetition penalty -> temperature -> top-k -> top-p ->
// multinomial. Vocab-sized work per token, negligible vs. the GPU step.
int64_t sample_token(const float* logits, int vocab, const SamplingParams& sp,
                     const std::vector<int64_t>& recent, std::mt19937_64& rng) {
  std::vector<float> l(logits, logits + vocab);

  if (sp.repetition_penalty != 1.0f) {
    for (int64_t id : recent) {
      if (id < 0 || id >= vocab) continue;
      float& v = l[id];
      v = v > 0 ? v / sp.repetition_penalty : v * sp.repetition_penalty;
    }
  }

  if (sp.temperature <= 0.0f) {
    return static_cast<int64_t>(
        std::max_element(l.begin(), l.end()) - l.begin());
  }

  for (float& v : l) v /= sp.temperature;

  // top-k filter.
  int k = std::min<int>(std::max(sp.top_k, 1), vocab);
  std::vector<int> idx(vocab);
  for (int i = 0; i < vocab; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return l[a] > l[b]; });
  idx.resize(k);

  // softmax over the kept candidates.
  float m = l[idx[0]];
  float sum = 0.0f;
  std::vector<float> prob(k);
  for (int i = 0; i < k; ++i) {
    prob[i] = std::exp(l[idx[i]] - m);
    sum += prob[i];
  }
  for (float& p : prob) p /= sum;

  // top-p (nucleus) truncation: candidates are already sorted desc.
  if (sp.top_p < 1.0f) {
    float cum = 0.0f;
    int keep = k;
    for (int i = 0; i < k; ++i) {
      cum += prob[i];
      if (cum >= sp.top_p) {
        keep = i + 1;
        break;
      }
    }
    idx.resize(keep);
    prob.resize(keep);
    float s2 = 0.0f;
    for (float p : prob) s2 += p;
    for (float& p : prob) p /= s2;
  }

  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  float r = uni(rng);
  float cum = 0.0f;
  for (int i = 0; i < static_cast<int>(prob.size()); ++i) {
    cum += prob[i];
    if (r <= cum) return idx[i];
  }
  return idx.back();
}

}  // namespace

Engine::Engine(const std::string& model_dir, const ModelOptions& options,
               bool enable_thinking)
    : renderer_(enable_thinking) {
  init_device();
  tokenizer_ = std::make_unique<Tokenizer>(path_join(model_dir, "tokenizer.json"));
  model_ = std::make_unique<Model>(model_dir, options);
  const ModelConfig& c = model_->config();
  kv_ = std::make_unique<KVCache>(c.num_hidden_layers, options.max_context,
                                  c.num_key_value_heads, c.head_dim);
  LOG_INFO("engine ready: vocab=%d layers=%d ctx=%d chunk=%d",
           c.vocab_size, c.num_hidden_layers, options.max_context,
           options.prefill_chunk);
}

Engine::~Engine() = default;

std::vector<int64_t> Engine::tokenize(const std::string& text) const {
  return tokenizer_->encode(text);
}

std::string Engine::generate(const std::string& prompt,
                             const SamplingParams& sp,
                             const TokenCallback& on_token) {
  return run(tokenizer_->encode(prompt), sp, on_token);
}

std::string Engine::generate_chat(const std::vector<ChatMessage>& messages,
                                  const SamplingParams& sp,
                                  const TokenCallback& on_token) {
  std::string prompt = renderer_.render(messages, /*add_generation_prompt=*/true);
  return run(tokenizer_->encode_with_special(prompt), sp, on_token);
}

std::string Engine::run(std::vector<int64_t> ids, const SamplingParams& sp,
                        const TokenCallback& on_token) {
  if (ids.empty()) throw std::runtime_error("empty prompt");
  if (static_cast<int>(ids.size()) > kv_->max_context() - sp.max_new_tokens) {
    throw std::runtime_error("prompt + max_new_tokens exceeds --max-context");
  }

  hipStream_t stream = 0;
  std::mt19937_64 rng(sp.seed != 0 ? sp.seed : std::random_device{}());
  const int V = model_->config().vocab_size;
  std::vector<int64_t> generated;
  std::string prev_text;

  // ---- Chunked prefill; the final chunk's logits seed the first token.
  const float* logits_dev = nullptr;
  const int chunk = model_->prefill_chunk();
  int pos = 0;
  while (pos < static_cast<int>(ids.size())) {
    int end = std::min(pos + chunk, static_cast<int>(ids.size()));
    std::vector<int64_t> part(ids.begin() + pos, ids.begin() + end);
    logits_dev = model_->forward(part, pos, *kv_, stream);
    pos = end;
  }

  // ---- Decode loop.
  for (int step = 0; step < sp.max_new_tokens; ++step) {
    std::vector<float> logits(V);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(logits.data(), logits_dev, V * sizeof(float),
                        hipMemcpyDeviceToHost));
    int64_t next = sample_token(logits.data(), V, sp, ids, rng);

    if (tokenizer_->is_special(next)) break;  // <|im_end|> etc.
    generated.push_back(next);
    ids.push_back(next);

    if (on_token) {
      std::string text = tokenizer_->decode(generated);
      if (text.size() > prev_text.size()) {
        on_token(text.substr(prev_text.size()));
      }
      prev_text = std::move(text);
    }

    if (step + 1 < sp.max_new_tokens) {
      logits_dev = model_->forward({next}, pos, *kv_, stream);
      pos += 1;
    }
  }

  if (on_token && prev_text.size() < tokenizer_->decode(generated).size()) {
    std::string text = tokenizer_->decode(generated);
    on_token(text.substr(prev_text.size()));
  }
  return tokenizer_->decode(generated);
}

}  // namespace ninfer
