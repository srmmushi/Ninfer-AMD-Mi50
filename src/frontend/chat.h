#pragma once
// Chat template for Qwen2/Qwen3-style models (ChatML). Replaces ninfer's
// jinja renderer with the concrete template the supported architectures use.

#include <string>
#include <vector>

namespace ninfer {

struct ChatMessage {
  std::string role;     // system / user / assistant
  std::string content;
  bool has_think_open = false;   // assistant continuation bookkeeping
  bool has_think_close = false;
};

class ChatRenderer {
 public:
  // `enable_thinking` mirrors Qwen3's template flag: it only affects the
  // empty <think></think> block appended to the last user turn.
  explicit ChatRenderer(bool enable_thinking = true)
      : enable_thinking_(enable_thinking) {}

  // Renders the full prompt for generation (no assistant prefill text).
  std::string render(const std::vector<ChatMessage>& messages,
                     bool add_generation_prompt = true) const;

  // Renders the prompt plus `assistant_prefix` (assistant continuation, used
  // by --assistant-prefix and by prefill-style steering).
  std::string render_with_prefix(const std::vector<ChatMessage>& messages,
                                 const std::string& assistant_prefix) const;

 private:
  bool enable_thinking_;
};

}  // namespace ninfer
