#include "frontend/chat.h"

namespace ninfer {

std::string ChatRenderer::render(const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt) const {
  std::string out;
  for (size_t i = 0; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    out += "<|im_start|>" + m.role + "\n" + m.content;
    if (m.role == "user" && i + 1 == messages.size() && enable_thinking_) {
      // Qwen3 thinking-mode marker on the final user turn.
      out += " <think>\n\n</think>\n\n";
    } else {
      out += "<|im_end|>\n";
    }
  }
  if (add_generation_prompt) {
    out += "<|im_start|>assistant\n";
  }
  return out;
}

std::string ChatRenderer::render_with_prefix(
    const std::vector<ChatMessage>& messages,
    const std::string& assistant_prefix) const {
  std::string out = render(messages, /*add_generation_prompt=*/false);
  out += "<|im_start|>assistant\n" + assistant_prefix;
  return out;
}

}  // namespace ninfer
