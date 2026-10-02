#pragma once
// Byte-level BPE tokenizer for Qwen2/Qwen3 checkpoints, loaded from a
// HuggingFace `tokenizer.json` (replaces ninfer's jinja-driven frontend
// resources with a direct HF-compatible path).

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json_path);

  int64_t eos_id() const { return eos_id_; }
  int64_t bos_id() const { return bos_id_; }
  int64_t pad_id() const { return pad_id_; }
  size_t vocab_size() const { return id_to_token_.size(); }

  // encodes raw text; special tokens in the text are NOT recognized.
  std::vector<int64_t> encode(const std::string& text) const;

  // encodes raw text, recognizing special tokens (chat markers etc.).
  std::vector<int64_t> encode_with_special(const std::string& text) const;

  std::string decode(const std::vector<int64_t>& ids) const;
  std::string decode_one(int64_t id) const;
  std::string token_text(int64_t id) const;
  bool is_special(int64_t id) const { return special_ids_.count(id) != 0; }

 private:
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int64_t> token_to_id_;
  std::map<std::pair<int64_t, int64_t>, int64_t> merge_ranks_;
  std::unordered_map<int64_t, std::string> special_id_to_text_;
  std::unordered_map<std::string, int64_t> special_text_to_id_;
  std::unordered_map<uint8_t, uint32_t> byte_to_unicode_;
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;
  int64_t eos_id_ = -1, bos_id_ = -1, pad_id_ = -1;

  std::vector<int64_t> bpe(const std::string& piece) const;
};

}  // namespace ninfer
