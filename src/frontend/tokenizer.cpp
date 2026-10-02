#include "frontend/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include "common/json.h"
#include "common/log.h"
#include "common/util.h"
#include "frontend/unicode.h"

namespace ninfer {
namespace {

// GPT-2 byte <-> printable-unicode mapping (used by all byte-level BPE
// vocabs, including Qwen's).
void build_byte_maps(std::unordered_map<uint8_t, uint32_t>& byte_to_unicode,
                     std::unordered_map<uint32_t, uint8_t>& unicode_to_byte) {
  std::vector<uint32_t> cp;
  for (uint32_t b = '!' ; b <= '~'; ++b) cp.push_back(b);
  for (uint32_t b = 0xA1; b <= 0xAC; ++b) cp.push_back(b);
  for (uint32_t b = 0xAE; b <= 0xFF; ++b) cp.push_back(b);
  uint32_t next = 0x100;
  for (int b = 0; b < 256; ++b) {
    uint32_t u = static_cast<uint32_t>(b);
    auto it = std::find(cp.begin(), cp.end(), u);
    if (it == cp.end()) u = next++;
    byte_to_unicode[static_cast<uint8_t>(b)] = u;
    unicode_to_byte[u] = static_cast<uint8_t>(b);
  }
}

std::string bytes_to_unicode_string(const std::string& raw,
                                    const std::unordered_map<uint8_t, uint32_t>& map) {
  std::string out;
  for (unsigned char b : raw) {
    utf8_encode(map.at(b), out);
  }
  return out;
}

// Applies the Qwen/GPT-4o-style pretokenizer:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
// | [^\r\n\p{L}\p{N}]?\p{L}+
// | \p{N}{1,3}
// |  ?[^\s\p{L}\p{N}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
std::vector<std::string> pretokenize(const std::string& text) {
  std::vector<std::string> pieces;
  size_t i = 0;
  auto is_space = [](uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f' || c == 0x85 || c == 0xA0 || c == 0x2028 || c == 0x2029;
  };
  auto is_punct = [&](uint32_t c) {
    return !is_space(c) && !is_unicode_letter(c) && !is_unicode_number(c);
  };
  auto match_contraction = [&]() -> bool {
    if (i >= text.size() || text[i] != '\'') return false;
    static const char* suffixes[] = {"re", "ll", "s", "t", "ve", "m", "d"};
    for (const char* suf : suffixes) {
      size_t n = std::strlen(suf);
      if (i + 1 + n <= text.size()) {
        bool ok = true;
        for (size_t k = 0; k < n; ++k) {
          char a = text[i + 1 + k];
          char b = suf[k];
          if (std::tolower(static_cast<unsigned char>(a)) != b) { ok = false; break; }
        }
        if (ok) { i += 1 + n; return true; }
      }
    }
    return false;
  };

  while (i < text.size()) {
    size_t start = i;
    uint32_t c = utf8_decode(text, i);

    if (match_contraction()) { pieces.push_back(text.substr(start, i - start)); continue; }

    // [^\r\n\p{L}\p{N}]? \p{L}+
    size_t save = i;
    if (c != '\r' && c != '\n' && is_punct(c) && i < text.size()) {
      uint32_t c2 = utf8_decode(text, i);
      if (is_unicode_letter(c2)) {
        while (i < text.size()) {
          size_t s2 = i;
          uint32_t cx = utf8_decode(text, i);
          if (!is_unicode_letter(cx)) { i = s2; break; }
        }
        pieces.push_back(text.substr(start, i - start));
        continue;
      }
      i = save;
      c = utf8_decode(text, i);
    } else {
      i = save;
    }
    // Re-check leading letter (no prefix char).
    size_t s2 = i;
    uint32_t cl = utf8_decode(text, i);
    if (is_unicode_letter(cl)) {
      while (i < text.size()) {
        size_t s3 = i;
        uint32_t cx = utf8_decode(text, i);
        if (!is_unicode_letter(cx)) { i = s3; break; }
      }
      pieces.push_back(text.substr(start, i - start));
      continue;
    }
    i = s2;

    // \p{N}{1,3}
    if (is_unicode_number(c)) {
      int n = 1;
      while (n < 3 && i < text.size()) {
        size_t s3 = i;
        uint32_t cx = utf8_decode(text, i);
        if (!is_unicode_number(cx)) { i = s3; break; }
        ++n;
      }
      pieces.push_back(text.substr(start, i - start));
      continue;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    size_t sym_start = i;
    if (c == ' ') {
      size_t s3 = i;
      uint32_t cx = utf8_decode(text, i);
      if (!is_space(cx) && !is_unicode_letter(cx) && !is_unicode_number(cx)) {
        sym_start = s3;
        while (i < text.size()) {
          size_t s4 = i;
          uint32_t cy = utf8_decode(text, i);
          if (is_space(cy) || is_unicode_letter(cy) || is_unicode_number(cy)) {
            i = s4;
            break;
          }
        }
        // [\r\n]*
        while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) ++i;
        pieces.push_back(text.substr(start, i - start));
        continue;
      }
      i = s3;
    } else if (is_punct(c)) {
      i = sym_start;
      while (i < text.size()) {
        size_t s4 = i;
        uint32_t cy = utf8_decode(text, i);
        if (is_space(cy) || is_unicode_letter(cy) || is_unicode_number(cy)) {
          i = s4;
          break;
        }
      }
      while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) ++i;
      pieces.push_back(text.substr(start, i - start));
      continue;
    }

    // \s*[\r\n]+
    if (is_space(c)) {
      size_t ws_start = i;
      size_t s3 = i;
      uint32_t cx = utf8_decode(text, i);
      if (cx == '\r' || cx == '\n') {
        // whitespace run before newlines
        size_t nl_start = s3;
        while (i < text.size()) {
          size_t s4 = i;
          uint32_t cy = utf8_decode(text, i);
          if (cy != '\r' && cy != '\n') { i = s4; break; }
        }
        pieces.push_back(text.substr(start, i - start));
        (void)nl_start;
        continue;
      }
      // \s+(?!\S) then \s+
      size_t ws_end = ws_start;
      while (ws_end < text.size()) {
        size_t s4 = ws_end;
        uint32_t cy = utf8_decode(text, ws_end);
        (void)cy;
        if (!is_space(cy)) break;
        (void)s4;
      }
      // ws_end now points past the whitespace run (or at first non-space).
      size_t run_len = ws_end - start;
      bool followed_by_nonspace = ws_end < text.size();
      if (followed_by_nonspace && run_len > 1) run_len -= 1;  // \s+(?!\S)
      if (run_len > 0) {
        pieces.push_back(text.substr(start, run_len));
        i = start + run_len;
        continue;
      }
      i = ws_start;
      // single whitespace followed by non-space: treat as prefix of next piece
      pieces.push_back(text.substr(start, i - start));
      continue;
    }

    // Fallback: single code point.
    pieces.push_back(text.substr(start, i - start));
  }
  return pieces;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& tokenizer_json_path) {
  build_byte_maps(byte_to_unicode_, unicode_to_byte_);

  Json root = Json::parse(read_file_text(tokenizer_json_path));

  // Vocab: byte-level token string -> id.
  const Json* vocab = root.find("model") != nullptr &&
                              root.at("model").find("vocab") != nullptr
                          ? &root.at("model").at("vocab")
                          : root.find("vocab");
  if (!vocab) throw std::runtime_error("tokenizer.json has no vocab");
  id_to_token_.resize(vocab->members().size());
  for (const auto& [tok, id] : vocab->members()) {
    int64_t i = id.as_int();
    token_to_id_[tok] = i;
    if (static_cast<size_t>(i) >= id_to_token_.size()) id_to_token_.resize(i + 1);
    id_to_token_[i] = tok;
  }

  // Merges.
  const Json* model = root.find("model");
  const Json* merges = model ? model->find("merges") : nullptr;
  if (merges && merges->is_array()) {
    for (const auto& m : merges->items()) {
      std::string pair = m.is_string() ? m.as_string() : m.dump();
      // either "a b" string or [a, b]
      if (m.is_string()) {
        size_t sp = m.as_string().rfind(' ');
        if (sp == std::string::npos) continue;
        std::string a = m.as_string().substr(0, sp);
        std::string b = m.as_string().substr(sp + 1);
        auto ia = token_to_id_.find(a);
        auto ib = token_to_id_.find(b);
        auto im = token_to_id_.find(a + b);
        if (ia == token_to_id_.end() || ib == token_to_id_.end() ||
            im == token_to_id_.end()) {
          continue;
        }
        merge_ranks_[{ia->second, ib->second}] = im->second;
      }
    }
  }

  // Added / special tokens.
  const Json* added = root.find("added_tokens");
  if (added && added->is_array()) {
    for (const auto& t : added->items()) {
      int64_t id = t.at("id").as_int();
      std::string content = t.at("content").as_string();
      bool special = t.find("special") ? t.at("special").as_bool() : false;
      if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(id + 1);
      id_to_token_[id] = content;
      token_to_id_[content] = id;
      if (special || content.rfind("<|", 0) == 0) {
        special_id_to_text_[id] = content;
        special_text_to_id_[content] = id;
      }
    }
  }

  auto lookup_special = [&](const std::string& key) -> int64_t {
    auto it = special_text_to_id_.find(key);
    return it == special_text_to_id_.end() ? -1 : it->second;
  };
  eos_id_ = lookup_special("<|im_end|>");
  if (eos_id_ < 0) eos_id_ = lookup_special("<|endoftext|>");
  bos_id_ = lookup_special("<|im_start|>");
  pad_id_ = eos_id_;

  LOG_DEBUG("tokenizer: %zu vocab, %zu merges, eos=%lld",
            token_to_id_.size(), merge_ranks_.size(),
            static_cast<long long>(eos_id_));
}

std::vector<int64_t> Tokenizer::bpe(const std::string& piece) const {
  std::string encoded = bytes_to_unicode_string(piece, byte_to_unicode_);

  // Split the piece into single code points as initial symbols.
  std::vector<std::string> parts;
  std::vector<int64_t> ids;
  size_t pos = 0;
  while (pos < encoded.size()) {
    size_t start = pos;
    utf8_decode(encoded, pos);
    parts.push_back(encoded.substr(start, pos - start));
  }
  ids.reserve(parts.size());
  for (const auto& p : parts) {
    auto it = token_to_id_.find(p);
    if (it == token_to_id_.end()) throw std::runtime_error("tokenizer: unmappable byte in '" + piece + "'");
    ids.push_back(it->second);
  }

  // Iteratively merge the lowest-rank adjacent pair.
  while (ids.size() > 1) {
    int64_t best_rank = -1;
    size_t best_idx = SIZE_MAX;
    for (size_t k = 0; k + 1 < ids.size(); ++k) {
      auto it = merge_ranks_.find({ids[k], ids[k + 1]});
      if (it != merge_ranks_.end() &&
          (best_rank < 0 || it->second < best_rank)) {
        best_rank = it->second;
        best_idx = k;
      }
    }
    if (best_idx == SIZE_MAX) break;
    ids[best_idx] = merge_ranks_.at({ids[best_idx], ids[best_idx + 1]});
    ids.erase(ids.begin() + best_idx + 1);
  }
  return ids;
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int64_t> out;
  for (const auto& piece : pretokenize(text)) {
    auto ids = bpe(piece);
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

std::vector<int64_t> Tokenizer::encode_with_special(const std::string& text) const {
  std::vector<int64_t> out;
  size_t pos = 0;
  // Find the earliest special-token occurrence; encode the gap as raw text.
  while (pos < text.size()) {
    size_t best_pos = std::string::npos;
    const std::string* best_tok = nullptr;
    for (const auto& [tok, id] : special_text_to_id_) {
      size_t p = text.find(tok, pos);
      if (p != std::string::npos && (best_pos == std::string::npos || p < best_pos)) {
        best_pos = p;
        best_tok = &tok;
      }
    }
    if (!best_tok) break;
    if (best_pos > pos) {
      auto ids = encode(text.substr(pos, best_pos - pos));
      out.insert(out.end(), ids.begin(), ids.end());
    }
    out.push_back(special_text_to_id_.at(*best_tok));
    pos = best_pos + best_tok->size();
  }
  if (pos < text.size()) {
    auto ids = encode(text.substr(pos));
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

std::string Tokenizer::token_text(int64_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "";
  return id_to_token_[id];
}

std::string Tokenizer::decode_one(int64_t id) const {
  auto sit = special_id_to_text_.find(id);
  if (sit != special_id_to_text_.end()) return sit->second;
  std::string tok = token_text(id);
  if (tok.empty()) return "";
  std::string out;
  size_t pos = 0;
  bool all_mapped = true;
  std::string raw;
  while (pos < tok.size()) {
    size_t start = pos;
    uint32_t cp = utf8_decode(tok, pos);
    auto it = unicode_to_byte_.find(cp);
    if (it == unicode_to_byte_.end()) { all_mapped = false; break; }
    raw.push_back(static_cast<char>(it->second));
  }
  if (all_mapped) return raw;
  return tok;  // special-token-like text: return as-is
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
  std::string out;
  for (int64_t id : ids) out += decode_one(id);
  return out;
}

}  // namespace ninfer
