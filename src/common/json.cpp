#include "common/json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : p_(text.data()), end_(text.data() + text.size()) {}

  Json parse() {
    Json v = parse_value();
    skip_ws();
    if (p_ != end_) fail("trailing characters");
    return v;
  }

 private:
  const char* p_;
  const char* end_;

  [[noreturn]] void fail(const char* msg) const {
    throw std::runtime_error(std::string("json: ") + msg);
  }

  void skip_ws() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
  }

  char peek() {
    if (p_ >= end_) fail("unexpected end of input");
    return *p_;
  }

  void expect(char c) {
    if (p_ >= end_ || *p_ != c) fail("unexpected character");
    ++p_;
  }

  Json parse_value() {
    skip_ws();
    char c = peek();
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Json(parse_string());
      case 't': expect_word("true"); return Json(true);
      case 'f': expect_word("false"); return Json(false);
      case 'n': expect_word("null"); return Json();
      default: return parse_number();
    }
  }

  void expect_word(const char* w) {
    size_t n = std::strlen(w);
    if (static_cast<size_t>(end_ - p_) < n || std::strncmp(p_, w, n) != 0) {
      fail("invalid literal");
    }
    p_ += n;
  }

  Json parse_object() {
    expect('{');
    Json obj = Json::object();
    skip_ws();
    if (peek() == '}') { ++p_; return obj; }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      obj[key] = parse_value();
      skip_ws();
      char c = peek();
      if (c == ',') { ++p_; continue; }
      if (c == '}') { ++p_; break; }
      fail("expected ',' or '}'");
    }
    return obj;
  }

  Json parse_array() {
    expect('[');
    Json arr = Json::array();
    skip_ws();
    if (peek() == ']') { ++p_; return arr; }
    while (true) {
      arr.push_back(parse_value());
      skip_ws();
      char c = peek();
      if (c == ',') { ++p_; continue; }
      if (c == ']') { ++p_; break; }
      fail("expected ',' or ']'");
    }
    return arr;
  }

  uint32_t parse_hex4() {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = peek();
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else fail("bad \\u escape");
      ++p_;
    }
    return v;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (p_ >= end_) fail("unterminated string");
      char c = *p_++;
      if (c == '"') break;
      if (c == '\\') {
        char e = peek();
        ++p_;
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            uint32_t cp = parse_hex4();
            if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 1 < end_ &&
                p_[0] == '\\' && p_[1] == 'u') {
              p_ += 2;
              uint32_t lo = parse_hex4();
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              }
            }
            append_utf8(out, cp);
            break;
          }
          default: fail("bad escape");
        }
      } else {
        out.push_back(c);
      }
    }
    return out;
  }

  Json parse_number() {
    const char* start = p_;
    if (p_ < end_ && (*p_ == '-' || *p_ == '+')) ++p_;
    while (p_ < end_ && (std::isdigit(static_cast<unsigned char>(*p_)) ||
                         *p_ == '.' || *p_ == 'e' || *p_ == 'E' ||
                         *p_ == '+' || *p_ == '-')) {
      ++p_;
    }
    if (p_ == start) fail("invalid number");
    return Json(std::strtod(std::string(start, p_).c_str(), nullptr));
  }
};

void dump_string(const std::string& s, std::string& out) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

void dump_value(const Json& j, std::string& out) {
  switch (j.type()) {
    case Json::Type::Null: out += "null"; break;
    case Json::Type::Bool: out += j.as_bool() ? "true" : "false"; break;
    case Json::Type::Number: {
      double v = j.as_number();
      if (std::isfinite(v) && v == static_cast<int64_t>(v) && std::fabs(v) < 1e15) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld",
                      static_cast<long long>(static_cast<int64_t>(v)));
        out += buf;
      } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        out += buf;
      }
      break;
    }
    case Json::Type::String: dump_string(j.as_string(), out); break;
    case Json::Type::Array: {
      out.push_back('[');
      bool first = true;
      for (const auto& item : j.items()) {
        if (!first) out.push_back(',');
        first = false;
        dump_value(item, out);
      }
      out.push_back(']');
      break;
    }
    case Json::Type::Object: {
      out.push_back('{');
      bool first = true;
      for (const auto& [key, val] : j.members()) {
        if (!first) out.push_back(',');
        first = false;
        dump_string(key, out);
        out.push_back(':');
        dump_value(val, out);
      }
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

Json Json::parse(const std::string& text) { return Parser(text).parse(); }

std::string Json::dump() const {
  std::string out;
  dump_value(*this, out);
  return out;
}

}  // namespace ninfer
