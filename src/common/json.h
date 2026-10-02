#pragma once
// Minimal self-contained JSON value (parse + serialize). Enough for
// config.json / tokenizer.json / safetensors headers / OpenAI-style payloads.
// Replaces the vendored nlohmann/json of the original ninfer to keep the
// refactor dependency-free.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ninfer {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Json() : type_(Type::Null) {}
  explicit Json(bool b) : type_(Type::Bool), bool_(b) {}
  explicit Json(double n) : type_(Type::Number), num_(n) {}
  explicit Json(int n) : type_(Type::Number), num_(static_cast<double>(n)) {}
  explicit Json(int64_t n) : type_(Type::Number), num_(static_cast<double>(n)) {}
  explicit Json(std::string s) : type_(Type::String), str_(std::move(s)) {}

  static Json array() { Json j; j.type_ = Type::Array; return j; }
  static Json object() { Json j; j.type_ = Type::Object; return j; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool(bool def = false) const {
    return type_ == Type::Bool ? bool_ : def;
  }
  double as_number(double def = 0.0) const {
    return type_ == Type::Number ? num_ : def;
  }
  int64_t as_int(int64_t def = 0) const {
    return type_ == Type::Number ? static_cast<int64_t>(num_) : def;
  }
  const std::string& as_string() const { return str_; }

  // Array access.
  const std::vector<Json>& items() const { return arr_; }
  std::vector<Json>& items() { return arr_; }
  void push_back(Json v) { arr_.push_back(std::move(v)); }
  const Json& at(size_t i) const { return arr_.at(i); }

  // Object access.
  const std::map<std::string, Json>& members() const { return obj_; }
  const Json* find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    auto it = obj_.find(key);
    return it == obj_.end() ? nullptr : &it->second;
  }
  Json& operator[](const std::string& key) {
    type_ = Type::Object;
    return obj_[key];
  }
  bool contains(const std::string& key) const { return find(key) != nullptr; }

  // Recursive-descent parser. Throws std::runtime_error on malformed input.
  static Json parse(const std::string& text);

  // Compact serialization.
  std::string dump() const;

 private:
  Type type_;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  std::vector<Json> arr_;
  std::map<std::string, Json> obj_;
};

}  // namespace ninfer
