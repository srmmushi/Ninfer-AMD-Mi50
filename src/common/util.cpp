#include "common/util.h"

#include <stdexcept>

namespace ninfer {

std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open file: " + path);
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (size > 0 && std::fread(data.data(), 1, data.size(), f) != data.size()) {
    std::fclose(f);
    throw std::runtime_error("short read on file: " + path);
  }
  std::fclose(f);
  return data;
}

std::string read_file_text(const std::string& path) {
  auto bytes = read_file_bytes(path);
  return std::string(bytes.begin(), bytes.end());
}

std::vector<uint8_t> read_file_range(const std::string& path, uint64_t offset,
                                     uint64_t bytes) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open file: " + path);
  std::vector<uint8_t> data(static_cast<size_t>(bytes));
  if (bytes > 0) {
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0) {
      std::fclose(f);
      throw std::runtime_error("seek failed on file: " + path);
    }
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) {
      std::fclose(f);
      throw std::runtime_error("short read on file: " + path);
    }
  }
  std::fclose(f);
  return data;
}

bool file_exists(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fclose(f);
  return true;
}

std::string path_join(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

}  // namespace ninfer
