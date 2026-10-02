#include "core/safetensors.h"

#include <stdexcept>

#include "common/json.h"
#include "common/log.h"
#include "common/util.h"

namespace ninfer {

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
  if (!file_exists(path)) {
    throw std::runtime_error("safetensors file not found: " + path);
  }
  // Layout: 8-byte little-endian header length, JSON header, then payloads.
  auto head = read_file_range(path, 0, 8);
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) {
    header_len = (header_len << 8) | head[i];
  }
  if (header_len == 0 || header_len > (1ull << 30)) {
    throw std::runtime_error("invalid safetensors header length in " + path);
  }
  auto header_bytes = read_file_range(path, 8, header_len);
  std::string header_text(header_bytes.begin(), header_bytes.end());
  Json header = Json::parse(header_text);

  const uint64_t data_base = 8 + header_len;
  for (const auto& [name, meta] : header.members()) {
    if (name == "__metadata__") continue;
    const Json* dtype = meta.find("dtype");
    const Json* shape = meta.find("shape");
    const Json* offsets = meta.find("data_offsets");
    if (!dtype || !shape || !offsets || !dtype->is_string() ||
        !shape->is_array() || !offsets->is_array() || offsets->items().size() != 2) {
      throw std::runtime_error("malformed safetensors entry for '" + name + "'");
    }
    StTensorInfo info;
    info.dtype = dtype->as_string();
    for (const auto& d : shape->items()) info.shape.push_back(d.as_int());
    info.data_offsets[0] = data_base + static_cast<uint64_t>(offsets->at(0).as_int());
    info.data_offsets[1] = data_base + static_cast<uint64_t>(offsets->at(1).as_int());
    index_[name] = std::move(info);
  }
  LOG_DEBUG("safetensors '%s': %zu tensors", path.c_str(), index_.size());
}

std::vector<uint8_t> SafeTensorsFile::read_raw(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) {
    throw std::runtime_error("tensor not found in checkpoint: " + name);
  }
  const StTensorInfo& info = it->second;
  uint64_t bytes = info.data_offsets[1] - info.data_offsets[0];
  return read_file_range(path_, info.data_offsets[0], bytes);
}

}  // namespace ninfer
