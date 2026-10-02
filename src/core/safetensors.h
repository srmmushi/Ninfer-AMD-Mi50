#pragma once
// safetensors reader: index-only at startup, per-tensor streaming reads.
// Replaces the ninfer artifact Reader/Materializer for the HuggingFace
// checkpoint flow (config.json + model.safetensors).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ninfer {

struct StTensorInfo {
  std::string dtype;      // F32 / F16 / BF16 / F64
  std::vector<int64_t> shape;
  uint64_t data_offsets[2] = {0, 0};  // absolute file offsets
  uint64_t num_elements() const {
    uint64_t n = 1;
    for (int64_t d : shape) n *= static_cast<uint64_t>(d);
    return n;
  }
};

class SafeTensorsFile {
 public:
  explicit SafeTensorsFile(const std::string& path);

  const std::map<std::string, StTensorInfo>& index() const { return index_; }

  bool has(const std::string& name) const {
    return index_.count(name) != 0;
  }
  const StTensorInfo& tensor_info(const std::string& name) const {
    return index_.at(name);
  }

  // Reads raw tensor bytes (converted later by core/device convert_to_fp16).
  std::vector<uint8_t> read_raw(const std::string& name) const;

 private:
  std::string path_;
  std::map<std::string, StTensorInfo> index_;
};

}  // namespace ninfer
