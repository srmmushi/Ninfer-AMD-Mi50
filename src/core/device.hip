#include "core/device.h"

#include <cmath>
#include <cstring>

#include "common/log.h"

namespace ninfer {

void init_device() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (count < 1) {
    throw std::runtime_error("no HIP device found; NInfer-HIP requires an AMD "
                             "GPU (gfx906 / MI50 series)");
  }
  HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (std::string(prop.gcnArchName).find("gfx906") == std::string::npos) {
    throw std::runtime_error(std::string("device '") + prop.name + "' (" +
                             prop.gcnArchName +
                             ") is not gfx906; NInfer-HIP targets MI50 only");
  }
  LOG_INFO("device: %s (%s), %d CUs, %.1f GiB HBM", prop.name, prop.gcnArchName,
           prop.multiProcessorCount,
           static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
}

DeviceInfo query_device() {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  size_t free_b = 0, total_b = 0;
  HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  DeviceInfo info;
  info.name = prop.name;
  info.cu_count = prop.multiProcessorCount;
  info.total_vram = prop.totalGlobalMem;
  info.free_vram = free_b;
  return info;
}

uint16_t float_to_half(float v) {
  __half h = __float2half(v);
  uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

float half_to_float(uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, sizeof(bits));
  return __half2float(h);
}

std::vector<uint16_t> convert_to_fp16(const void* src, const std::string& src_dtype,
                                      size_t count) {
  std::vector<uint16_t> out(count);
  if (src_dtype == "F16") {
    std::memcpy(out.data(), src, count * sizeof(uint16_t));
    return out;
  }
  if (src_dtype == "F32") {
    const float* p = static_cast<const float*>(src);
    for (size_t i = 0; i < count; ++i) out[i] = float_to_half(p[i]);
    return out;
  }
  if (src_dtype == "F64") {
    const double* p = static_cast<const double*>(src);
    for (size_t i = 0; i < count; ++i) out[i] = float_to_half(static_cast<float>(p[i]));
    return out;
  }
  if (src_dtype == "BF16") {
    // BF16 shares the upper 16 bits with FP32; widen then narrow with
    // round-to-nearest via the fp32 path.
    const uint16_t* p = static_cast<const uint16_t*>(src);
    for (size_t i = 0; i < count; ++i) {
      uint32_t bits = static_cast<uint32_t>(p[i]) << 16;
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      out[i] = float_to_half(f);
    }
    return out;
  }
  throw std::runtime_error("unsupported safetensors dtype for gfx906 (no bf16 "
                           "units): " + src_dtype);
}

}  // namespace ninfer
