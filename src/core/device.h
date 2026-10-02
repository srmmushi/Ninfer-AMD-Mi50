#pragma once
// HIP device management and device-memory RAII (mirrors ninfer src/core/
// device.cu + arena: one GPU, startup-fixed residency).

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HIP_CHECK(expr)                                                    \
  do {                                                                     \
    hipError_t err__ = (expr);                                             \
    if (err__ != hipSuccess) {                                             \
      throw std::runtime_error(std::string("HIP error ") +                 \
                               hipGetErrorString(err__) + " at " +         \
                               std::string(__FILE__) + ":" +               \
                               std::to_string(__LINE__) + " (" + #expr +   \
                               ")");                                       \
    }                                                                      \
  } while (0)

namespace ninfer {

// Selects and pins device 0 at process startup (one GPU, one resident model).
void init_device();

// RAII device switch: all hipMalloc / kernel launches happen on the current
// device, so every cross-device code path must hold a DeviceGuard.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    hipGetDevice(&prev_);
    hipSetDevice(device);
  }
  ~DeviceGuard() { hipSetDevice(prev_); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int prev_ = 0;
};

struct DeviceInfo {
  std::string name;
  int cu_count = 0;
  size_t total_vram = 0;
  size_t free_vram = 0;
};
DeviceInfo query_device();

// ---------------------------------------------------------------------------
// DeviceBuffer: raw device allocation with RAII.
// ---------------------------------------------------------------------------
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { alloc(bytes); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = std::exchange(o.ptr_, nullptr);
      size_ = std::exchange(o.size_, size_t(0));
    }
    return *this;
  }

  ~DeviceBuffer() { reset(); }

  void alloc(size_t bytes) {
    reset();
    if (bytes > 0) HIP_CHECK(hipMalloc(&ptr_, bytes));
    size_ = bytes;
  }

  void reset() {
    if (ptr_) {
      (void)hipFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = 0;
  }

  void* data() { return ptr_; }
  const void* data() const { return ptr_; }
  size_t bytes() const { return size_; }

  void copy_from_host(const void* src, size_t bytes, hipStream_t stream = 0) {
    HIP_CHECK(hipMemcpyAsync(ptr_, src, bytes, hipMemcpyHostToDevice, stream));
  }
  void copy_to_host(void* dst, size_t bytes, hipStream_t stream = 0) const {
    HIP_CHECK(hipMemcpyAsync(dst, ptr_, bytes, hipMemcpyDeviceToHost, stream));
  }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// half helpers (gfx906: storage fp16, conversions on host for weight loading).
// ---------------------------------------------------------------------------
uint16_t float_to_half(float v);
float half_to_float(uint16_t h);

// Host-side fp16 quantization of a fp32/fp64/bf16 tensor. `src_dtype` follows
// safetensors dtype strings: "F32", "F64", "BF16", "F16".
std::vector<uint16_t> convert_to_fp16(const void* src, const std::string& src_dtype,
                                      size_t count);

}  // namespace ninfer
