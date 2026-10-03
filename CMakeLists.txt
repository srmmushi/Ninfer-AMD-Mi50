cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# NInfer-HIP: a from-scratch C++/HIP inference engine, AMD MI50 (gfx906) only.
# Mirrors the ninfer/ layering: core / ops / models / frontend / runtime / apps.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "HIP architectures to build")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx906")
  message(FATAL_ERROR
    "NInfer-HIP supports only CMAKE_HIP_ARCHITECTURES=gfx906 (AMD MI50); got "
    "'${CMAKE_HIP_ARCHITECTURES}'")
endif()

project(ninfer_hip LANGUAGES C CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ROCm toolchain discovery (validated against ROCm 6.3.3).
if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  else()
    set(ROCM_PATH /opt/rocm)
  endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(NINFER_BUILD_SHARED "Build libninfer.so (C API for vLLM integration)" ON)

# libninfer.so / static libninfer_core must be relocatable.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# gfx906 has no BF16 units: the engine computes in FP16 with FP32 accumulation.
add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:NINFER_GFX906=1>)

add_library(ninfer_core STATIC
  src/common/log.cpp
  src/common/util.cpp
  src/common/json.cpp
  src/core/device.hip
  src/core/safetensors.cpp
  src/frontend/unicode.cpp
  src/frontend/tokenizer.cpp
  src/frontend/chat.cpp
  src/ops/linear.hip
  src/ops/kernels.hip
  src/ops/gemv.hip
  src/ops/quant.hip
  src/ops/attention.hip
  src/ops/moe.hip
  src/models/config.cpp
  src/models/model.hip
  src/runtime/kv_cache.cpp
  src/runtime/engine.cpp
)
target_include_directories(ninfer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ninfer_core PUBLIC hip::host roc::rocblas)
target_compile_options(ninfer_core PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3 -w>)

if(NINFER_BUILD_APPS)
  add_executable(ninfer src/apps/ninfer.cpp)
  target_link_libraries(ninfer PRIVATE ninfer_core)

  add_executable(ninfer-serve src/apps/ninfer_serve.cpp src/apps/http_server.cpp)
  target_link_libraries(ninfer-serve PRIVATE ninfer_core)
endif()

if(NINFER_BUILD_SHARED)
  add_library(ninfer SHARED src/api/ninfer_c_api.cpp)
  target_link_libraries(ninfer PUBLIC ninfer_core)
  set_target_properties(ninfer PROPERTIES PUBLIC_HEADER "src/api/ninfer_c_api.h")
endif()
