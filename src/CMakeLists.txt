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

# gfx906 has no BF16 units: the engine computes in FP16 with FP32 accumulation.
add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:NINFER_GFX906=1>)

add_library(ninfer_core STATIC
  common/log.cpp
  common/util.cpp
  common/json.cpp
  core/device.hip
  core/safetensors.cpp
  frontend/unicode.cpp
  frontend/tokenizer.cpp
  frontend/chat.cpp
  ops/linear.hip
  ops/kernels.hip
  ops/attention.hip
  ops/moe.hip
  models/config.cpp
  models/model.hip
  runtime/kv_cache.cpp
  runtime/engine.cpp
)
target_include_directories(ninfer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ninfer_core PUBLIC hip::host roc::rocblas)
target_compile_options(ninfer_core PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3 -w>)

if(NINFER_BUILD_APPS)
  add_executable(ninfer apps/ninfer.cpp)
  target_link_libraries(ninfer PRIVATE ninfer_core)

  add_executable(ninfer-serve apps/ninfer_serve.cpp apps/http_server.cpp)
  target_link_libraries(ninfer-serve PRIVATE ninfer_core)
endif()
