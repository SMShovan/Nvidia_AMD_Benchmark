cmake_minimum_required(VERSION 3.24)
project(nvidia_amd_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Backend: AUTO (detect), CUDA (H100), HIP (MI300A), or CPU (portable smoke test)
set(BACKEND "AUTO" CACHE STRING "Compute backend: AUTO, CUDA, HIP, or CPU")
include(${CMAKE_SOURCE_DIR}/cmake/SelectBackend.cmake)

if(BENCH_BACKEND STREQUAL "CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)   # H100 = sm_90 (override with -DCMAKE_CUDA_ARCHITECTURES=..)
  endif()
  add_executable(gemm_bench src/cuda/main.cu)
  target_compile_definitions(gemm_bench PRIVATE BENCH_BACKEND_CUDA)
elseif(BENCH_BACKEND STREQUAL "HIP")
  enable_language(HIP)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx942) # MI300A
  endif()
  find_package(hip REQUIRED)
  add_executable(gemm_bench src/hip/main.hip)
  target_compile_definitions(gemm_bench PRIVATE BENCH_BACKEND_HIP)
  target_link_libraries(gemm_bench PRIVATE hip::device)
else()
  add_executable(gemm_bench src/cpu/main.cpp)
  target_compile_definitions(gemm_bench PRIVATE BENCH_BACKEND_CPU)
endif()

target_include_directories(gemm_bench PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)

message(STATUS "Configured gemm_bench  [backend=${BENCH_BACKEND}]")
