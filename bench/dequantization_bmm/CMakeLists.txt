# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(dequant_bmm_probe LANGUAGES C CXX ASM)

# Standalone Android cross-build. No LiteRT or model files are required.
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "Build XNNPACK unit tests")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "Build other XNNPACK benchmarks")
set(XNNPACK_BUILD_ALL_MICROKERNELS OFF CACHE BOOL "Build unused microkernels")
set(XNNPACK_BUILD_YNNPACK OFF CACHE BOOL "Build YNNPACK")
set(XNNPACK_LIBRARY_TYPE static CACHE STRING "XNNPACK library type")
get_filename_component(XNNPACK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_subdirectory("${XNNPACK_ROOT}" "${CMAKE_BINARY_DIR}/xnnpack")

add_executable(dequant_bmm_probe probe.cc)
target_compile_features(dequant_bmm_probe PRIVATE cxx_std_17)
target_include_directories(dequant_bmm_probe PRIVATE "${XNNPACK_ROOT}")
# Internal dispatch structs depend on feature macros (KleidiAI changes
# XNN_MAX_MR). Match the library's definitions even on CPUs lacking that ISA.
get_directory_property(XNNPACK_PRIVATE_DEFINITIONS
  DIRECTORY "${XNNPACK_ROOT}" COMPILE_DEFINITIONS)
target_compile_definitions(dequant_bmm_probe PRIVATE ${XNNPACK_PRIVATE_DEFINITIONS})
target_link_libraries(dequant_bmm_probe PRIVATE XNNPACK pthreadpool ${CMAKE_DL_LIBS})
if(ANDROID)
  target_link_libraries(dequant_bmm_probe PRIVATE log)
endif()
