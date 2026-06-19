# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FIND_PACKAGE(Python3 REQUIRED)

INCLUDE(../arch_copts.cmake)

ADD_LIBRARY(ynnpack_kernels_binary STATIC binary.cc)
TARGET_INCLUDE_DIRECTORIES(ynnpack_kernels_binary PUBLIC . ${CMAKE_BINARY_DIR})
TARGET_LINK_LIBRARIES(ynnpack_kernels_binary PUBLIC ynnpack_base ynnpack_simd)

FUNCTION(YNN_GENERATE_BINARY_KERNEL NAME ARCH)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    SET(OUTPUT_SRC "${CMAKE_BINARY_DIR}/ynnpack/kernels/binary/${NAME}.cc")
    SET(OUTPUT_INC "${CMAKE_BINARY_DIR}/ynnpack/kernels/binary/${NAME}.inc")

    ADD_CUSTOM_COMMAND(
      OUTPUT "${OUTPUT_SRC}" "${OUTPUT_INC}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ynnpack/kernels/binary"
      COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}"
              ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
              "${OUTPUT_SRC}" "${OUTPUT_INC}" "${ARCH}" ${ARGN}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/kernels.py"
              "${CMAKE_SOURCE_DIR}/ynnpack/kernels/elementwise/generator.py"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )

    SET(ARCH_LIB "ynnpack_kernels_binary_${NAME}")
    ADD_LIBRARY(${ARCH_LIB} STATIC "${OUTPUT_SRC}")
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    YNN_GET_ARCH_DEFINES(${ARCH} ARCH_DEFINES)
    TARGET_COMPILE_OPTIONS(${ARCH_LIB} PRIVATE ${ARCH_COPTS})
    TARGET_COMPILE_DEFINITIONS(${ARCH_LIB} PUBLIC ${ARCH_DEFINES})
    TARGET_INCLUDE_DIRECTORIES(${ARCH_LIB} PUBLIC . ${CMAKE_BINARY_DIR})
    TARGET_LINK_LIBRARIES(${ARCH_LIB} PUBLIC ynnpack_base ynnpack_simd)
    TARGET_LINK_LIBRARIES(ynnpack_kernels_binary PUBLIC ${ARCH_LIB})
  ENDIF()
ENDFUNCTION()

# x86
YNN_GENERATE_BINARY_KERNEL(x86_sse2 x86_sse2)
YNN_GENERATE_BINARY_KERNEL(x86_avx x86_avx)
YNN_GENERATE_BINARY_KERNEL(x86_avx2 x86_avx2)
YNN_GENERATE_BINARY_KERNEL(x86_avx512 x86_avx512)
YNN_GENERATE_BINARY_KERNEL(x86_avx512bf16 x86_avx512bf16)

# ARM
YNN_GENERATE_BINARY_KERNEL(arm64_neon arm64_neon)
YNN_GENERATE_BINARY_KERNEL(arm_neon arm_neon)
YNN_GENERATE_BINARY_KERNEL(arm_neonbf16 arm_neonbf16)

# WASM
YNN_GENERATE_BINARY_KERNEL(wasm_simd128 wasm_simd128)



IF(YNNPACK_BUILD_TESTS)
  ADD_LIBRARY(ynnpack_kernels_binary_reference STATIC reference.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_binary_reference PUBLIC ynnpack ynnpack_base ynnpack_base_test_tensor gtest)

  ADD_EXECUTABLE(ynnpack_kernels_binary_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_binary_test ynnpack_kernels_binary ynnpack_kernels_binary_reference ynnpack ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_binary_test COMMAND ynnpack_kernels_binary_test)
ENDIF()

IF(YNNPACK_BUILD_BENCHMARKS)
  ADD_EXECUTABLE(ynnpack_kernels_binary_bench bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_binary_bench ynnpack_kernels_binary ynnpack_base ynnpack_base_test_tensor benchmark::benchmark_main)
ENDIF()
