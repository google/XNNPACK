# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FIND_PACKAGE(Python3 REQUIRED)

INCLUDE(../arch_copts.cmake)

ADD_LIBRARY(ynnpack_kernels_unary STATIC unary.cc)
TARGET_INCLUDE_DIRECTORIES(ynnpack_kernels_unary PUBLIC . ${CMAKE_BINARY_DIR})
TARGET_LINK_LIBRARIES(ynnpack_kernels_unary PUBLIC ynnpack_base ynnpack_simd)

FUNCTION(YNN_GENERATE_UNARY_KERNEL NAME ARCH)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    SET(OUTPUT_SRC "${CMAKE_BINARY_DIR}/ynnpack/kernels/unary/${NAME}.cc")
    SET(OUTPUT_INC "${CMAKE_BINARY_DIR}/ynnpack/kernels/unary/${NAME}.inc")

    ADD_CUSTOM_COMMAND(
      OUTPUT "${OUTPUT_SRC}" "${OUTPUT_INC}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ynnpack/kernels/unary"
      COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}"
              ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
              "${OUTPUT_SRC}" "${OUTPUT_INC}" "${ARCH}" ${ARGN}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/kernels.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/convert.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/erf.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/exp.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/log.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/sigmoid.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/sine_cosine.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/tanh.py"
              "${CMAKE_SOURCE_DIR}/ynnpack/kernels/elementwise/generator.py"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )

    SET(ARCH_LIB "ynnpack_kernels_unary_${NAME}")
    ADD_LIBRARY(${ARCH_LIB} STATIC "${OUTPUT_SRC}")
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    YNN_GET_ARCH_DEFINES(${ARCH} ARCH_DEFINES)
    TARGET_COMPILE_OPTIONS(${ARCH_LIB} PRIVATE ${ARCH_COPTS})
    TARGET_COMPILE_DEFINITIONS(${ARCH_LIB} PUBLIC ${ARCH_DEFINES})
    TARGET_INCLUDE_DIRECTORIES(${ARCH_LIB} PUBLIC . ${CMAKE_BINARY_DIR})
    TARGET_LINK_LIBRARIES(${ARCH_LIB} PUBLIC ynnpack_base ynnpack_simd)
    TARGET_LINK_LIBRARIES(ynnpack_kernels_unary PUBLIC ${ARCH_LIB})
  ENDIF()
ENDFUNCTION()

# x86
YNN_GENERATE_UNARY_KERNEL(x86_sse2 x86_sse2)
YNN_GENERATE_UNARY_KERNEL(x86_sse2_fma x86_sse2_fma)
YNN_GENERATE_UNARY_KERNEL(x86_sse41 x86_sse41)
YNN_GENERATE_UNARY_KERNEL(x86_avx x86_avx)
YNN_GENERATE_UNARY_KERNEL(x86_avx2 x86_avx2)
YNN_GENERATE_UNARY_KERNEL(x86_avx2_fma3 x86_avx2_fma3)
YNN_GENERATE_UNARY_KERNEL(x86_fma3 x86_fma3)
YNN_GENERATE_UNARY_KERNEL(x86_f16c x86_f16c)
YNN_GENERATE_UNARY_KERNEL(x86_avx512 x86_avx512)
YNN_GENERATE_UNARY_KERNEL(x86_avx512bf16 x86_avx512bf16)

# ARM
YNN_GENERATE_UNARY_KERNEL(arm64_neon arm64_neon)
YNN_GENERATE_UNARY_KERNEL(arm_neon arm_neon)
YNN_GENERATE_UNARY_KERNEL(arm_neonbf16 arm_neonbf16)
YNN_GENERATE_UNARY_KERNEL(arm_neonfp16 arm_neonfp16)

# WASM
YNN_GENERATE_UNARY_KERNEL(wasm_simd128 wasm_simd128)



IF(YNNPACK_BUILD_TESTS)
  ADD_LIBRARY(ynnpack_kernels_unary_reference STATIC reference.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_unary_reference PUBLIC ynnpack ynnpack_base ynnpack_base_test_tensor gtest)

  ADD_EXECUTABLE(ynnpack_kernels_unary_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_unary_test ynnpack_kernels_unary ynnpack_kernels_unary_reference ynnpack ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_unary_test COMMAND ynnpack_kernels_unary_test)

  ADD_EXECUTABLE(ynnpack_kernels_unary_consistent_arithmetic_test consistent_arithmetic_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_unary_consistent_arithmetic_test ynnpack_kernels_unary ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_unary_consistent_arithmetic_test COMMAND ynnpack_kernels_unary_consistent_arithmetic_test)
ENDIF()

IF(YNNPACK_BUILD_BENCHMARKS)
  ADD_EXECUTABLE(ynnpack_kernels_unary_bench bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_unary_bench ynnpack_kernels_unary ynnpack_base ynnpack_base_test_tensor benchmark::benchmark_main)
ENDIF()
