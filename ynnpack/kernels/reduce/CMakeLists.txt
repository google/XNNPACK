# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_kernels_reduce STATIC reduce.cc)
TARGET_LINK_LIBRARIES(ynnpack_kernels_reduce PUBLIC ynnpack_base ynnpack_simd)

INCLUDE(../arch_copts.cmake)

FUNCTION(YNN_ADD_REDUCE_SRC ARCH SRC)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    SET(ARCH_LIB "ynnpack_kernels_reduce_${ARCH}")
    ADD_LIBRARY(${ARCH_LIB} STATIC ${SRC})
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    YNN_GET_ARCH_DEFINES(${ARCH} ARCH_DEFINES)
    TARGET_COMPILE_OPTIONS(${ARCH_LIB} PRIVATE ${ARCH_COPTS})
    TARGET_COMPILE_DEFINITIONS(${ARCH_LIB} PUBLIC ${ARCH_DEFINES})
    TARGET_LINK_LIBRARIES(${ARCH_LIB} PUBLIC ynnpack_base ynnpack_simd)
    TARGET_LINK_LIBRARIES(ynnpack_kernels_reduce PUBLIC ${ARCH_LIB})
  ENDIF()
ENDFUNCTION()

# ARM
YNN_ADD_REDUCE_SRC(arm_neon arm_neon.cc)
YNN_ADD_REDUCE_SRC(arm_neonbf16 arm_neonbf16.cc)
YNN_ADD_REDUCE_SRC(arm_neondot arm_neondot.cc)
YNN_ADD_REDUCE_SRC(arm_neonfma arm_neonfma.cc)
YNN_ADD_REDUCE_SRC(arm_neonfp16 arm_neonfp16.cc)
YNN_ADD_REDUCE_SRC(arm64_neon arm64_neon.cc)

# x86
YNN_ADD_REDUCE_SRC(x86_sse2 x86_sse2.cc)
YNN_ADD_REDUCE_SRC(x86_sse41 x86_sse41.cc)
YNN_ADD_REDUCE_SRC(x86_ssse3 x86_ssse3.cc)
YNN_ADD_REDUCE_SRC(x86_avx x86_avx.cc)
YNN_ADD_REDUCE_SRC(x86_avx2 x86_avx2.cc)
YNN_ADD_REDUCE_SRC(x86_avx2_fma3 x86_avx2_fma3.cc)
YNN_ADD_REDUCE_SRC(x86_avx512 x86_avx512.cc)
YNN_ADD_REDUCE_SRC(x86_avx512bf16 x86_avx512bf16.cc)
YNN_ADD_REDUCE_SRC(x86_f16c x86_f16c.cc)

# Others
YNN_ADD_REDUCE_SRC(hexagon_hvx hexagon_hvx.cc)
YNN_ADD_REDUCE_SRC(wasm_simd128 wasm_simd128.cc)



IF(YNNPACK_BUILD_TESTS)
  ADD_EXECUTABLE(ynnpack_kernels_reduce_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_reduce_test ynnpack_kernels_reduce ynnpack ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_reduce_test COMMAND ynnpack_kernels_reduce_test)

  ADD_EXECUTABLE(ynnpack_kernels_consistent_reduce_test consistent_reduce_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_consistent_reduce_test ynnpack_kernels_reduce ynnpack ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_consistent_reduce_test COMMAND ynnpack_kernels_consistent_reduce_test)

  ADD_EXECUTABLE(ynnpack_kernels_reduce_pi_test pi_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_reduce_pi_test ynnpack_kernels_reduce ynnpack_base gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_reduce_pi_test COMMAND ynnpack_kernels_reduce_pi_test)
ENDIF()

IF(YNNPACK_BUILD_BENCHMARKS)
  ADD_EXECUTABLE(ynnpack_kernels_reduce_bench bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_reduce_bench ynnpack_kernels_reduce ynnpack_base ynnpack_base_test_tensor benchmark::benchmark)
ENDIF()
