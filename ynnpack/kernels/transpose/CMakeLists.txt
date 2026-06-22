# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_kernels_transpose STATIC
  interleave.cc
  transpose.cc
)

TARGET_LINK_LIBRARIES(ynnpack_kernels_transpose PUBLIC ynnpack_base ynnpack_simd)

INCLUDE(../arch_copts.cmake)

FUNCTION(YNN_ADD_TRANSPOSE_SRC ARCH SRC)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    SET(ARCH_LIB "ynnpack_kernels_transpose_${ARCH}")
    ADD_LIBRARY(${ARCH_LIB} STATIC ${SRC})
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    YNN_GET_ARCH_DEFINES(${ARCH} ARCH_DEFINES)
    TARGET_COMPILE_OPTIONS(${ARCH_LIB} PRIVATE ${ARCH_COPTS})
    TARGET_COMPILE_DEFINITIONS(${ARCH_LIB} PUBLIC ${ARCH_DEFINES})
    TARGET_LINK_LIBRARIES(${ARCH_LIB} PUBLIC ynnpack_base ynnpack_simd)
    TARGET_LINK_LIBRARIES(ynnpack_kernels_transpose PUBLIC ${ARCH_LIB})
  ENDIF()
ENDFUNCTION()

# ARM
YNN_ADD_TRANSPOSE_SRC(arm_neon arm_neon.cc)
YNN_ADD_TRANSPOSE_SRC(arm64_sve arm64_sve.cc)

# x86
YNN_ADD_TRANSPOSE_SRC(x86_sse2 x86_sse2.cc)
YNN_ADD_TRANSPOSE_SRC(x86_avx x86_avx.cc)
YNN_ADD_TRANSPOSE_SRC(x86_avx2 x86_avx2.cc)
YNN_ADD_TRANSPOSE_SRC(x86_avx512 x86_avx512.cc)

# Others
YNN_ADD_TRANSPOSE_SRC(hexagon_hvx hexagon_hvx.cc)
YNN_ADD_TRANSPOSE_SRC(wasm_simd128 wasm_simd128.cc)



IF(YNNPACK_BUILD_TESTS)
  ADD_EXECUTABLE(ynnpack_kernels_transpose_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_transpose_test ynnpack_kernels_transpose ynnpack_base ynnpack_base_test_tensor ynnpack_base_test_util gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_transpose_test COMMAND ynnpack_kernels_transpose_test)
ENDIF()

IF(YNNPACK_BUILD_BENCHMARKS)
  ADD_EXECUTABLE(ynnpack_kernels_transpose_bench bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_transpose_bench ynnpack_kernels_transpose ynnpack_base ynnpack_base_test_tensor benchmark::benchmark_main)
ENDIF()
