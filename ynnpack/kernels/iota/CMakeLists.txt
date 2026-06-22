# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

INCLUDE(../arch_copts.cmake)

ADD_LIBRARY(ynnpack_kernels_iota STATIC iota.cc)
TARGET_INCLUDE_DIRECTORIES(ynnpack_kernels_iota PUBLIC . ${CMAKE_BINARY_DIR})
TARGET_LINK_LIBRARIES(ynnpack_kernels_iota PUBLIC ynnpack_base ynnpack_simd)

FUNCTION(YNN_ADD_IOTA_SRC ARCH SRC)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    SET(ARCH_LIB "ynnpack_kernels_iota_${ARCH}")
    ADD_LIBRARY(${ARCH_LIB} STATIC ${SRC})
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    YNN_GET_ARCH_DEFINES(${ARCH} ARCH_DEFINES)
    TARGET_COMPILE_OPTIONS(${ARCH_LIB} PRIVATE ${ARCH_COPTS})
    TARGET_COMPILE_DEFINITIONS(${ARCH_LIB} PUBLIC ${ARCH_DEFINES})
    TARGET_INCLUDE_DIRECTORIES(${ARCH_LIB} PUBLIC . ${CMAKE_BINARY_DIR})
    TARGET_LINK_LIBRARIES(${ARCH_LIB} PUBLIC ynnpack_base ynnpack_simd)
    TARGET_LINK_LIBRARIES(ynnpack_kernels_iota PUBLIC ${ARCH_LIB})
  ENDIF()
ENDFUNCTION()

YNN_ADD_IOTA_SRC(arm_neon arm_neon.cc)
YNN_ADD_IOTA_SRC(x86_avx2 x86_avx2.cc)
YNN_ADD_IOTA_SRC(x86_avx512 x86_avx512.cc)
YNN_ADD_IOTA_SRC(x86_sse2 x86_sse2.cc)
YNN_ADD_IOTA_SRC(x86_sse41 x86_sse41.cc)



IF(YNNPACK_BUILD_TESTS)
  ADD_EXECUTABLE(ynnpack_kernels_iota_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_iota_test ynnpack_kernels_iota ynnpack ynnpack_base ynnpack_base_test_tensor ynnpack_base_test_util gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_iota_test COMMAND ynnpack_kernels_iota_test)
ENDIF()

IF(YNNPACK_BUILD_BENCHMARKS)
  ADD_EXECUTABLE(ynnpack_kernels_iota_bench bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_iota_bench ynnpack_kernels_iota ynnpack_base ynnpack_base_test_tensor benchmark::benchmark_main)
ENDIF()
