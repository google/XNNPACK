# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_kernels_transpose STATIC
  interleave.cc
  transpose.cc
)

TARGET_LINK_LIBRARIES(ynnpack_kernels_transpose PUBLIC ynnpack_base)

INCLUDE(../arch_copts.cmake)

SET(TRANSPOSE_KERNEL_SRCS)

FUNCTION(YNN_ADD_TRANSPOSE_SRC ARCH SRC)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES COMPILE_OPTIONS "${ARCH_COPTS}")
    LIST(APPEND TRANSPOSE_KERNEL_SRCS ${SRC})
    SET(TRANSPOSE_KERNEL_SRCS ${TRANSPOSE_KERNEL_SRCS} PARENT_SCOPE)
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

TARGET_SOURCES(ynnpack_kernels_transpose PRIVATE ${TRANSPOSE_KERNEL_SRCS})

IF(YNNPACK_BUILD_TESTS)
  ADD_EXECUTABLE(ynnpack_kernels_transpose_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_transpose_test ynnpack_kernels_transpose ynnpack_base ynnpack_base_test_tensor ynnpack_base_test_util gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_transpose_test COMMAND ynnpack_kernels_transpose_test)
ENDIF()
