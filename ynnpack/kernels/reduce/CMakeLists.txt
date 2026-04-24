# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_kernels_reduce STATIC reduce.cc)
TARGET_LINK_LIBRARIES(ynnpack_kernels_reduce PUBLIC ynnpack_base)

INCLUDE(../arch_copts.cmake)

SET(REDUCE_KERNEL_SRCS)

FUNCTION(YNN_ADD_REDUCE_SRC ARCH SRC)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES COMPILE_OPTIONS "${ARCH_COPTS}")
    LIST(APPEND REDUCE_KERNEL_SRCS ${SRC})
    SET(REDUCE_KERNEL_SRCS ${REDUCE_KERNEL_SRCS} PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

# ARM
YNN_ADD_REDUCE_SRC(arm_neon arm_neon.cc)
YNN_ADD_REDUCE_SRC(arm_neonbf16 arm_neonbf16.cc)
YNN_ADD_REDUCE_SRC(arm_neondot arm_neondot.cc)
YNN_ADD_REDUCE_SRC(arm_neonfma arm_neonfma.cc)
YNN_ADD_REDUCE_SRC(arm_neonfp16 arm_neonfp16.cc)

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

TARGET_SOURCES(ynnpack_kernels_reduce PRIVATE ${REDUCE_KERNEL_SRCS})
