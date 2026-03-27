# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FIND_PACKAGE(Python3 REQUIRED)

INCLUDE(../arch_copts.cmake)

# Generates dot kernels from a script.
FUNCTION(YNN_GENERATE_DOT_KERNELS SCRIPT)
  SET(OUTPUTS)
  FOREACH(NAME ${ARGN})
    LIST(APPEND OUTPUTS "${CMAKE_BINARY_DIR}/ynnpack/kernels/dot/${NAME}.cc" "${CMAKE_BINARY_DIR}/ynnpack/kernels/dot/${NAME}.inc")
  ENDFOREACH()

  ADD_CUSTOM_COMMAND(
    OUTPUT ${OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ynnpack/kernels/dot"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/generator/${SCRIPT}.py" ${OUTPUTS}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generator/${SCRIPT}.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/generator/dot_base.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/generator/x86.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/generator/arm.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
ENDFUNCTION()

SET(DOT_KERNEL_SRCS)

# Helper to collect variants and generate them if any are enabled.
FUNCTION(YNN_ADD_DOT_KERNELS SCRIPT)
  SET(ACTIVE_VARIANTS)
  SET(ACTIVE_ARCHS)
  # Arguments after SCRIPT are pairs of (ARCH, VARIANT_NAME)
  SET(INDEX 0)
  LIST(LENGTH ARGN ARGC)
  WHILE(INDEX LESS ARGC)
    LIST(GET ARGN ${INDEX} ARCH_NAME)
    MATH(EXPR INDEX "${INDEX} + 1")
    LIST(GET ARGN ${INDEX} VARIANT_NAME)
    MATH(EXPR INDEX "${INDEX} + 1")

    YNN_CHECK_ARCH_ENABLED(${ARCH_NAME} ENABLED)
    IF(ENABLED)
      LIST(APPEND ACTIVE_VARIANTS ${VARIANT_NAME})
      LIST(APPEND ACTIVE_ARCHS ${ARCH_NAME})
    ENDIF()
  ENDWHILE()

  IF(ACTIVE_VARIANTS)
    YNN_GENERATE_DOT_KERNELS(${SCRIPT} ${ACTIVE_VARIANTS})

    LIST(LENGTH ACTIVE_VARIANTS LEN)
    MATH(EXPR LIMIT "${LEN} - 1")
    FOREACH(I RANGE ${LIMIT})
      LIST(GET ACTIVE_VARIANTS ${I} V_NAME)
      LIST(GET ACTIVE_ARCHS ${I} A_NAME)
      SET(OUT_CC "${CMAKE_BINARY_DIR}/ynnpack/kernels/dot/${V_NAME}.cc")
      YNN_GET_ARCH_COPTS(${A_NAME} ARCH_COPTS)
      SET_SOURCE_FILES_PROPERTIES("${OUT_CC}" PROPERTIES COMPILE_OPTIONS "${ARCH_COPTS}")
      LIST(APPEND DOT_KERNEL_SRCS "${OUT_CC}")
    ENDFOREACH()
    SET(DOT_KERNEL_SRCS ${DOT_KERNEL_SRCS} PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

# Helper for hand-written sources.
FUNCTION(YNN_ADD_DOT_SRC ARCH SRC)
  YNN_CHECK_ARCH_ENABLED(${ARCH} ENABLED)
  IF(ENABLED)
    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES COMPILE_OPTIONS "${ARCH_COPTS}")
    LIST(APPEND DOT_KERNEL_SRCS ${SRC})
    SET(DOT_KERNEL_SRCS ${DOT_KERNEL_SRCS} PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

# --- x86 kernels ---

YNN_ADD_DOT_KERNELS(x86_fp32
  x86_sse2 x86_sse2_fp32
  x86_avx x86_avx_fp32
  x86_fma3 x86_fma3_fp32
  x86_avx512 x86_avx512_fp32
)

YNN_ADD_DOT_KERNELS(x86_fp32_k2
  x86_avx2 x86_avx2_fp32_k2
  x86_avx2_fma3 x86_avx2_fma3_fp32_k2
  x86_avx512 x86_avx512_fp32_k2
)

YNN_ADD_DOT_KERNELS(x86_fp32_k4
  x86_avx512 x86_avx512_fp32_k4
)

YNN_ADD_DOT_KERNELS(x86_fp64
  x86_avx x86_avx_fp64
  x86_fma3 x86_fma3_fp64
  x86_avx512 x86_avx512_fp64
)

YNN_ADD_DOT_KERNELS(x86_fp16_fp16_fp32
  x86_f16c x86_f16c_fp16_fp16_fp32
  x86_f16c_fma3 x86_f16c_fma3_fp16_fp16_fp32
  x86_avx512fp16 x86_avx512_fp16_fp16_fp32
)

YNN_ADD_DOT_KERNELS(x86_bf16_bf16_fp32
  x86_avx2_fma3 x86_avx2_fma3_bf16_bf16_fp32
  x86_avx512 x86_avx512_bf16_bf16_fp32
  x86_avx512bf16 x86_avx512bf16_bf16_bf16_fp32
)

YNN_ADD_DOT_KERNELS(x86_bf16_bf16_fp32_k1
  x86_avx2_fma3 x86_avx2_fma3_bf16_bf16_fp32_k1
  x86_avx512 x86_avx512_bf16_bf16_fp32_k1
)

YNN_ADD_DOT_KERNELS(x86_int8_int8_int32
  x86_avx2 x86_avx2_int8_int8_int32
  x86_avx512 x86_avx512_int8_int8_int32
)

YNN_ADD_DOT_KERNELS(x86_int8_int8_int32_k1
  x86_avx2 x86_avx2_int8_int8_int32_k1
  x86_avx512 x86_avx512_int8_int8_int32_k1
)

YNN_ADD_DOT_KERNELS(x86_int8_int8_int32_k16
  x86_avx512 x86_avx512_int8_int8_int32_k16
)

YNN_ADD_DOT_KERNELS(x86_uint8_int8_int32
  x86_avx512vnni x86_avx512vnni_uint8_int8_int32
)

YNN_ADD_DOT_KERNELS(x86_uint8_int8_int32_k16
  x86_avx512vnni x86_avx512vnni_uint8_int8_int32_k16
)

YNN_ADD_DOT_SRC(x86_amxbf16 x86_amxbf16.cc)
YNN_ADD_DOT_SRC(x86_amxfp16 x86_amxfp16.cc)
YNN_ADD_DOT_SRC(x86_amxint8 x86_amxint8.cc)

# --- ARM kernels ---

YNN_ADD_DOT_KERNELS(arm_fp32
  arm64_neon arm64_neon_fp32
)

YNN_ADD_DOT_KERNELS(arm_fp64
  arm64_neon arm64_neon_fp64
)

YNN_ADD_DOT_KERNELS(arm_bf16_bf16_fp32
  arm64_neon arm64_neon_bf16_bf16_fp32
)

YNN_ADD_DOT_KERNELS(arm_int8_int8_int32
  arm64_neoni8mm arm64_neoni8mm_int8_int8_int32
  arm_neon arm_neon_int8_int8_int32
  arm_neondot arm_neondot_int8_int8_int32
)

YNN_ADD_DOT_KERNELS(arm_bf16_bf16_fp32_k2
  arm_neonbf16 arm_neonbf16_bf16_bf16_fp32_k2
)

YNN_ADD_DOT_KERNELS(arm_bf16_bf16_fp32_k4
  arm_neonbf16 arm_neonbf16_bf16_bf16_fp32_k4
)

YNN_ADD_DOT_SRC(arm64_sme arm64_sme.cc)
YNN_ADD_DOT_SRC(arm64_sme2 arm64_sme2.cc)

ADD_LIBRARY(ynnpack_kernels_dot STATIC
  dot.cc
  pack.cc
  schedule.cc
  ${DOT_KERNEL_SRCS}
)

TARGET_INCLUDE_DIRECTORIES(ynnpack_kernels_dot PUBLIC . ${CMAKE_BINARY_DIR})
TARGET_LINK_LIBRARIES(ynnpack_kernels_dot PUBLIC ynnpack_base ynnpack_kernels_transpose)

IF(YNNPACK_TARGET_PROCESSOR STREQUAL "arm64")
  TARGET_SOURCES(ynnpack_kernels_dot PRIVATE arm64_sme_stubs.cc)
ENDIF()

IF(YNNPACK_BUILD_TESTS)
  ADD_LIBRARY(ynnpack_kernels_dot_test_tensor INTERFACE)
  TARGET_INCLUDE_DIRECTORIES(ynnpack_kernels_dot_test_tensor INTERFACE .)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_dot_test_tensor INTERFACE ynnpack_kernels_dot ynnpack_base ynnpack_base_test_buffer ynnpack_base_test_tensor)

  ADD_EXECUTABLE(ynnpack_kernels_dot_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_dot_test ynnpack_kernels_dot ynnpack_kernels_dot_test_tensor ynnpack ynnpack_base ynnpack_base_test_buffer ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_dot_test COMMAND ynnpack_kernels_dot_test)

  ADD_EXECUTABLE(ynnpack_kernels_dot_consistent_arithmetic_test consistent_arithmetic_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_dot_consistent_arithmetic_test ynnpack_kernels_dot ynnpack_kernels_dot_test_tensor ynnpack_base ynnpack_base_test_buffer ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_dot_consistent_arithmetic_test COMMAND ynnpack_kernels_dot_consistent_arithmetic_test)

  ADD_EXECUTABLE(ynnpack_kernels_dot_schedule_test schedule_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_dot_schedule_test ynnpack_kernels_dot gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_dot_schedule_test COMMAND ynnpack_kernels_dot_schedule_test)

  ADD_EXECUTABLE(ynnpack_kernels_dot_pack_test pack_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_dot_pack_test ynnpack_kernels_dot ynnpack_base gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_dot_pack_test COMMAND ynnpack_kernels_dot_pack_test)

  ADD_EXECUTABLE(ynnpack_kernels_dot_get_dot_kernel_test get_dot_kernel_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_dot_get_dot_kernel_test ynnpack_kernels_dot ynnpack ynnpack_base gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_dot_get_dot_kernel_test COMMAND ynnpack_kernels_dot_get_dot_kernel_test)
ENDIF()
