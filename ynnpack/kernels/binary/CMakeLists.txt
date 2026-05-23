# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FIND_PACKAGE(Python3 REQUIRED)

INCLUDE(../arch_copts.cmake)

SET(BINARY_KERNEL_SRCS)

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

    YNN_GET_ARCH_COPTS(${ARCH} ARCH_COPTS)
    SET_SOURCE_FILES_PROPERTIES("${OUTPUT_SRC}" PROPERTIES COMPILE_OPTIONS "${ARCH_COPTS}")

    SET(${NAME}_SRC "${OUTPUT_SRC}" PARENT_SCOPE)
    LIST(APPEND BINARY_KERNEL_SRCS "${OUTPUT_SRC}")
    SET(BINARY_KERNEL_SRCS ${BINARY_KERNEL_SRCS} PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

# x86
YNN_GENERATE_BINARY_KERNEL(x86_sse2 x86_sse2)
YNN_GENERATE_BINARY_KERNEL(x86_avx x86_avx)
YNN_GENERATE_BINARY_KERNEL(x86_avx2 x86_avx2)
YNN_GENERATE_BINARY_KERNEL(x86_avx512 x86_avx512)
YNN_GENERATE_BINARY_KERNEL(x86_avx512bf16 x86_avx512bf16)

# ARM
YNN_GENERATE_BINARY_KERNEL(arm_neon arm_neon)

ADD_LIBRARY(ynnpack_kernels_binary STATIC
  binary.cc
  ${BINARY_KERNEL_SRCS}
)

TARGET_INCLUDE_DIRECTORIES(ynnpack_kernels_binary PUBLIC . ${CMAKE_BINARY_DIR})
TARGET_LINK_LIBRARIES(ynnpack_kernels_binary PUBLIC ynnpack_base)

IF(YNNPACK_BUILD_TESTS)
  ADD_LIBRARY(ynnpack_kernels_binary_reference STATIC reference.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_binary_reference PUBLIC ynnpack ynnpack_base ynnpack_base_test_tensor gtest)

  ADD_EXECUTABLE(ynnpack_kernels_binary_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_binary_test ynnpack_kernels_binary ynnpack_kernels_binary_reference ynnpack ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_binary_test COMMAND ynnpack_kernels_binary_test)
ENDIF()
