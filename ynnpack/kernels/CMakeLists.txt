# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Override ADD_LIBRARY to automatically apply -O2 to all kernel libraries.
MACRO(ADD_LIBRARY TARGET)
  _ADD_LIBRARY(${TARGET} ${ARGN})
  SET(IS_INTERFACE OFF)
  FOREACH(ARG ${ARGN})
    IF(ARG STREQUAL "INTERFACE")
      SET(IS_INTERFACE ON)
      BREAK()
    ENDIF()
  ENDFOREACH()
  IF("${TARGET}" MATCHES "^ynnpack_kernels_" AND NOT IS_INTERFACE)
    TARGET_COMPILE_OPTIONS(${TARGET} PRIVATE "-O2")
  ENDIF()
ENDMACRO()

ADD_SUBDIRECTORY(binary)
ADD_SUBDIRECTORY(dequantize_dot)
ADD_SUBDIRECTORY(dot)
ADD_SUBDIRECTORY(iota)
ADD_SUBDIRECTORY(lut)
ADD_SUBDIRECTORY(reduce)
ADD_SUBDIRECTORY(ternary)
ADD_SUBDIRECTORY(transpose)
ADD_SUBDIRECTORY(unary)
