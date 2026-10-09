# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(slinky-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
  CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

INCLUDE(ExternalProject)
ExternalProject_Add(slinky
  URL https://github.com/dsharlet/slinky/archive/a3eec6fcfe3093be04d6c3d7dcab7d4e70bdd0dc.zip
  URL_HASH SHA256=cca25390873ad01a6e461edd39d220fb16c912beba2ebe02ae3b897ca63f30ef
  SOURCE_DIR "${CMAKE_BINARY_DIR}/slinky-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/slinky"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
