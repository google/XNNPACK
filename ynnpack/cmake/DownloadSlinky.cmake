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
  URL https://github.com/dsharlet/slinky/archive/8eda4a050105a7148d3d6cb77960077790d69b01.zip
  URL_HASH SHA256=eee999088e6105043663abf22a00268b22013151166109fafb15635da81f2de6
  SOURCE_DIR "${CMAKE_BINARY_DIR}/slinky-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/slinky"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
