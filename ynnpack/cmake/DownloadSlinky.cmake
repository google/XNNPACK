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
  URL https://github.com/dsharlet/slinky/archive/9b2ee975f1a8a67a4daf1df6ce0911b66e6faf9f.zip
  URL_HASH SHA256=56793854bfd71f81597a59e4590d2fa4c444319c917ac387f5bde8c83b407a0a
  SOURCE_DIR "${CMAKE_BINARY_DIR}/slinky-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/slinky"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
