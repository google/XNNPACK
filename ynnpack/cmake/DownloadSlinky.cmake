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
  URL https://github.com/dsharlet/slinky/archive/8d77e68e391fd3a87d7cfb36dbe5efd6f9af73ca.zip
  URL_HASH SHA256=099510dcf18ade121331a94a536d1a4d50a97fd202934d419bd02faf2faecf38
  SOURCE_DIR "${CMAKE_BINARY_DIR}/slinky-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/slinky"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
