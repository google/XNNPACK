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
  URL https://github.com/dsharlet/slinky/archive/b6b9cdfba51ad17959a8314f338e2931c55f74d6.zip
  URL_HASH SHA256=71584e401c60a576cc934353d5d437b49b032dd315820fd810bb44629acff2bf
  SOURCE_DIR "${CMAKE_BINARY_DIR}/slinky-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/slinky"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
