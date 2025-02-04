# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(googlebenchmark-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
  CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(googlebenchmark
  URL https://github.com/google/benchmark/archive/4a805f9f0f468bd4d499d060a1a1c6bd5d6b6b73.zip
  URL_HASH SHA256=a3f2e783628cee5b75166cb02af43fd1220479553c397ba4a35abd1d19d19ad3
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:benchmark)
