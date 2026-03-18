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
  URL https://github.com/google/benchmark/archive/7da00e8f6763d6e8c284d172c9cfcc5ae0ce9b7a.zip
  URL_HASH SHA256=3f676d4adb76e2eb0d84f997d8223705d2c6868eac35c0c65f76e5332a84e838
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:benchmark)
