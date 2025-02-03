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
  URL https://github.com/google/benchmark/archive/c58e6d0710581e3a08d65c349664128a8d9a2461.zip
  URL_HASH SHA256=3eb1d0e2f8abc548d34d65a8a16a8edd6fe1a949138a816dcf4955c74dfc492b
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:benchmark)
