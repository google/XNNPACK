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
  URL https://github.com/dsharlet/benchmark/archive/7fb2aa909f2d21a343e4bf7344f1478c7f07e8e8.zip
  #URL https://github.com/google/benchmark/archive/559b7cc1aec1950a9e3f4e879b08cf0b00f796f0.zip
  #URL_HASH SHA256=fddfa6ce9a011c37564a7feeceba3c7ce74d6a05d0c70142a61c3a2f5ffefd58
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:benchmark)
