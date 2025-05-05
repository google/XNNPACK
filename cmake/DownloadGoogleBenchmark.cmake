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
  URL https://github.com/google/benchmark/archive/8d4fdd6e6e003867045e0bb3473b5b423818e4b7.zip
  URL_HASH SHA256=28c7cac12cc25d87d3dcc8c5fb7d1bd0971b41a599a5c4787f8742cb39ca47db
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:benchmark)
