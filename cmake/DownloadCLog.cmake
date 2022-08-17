# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(clog-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(clog
  URL https://github.com/pytorch/cpuinfo/archive/49610f89b8b1eb52d75d1eda7a2c40c1e86a78e7.zip
  URL_HASH SHA256=25843b5f21c32cba89f9b921c0500ab5cd0c2cb8fb0f345e5b5e4678329386c7
  SOURCE_DIR "${CMAKE_BINARY_DIR}/clog-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/clog"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
