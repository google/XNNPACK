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
  URL https://github.com/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz
  URL_HASH SHA256=3f2dc1970f397a0e59db72f9fca6ff144b216895c1d606f6c94a507c1e53a025
  SOURCE_DIR "${CMAKE_BINARY_DIR}/clog-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/clog"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
