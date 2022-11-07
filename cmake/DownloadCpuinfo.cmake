# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(cpuinfo-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(cpuinfo
  URL https://github.com/Maratyszcza/cpuinfo/archive/0a38bc5cf17837bf3b536b57b9d35a259b6b2283.zip
  URL_HASH SHA256=fc79c33f10b7dcb710c5eb0fcd7fe4467bf98cdc6ff1925883b175fbb800c53e
  SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/cpuinfo"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
