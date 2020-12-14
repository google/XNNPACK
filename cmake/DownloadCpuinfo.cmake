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
  URL https://github.com/pytorch/cpuinfo/archive/ed8b86a253800bafdb7b25c5c399f91bff9cb1f3.zip
  URL_HASH SHA256=a7f9a188148a1660149878f737f42783e72f33a4f842f3e362fee2c981613e53
  SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/cpuinfo"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND patch -p0 -i "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpuinfo.patch"
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
