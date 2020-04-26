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
  URL https://github.com/pytorch/cpuinfo/archive/a1e0b9571b51131cf80613d061d2aa123876bd0a.zip
  URL_HASH SHA256=18918d39a77616bdcf948c99ed57d03d52109a820487fcda9fb5ee9b530d3d64
  SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/cpuinfo"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND "patch -p0 -i ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpuinfo.patch"
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
