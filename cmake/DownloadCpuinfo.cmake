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
  URL https://github.com/pytorch/cpuinfo/archive/c2092219e7c874783a00a62edb94ddc672f57ab3.zip
  URL_HASH SHA256=ea56c399a4f6ca5f749e71acb6a7bfdc653eb65d8f658cb2e414a2fcdca1fe8b
  SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/cpuinfo"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND "patch -p0 -i ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpuinfo.patch"
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
