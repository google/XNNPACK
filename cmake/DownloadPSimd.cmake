# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(psimd-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(psimd
  URL https://github.com/Maratyszcza/psimd/archive/85427dd4c8521cc037a1ffa6fcd25c55fafc8a00.zip
  URL_HASH SHA256=db23c2bc4a58d6f40c181797e43103300edac7cf9d286ca81590543f66ab95d2
  SOURCE_DIR "${CMAKE_BINARY_DIR}/psimd-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/psimd"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
