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
  URL https://github.com/Maratyszcza/psimd/archive/4f2c53947184b56f58607b9e777416bb63ebbde1.tar.gz
  URL_HASH SHA256=7d1795ebf289af26e404cff5877c284775e491414cf41d7d99ab850ceaced458
  SOURCE_DIR "${CMAKE_BINARY_DIR}/psimd-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/psimd"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
