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
  URL https://github.com/Maratyszcza/psimd/archive/8fd2884b88848180904a40c452a362d1ee429ad5.tar.gz
  URL_HASH SHA256=9d4f05bc5a93a0ab8bcef12027ebe54cfddd0050d4862442449c8de11b4e8c17
  SOURCE_DIR "${CMAKE_BINARY_DIR}/psimd-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/psimd"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
