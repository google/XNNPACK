# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(ruy-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(ruy
  URL https://github.com/bjacob/ruy/archive/de0ac9792bb502f816f9606ef6007c5f2f25b8ec.zip
  URL_HASH SHA256=ec6755b8f6e6fea37e72cadf93e2590a980db7a4133d1817c40835dd9837dd56
  SOURCE_DIR "${CMAKE_BINARY_DIR}/ruy-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/ruy"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
