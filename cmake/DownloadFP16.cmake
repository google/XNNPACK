# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(fp16-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(fp16
  URL https://github.com/Maratyszcza/FP16/archive/ba1d31f5eed2eb4a69e4dea3870a68c7c95f998f.tar.gz
  URL_HASH SHA256=9764297a339ad73b0717331a2c3e9c42a52105cd04cab62cb160e2b4598d2ea6
  SOURCE_DIR "${CMAKE_BINARY_DIR}/FP16-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/FP16"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
