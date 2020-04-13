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
  URL https://github.com/Maratyszcza/FP16/archive/3c54eacb74f6f5e39077300c5564156c424d77ba.zip
  URL_HASH SHA256=0d56bb92f649ec294dbccb13e04865e3c82933b6f6735d1d7145de45da700156
  SOURCE_DIR "${CMAKE_BINARY_DIR}/FP16-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/FP16"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
