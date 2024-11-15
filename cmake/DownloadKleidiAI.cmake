# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(kleidiai-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
  CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

INCLUDE(ExternalProject)
ExternalProject_Add(kleidiai
  URL https://gitlab.arm.com/kleidi/kleidiai/-/archive/40a926833857fb64786e02f97703e42b1537cb57/kleidiai-40a926833857fb64786e02f97703e42b1537cb57.zip
  URL_HASH SHA256=ad37707084a6d4ff41be10cbe8540c75bea057ba79d0de6c367c1bfac6ba0852
  SOURCE_DIR "${CMAKE_BINARY_DIR}/kleidiai-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/kleidiai"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
