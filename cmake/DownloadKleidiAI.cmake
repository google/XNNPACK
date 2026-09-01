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

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(kleidiai
  URL https://gitlab.arm.com/kleidi/kleidiai/-/archive/4bc7bd457930de119f8b826ecfdebfbd9c08cf8c/kleidiai-4bc7bd457930de119f8b826ecfdebfbd9c08cf8c.zip
  URL_HASH SHA256=652ee046ebf99bee5ed78aaf8961ea1df6ac09b6ebbd3a33106d9ce255efba69
  SOURCE_DIR "${CMAKE_BINARY_DIR}/kleidiai-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/kleidiai"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:kleidiai)
