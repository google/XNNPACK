# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(kleidiai-qmx-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
  CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(kleidiai
  URL https://github.com/qualcomm/kleidiai/archive/3e605b85be4d3b411011bc8b8d09b12cc663ade3.zip 
  URL_HASH SHA256=e0ed9626832b3b4dcd061bdf3eaa64a9e2903319c4c834362de9877756056243
  SOURCE_DIR "${CMAKE_BINARY_DIR}/kleidiai-qmx-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/kleidiai-qmx"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:kleidiai)
