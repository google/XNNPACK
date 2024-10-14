# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(googletest-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
  CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(googletest
  URL https://github.com/google/googletest/archive/62df7bdbc10887e094661e07ec2595b7920376fd.zip
  URL_HASH SHA256=5a5f23d3d788a0b6e10c45f30128e126a7f3db01c5c8026f0f68e64d6d7a26f0
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googletest"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../WORKSPACE.bazel)
