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
  URL https://github.com/google/googletest/archive/56efe3983185e3f37e43415d1afa97e3860f187f.zip
  URL_HASH SHA256=47925a240670c819eda1df2590a40e58c68133aa88022df5a2b13c84251f62f5
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googletest"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:googletest)
