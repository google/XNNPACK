# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(pthreadpool-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
  CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(pthreadpool
  URL https://github.com/google/pthreadpool/archive/c2ba5c50bb58d1397b693740cf75fad836a0d1bf.zip
  URL_HASH SHA256=516ba8d05c30e016d7fd7af6a7fc74308273883f857faf92bc9bb630ab6dba2c
  SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/pthreadpool"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:pthreadpool)
