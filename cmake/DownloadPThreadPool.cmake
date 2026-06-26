# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019-2025 Google LLC
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
  URL https://github.com/google/pthreadpool/archive/02460584c6092e527c8b89f7df4de143d70e801f.zip
  URL_HASH SHA256=5ab4e8f63e3dcf62048360c216532bdf62f00dc204883a52d91230402f0feb6a
  SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/pthreadpool"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:pthreadpool)
