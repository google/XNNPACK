# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
  URL https://gitlab.arm.com/kleidi/kleidiai/-/archive/02f7b3df98c39df1884eead7461c3269feca4ddc/kleidiai-02f7b3df98c39df1884eead7461c3269feca4ddc.zip
  URL_HASH SHA256=7408d45fabd3fef3b3b7b59638f4b69e3613603d2526b80300a77f85370c2936
  SOURCE_DIR "${CMAKE_BINARY_DIR}/kleidiai-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/kleidiai"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../MODULE.bazel:kleidiai)
