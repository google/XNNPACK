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
  URL https://gitlab.arm.com/kleidi/kleidiai/-/archive/aa070773b54d49d9740b47122c99fa1c55a370e1/kleidiai-cddf991af5de49fd34949fa39690e4e906e04074.zip
  URL_HASH SHA256=f546dc7f176603bb8f7d5477b1e06a4f2c2f4dd2bbc3a53955f3259adb8d7f50
  SOURCE_DIR "${CMAKE_BINARY_DIR}/kleidiai-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/kleidiai"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
