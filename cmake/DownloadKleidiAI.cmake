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
  URL https://gitlab.arm.com/kleidi/kleidiai/-/archive/8c6cf04366a3602ca974c4b262e29ae24b699556/kleidiai-8c6cf04366a3602ca974c4b262e29ae24b699556.zip
  URL_HASH SHA256=e6f2b475378173e5f5a41147d8255bdeee0d264f80ec6d23409bd0ea8dc88cd1
  SOURCE_DIR "${CMAKE_BINARY_DIR}/kleidiai-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/kleidiai"
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
