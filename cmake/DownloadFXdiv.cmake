# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(fxdiv-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(fxdiv
  URL https://github.com/Maratyszcza/FXdiv/archive/f7dd0576a1c8289ef099d4fd8b136b1c4487a873.zip
  URL_HASH SHA256=6e4b6e3c58e67c3bb090e286c4f235902c89b98cf3e67442a18f9167963aa286
  SOURCE_DIR "${CMAKE_BINARY_DIR}/FXdiv-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/FXdiv"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
