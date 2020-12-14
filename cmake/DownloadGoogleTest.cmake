# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(googletest-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googletest
  URL https://github.com/google/googletest/archive/5a509dbd2e5a6c694116e329c5a20dc190653724.zip
  URL_HASH SHA256=fcfac631041fce253eba4fc014c28fd620e33e3758f64f8ed5487cc3e1840e3d
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googletest"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
