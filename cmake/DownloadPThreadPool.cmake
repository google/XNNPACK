# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(pthreadpool-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(pthreadpool
  URL https://github.com/Maratyszcza/pthreadpool/archive/76042155a8b1e189c8f141429fd72219472c32e1.tar.gz
  URL_HASH SHA256=91c7b00c16c60c96f23d1966d524879c0f6044caf4bc5e9fc06518dda643e07e
  SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/pthreadpool"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
