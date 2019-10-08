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
  URL https://github.com/Maratyszcza/pthreadpool/archive/0e275fe56094626349c55a524ea8b71a85daa64b.tar.gz
  URL_HASH SHA256=c2328fdf9e48ac9b928953bcbc442eb14402d393e4cfae0541581a3d39efca9d
  SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/pthreadpool"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
