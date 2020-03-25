# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(cpuinfo-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(cpuinfo
  URL https://github.com/pytorch/cpuinfo/archive/0cc563acb9baac39f2c1349bc42098c4a1da59e3.tar.gz
  URL_HASH SHA256=80625d0b69a3d69b70c2236f30db2c542d0922ccf9bb51a61bc39c49fac91a35
  SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/cpuinfo"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
