# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(pthreadpool-download NONE)

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(pthreadpool
  URL https://github.com/Maratyszcza/pthreadpool/archive/b1b6536539d5ec04416d41a04af636f50a2f2b7a.zip
  URL_HASH SHA256=30cb30fb77cd903eda901651e63be8db5e2e33b1d06e4bbdd3f90a9a4432ae9a
  SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/pthreadpool"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
# LINT.ThenChange(../WORKSPACE.bazel)
