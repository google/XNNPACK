# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(clog-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(clog
  URL https://github.com/pytorch/cpuinfo/archive/4b5a76c4de21265ddba98fc8f259e136ad11411b.zip
  URL_HASH SHA256=6000cf2a0befe428d97ea921372397d049889cbd8a4cd5b93390c71415dd3b68
  SOURCE_DIR "${CMAKE_BINARY_DIR}/clog-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/clog"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
