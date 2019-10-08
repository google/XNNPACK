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
  URL https://github.com/Maratyszcza/FXdiv/archive/f8c5354679ec2597792bc70a9e06eff50c508b9a.tar.gz
  URL_HASH SHA256=7d3215bea832fe77091ec5666200b91156df6724da1e348205078346325fc45e
  SOURCE_DIR "${CMAKE_BINARY_DIR}/FXdiv-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/FXdiv"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
