#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

if ! command -v emcmake >/dev/null 2>&1; then
  echo "emcmake not found in PATH, please install emscripten SDK and add it to PATH"
  exit 1
fi

mkdir -p build/wasm

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

CMAKE_ARGS+=("-DXNNPACK_LIBRARY_TYPE=static")

CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")


# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/wasm && emcmake cmake ../.. \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ "$(uname)" == "Darwin" ]
then
  cmake --build . -- "-j$((2*$(sysctl -n hw.ncpu)))"
else
  cmake --build . -- "-j$((2*$(nproc)))"
fi
