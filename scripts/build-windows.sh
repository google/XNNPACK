#!/usr/bin/env bash
#
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GitHub Actions Windows runner will run this using Git Bash.
set -e

mkdir -p build/local

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
fi

CMAKE_ARGS+=("-DXNNPACK_LIBRARY_TYPE=static")

# We run out of disk space and timeout on Windows, so build less.
CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=OFF")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=OFF")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/local && cmake ../.. \
    "${CMAKE_ARGS[@]}"

cmake --build .
