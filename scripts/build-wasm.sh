#!/usr/bin/env bash
#
# Copyright (c) 2021 Adobe
# All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

rm -rf build/wasm
mkdir -p build/wasm

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DXNNPACK_LIBRARY_TYPE=static")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
fi

CMAKE_ARGS+=()

CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=OFF")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=OFF")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

if [[ "$*" == "-DXNNPACK_BUILD_TESTS=ON" ]]
then
    CMAKE_ARGS+=("-DCMAKE_EXE_LINKER_FLAGS=-s ASSERTIONS=2 -s ERROR_ON_UNDEFINED_SYMBOLS=1 -s DEMANGLE_SUPPORT=1 -s EXIT_RUNTIME=1 -s ALLOW_MEMORY_GROWTH=1")
    CMAKE_ARGS+=("-DCMAKE_EXECUTABLE_SUFFIX=\".html\"")
fi

if [[ "$*" == "-DXNNPACK_BUILD_BENCHMARKS=ON" ]]
then
    CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")
    CMAKE_ARGS+=("-DCMAKE_EXE_LINKER_FLAGS=-s ASSERTIONS=2 -s ERROR_ON_UNDEFINED_SYMBOLS=1 -s EXIT_RUNTIME=1 -s ALLOW_MEMORY_GROWTH=1 -s TOTAL_MEMORY=436207616")
    CMAKE_ARGS+=("-DCMAKE_EXECUTABLE_SUFFIX=\".html\"")
fi

cd build/wasm && emcmake cmake ../.. \
  "${CMAKE_ARGS[@]}"

cmake --build .
