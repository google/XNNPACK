#!/usr/bin/env bash
#
# Copyright 2021 Adobe
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

if [ "$1"  ==  "build_only" ] || [ "$2"  ==  "build_only" ] || [ "$3"  ==  "build_only" ];
then
    CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=OFF")
    CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=OFF")
else
    CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
    CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")
fi

if [ "$1"  ==  "enable_simd" ] || [ "$2"  ==  "enable_simd" ] || [ "$3"  ==  "enable_simd" ];
then
    CMAKE_ARGS+=("-DXNNPACK_BUILD_WASM_WITH_SIMD=ON")
fi

if [ "$1"  ==  "enable_threads" ] || [ "$2"  ==  "enable_threads" ] || [ "$3"  ==  "enable_threads" ];
then
    CMAKE_ARGS+=("-DXNNPACK_BUILD_WASM_WITH_PTHREAD=ON")
fi

CMAKE_ARGS+=()

cd build/local && emcmake cmake ../.. \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ "$(uname)" == "Darwin" ]
then
  emmake make "-j$(sysctl -n hw.ncpu)"
else
  emmake make "-j$(nproc)"
fi
