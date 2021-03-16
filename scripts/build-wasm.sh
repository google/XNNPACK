#!/usr/bin/env bash
#
# Copyright 2019 Adobe
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

if [ "$1" == "enable_tests" ] || [ "$2" == "enable_tests" ] || [ "$3" == "enable_tests" ] || [ "$4" == "enable_tests" ]; then
  echo "Building tests"
  CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")
else
  echo "Skipping tests"
  CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=OFF")
fi

if [ "$1" == "enable_benchmarks" ] || [ "$2" == "enable_benchmarks" ] || [ "$3" == "enable_benchmarks" ] || [ "$4" == "enable_benchmarks" ]; then
  echo "Building benchmarks"
  CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
else
  echo "Skipping benchmarks"
  CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=OFF")
fi

if [ "$1" == "enable_simd" ] || [ "$2" == "enable_simd" ] || [ "$3" == "enable_simd" ] || [ "$4" == "enable_simd" ]; then
  echo "Building with SIMD enabled"
  CMAKE_ARGS+=("-DXNNPACK_BUILD_WASM_WITH_SIMD=ON")
else
  CMAKE_ARGS+=("-DXNNPACK_BUILD_WASM_WITH_SIMD=OFF")
fi

if [ "$1" == "enable_threads" ] || [ "$2" == "enable_threads" ] || [ "$3" == "enable_threads" ] || [ "$4" == "enable_threads" ]; then
  echo "Building with threading support"
  CMAKE_ARGS+=("-DXNNPACK_BUILD_WASM_WITH_PTHREAD=ON")
else
  CMAKE_ARGS+=("-DXNNPACK_BUILD_WASM_WITH_PTHREAD=OFF")
fi

CMAKE_ARGS+=()

cd build/wasm && emcmake cmake ../.. \
  "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ "$(uname)" == "Darwin" ]; then
  emmake make "-j$(sysctl -n hw.ncpu)" --ignore-errors
else
  emmake make "-j$(nproc)"
fi
