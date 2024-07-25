#!/usr/bin/env bash
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

HEXAGON_ARCH=v68
mkdir -p build/qurt/${HEXAGON_ARCH}

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/hexagon.toolchain")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
fi

CMAKE_ARGS+=("-DXNNPACK_BUILD_LIBRARY=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")
CMAKE_ARGS+=("-DXNNPACK_ENABLE_RISCV_VECTOR=OFF")

# Cross-compilation options for Google Benchmark
CMAKE_ARGS+=("-DHAVE_STEADY_CLOCK=0")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/qurt/${HEXAGON_ARCH} && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

cmake --build . -- "-j$((2*$(nproc)))"
