#!/usr/bin/env bash
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

mkdir -p build/qurt/v66

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/hexagon.toolchain")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
fi

CMAKE_ARGS+=("-DXNNPACK_BUILD_LIBRARY=OFF")
CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")

# Cross-compilation options for Google Benchmark
CMAKE_ARGS+=("-DHAVE_STEADY_CLOCK=0")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/qurt/v66 && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

cmake --build . -- "-j$(nproc)"
