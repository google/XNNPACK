#!/usr/bin/env bash
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

mkdir -p build/linux/armhf

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/armhf.toolchain")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
fi

CMAKE_ARGS+=("-DXNNPACK_LIBRARY_TYPE=static")

CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/linux/armhf && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

cmake --build . -- "-j$((2*$(nproc)))"
