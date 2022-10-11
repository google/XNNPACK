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

if [ ! -d "/Library/Developer/CommandLineTools" ] || [ ! -x "$(command -v xcodebuild)" ] 
then
  echo "Have you installed Xcode?"
  exit 1
fi

mkdir -p build/iOS/arm64

# Create Tool-chain file
IOS_TOOL_CHAIN=build/iOS/arm64/ios_tmp.toolchain.cmake
cat << EOF > ${IOS_TOOL_CHAIN}
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED NO)
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED NO)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")
EOF

CMAKE_ARGS=()
# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$IOS_TOOL_CHAIN")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
else
  CMAKE_ARGS+=("-GXcode")
fi

CMAKE_ARGS+=("-DXNNPACK_LIBRARY_TYPE=static")

CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")

# Cross-compilation options for Google Benchmark
CMAKE_ARGS+=("-DHAVE_POSIX_REGEX=0")
CMAKE_ARGS+=("-DHAVE_STEADY_CLOCK=0")
CMAKE_ARGS+=("-DHAVE_STD_REGEX=0")

# iOS-specific options
CMAKE_ARGS+=("-DIOS_ARCH=arm64")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/iOS/arm64 && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

cmake --build .
