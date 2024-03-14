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

if [ -z "$ANDROID_NDK" ]
then
  echo "ANDROID_NDK not set; please set it to the Android NDK directory"
  exit 1
fi

if [ ! -d "$ANDROID_NDK" ]
then
  echo "ANDROID_NDK not a directory; did you install it under ${ANDROID_NDK}?"
  exit 1
fi

mkdir -p build/android/armeabi-v7a

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
fi

CMAKE_ARGS+=("-DXNNPACK_LIBRARY_TYPE=static")

CMAKE_ARGS+=("-DXNNPACK_BUILD_BENCHMARKS=ON")
CMAKE_ARGS+=("-DXNNPACK_BUILD_TESTS=ON")

# Cross-compilation options for Google Benchmark
CMAKE_ARGS+=("-DHAVE_POSIX_REGEX=0")
CMAKE_ARGS+=("-DHAVE_STEADY_CLOCK=0")
CMAKE_ARGS+=("-DHAVE_STD_REGEX=0")

# Android-specific options
CMAKE_ARGS+=("-DANDROID_NDK=$ANDROID_NDK")
CMAKE_ARGS+=("-DANDROID_ABI=armeabi-v7a")
CMAKE_ARGS+=("-DANDROID_PLATFORM=android-14")
CMAKE_ARGS+=("-DANDROID_PIE=ON")
CMAKE_ARGS+=("-DANDROID_STL=c++_static")
CMAKE_ARGS+=("-DANDROID_CPP_FEATURES=exceptions")

# BF16 instructions cause ICE in Android NDK compiler
CMAKE_ARGS+=("-DXNNPACK_ENABLE_ARM_BF16=OFF")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/android/armeabi-v7a && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ "$(uname)" == "Darwin" ]
then
  cmake --build . -- "-j$((2*$(sysctl -n hw.ncpu)))"
else
  cmake --build . -- "-j$((2*$(nproc)))"
fi
