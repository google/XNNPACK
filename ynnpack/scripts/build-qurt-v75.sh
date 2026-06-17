#!/usr/bin/env bash
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build YNNPACK for Hexagon v75 (QURT) target.
#
# Run from the ynnpack/ directory:
#   ./scripts/build-qurt-v75.sh
#
# Prerequisites:
#   HEXAGON_SDK_ROOT  - path to Hexagon SDK (e.g. 5.4.0.3)
#   HEXAGON_TOOLS_ROOT - path to Hexagon tools (e.g. HEXAGON_Tools/8.7.03)

set -e

if [ -z "$HEXAGON_SDK_ROOT" ]; then
  echo "ERROR: HEXAGON_SDK_ROOT must be set"
  exit 1
fi

if [ -z "$HEXAGON_TOOLS_ROOT" ]; then
  echo "ERROR: HEXAGON_TOOLS_ROOT must be set"
  exit 1
fi

HEXAGON_ARCH=v75
BUILD_DIR="build/qurt/${HEXAGON_ARCH}"
mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/hexagon.toolchain.${HEXAGON_ARCH}")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]; then
  CMAKE_ARGS+=("-GNinja")
fi

# YNNPACK build options
CMAKE_ARGS+=("-DYNNPACK_BUILD_TESTS=ON")
CMAKE_ARGS+=("-DYNNPACK_BUILD_BENCHMARKS=ON")

# Hexagon HVX is enabled automatically by the toolchain (YNN_ENABLE_HVX=ON),
# but be explicit here for clarity
CMAKE_ARGS+=("-DYNN_ENABLE_HVX=ON")

# Disable ISAs that don't apply to Hexagon
CMAKE_ARGS+=("-DYNN_ENABLE_X86_SSE=OFF")
CMAKE_ARGS+=("-DYNN_ENABLE_ARM_NEON=OFF")
CMAKE_ARGS+=("-DYNN_ENABLE_ARM64=OFF")
CMAKE_ARGS+=("-DYNN_ENABLE_WASM_SIMD128=OFF")

# cpuinfo is not applicable on the DSP target
CMAKE_ARGS+=("-DYNN_ENABLE_CPUINFO=OFF")

# Cross-compilation option: Google Benchmark steady_clock is unavailable on QURT
CMAKE_ARGS+=("-DHAVE_STEADY_CLOCK=0")

# User-specified CMake arguments go last to allow overriding defaults
CMAKE_ARGS+=("$@")

cd "${BUILD_DIR}" && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

cmake --build . -- "-j$(nproc)"
