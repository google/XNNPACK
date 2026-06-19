#!/bin/bash
#
# This is a utility to deploy and run a simple Hexagon-based binary to a
# Hexagon device connected via adb.
#
# Note that the binary must be built with the same HEXAGON_SDK_ROOT,
# HEXAGON_TOOLS_ROOT, HEXAGON_SDK_VER, and HEXAGON_TOOL_VER specified here,
# otherwise runtime errors will likely occur.
#
# Syntax: run-on-hexagon-device.sh path-to-binary

set -e

if [ -z "$HEXAGON_SDK_ROOT" ]; then
  echo "HEXAGON_SDK_ROOT must be set!"
  exit 1
fi

if [ -z "$HEXAGON_TOOLS_ROOT" ]; then
  echo "HEXAGON_TOOLS_ROOT must be set!"
  exit 1
fi

# These should match the versions specified in hexagon.toolchain
: "${HEXAGON_ARCH:=v68}"
: "${HEXAGON_SDK_VER:=5.3.0}"
: "${HEXAGON_TOOL_VER:=v86}"
: "${HEXAGON_DEVICE_FOLDER:=/data/local/tmp/run_main_on_hexagon}"

BINARY_PATH=$1
[ -z "$BINARY_PATH" ] && echo "You must specify a path to the binary as the first argument" && exit 1
shift

echo "Pushing to device..."
DSP_DIR=${HEXAGON_DEVICE_FOLDER}/dsp
BINARY_NAME=$(basename ${BINARY_PATH})
adb shell mkdir -p ${HEXAGON_DEVICE_FOLDER}
adb shell mkdir -p ${DSP_DIR}

LIB_EXPORT_STRING="ADSP_LIBRARY_PATH=${DSP_DIR}"
EXECUTION_STRING="chmod +x ${HEXAGON_DEVICE_FOLDER}/run_main_on_hexagon; ${LIB_EXPORT_STRING} ${HEXAGON_DEVICE_FOLDER}/run_main_on_hexagon 3 ${BINARY_NAME}"

adb shell "echo "0x1f" > ${DSP_DIR}/run_main_on_hexagon.farf"
adb push --sync ${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon ${HEXAGON_DEVICE_FOLDER}
adb push --sync ${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/hexagon_tool${HEXAGON_TOOL_VER}_${HEXAGON_ARCH}/librun_main_on_hexagon_skel.so ${DSP_DIR}
adb push --sync ${BINARY_PATH} ${DSP_DIR}

echo "Running..."
adb shell ${EXECUTION_STRING} $@

echo "Done!"
