#!/bin/bash
#
# This is a utility to deploy and run a simple Hexagon-based binary to the
# Hexagon simulator.
#
# Note that the binary must be built with the same HEXAGON_SDK_ROOT,
# HEXAGON_TOOLS_ROOT, HEXAGON_ARCH, and HEXAGON_TOOL_VER specified here,
# otherwise runtime errors will likely occur.
#
# Syntax: run-on-hexagon-sim.sh path-to-binary

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
: "${HEXAGON_TOOL_VER:=v86}"

BINARY_PATH=$1
[ -z "$BINARY_PATH" ] && echo "You must specify a path to the binary as the first argument" && exit 1
shift

EXE_NAME="$(basename ${BINARY_PATH})"
HEXAGON_LIBS_DIR=${HEXAGON_TOOLS_ROOT}/lib
HEXAGON_LIBC_DIR="${HEXAGON_TOOLS_ROOT}/target/hexagon/lib/${HEXAGON_ARCH}/G0/pic"
HEXAGON_SIM="${HEXAGON_TOOLS_ROOT}/bin/hexagon-sim"
SIM_STACK_SIZE=0x400000 # 4 MB

# Copy everything into a temp dir to avoid scrambling our build (etc) dirs.
HVX_SIM_TEMP_DIR=$(mktemp -d)

### Create configuration files for Hexagon simulator's cosimulation plugins
echo "${HVX_SIM_TEMP_DIR}/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > ${HVX_SIM_TEMP_DIR}/q6ss.cfg
echo "${HVX_SIM_TEMP_DIR}/l2vic.so 32 0xFC910000" >> ${HVX_SIM_TEMP_DIR}/q6ss.cfg

### Create RTOS config files
echo "${HVX_SIM_TEMP_DIR}/qurt_model.so" > ${HVX_SIM_TEMP_DIR}/osam.cfg

# Copy the executable and a bunch of helper libs into the same dir;
# trying to get the library loading correct otherwise is far too tricky.
cp -f ${BINARY_PATH} ${HVX_SIM_TEMP_DIR}/${EXE_NAME}
cp -f ${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/hexagon_tool${HEXAGON_TOOL_VER}_${HEXAGON_ARCH}/run_main_on_hexagon_sim ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_SDK_ROOT}/rtos/qurt/compute${HEXAGON_ARCH}/debugger/lnx64/qurt_model.so  ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_SDK_ROOT}/rtos/qurt/compute${HEXAGON_ARCH}/sdksim_bin/runelf.pbn  ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_LIBS_DIR}/iss/l2vic.so ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_LIBS_DIR}/iss/qtimer.so ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_LIBC_DIR}/libc++abi.so.1 ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_LIBC_DIR}/libc++abi.so.1.0 ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_LIBC_DIR}/libc++.so.1 ${HVX_SIM_TEMP_DIR}/
cp -f ${HEXAGON_LIBC_DIR}/libc++.so.1.0 ${HVX_SIM_TEMP_DIR}/

### Invoke the simulator for executable
cd ${HVX_SIM_TEMP_DIR}

LD_LIBRARY_PATH=${HVX_SIM_TEMP_DIR}:${LD_LIBRARY_PATH} \
  ${HEXAGON_SIM} \
  -m${HEXAGON_ARCH} \
  --simulated_returnval \
  --usefs . \
  --pmu_statsfile ${HVX_SIM_TEMP_DIR}/pmu_stats.txt \
  --cosim_file ${HVX_SIM_TEMP_DIR}/q6ss.cfg \
  --l2tcm_base 0xd800 \
  --rtos ${HVX_SIM_TEMP_DIR}/osam.cfg \
  ${HVX_SIM_TEMP_DIR}/runelf.pbn \
  -- \
  ${HVX_SIM_TEMP_DIR}/run_main_on_hexagon_sim \
  stack_size=${SIM_STACK_SIZE} \
  -- \
  ${HVX_SIM_TEMP_DIR}/${EXE_NAME} $@

if (($? != 0 )); then
  echo "Simulator run failed!"
  rm -rf "${HVX_SIM_TEMP_DIR}"
  exit 1
fi

### Cleanup
rm -rf "${HVX_SIM_TEMP_DIR}"
