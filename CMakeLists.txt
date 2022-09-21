# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.15 FATAL_ERROR)

# MSVC runtime library flags are selected by an abstraction.
CMAKE_POLICY(SET CMP0091 NEW)

# ---[ Project and semantic versioning.
PROJECT(XNNPACK C CXX ASM)

SET(CMAKE_C_STANDARD 99)
SET(CMAKE_C_EXTENSIONS NO)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED YES)
SET(CMAKE_CXX_EXTENSIONS NO)

# ---[ Options.
SET(XNNPACK_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE XNNPACK_LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(XNNPACK_ENABLE_ASSEMBLY "Build XNNPACK with assembly micro-kernels" ON)
OPTION(XNNPACK_ENABLE_JIT "Build XNNPACK with JIT micro-kernels" OFF)
OPTION(XNNPACK_ENABLE_MEMOPT "Build XNNPACK with optimized memory allocation scheme" ON)
OPTION(XNNPACK_ENABLE_SPARSE "Build XNNPACK with graph rewriting for sparse inference" ON)
OPTION(XNNPACK_ENABLE_GEMM_M_SPECIALIZATION "Build XNNPACK with support for selecting microkernel with different MR" ON)
OPTION(XNNPACK_BUILD_TESTS "Build XNNPACK unit tests" ON)
OPTION(XNNPACK_BUILD_BENCHMARKS "Build XNNPACK benchmarks" ON)
OPTION(XNNPACK_USE_SYSTEM_LIBS "Use system-provided dependency libraries" OFF)

# --- [ Determine target processor
SET(XNNPACK_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
IF(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|arm64|arm64e)$")
  SET(XNNPACK_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
ENDIF()

# --- [ Processor-specific options
OPTION(XNNPACK_ENABLE_ARM_FP16 "Build XNNPACK with ARM FP16 (FP16 data processing) micro-kernels" ON)
OPTION(XNNPACK_ENABLE_ARM_BF16 "Build XNNPACK with ARM BF16 (BFLOAT16) micro-kernels" ON)
OPTION(XNNPACK_ENABLE_ARM_DOTPROD "Build XNNPACK with ARM DotProd (integer dot product) micro-kernels" ON)

# ---[ CMake options
INCLUDE(GNUInstallDirs)

IF(XNNPACK_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

ADD_COMPILE_DEFINITIONS("XNN_ENABLE_ARM_FP16=$<BOOL:${XNNPACK_ENABLE_ARM_FP16}>")
ADD_COMPILE_DEFINITIONS("XNN_ENABLE_ARM_BF16=$<BOOL:${XNNPACK_ENABLE_ARM_BF16}>")
ADD_COMPILE_DEFINITIONS("XNN_ENABLE_ARM_DOTPROD=$<BOOL:${XNNPACK_ENABLE_ARM_DOTPROD}>")
ADD_COMPILE_DEFINITIONS("XNN_ENABLE_ASSEMBLY=$<BOOL:${XNNPACK_ENABLE_ASSEMBLY}>")
ADD_COMPILE_DEFINITIONS("XNN_ENABLE_JIT=$<BOOL:${XNNPACK_ENABLE_JIT}>")
ADD_COMPILE_DEFINITIONS("XNN_ENABLE_MEMOPT=$<BOOL:${XNNPACK_ENABLE_MEMOPT}>")
ADD_COMPILE_DEFINITIONS("XNN_ENABLE_SPARSE=$<BOOL:${XNNPACK_ENABLE_SPARSE}>")
ADD_COMPILE_DEFINITIONS("XNN_ENABLE_GEMM_M_SPECIALIZATION=$<BOOL:${XNNPACK_ENABLE_GEMM_M_SPECIALIZATION}>")

IF(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  # Disable "unary minus operator applied to unsigned type, result still unsigned" warning
  ADD_COMPILE_OPTIONS("/wd4146")
ENDIF()

IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Disable "note: parameter passing for argument of type ... changed/will change in ..."
  ADD_COMPILE_OPTIONS("-Wno-psabi")
ENDIF()

# ---[ Build flags
IF(NOT CMAKE_SYSTEM_PROCESSOR)
  IF(IOS)
    LIST(LENGTH IOS_ARCH IOS_ARCH_COUNT)
    IF(IOS_ARCH_COUNT GREATER 1)
      MESSAGE(FATAL_ERROR "Unsupported XNNPACK build with multiple iOS architectures (${IOS_ARCH}). "
        "Specify a single architecture in IOS_ARCH and re-configure. ")
    ENDIF()
    IF(NOT IOS_ARCH MATCHES "^(i386|x86_64|AMD64|armv7.*|arm64.*)$")
      MESSAGE(FATAL_ERROR "Unrecognized IOS_ARCH = ${IOS_ARCH}")
    ENDIF()
  ELSE()
    MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR is not defined")
  ENDIF()
ELSEIF(NOT XNNPACK_TARGET_PROCESSOR MATCHES "^(i[3-6]86|x86|x86_64|AMD64|armv[5-8].*|aarch64|arm64.*|riscv(32|64|128))$")
  MESSAGE(FATAL_ERROR "Unrecognized XNNPACK_TARGET_PROCESSOR = ${XNNPACK_TARGET_PROCESSOR}")
ENDIF()

IF(NOT CMAKE_SYSTEM_NAME)
  MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_NAME not defined")
ELSEIF(NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux|Android|Windows|CYGWIN|MSYS)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
ENDIF()

# ---[ Download deps
IF(NOT XNNPACK_USE_SYSTEM_LIBS)
  IF(NOT DEFINED CLOG_SOURCE_DIR)
    MESSAGE(STATUS "Downloading clog to ${CMAKE_BINARY_DIR}/clog-source (define CLOG_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadCLog.cmake "${CMAKE_BINARY_DIR}/clog-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/clog-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/clog-download")
    SET(CLOG_SOURCE_DIR "${CMAKE_BINARY_DIR}/clog-source" CACHE STRING "clog source directory")
  ENDIF()

  IF(NOT DEFINED CPUINFO_SOURCE_DIR)
    MESSAGE(STATUS "Downloading cpuinfo to ${CMAKE_BINARY_DIR}/cpuinfo-source (define CPUINFO_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadCpuinfo.cmake "${CMAKE_BINARY_DIR}/cpuinfo-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/cpuinfo-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/cpuinfo-download")
    SET(CPUINFO_SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo-source" CACHE STRING "cpuinfo source directory")
  ENDIF()

  IF(NOT DEFINED FP16_SOURCE_DIR)
    MESSAGE(STATUS "Downloading FP16 to ${CMAKE_BINARY_DIR}/FP16-source (define FP16_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadFP16.cmake "${CMAKE_BINARY_DIR}/FP16-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/FP16-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/FP16-download")
    SET(FP16_SOURCE_DIR "${CMAKE_BINARY_DIR}/FP16-source" CACHE STRING "FP16 source directory")
  ENDIF()

  IF(NOT DEFINED FXDIV_SOURCE_DIR)
    MESSAGE(STATUS "Downloading FXdiv to ${CMAKE_BINARY_DIR}/FXdiv-source (define FXDIV_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadFXdiv.cmake "${CMAKE_BINARY_DIR}/FXdiv-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/FXdiv-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/FXdiv-download")
    SET(FXDIV_SOURCE_DIR "${CMAKE_BINARY_DIR}/FXdiv-source" CACHE STRING "FXdiv source directory")
  ENDIF()

  IF(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
    MESSAGE(STATUS "Downloading pthreadpool to ${CMAKE_BINARY_DIR}/pthreadpool-source (define PTHREADPOOL_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadPThreadPool.cmake "${CMAKE_BINARY_DIR}/pthreadpool-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")
    SET(PTHREADPOOL_SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source" CACHE STRING "pthreadpool source directory")
  ENDIF()

  IF(XNNPACK_BUILD_TESTS AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Test to ${CMAKE_BINARY_DIR}/googletest-source (define GOOGLETEST_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
    SET(GOOGLETEST_SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source" CACHE STRING "Google Test source directory")
  ENDIF()

  IF(XNNPACK_BUILD_BENCHMARKS AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Benchmark to ${CMAKE_BINARY_DIR}/googlebenchmark-source (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
    SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source" CACHE STRING "Google Benchmark source directory")
  ENDIF()
ENDIF()

# ---[ XNNPACK library
SET(OPERATOR_SRCS
  src/operator-delete.c
  src/operator-run.c
  src/operator-utils.c
  src/operators/argmax-pooling-nhwc.c
  src/operators/average-pooling-nhwc.c
  src/operators/binary-elementwise-nd.c
  src/operators/channel-shuffle-nc.c
  src/operators/constant-pad-nd.c
  src/operators/convolution-nchw.c
  src/operators/convolution-nhwc.c
  src/operators/deconvolution-nhwc.c
  src/operators/fully-connected-nc.c
  src/operators/global-average-pooling-ncw.c
  src/operators/global-average-pooling-nwc.c
  src/operators/lut-elementwise-nc.c
  src/operators/max-pooling-nhwc.c
  src/operators/prelu-nc.c
  src/operators/resize-bilinear-nchw.c
  src/operators/resize-bilinear-nhwc.c
  src/operators/softmax-nc.c
  src/operators/transpose-nd.c
  src/operators/unary-elementwise-nc.c
  src/operators/unpooling-nhwc.c)

SET(SUBGRAPH_SRCS
  src/memory-planner.c
  src/runtime.c
  src/subgraph.c
  src/subgraph/abs.c
  src/subgraph/add2.c
  src/subgraph/argmax-pooling-2d.c
  src/subgraph/average-pooling-2d.c
  src/subgraph/bankers-rounding.c
  src/subgraph/ceiling.c
  src/subgraph/clamp.c
  src/subgraph/concatenate.c
  src/subgraph/convert.c
  src/subgraph/convolution-2d.c
  src/subgraph/copy.c
  src/subgraph/deconvolution-2d.c
  src/subgraph/depth-to-space.c
  src/subgraph/depthwise-convolution-2d.c
  src/subgraph/divide.c
  src/subgraph/elu.c
  src/subgraph/even-split.c
  src/subgraph/floor.c
  src/subgraph/fully-connected.c
  src/subgraph/global-average-pooling.c
  src/subgraph/hardswish.c
  src/subgraph/leaky-relu.c
  src/subgraph/max-pooling-2d.c
  src/subgraph/maximum2.c
  src/subgraph/minimum2.c
  src/subgraph/multiply2.c
  src/subgraph/negate.c
  src/subgraph/prelu.c
  src/subgraph/sigmoid.c
  src/subgraph/softmax.c
  src/subgraph/square-root.c
  src/subgraph/square.c
  src/subgraph/squared-difference.c
  src/subgraph/static-constant-pad.c
  src/subgraph/static-reshape.c
  src/subgraph/static-resize-bilinear-2d.c
  src/subgraph/static-transpose.c
  src/subgraph/subtract.c
  src/subgraph/unpooling-2d.c
  src/subgraph/validation.c
  src/tensor.c)

SET(LOGGING_SRCS
  src/datatype-strings.c
  src/node-type.c
  src/operator-strings.c
  src/ukernel-strings.c)

SET(COLD_SRCS
  src/init.c
  src/log.c
  src/params.c)

SET(HOT_SRCS
  src/indirection.c
  src/packing.c)

SET(ALLOCATOR_SRCS
  src/allocator.c
  src/memory.c)

SET(TABLE_SRCS
  src/tables/exp2-k-over-64.c
  src/tables/exp2-k-over-2048.c
  src/tables/exp2minus-k-over-4.c
  src/tables/exp2minus-k-over-8.c
  src/tables/exp2minus-k-over-16.c
  src/tables/exp2minus-k-over-64.c
  src/tables/exp2minus-k-over-2048.c
  src/tables/vlog.c)

SET(JIT_SRCS
  src/jit/aarch32-assembler.cc
  src/jit/aarch64-assembler.cc
  src/jit/assembler.cc)

SET(JIT_AARCH32_SRCS
  src/f32-gemm/4x8-aarch32-neon-cortex-a7.cc
  src/f32-gemm/4x8-aarch32-neon-cortex-a53.cc
  src/f32-gemm/4x8-aarch32-neon-cortex-a55.cc
  src/f32-gemm/4x8-aarch32-neon-cortex-a75.cc
  src/f32-gemm/4x8-aarch32-neon-ld64.cc
  src/f32-igemm/4x8-aarch32-neon-cortex-a7.cc
  src/f32-igemm/4x8-aarch32-neon-cortex-a53.cc
  src/f32-igemm/4x8-aarch32-neon-cortex-a55.cc
  src/f32-igemm/4x8-aarch32-neon-cortex-a75.cc
  src/f32-igemm/4x8-aarch32-neon-ld64.cc
  src/qc8-gemm/4x8-fp32-aarch32-neonv8-mlal-lane-ld64.cc
  src/qc8-gemm/4x8c4-fp32-aarch32-neondot-ld64.cc
  src/qc8-igemm/4x8-fp32-aarch32-neonv8-mlal-lane-ld64.cc
  src/qc8-igemm/4x8c4-fp32-aarch32-neondot-ld64.cc
  src/qs8-gemm/4x8-rndnu-aarch32-neon-mlal-lane-ld64.cc
  src/qs8-gemm/4x8c4-rndnu-aarch32-neondot-ld64.cc
  src/qs8-igemm/4x8-rndnu-aarch32-neon-mlal-lane-ld64.cc
  src/qs8-igemm/4x8c4-rndnu-aarch32-neondot-ld64.cc)

SET(JIT_AARCH64_SRCS
  src/f32-gemm/upto6x8-aarch64-neonfma-cortex-a75.cc
  src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.cc
  src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.cc
  src/f32-gemm/6x8-aarch64-neonfma-ld128.cc
  src/f32-igemm/upto6x8-aarch64-neonfma-cortex-a75.cc
  src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.cc
  src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.cc
  src/f32-igemm/6x8-aarch64-neonfma-ld128.cc)

SET(PROD_SCALAR_PORTABLE_MICROKERNEL_SRCS
  src/u8-lut32norm/scalar.c
  src/xx-copy/memcpy.c
  src/xx-transpose/1x1-memcpy.c
  src/x8-lut/gen/lut-scalar-x4.c)

SET(PROD_SCALAR_AARCH32_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-scalar-x4.c
  src/f32-argmaxpool/4x-scalar-c1.c
  src/f32-argmaxpool/9p8x-scalar-c1.c
  src/f32-argmaxpool/9x-scalar-c1.c
  src/f32-avgpool/9p8x-minmax-scalar-c1.c
  src/f32-avgpool/9x-minmax-scalar-c1.c
  src/f32-conv-hwc/3x3s2p0p1c3x4-scalar-1x1.c
  src/f32-conv-hwc/3x3s2p1c3x4-scalar-1x1.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-scalar-1x1.c
  src/f32-dwconv/gen/up1x3-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x3-scalar-acc2.c
  src/f32-dwconv/gen/up1x4-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x4-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-scalar-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-4x1.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-2x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-2x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-2x1-acc2.c
  src/f32-f16-vcvt/gen/vcvt-scalar-fabsf-x2.c
  src/f32-gavgpool-cw/scalar-x1.c
  src/f32-gavgpool/7p7x-minmax-scalar-c1.c
  src/f32-gavgpool/7x-minmax-scalar-c1.c
  src/f32-gemm/gen/1x4-minmax-scalar.c
  src/f32-gemm/gen/1x4-relu-scalar.c
  src/f32-gemm/gen/1x4-scalar.c
  src/f32-gemm/gen/4x2-minmax-scalar.c
  src/f32-gemm/gen/4x2-scalar.c
  src/f32-gemm/gen/4x4-minmax-scalar.c
  src/f32-gemm/gen/4x4-relu-scalar.c
  src/f32-gemm/gen/4x4-scalar.c
  src/f32-ibilinear-chw/gen/scalar-p4.c
  src/f32-ibilinear/gen/scalar-c2.c
  src/f32-igemm/gen/1x4-minmax-scalar.c
  src/f32-igemm/gen/1x4-relu-scalar.c
  src/f32-igemm/gen/1x4-scalar.c
  src/f32-igemm/gen/4x2-minmax-scalar.c
  src/f32-igemm/gen/4x2-scalar.c
  src/f32-igemm/gen/4x4-minmax-scalar.c
  src/f32-igemm/gen/4x4-relu-scalar.c
  src/f32-igemm/gen/4x4-scalar.c
  src/f32-maxpool/9p8x-minmax-scalar-c1.c
  src/f32-pavgpool/9p8x-minmax-scalar-c1.c
  src/f32-pavgpool/9x-minmax-scalar-c1.c
  src/f32-prelu/gen/scalar-2x4.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x4.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x4.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x4-acc2.c
  src/f32-rmax/scalar.c
  src/f32-spmm/gen/8x1-minmax-scalar.c
  src/f32-spmm/gen/8x2-minmax-scalar.c
  src/f32-spmm/gen/8x4-minmax-scalar.c
  src/f32-vbinary/gen/vadd-minmax-scalar-x8.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x2.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vmax-scalar-x8.c
  src/f32-vbinary/gen/vmaxc-scalar-x8.c
  src/f32-vbinary/gen/vmin-scalar-x8.c
  src/f32-vbinary/gen/vminc-scalar-x8.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x8.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vsqrdiff-scalar-x8.c
  src/f32-vbinary/gen/vsqrdiffc-scalar-x8.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x8.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x8.c
  src/f32-vclamp/gen/vclamp-scalar-x4.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x4.c
  src/f32-vhswish/gen/vhswish-scalar-x4.c
  src/f32-vlrelu/gen/vlrelu-scalar-x4.c
  src/f32-vmulcaddc/gen/c1-minmax-scalar-2x.c
  src/f32-vrelu/gen/vrelu-scalar-x8.c
  src/f32-vrnd/gen/vrndd-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndne-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndu-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndz-scalar-libm-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut64-p2-div-x2.c
  src/f32-vsqrt/gen/scalar-sqrt-x1.c
  src/f32-vunary/gen/vabs-scalar-x4.c
  src/f32-vunary/gen/vneg-scalar-x4.c
  src/f32-vunary/gen/vsqr-scalar-x4.c
  src/qc8-dwconv/gen/up1x3-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/1x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/1x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c
  src/qs8-f32-vcvt/gen/vcvt-scalar-x4.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c1.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c1.c
  src/qs8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qs8-vadd/gen/minmax-scalar-x1.c
  src/qs8-vaddc/gen/minmax-scalar-x1.c
  src/qs8-vmul/gen/minmax-fp32-scalar-x4.c
  src/qs8-vmulc/gen/minmax-fp32-scalar-x4.c
  src/qu8-avgpool/9p8x-minmax-scalar-c1.c
  src/qu8-avgpool/9x-minmax-scalar-c1.c
  src/qu8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c
  src/qu8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c
  src/qu8-f32-vcvt/gen/vcvt-scalar-x4.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c1.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c1.c
  src/qu8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qu8-vadd/gen/minmax-scalar-x1.c
  src/qu8-vaddc/gen/minmax-scalar-x1.c
  src/qu8-vmul/gen/minmax-fp32-scalar-x4.c
  src/qu8-vmulc/gen/minmax-fp32-scalar-x4.c
  src/s8-ibilinear/gen/scalar-c1.c
  src/s8-maxpool/9p8x-minmax-scalar-c1.c
  src/s8-vclamp/scalar-x4.c
  src/u8-ibilinear/gen/scalar-c1.c
  src/u8-maxpool/9p8x-minmax-scalar-c1.c
  src/u8-rmax/scalar.c
  src/u8-vclamp/scalar-x4.c
  src/xx-fill/scalar-x16.c
  src/xx-pad/scalar.c
  src/x8-transposec/gen/2x4-scalar-int.c
  src/x8-zip/xm-scalar.c
  src/x8-zip/x2-scalar.c
  src/x8-zip/x3-scalar.c
  src/x8-zip/x4-scalar.c
  src/x16-transposec/gen/2x4-scalar-int.c
  src/x32-packx/x2-scalar.c
  src/x32-packx/x3-scalar.c
  src/x32-packx/x4-scalar.c
  src/x32-transposec/gen/2x4-scalar-int.c
  src/x32-unpool/scalar.c
  src/x32-zip/xm-scalar.c
  src/x32-zip/x2-scalar.c
  src/x32-zip/x3-scalar.c
  src/x32-zip/x4-scalar.c)

SET(PROD_SCALAR_RISCV_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-scalar-x4.c
  src/f32-argmaxpool/4x-scalar-c1.c
  src/f32-argmaxpool/9p8x-scalar-c1.c
  src/f32-argmaxpool/9x-scalar-c1.c
  src/f32-avgpool/9p8x-minmax-scalar-c1.c
  src/f32-avgpool/9x-minmax-scalar-c1.c
  src/f32-conv-hwc/3x3s2p0p1c3x4-scalar-1x1.c
  src/f32-conv-hwc/3x3s2p1c3x4-scalar-1x1.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-scalar-1x1.c
  src/f32-dwconv/gen/up1x3-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x3-scalar-acc2.c
  src/f32-dwconv/gen/up1x4-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x4-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-scalar-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-2x1-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc5.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc5.c
  src/f32-f16-vcvt/gen/vcvt-scalar-fabsf-x2.c
  src/f32-gavgpool-cw/scalar-x1.c
  src/f32-gavgpool/7p7x-minmax-scalar-c1.c
  src/f32-gavgpool/7x-minmax-scalar-c1.c
  src/f32-gemm/gen/1x4-minmax-scalar.c
  src/f32-gemm/gen/1x4-relu-scalar.c
  src/f32-gemm/gen/1x4-scalar.c
  src/f32-gemm/gen/4x2-minmax-scalar.c
  src/f32-gemm/gen/4x2-scalar.c
  src/f32-gemm/gen/4x4-minmax-scalar.c
  src/f32-gemm/gen/4x4-relu-scalar.c
  src/f32-gemm/gen/4x4-scalar.c
  src/f32-ibilinear-chw/gen/scalar-p4.c
  src/f32-ibilinear/gen/scalar-c2.c
  src/f32-igemm/gen/1x4-minmax-scalar.c
  src/f32-igemm/gen/1x4-relu-scalar.c
  src/f32-igemm/gen/1x4-scalar.c
  src/f32-igemm/gen/4x2-minmax-scalar.c
  src/f32-igemm/gen/4x2-scalar.c
  src/f32-igemm/gen/4x4-minmax-scalar.c
  src/f32-igemm/gen/4x4-relu-scalar.c
  src/f32-igemm/gen/4x4-scalar.c
  src/f32-maxpool/9p8x-minmax-scalar-c1.c
  src/f32-pavgpool/9p8x-minmax-scalar-c1.c
  src/f32-pavgpool/9x-minmax-scalar-c1.c
  src/f32-prelu/gen/scalar-2x4.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x4.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x4.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x4-acc2.c
  src/f32-rmax/scalar.c
  src/f32-spmm/gen/8x1-minmax-scalar.c
  src/f32-spmm/gen/8x2-minmax-scalar.c
  src/f32-spmm/gen/8x4-minmax-scalar.c
  src/f32-vbinary/gen/vadd-minmax-scalar-x8.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x2.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vmax-scalar-x8.c
  src/f32-vbinary/gen/vmaxc-scalar-x8.c
  src/f32-vbinary/gen/vmin-scalar-x8.c
  src/f32-vbinary/gen/vminc-scalar-x8.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x8.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vsqrdiff-scalar-x8.c
  src/f32-vbinary/gen/vsqrdiffc-scalar-x8.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x8.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x8.c
  src/f32-vclamp/gen/vclamp-scalar-x4.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x4.c
  src/f32-vhswish/gen/vhswish-scalar-x4.c
  src/f32-vlrelu/gen/vlrelu-scalar-x4.c
  src/f32-vmulcaddc/gen/c1-minmax-scalar-2x.c
  src/f32-vrelu/gen/vrelu-scalar-x8.c
  src/f32-vrnd/gen/vrndd-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndne-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndu-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndz-scalar-libm-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut64-p2-div-x2.c
  src/f32-vsqrt/gen/scalar-sqrt-x1.c
  src/f32-vunary/gen/vabs-scalar-x4.c
  src/f32-vunary/gen/vneg-scalar-x4.c
  src/f32-vunary/gen/vsqr-scalar-x4.c
  src/qc8-dwconv/gen/up2x3-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c
  src/qs8-f32-vcvt/gen/vcvt-scalar-x4.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c1.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c1.c
  src/qs8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qs8-vadd/gen/minmax-scalar-x4.c
  src/qs8-vaddc/gen/minmax-scalar-x4.c
  src/qs8-vcvt/gen/vcvt-scalar-x4.c
  src/qs8-vlrelu/gen/vlrelu-scalar-andxor-x4.c
  src/qs8-vmul/gen/minmax-fp32-scalar-x4.c
  src/qs8-vmulc/gen/minmax-fp32-scalar-x4.c
  src/qu8-avgpool/9p8x-minmax-scalar-c1.c
  src/qu8-avgpool/9x-minmax-scalar-c1.c
  src/qu8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c
  src/qu8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c
  src/qu8-f32-vcvt/gen/vcvt-scalar-x4.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c1.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c1.c
  src/qu8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qu8-vadd/gen/minmax-scalar-x4.c
  src/qu8-vaddc/gen/minmax-scalar-x4.c
  src/qu8-vcvt/gen/vcvt-scalar-x4.c
  src/qu8-vlrelu/gen/vlrelu-scalar-andxor-x4.c
  src/qu8-vmul/gen/minmax-fp32-scalar-x4.c
  src/qu8-vmulc/gen/minmax-fp32-scalar-x4.c
  src/s8-ibilinear/gen/scalar-c1.c
  src/s8-maxpool/9p8x-minmax-scalar-c1.c
  src/s8-vclamp/scalar-x4.c
  src/u8-ibilinear/gen/scalar-c1.c
  src/u8-maxpool/9p8x-minmax-scalar-c1.c
  src/u8-rmax/scalar.c
  src/u8-vclamp/scalar-x4.c
  src/xx-fill/scalar-x16.c
  src/xx-pad/scalar.c
  src/x8-transposec/gen/2x4-scalar-int.c
  src/x8-zip/xm-scalar.c
  src/x8-zip/x2-scalar.c
  src/x8-zip/x3-scalar.c
  src/x8-zip/x4-scalar.c
  src/x16-transposec/gen/2x4-scalar-int.c
  src/x32-packx/x2-scalar.c
  src/x32-packx/x3-scalar.c
  src/x32-packx/x4-scalar.c
  src/x32-transposec/gen/2x4-scalar-int.c
  src/x32-unpool/scalar.c
  src/x32-zip/xm-scalar.c
  src/x32-zip/x2-scalar.c
  src/x32-zip/x3-scalar.c
  src/x32-zip/x4-scalar.c)

SET(ALL_SCALAR_MICROKERNEL_SRCS
  src/cs16-bfly4/gen/scalar-x1.c
  src/cs16-bfly4/gen/scalar-x2.c
  src/cs16-bfly4/gen/scalar-x4.c
  src/cs16-bfly4/samples1-scalar.c
  src/cs16-bfly4/samples4-scalar.c
  src/cs16-fftr/gen/scalar-x1.c
  src/cs16-fftr/gen/scalar-x2.c
  src/cs16-fftr/gen/scalar-x4.c
  src/cs16-vsquareabs/gen/scalar-x1.c
  src/cs16-vsquareabs/gen/scalar-x2.c
  src/cs16-vsquareabs/gen/scalar-x3.c
  src/cs16-vsquareabs/gen/scalar-x4.c
  src/f16-f32-vcvt/gen/vcvt-scalar-x1.c
  src/f16-f32-vcvt/gen/vcvt-scalar-x2.c
  src/f16-f32-vcvt/gen/vcvt-scalar-x3.c
  src/f16-f32-vcvt/gen/vcvt-scalar-x4.c
  src/f32-argmaxpool/4x-scalar-c1.c
  src/f32-argmaxpool/9p8x-scalar-c1.c
  src/f32-argmaxpool/9x-scalar-c1.c
  src/f32-avgpool/9p8x-minmax-scalar-c1.c
  src/f32-avgpool/9x-minmax-scalar-c1.c
  src/f32-conv-hwc/3x3s2p0p1c3x4-scalar-1x1.c
  src/f32-conv-hwc/3x3s2p1c3x4-scalar-1x1.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-scalar-1x1.c
  src/f32-dwconv/gen/up1x3-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x3-minmax-scalar.c
  src/f32-dwconv/gen/up1x3-scalar-acc2.c
  src/f32-dwconv/gen/up1x3-scalar.c
  src/f32-dwconv/gen/up1x4-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x4-minmax-scalar.c
  src/f32-dwconv/gen/up1x4-scalar-acc2.c
  src/f32-dwconv/gen/up1x4-scalar.c
  src/f32-dwconv/gen/up1x9-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-minmax-scalar.c
  src/f32-dwconv/gen/up1x9-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-scalar.c
  src/f32-dwconv/gen/up1x25-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-minmax-scalar.c
  src/f32-dwconv/gen/up1x25-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-scalar.c
  src/f32-dwconv/gen/up2x3-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up2x3-minmax-scalar.c
  src/f32-dwconv/gen/up2x3-scalar-acc2.c
  src/f32-dwconv/gen/up2x3-scalar.c
  src/f32-dwconv/gen/up2x4-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up2x4-minmax-scalar.c
  src/f32-dwconv/gen/up2x4-scalar-acc2.c
  src/f32-dwconv/gen/up2x4-scalar.c
  src/f32-dwconv/gen/up2x9-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up2x9-minmax-scalar.c
  src/f32-dwconv/gen/up2x9-scalar-acc2.c
  src/f32-dwconv/gen/up2x9-scalar.c
  src/f32-dwconv/gen/up2x25-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up2x25-minmax-scalar.c
  src/f32-dwconv/gen/up2x25-scalar-acc2.c
  src/f32-dwconv/gen/up2x25-scalar.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1-acc3.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1-acc4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-2x1-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-2x1.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-3x1.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-4x1.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-5x1.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-6x1.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1-acc3.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1-acc4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-2x1-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-2x1.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-3x1.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-4x1.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc5.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-2x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-2x1-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-2x1.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-3x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-3x1.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc5.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-2x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-2x1-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-2x1.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-3x1-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-3x1.c
  src/f32-f16-vcvt/gen/vcvt-scalar-bitcast-x1.c
  src/f32-f16-vcvt/gen/vcvt-scalar-bitcast-x2.c
  src/f32-f16-vcvt/gen/vcvt-scalar-bitcast-x3.c
  src/f32-f16-vcvt/gen/vcvt-scalar-bitcast-x4.c
  src/f32-f16-vcvt/gen/vcvt-scalar-fabsf-x1.c
  src/f32-f16-vcvt/gen/vcvt-scalar-fabsf-x2.c
  src/f32-f16-vcvt/gen/vcvt-scalar-fabsf-x3.c
  src/f32-f16-vcvt/gen/vcvt-scalar-fabsf-x4.c
  src/f32-gavgpool-cw/scalar-x1.c
  src/f32-gavgpool/7p7x-minmax-scalar-c1.c
  src/f32-gavgpool/7x-minmax-scalar-c1.c
  src/f32-gemm/gen-inc/1x4inc-minmax-scalar.c
  src/f32-gemm/gen-inc/2x4inc-minmax-scalar.c
  src/f32-gemm/gen-inc/4x4inc-minmax-scalar.c
  src/f32-gemm/gen/1x4-minmax-scalar.c
  src/f32-gemm/gen/1x4-relu-scalar.c
  src/f32-gemm/gen/1x4-scalar.c
  src/f32-gemm/gen/2x4-minmax-scalar.c
  src/f32-gemm/gen/2x4-relu-scalar.c
  src/f32-gemm/gen/2x4-scalar.c
  src/f32-gemm/gen/4x2-minmax-scalar.c
  src/f32-gemm/gen/4x2-relu-scalar.c
  src/f32-gemm/gen/4x2-scalar.c
  src/f32-gemm/gen/4x4-minmax-scalar.c
  src/f32-gemm/gen/4x4-relu-scalar.c
  src/f32-gemm/gen/4x4-scalar.c
  src/f32-ibilinear-chw/gen/scalar-p1.c
  src/f32-ibilinear-chw/gen/scalar-p2.c
  src/f32-ibilinear-chw/gen/scalar-p4.c
  src/f32-ibilinear/gen/scalar-c1.c
  src/f32-ibilinear/gen/scalar-c2.c
  src/f32-ibilinear/gen/scalar-c4.c
  src/f32-igemm/gen/1x4-minmax-scalar.c
  src/f32-igemm/gen/1x4-relu-scalar.c
  src/f32-igemm/gen/1x4-scalar.c
  src/f32-igemm/gen/2x4-minmax-scalar.c
  src/f32-igemm/gen/2x4-relu-scalar.c
  src/f32-igemm/gen/2x4-scalar.c
  src/f32-igemm/gen/4x2-minmax-scalar.c
  src/f32-igemm/gen/4x2-relu-scalar.c
  src/f32-igemm/gen/4x2-scalar.c
  src/f32-igemm/gen/4x4-minmax-scalar.c
  src/f32-igemm/gen/4x4-relu-scalar.c
  src/f32-igemm/gen/4x4-scalar.c
  src/f32-maxpool/9p8x-minmax-scalar-c1.c
  src/f32-pavgpool/9p8x-minmax-scalar-c1.c
  src/f32-pavgpool/9x-minmax-scalar-c1.c
  src/f32-ppmm/gen/2x4-minmax-scalar.c
  src/f32-ppmm/gen/3x3-minmax-scalar.c
  src/f32-ppmm/gen/4x2-minmax-scalar.c
  src/f32-ppmm/gen/4x4-minmax-scalar.c
  src/f32-prelu/gen/scalar-2x1.c
  src/f32-prelu/gen/scalar-2x4.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x1.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x2.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x3.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x4.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x1.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x2.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x3.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x4.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x1.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x2.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x3.c
  src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x4.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x1.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x2.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x3.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x4.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x1.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x2.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x3.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x4.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x1.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x2.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x3.c
  src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x4.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-lut64-p2-x1.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-lut64-p2-x2-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-lut64-p2-x2.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-lut64-p2-x4-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-lut64-p2-x4-acc4.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x1.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x2-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x2.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x4-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x4-acc4.c
  src/f32-raddstoreexpminusmax/gen/scalar-rr2-p5-x4.c
  src/f32-rmax/scalar.c
  src/f32-spmm/gen/1x1-minmax-scalar-pipelined.c
  src/f32-spmm/gen/1x1-minmax-scalar.c
  src/f32-spmm/gen/2x1-minmax-scalar-pipelined.c
  src/f32-spmm/gen/2x1-minmax-scalar.c
  src/f32-spmm/gen/4x1-minmax-scalar-pipelined.c
  src/f32-spmm/gen/4x1-minmax-scalar.c
  src/f32-spmm/gen/8x1-minmax-scalar-pipelined.c
  src/f32-spmm/gen/8x1-minmax-scalar.c
  src/f32-spmm/gen/8x2-minmax-scalar.c
  src/f32-spmm/gen/8x4-minmax-scalar.c
  src/f32-vbinary/gen/vadd-minmax-scalar-x1.c
  src/f32-vbinary/gen/vadd-minmax-scalar-x2.c
  src/f32-vbinary/gen/vadd-minmax-scalar-x4.c
  src/f32-vbinary/gen/vadd-minmax-scalar-x8.c
  src/f32-vbinary/gen/vadd-relu-scalar-x1.c
  src/f32-vbinary/gen/vadd-relu-scalar-x2.c
  src/f32-vbinary/gen/vadd-relu-scalar-x4.c
  src/f32-vbinary/gen/vadd-relu-scalar-x8.c
  src/f32-vbinary/gen/vadd-scalar-x1.c
  src/f32-vbinary/gen/vadd-scalar-x2.c
  src/f32-vbinary/gen/vadd-scalar-x4.c
  src/f32-vbinary/gen/vadd-scalar-x8.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vaddc-relu-scalar-x1.c
  src/f32-vbinary/gen/vaddc-relu-scalar-x2.c
  src/f32-vbinary/gen/vaddc-relu-scalar-x4.c
  src/f32-vbinary/gen/vaddc-relu-scalar-x8.c
  src/f32-vbinary/gen/vaddc-scalar-x1.c
  src/f32-vbinary/gen/vaddc-scalar-x2.c
  src/f32-vbinary/gen/vaddc-scalar-x4.c
  src/f32-vbinary/gen/vaddc-scalar-x8.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x1.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x2.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x4.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x8.c
  src/f32-vbinary/gen/vdiv-relu-scalar-x1.c
  src/f32-vbinary/gen/vdiv-relu-scalar-x2.c
  src/f32-vbinary/gen/vdiv-relu-scalar-x4.c
  src/f32-vbinary/gen/vdiv-relu-scalar-x8.c
  src/f32-vbinary/gen/vdiv-scalar-x1.c
  src/f32-vbinary/gen/vdiv-scalar-x2.c
  src/f32-vbinary/gen/vdiv-scalar-x4.c
  src/f32-vbinary/gen/vdiv-scalar-x8.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vdivc-relu-scalar-x1.c
  src/f32-vbinary/gen/vdivc-relu-scalar-x2.c
  src/f32-vbinary/gen/vdivc-relu-scalar-x4.c
  src/f32-vbinary/gen/vdivc-relu-scalar-x8.c
  src/f32-vbinary/gen/vdivc-scalar-x1.c
  src/f32-vbinary/gen/vdivc-scalar-x2.c
  src/f32-vbinary/gen/vdivc-scalar-x4.c
  src/f32-vbinary/gen/vdivc-scalar-x8.c
  src/f32-vbinary/gen/vmax-scalar-x1.c
  src/f32-vbinary/gen/vmax-scalar-x2.c
  src/f32-vbinary/gen/vmax-scalar-x4.c
  src/f32-vbinary/gen/vmax-scalar-x8.c
  src/f32-vbinary/gen/vmaxc-scalar-x1.c
  src/f32-vbinary/gen/vmaxc-scalar-x2.c
  src/f32-vbinary/gen/vmaxc-scalar-x4.c
  src/f32-vbinary/gen/vmaxc-scalar-x8.c
  src/f32-vbinary/gen/vmin-scalar-x1.c
  src/f32-vbinary/gen/vmin-scalar-x2.c
  src/f32-vbinary/gen/vmin-scalar-x4.c
  src/f32-vbinary/gen/vmin-scalar-x8.c
  src/f32-vbinary/gen/vminc-scalar-x1.c
  src/f32-vbinary/gen/vminc-scalar-x2.c
  src/f32-vbinary/gen/vminc-scalar-x4.c
  src/f32-vbinary/gen/vminc-scalar-x8.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x1.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x2.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x4.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x8.c
  src/f32-vbinary/gen/vmul-relu-scalar-x1.c
  src/f32-vbinary/gen/vmul-relu-scalar-x2.c
  src/f32-vbinary/gen/vmul-relu-scalar-x4.c
  src/f32-vbinary/gen/vmul-relu-scalar-x8.c
  src/f32-vbinary/gen/vmul-scalar-x1.c
  src/f32-vbinary/gen/vmul-scalar-x2.c
  src/f32-vbinary/gen/vmul-scalar-x4.c
  src/f32-vbinary/gen/vmul-scalar-x8.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vmulc-relu-scalar-x1.c
  src/f32-vbinary/gen/vmulc-relu-scalar-x2.c
  src/f32-vbinary/gen/vmulc-relu-scalar-x4.c
  src/f32-vbinary/gen/vmulc-relu-scalar-x8.c
  src/f32-vbinary/gen/vmulc-scalar-x1.c
  src/f32-vbinary/gen/vmulc-scalar-x2.c
  src/f32-vbinary/gen/vmulc-scalar-x4.c
  src/f32-vbinary/gen/vmulc-scalar-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vrdivc-relu-scalar-x1.c
  src/f32-vbinary/gen/vrdivc-relu-scalar-x2.c
  src/f32-vbinary/gen/vrdivc-relu-scalar-x4.c
  src/f32-vbinary/gen/vrdivc-relu-scalar-x8.c
  src/f32-vbinary/gen/vrdivc-scalar-x1.c
  src/f32-vbinary/gen/vrdivc-scalar-x2.c
  src/f32-vbinary/gen/vrdivc-scalar-x4.c
  src/f32-vbinary/gen/vrdivc-scalar-x8.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vrsubc-relu-scalar-x1.c
  src/f32-vbinary/gen/vrsubc-relu-scalar-x2.c
  src/f32-vbinary/gen/vrsubc-relu-scalar-x4.c
  src/f32-vbinary/gen/vrsubc-relu-scalar-x8.c
  src/f32-vbinary/gen/vrsubc-scalar-x1.c
  src/f32-vbinary/gen/vrsubc-scalar-x2.c
  src/f32-vbinary/gen/vrsubc-scalar-x4.c
  src/f32-vbinary/gen/vrsubc-scalar-x8.c
  src/f32-vbinary/gen/vsqrdiff-scalar-x1.c
  src/f32-vbinary/gen/vsqrdiff-scalar-x2.c
  src/f32-vbinary/gen/vsqrdiff-scalar-x4.c
  src/f32-vbinary/gen/vsqrdiff-scalar-x8.c
  src/f32-vbinary/gen/vsqrdiffc-scalar-x1.c
  src/f32-vbinary/gen/vsqrdiffc-scalar-x2.c
  src/f32-vbinary/gen/vsqrdiffc-scalar-x4.c
  src/f32-vbinary/gen/vsqrdiffc-scalar-x8.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x1.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x2.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x4.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x8.c
  src/f32-vbinary/gen/vsub-relu-scalar-x1.c
  src/f32-vbinary/gen/vsub-relu-scalar-x2.c
  src/f32-vbinary/gen/vsub-relu-scalar-x4.c
  src/f32-vbinary/gen/vsub-relu-scalar-x8.c
  src/f32-vbinary/gen/vsub-scalar-x1.c
  src/f32-vbinary/gen/vsub-scalar-x2.c
  src/f32-vbinary/gen/vsub-scalar-x4.c
  src/f32-vbinary/gen/vsub-scalar-x8.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x8.c
  src/f32-vbinary/gen/vsubc-relu-scalar-x1.c
  src/f32-vbinary/gen/vsubc-relu-scalar-x2.c
  src/f32-vbinary/gen/vsubc-relu-scalar-x4.c
  src/f32-vbinary/gen/vsubc-relu-scalar-x8.c
  src/f32-vbinary/gen/vsubc-scalar-x1.c
  src/f32-vbinary/gen/vsubc-scalar-x2.c
  src/f32-vbinary/gen/vsubc-scalar-x4.c
  src/f32-vbinary/gen/vsubc-scalar-x8.c
  src/f32-vclamp/gen/vclamp-scalar-x1.c
  src/f32-vclamp/gen/vclamp-scalar-x2.c
  src/f32-vclamp/gen/vclamp-scalar-x4.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x1.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x2.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x3.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x4.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x5.c
  src/f32-velu/gen/velu-scalar-rr2-lut16-p3-x6.c
  src/f32-velu/gen/velu-scalar-rr2-p6-x1.c
  src/f32-velu/gen/velu-scalar-rr2-p6-x2.c
  src/f32-velu/gen/velu-scalar-rr2-p6-x3.c
  src/f32-velu/gen/velu-scalar-rr2-p6-x4.c
  src/f32-velu/gen/velu-scalar-rr2-p6-x5.c
  src/f32-velu/gen/velu-scalar-rr2-p6-x6.c
  src/f32-vhswish/gen/vhswish-scalar-x1.c
  src/f32-vhswish/gen/vhswish-scalar-x2.c
  src/f32-vhswish/gen/vhswish-scalar-x4.c
  src/f32-vlrelu/gen/vlrelu-scalar-x1.c
  src/f32-vlrelu/gen/vlrelu-scalar-x2.c
  src/f32-vlrelu/gen/vlrelu-scalar-x4.c
  src/f32-vmulcaddc/gen/c1-minmax-scalar-2x.c
  src/f32-vmulcaddc/gen/c2-minmax-scalar-2x.c
  src/f32-vmulcaddc/gen/c4-minmax-scalar-2x.c
  src/f32-vrelu/gen/vrelu-scalar-x1.c
  src/f32-vrelu/gen/vrelu-scalar-x2.c
  src/f32-vrelu/gen/vrelu-scalar-x4.c
  src/f32-vrelu/gen/vrelu-scalar-x8.c
  src/f32-vrnd/gen/vrndd-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndd-scalar-libm-x2.c
  src/f32-vrnd/gen/vrndd-scalar-libm-x4.c
  src/f32-vrnd/gen/vrndne-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndne-scalar-libm-x2.c
  src/f32-vrnd/gen/vrndne-scalar-libm-x4.c
  src/f32-vrnd/gen/vrndu-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndu-scalar-libm-x2.c
  src/f32-vrnd/gen/vrndu-scalar-libm-x4.c
  src/f32-vrnd/gen/vrndz-scalar-libm-x1.c
  src/f32-vrnd/gen/vrndz-scalar-libm-x2.c
  src/f32-vrnd/gen/vrndz-scalar-libm-x4.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut64-p2-div-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut64-p2-div-x2.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut64-p2-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut2048-p1-div-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut2048-p1-div-x2.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-lut2048-p1-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-p5-div-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-p5-div-x2.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-rr2-p5-div-x4.c
  src/f32-vsqrt/gen/scalar-sqrt-x1.c
  src/f32-vsqrt/gen/scalar-sqrt-x2.c
  src/f32-vsqrt/gen/scalar-sqrt-x4.c
  src/f32-vunary/gen/vabs-scalar-x1.c
  src/f32-vunary/gen/vabs-scalar-x2.c
  src/f32-vunary/gen/vabs-scalar-x4.c
  src/f32-vunary/gen/vneg-scalar-x1.c
  src/f32-vunary/gen/vneg-scalar-x2.c
  src/f32-vunary/gen/vneg-scalar-x4.c
  src/f32-vunary/gen/vsqr-scalar-x1.c
  src/f32-vunary/gen/vsqr-scalar-x2.c
  src/f32-vunary/gen/vsqr-scalar-x4.c
  src/math/cvt-f32-f16-scalar-bitcast.c
  src/math/cvt-f32-f16-scalar-fabsf.c
  src/math/expminus-f32-scalar-rr2-lut64-p2.c
  src/math/expminus-f32-scalar-rr2-lut2048-p1.c
  src/math/expminus-f32-scalar-rr2-p5.c
  src/math/expm1minus-f32-scalar-rr2-lut4-p4.c
  src/math/expm1minus-f32-scalar-rr2-lut8-p3.c
  src/math/expm1minus-f32-scalar-rr2-lut8-p4.c
  src/math/expm1minus-f32-scalar-rr2-lut16-p3.c
  src/math/expm1minus-f32-scalar-rr2-lut16-p4.c
  src/math/expm1minus-f32-scalar-rr2-p5.c
  src/math/expm1minus-f32-scalar-rr2-p6.c
  src/math/roundd-scalar-addsub.c
  src/math/roundd-scalar-cvt.c
  src/math/roundd-scalar-floor.c
  src/math/roundne-scalar-addsub.c
  src/math/roundne-scalar-nearbyint.c
  src/math/roundne-scalar-rint.c
  src/math/roundu-scalar-addsub.c
  src/math/roundu-scalar-ceil.c
  src/math/roundu-scalar-cvt.c
  src/math/roundz-scalar-addsub.c
  src/math/roundz-scalar-cvt.c
  src/math/roundz-scalar-trunc.c
  src/math/sigmoid-f32-scalar-rr2-lut64-p2-div.c
  src/math/sigmoid-f32-scalar-rr2-lut2048-p1-div.c
  src/math/sigmoid-f32-scalar-rr2-p5-div.c
  src/math/sqrt-u32-scalar-bitmanip.c
  src/math/sqrt-u32-scalar-clz-binsearch.c
  src/math/sqrt-u32-scalar-clz-newton.c
  src/math/sqrt-u32-scalar-cvti32-sqrt-lrint.c
  src/math/sqrt-u32-scalar-cvti64-sqrt-lrint.c
  src/math/sqrt-u32-scalar-cvti64-sqrtf-lrintf.c
  src/math/sqrt-u32-scalar-cvtu32-sqrt-lrint.c
  src/math/sqrt-u32-scalar-cvtu32-sqrtf-lrintf.c
  src/math/sqrt-u32-scalar-hashemian.c
  src/math/sqrt-u32-scalar-tflm.c
  src/math/sqrt-u64-scalar-cvtu32-sqrt-cvtsatu32f64.c
  src/math/sqrt-u64-scalar-cvtu32-sqrt-llrint.c
  src/math/sqrt-u64-scalar-cvtu64-sqrt-llrint.c
  src/qc8-dwconv/gen/up1x3-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up1x9-minmax-fp32-scalar-imagic.c
  src/qc8-dwconv/gen/up1x9-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up1x25-minmax-fp32-scalar-imagic.c
  src/qc8-dwconv/gen/up1x25-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up2x3-minmax-fp32-scalar-imagic.c
  src/qc8-dwconv/gen/up2x3-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up2x9-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up2x9-minmax-fp32-scalar-imagic.c
  src/qc8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up2x25-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up2x25-minmax-fp32-scalar-imagic.c
  src/qc8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up4x9-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up4x9-minmax-fp32-scalar-imagic.c
  src/qc8-dwconv/gen/up4x9-minmax-fp32-scalar-lrintf.c
  src/qc8-dwconv/gen/up4x25-minmax-fp32-scalar-fmagic.c
  src/qc8-dwconv/gen/up4x25-minmax-fp32-scalar-imagic.c
  src/qc8-dwconv/gen/up4x25-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/1x2-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/1x2-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/1x4-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/1x4-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/2x2-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/2x2-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/2x4-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/2x4-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/2x4-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/3x2-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/3x2-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/3x2-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/3x4-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/3x4-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/4x2-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/4x2-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/4x2-minmax-fp32-scalar-lrintf.c
  src/qc8-gemm/gen/4x4-minmax-fp32-scalar-fmagic.c
  src/qc8-gemm/gen/4x4-minmax-fp32-scalar-imagic.c
  src/qc8-gemm/gen/4x4-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/1x2-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/1x2-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/1x4-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/1x4-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/2x2-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/2x2-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/2x4-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/2x4-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/2x4-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/3x2-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/3x2-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/3x2-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/3x4-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/3x4-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/4x2-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/4x2-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/4x2-minmax-fp32-scalar-lrintf.c
  src/qc8-igemm/gen/4x4-minmax-fp32-scalar-fmagic.c
  src/qc8-igemm/gen/4x4-minmax-fp32-scalar-imagic.c
  src/qc8-igemm/gen/4x4-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up1x9-minmax-fp32-scalar-imagic.c
  src/qs8-dwconv/gen/up1x9-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up1x9-minmax-rndnu-scalar.c
  src/qs8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up1x25-minmax-fp32-scalar-imagic.c
  src/qs8-dwconv/gen/up1x25-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up2x9-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up2x9-minmax-fp32-scalar-imagic.c
  src/qs8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up2x9-minmax-rndnu-scalar.c
  src/qs8-dwconv/gen/up2x25-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up2x25-minmax-fp32-scalar-imagic.c
  src/qs8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up4x9-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up4x9-minmax-fp32-scalar-imagic.c
  src/qs8-dwconv/gen/up4x9-minmax-fp32-scalar-lrintf.c
  src/qs8-dwconv/gen/up4x9-minmax-rndnu-scalar.c
  src/qs8-dwconv/gen/up4x25-minmax-fp32-scalar-fmagic.c
  src/qs8-dwconv/gen/up4x25-minmax-fp32-scalar-imagic.c
  src/qs8-dwconv/gen/up4x25-minmax-fp32-scalar-lrintf.c
  src/qs8-f32-vcvt/gen/vcvt-scalar-x1.c
  src/qs8-f32-vcvt/gen/vcvt-scalar-x2.c
  src/qs8-f32-vcvt/gen/vcvt-scalar-x3.c
  src/qs8-f32-vcvt/gen/vcvt-scalar-x4.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-fmagic-c1.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-fmagic-c2.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-fmagic-c4.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c1.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c2.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c4.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-lrintf-c1.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-lrintf-c2.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-scalar-lrintf-c4.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-fmagic-c1.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-fmagic-c2.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-fmagic-c4.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c1.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c2.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c4.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-lrintf-c1.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-lrintf-c2.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-scalar-lrintf-c4.c
  src/qs8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/1x2-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/1x2-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/1x2-minmax-rndnu-scalar.c
  src/qs8-gemm/gen/1x4-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/1x4-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/1x4-minmax-rndnu-scalar.c
  src/qs8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/2x2-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/2x2-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/2x2-minmax-rndnu-scalar.c
  src/qs8-gemm/gen/2x4-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/2x4-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/2x4-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/2x4-minmax-rndnu-scalar.c
  src/qs8-gemm/gen/3x2-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/3x2-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/3x2-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/3x2-minmax-rndnu-scalar.c
  src/qs8-gemm/gen/3x4-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/3x4-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/3x4-minmax-rndnu-scalar.c
  src/qs8-gemm/gen/4x2-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/4x2-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/4x2-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/4x2-minmax-rndnu-scalar.c
  src/qs8-gemm/gen/4x4-minmax-fp32-scalar-fmagic.c
  src/qs8-gemm/gen/4x4-minmax-fp32-scalar-imagic.c
  src/qs8-gemm/gen/4x4-minmax-fp32-scalar-lrintf.c
  src/qs8-gemm/gen/4x4-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/1x2-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/1x2-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/1x2-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/1x4-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/1x4-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/1x4-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/2x2-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/2x2-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/2x2-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/2x4-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/2x4-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/2x4-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/2x4-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/3x2-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/3x2-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/3x2-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/3x2-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/3x4-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/3x4-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/3x4-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/4x2-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/4x2-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/4x2-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/4x2-minmax-rndnu-scalar.c
  src/qs8-igemm/gen/4x4-minmax-fp32-scalar-fmagic.c
  src/qs8-igemm/gen/4x4-minmax-fp32-scalar-imagic.c
  src/qs8-igemm/gen/4x4-minmax-fp32-scalar-lrintf.c
  src/qs8-igemm/gen/4x4-minmax-rndnu-scalar.c
  src/qs8-requantization/fp32-scalar-fmagic.c
  src/qs8-requantization/fp32-scalar-lrintf.c
  src/qs8-requantization/gemmlowp-scalar.c
  src/qs8-requantization/rndna-scalar-signed64.c
  src/qs8-requantization/rndna-scalar-unsigned32.c
  src/qs8-requantization/rndna-scalar-unsigned64.c
  src/qs8-requantization/rndnu-scalar.c
  src/qs8-vadd/gen/minmax-scalar-x1.c
  src/qs8-vadd/gen/minmax-scalar-x2.c
  src/qs8-vadd/gen/minmax-scalar-x4.c
  src/qs8-vaddc/gen/minmax-scalar-x1.c
  src/qs8-vaddc/gen/minmax-scalar-x2.c
  src/qs8-vaddc/gen/minmax-scalar-x4.c
  src/qs8-vcvt/gen/vcvt-scalar-x1.c
  src/qs8-vcvt/gen/vcvt-scalar-x2.c
  src/qs8-vcvt/gen/vcvt-scalar-x4.c
  src/qs8-vlrelu/gen/vlrelu-scalar-andxor-x1.c
  src/qs8-vlrelu/gen/vlrelu-scalar-andxor-x2.c
  src/qs8-vlrelu/gen/vlrelu-scalar-andxor-x4.c
  src/qs8-vlrelu/gen/vlrelu-scalar-select-x1.c
  src/qs8-vlrelu/gen/vlrelu-scalar-select-x2.c
  src/qs8-vlrelu/gen/vlrelu-scalar-select-x4.c
  src/qs8-vmul/gen/minmax-fp32-scalar-x1.c
  src/qs8-vmul/gen/minmax-fp32-scalar-x2.c
  src/qs8-vmul/gen/minmax-fp32-scalar-x4.c
  src/qs8-vmulc/gen/minmax-fp32-scalar-x1.c
  src/qs8-vmulc/gen/minmax-fp32-scalar-x2.c
  src/qs8-vmulc/gen/minmax-fp32-scalar-x4.c
  src/qu8-avgpool/9p8x-minmax-scalar-c1.c
  src/qu8-avgpool/9x-minmax-scalar-c1.c
  src/qu8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c
  src/qu8-dwconv/gen/up1x9-minmax-fp32-scalar-imagic.c
  src/qu8-dwconv/gen/up1x9-minmax-fp32-scalar-lrintf.c
  src/qu8-dwconv/gen/up1x9-minmax-rndnu-scalar.c
  src/qu8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c
  src/qu8-dwconv/gen/up1x25-minmax-fp32-scalar-imagic.c
  src/qu8-dwconv/gen/up1x25-minmax-fp32-scalar-lrintf.c
  src/qu8-dwconv/gen/up2x9-minmax-fp32-scalar-fmagic.c
  src/qu8-dwconv/gen/up2x9-minmax-fp32-scalar-imagic.c
  src/qu8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c
  src/qu8-dwconv/gen/up2x9-minmax-rndnu-scalar.c
  src/qu8-dwconv/gen/up2x25-minmax-fp32-scalar-fmagic.c
  src/qu8-dwconv/gen/up2x25-minmax-fp32-scalar-imagic.c
  src/qu8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c
  src/qu8-dwconv/gen/up4x9-minmax-fp32-scalar-fmagic.c
  src/qu8-dwconv/gen/up4x9-minmax-fp32-scalar-imagic.c
  src/qu8-dwconv/gen/up4x9-minmax-fp32-scalar-lrintf.c
  src/qu8-dwconv/gen/up4x9-minmax-rndnu-scalar.c
  src/qu8-dwconv/gen/up4x25-minmax-fp32-scalar-fmagic.c
  src/qu8-dwconv/gen/up4x25-minmax-fp32-scalar-imagic.c
  src/qu8-dwconv/gen/up4x25-minmax-fp32-scalar-lrintf.c
  src/qu8-f32-vcvt/gen/vcvt-scalar-x1.c
  src/qu8-f32-vcvt/gen/vcvt-scalar-x2.c
  src/qu8-f32-vcvt/gen/vcvt-scalar-x3.c
  src/qu8-f32-vcvt/gen/vcvt-scalar-x4.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-fmagic-c1.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-fmagic-c2.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-fmagic-c4.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c1.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c2.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-imagic-c4.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-lrintf-c1.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-lrintf-c2.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-scalar-lrintf-c4.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-fmagic-c1.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-fmagic-c2.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-fmagic-c4.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c1.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c2.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-imagic-c4.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-lrintf-c1.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-lrintf-c2.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-scalar-lrintf-c4.c
  src/qu8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/1x2-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/1x2-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/1x2-minmax-rndnu-scalar.c
  src/qu8-gemm/gen/1x4-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/1x4-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/1x4-minmax-rndnu-scalar.c
  src/qu8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/2x2-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/2x2-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/2x2-minmax-rndnu-scalar.c
  src/qu8-gemm/gen/2x4-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/2x4-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/2x4-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/2x4-minmax-rndnu-scalar.c
  src/qu8-gemm/gen/3x2-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/3x2-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/3x2-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/3x2-minmax-rndnu-scalar.c
  src/qu8-gemm/gen/3x4-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/3x4-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/3x4-minmax-rndnu-scalar.c
  src/qu8-gemm/gen/4x2-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/4x2-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/4x2-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/4x2-minmax-rndnu-scalar.c
  src/qu8-gemm/gen/4x4-minmax-fp32-scalar-fmagic.c
  src/qu8-gemm/gen/4x4-minmax-fp32-scalar-imagic.c
  src/qu8-gemm/gen/4x4-minmax-fp32-scalar-lrintf.c
  src/qu8-gemm/gen/4x4-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/1x2-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/1x2-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/1x2-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/1x2-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/1x4-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/1x4-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/1x4-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/1x4-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/2x2-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/2x2-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/2x2-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/2x2-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/2x4-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/2x4-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/2x4-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/2x4-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/3x2-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/3x2-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/3x2-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/3x2-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/3x4-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/3x4-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/3x4-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/3x4-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/4x2-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/4x2-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/4x2-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/4x2-minmax-rndnu-scalar.c
  src/qu8-igemm/gen/4x4-minmax-fp32-scalar-fmagic.c
  src/qu8-igemm/gen/4x4-minmax-fp32-scalar-imagic.c
  src/qu8-igemm/gen/4x4-minmax-fp32-scalar-lrintf.c
  src/qu8-igemm/gen/4x4-minmax-rndnu-scalar.c
  src/qu8-requantization/fp32-scalar-fmagic.c
  src/qu8-requantization/fp32-scalar-lrintf.c
  src/qu8-requantization/gemmlowp-scalar.c
  src/qu8-requantization/rndna-scalar-signed64.c
  src/qu8-requantization/rndna-scalar-unsigned32.c
  src/qu8-requantization/rndna-scalar-unsigned64.c
  src/qu8-vadd/gen/minmax-scalar-x1.c
  src/qu8-vadd/gen/minmax-scalar-x2.c
  src/qu8-vadd/gen/minmax-scalar-x4.c
  src/qu8-vaddc/gen/minmax-scalar-x1.c
  src/qu8-vaddc/gen/minmax-scalar-x2.c
  src/qu8-vaddc/gen/minmax-scalar-x4.c
  src/qu8-vcvt/gen/vcvt-scalar-x1.c
  src/qu8-vcvt/gen/vcvt-scalar-x2.c
  src/qu8-vcvt/gen/vcvt-scalar-x4.c
  src/qu8-vlrelu/gen/vlrelu-scalar-andxor-x1.c
  src/qu8-vlrelu/gen/vlrelu-scalar-andxor-x2.c
  src/qu8-vlrelu/gen/vlrelu-scalar-andxor-x4.c
  src/qu8-vlrelu/gen/vlrelu-scalar-select-x1.c
  src/qu8-vlrelu/gen/vlrelu-scalar-select-x2.c
  src/qu8-vlrelu/gen/vlrelu-scalar-select-x4.c
  src/qu8-vmul/gen/minmax-fp32-scalar-x1.c
  src/qu8-vmul/gen/minmax-fp32-scalar-x2.c
  src/qu8-vmul/gen/minmax-fp32-scalar-x4.c
  src/qu8-vmulc/gen/minmax-fp32-scalar-x1.c
  src/qu8-vmulc/gen/minmax-fp32-scalar-x2.c
  src/qu8-vmulc/gen/minmax-fp32-scalar-x4.c
  src/s8-ibilinear/gen/scalar-c1.c
  src/s8-ibilinear/gen/scalar-c2.c
  src/s8-ibilinear/gen/scalar-c4.c
  src/s8-maxpool/9p8x-minmax-scalar-c1.c
  src/s8-vclamp/scalar-x4.c
  src/s16-rmaxabs/gen/scalar-x1.c
  src/s16-rmaxabs/gen/scalar-x2.c
  src/s16-rmaxabs/gen/scalar-x3.c
  src/s16-rmaxabs/gen/scalar-x4.c
  src/s16-vlshift/gen/scalar-x1.c
  src/s16-vlshift/gen/scalar-x2.c
  src/s16-vlshift/gen/scalar-x3.c
  src/s16-vlshift/gen/scalar-x4.c
  src/s16-window/gen/scalar-x1.c
  src/s16-window/gen/scalar-x2.c
  src/s16-window/gen/scalar-x3.c
  src/s16-window/gen/scalar-x4.c
  src/u8-ibilinear/gen/scalar-c1.c
  src/u8-ibilinear/gen/scalar-c2.c
  src/u8-ibilinear/gen/scalar-c4.c
  src/u8-lut32norm/scalar.c
  src/u8-maxpool/9p8x-minmax-scalar-c1.c
  src/u8-rmax/scalar.c
  src/u8-vclamp/scalar-x4.c
  src/u32-filterbank-accumulate/gen/scalar-x1.c
  src/u32-filterbank-subtract/scalar-x2.c
  src/u32-vlog/gen/scalar-x1.c
  src/u32-vlog/gen/scalar-x2.c
  src/u32-vlog/gen/scalar-x3.c
  src/u32-vlog/gen/scalar-x4.c
  src/u64-u32-vsqrtshift/scalar-cvtu32-sqrt-cvtu32f64-x1.c
  src/xx-copy/memcpy.c
  src/xx-fill/scalar-x16.c
  src/xx-pad/scalar.c
  src/xx-transpose/1x1-memcpy.c
  src/x8-lut/gen/lut-scalar-x1.c
  src/x8-lut/gen/lut-scalar-x2.c
  src/x8-lut/gen/lut-scalar-x4.c
  src/x8-lut/gen/lut-scalar-x8.c
  src/x8-lut/gen/lut-scalar-x16.c
  src/x8-transposec/gen/1x2-scalar-int.c
  src/x8-transposec/gen/1x4-scalar-int.c
  src/x8-transposec/gen/2x1-scalar-int.c
  src/x8-transposec/gen/2x2-scalar-int.c
  src/x8-transposec/gen/2x4-scalar-int.c
  src/x8-transposec/gen/4x1-scalar-int.c
  src/x8-transposec/gen/4x2-scalar-int.c
  src/x8-transposec/gen/4x4-scalar-int.c
  src/x8-zip/xm-scalar.c
  src/x8-zip/x2-scalar.c
  src/x8-zip/x3-scalar.c
  src/x8-zip/x4-scalar.c
  src/x16-transposec/gen/1x2-scalar-int.c
  src/x16-transposec/gen/1x4-scalar-int.c
  src/x16-transposec/gen/2x1-scalar-int.c
  src/x16-transposec/gen/2x2-scalar-int.c
  src/x16-transposec/gen/2x4-scalar-int.c
  src/x16-transposec/gen/4x1-scalar-int.c
  src/x16-transposec/gen/4x2-scalar-int.c
  src/x16-transposec/gen/4x4-scalar-int.c
  src/x24-transposec/gen/1x2-scalar.c
  src/x24-transposec/gen/1x4-scalar.c
  src/x24-transposec/gen/2x1-scalar.c
  src/x24-transposec/gen/2x2-scalar.c
  src/x24-transposec/gen/2x4-scalar.c
  src/x24-transposec/gen/4x1-scalar.c
  src/x24-transposec/gen/4x2-scalar.c
  src/x24-transposec/gen/4x4-scalar.c
  src/x32-packx/x2-scalar.c
  src/x32-packx/x3-scalar.c
  src/x32-packx/x4-scalar.c
  src/x32-transposec/gen/1x2-scalar-float.c
  src/x32-transposec/gen/1x2-scalar-int.c
  src/x32-transposec/gen/1x4-scalar-float.c
  src/x32-transposec/gen/1x4-scalar-int.c
  src/x32-transposec/gen/2x1-scalar-float.c
  src/x32-transposec/gen/2x1-scalar-int.c
  src/x32-transposec/gen/2x2-scalar-float.c
  src/x32-transposec/gen/2x2-scalar-int.c
  src/x32-transposec/gen/2x4-scalar-float.c
  src/x32-transposec/gen/2x4-scalar-int.c
  src/x32-transposec/gen/4x1-scalar-float.c
  src/x32-transposec/gen/4x1-scalar-int.c
  src/x32-transposec/gen/4x2-scalar-float.c
  src/x32-transposec/gen/4x2-scalar-int.c
  src/x32-transposec/gen/4x4-scalar-float.c
  src/x32-transposec/gen/4x4-scalar-int.c
  src/x32-unpool/scalar.c
  src/x32-zip/xm-scalar.c
  src/x32-zip/x2-scalar.c
  src/x32-zip/x3-scalar.c
  src/x32-zip/x4-scalar.c
  src/x64-transposec/gen/1x2-scalar-float.c
  src/x64-transposec/gen/1x2-scalar-int.c
  src/x64-transposec/gen/2x1-scalar-float.c
  src/x64-transposec/gen/2x1-scalar-int.c
  src/x64-transposec/gen/2x2-scalar-float.c
  src/x64-transposec/gen/2x2-scalar-int.c
  src/x64-transposec/gen/4x1-scalar-float.c
  src/x64-transposec/gen/4x1-scalar-int.c
  src/x64-transposec/gen/4x2-scalar-float.c
  src/x64-transposec/gen/4x2-scalar-int.c)

SET(PROD_ARMSIMD32_MICROKERNEL_SRCS
  src/qc8-gemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qc8-gemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qc8-igemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qc8-igemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qs8-gemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qs8-gemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qs8-igemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qs8-igemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qs8-vcvt/gen/vcvt-armsimd32-x8.c
  src/qs8-vlrelu/gen/vlrelu-armsimd32-x4.c
  src/qu8-gemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qu8-gemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qu8-igemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qu8-igemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qu8-vcvt/gen/vcvt-armsimd32-x8.c
  src/qu8-vlrelu/gen/vlrelu-armsimd32-x4.c)

SET(ALL_ARMSIMD32_MICROKERNEL_SRCS
  src/qc8-gemm/gen/1x1c4-minmax-fp32-armsimd32.c
  src/qc8-gemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qc8-gemm/gen/2x1c4-minmax-fp32-armsimd32.c
  src/qc8-gemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qc8-igemm/gen/1x1c4-minmax-fp32-armsimd32.c
  src/qc8-igemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qc8-igemm/gen/2x1c4-minmax-fp32-armsimd32.c
  src/qc8-igemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qs8-gemm/gen/1x1c4-minmax-fp32-armsimd32.c
  src/qs8-gemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qs8-gemm/gen/2x1c4-minmax-fp32-armsimd32.c
  src/qs8-gemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qs8-igemm/gen/1x1c4-minmax-fp32-armsimd32.c
  src/qs8-igemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qs8-igemm/gen/2x1c4-minmax-fp32-armsimd32.c
  src/qs8-igemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qs8-vcvt/gen/vcvt-armsimd32-x4.c
  src/qs8-vcvt/gen/vcvt-armsimd32-x8.c
  src/qs8-vlrelu/gen/vlrelu-armsimd32-x4.c
  src/qs8-vlrelu/gen/vlrelu-armsimd32-x8.c
  src/qu8-gemm/gen/1x1c4-minmax-fp32-armsimd32.c
  src/qu8-gemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qu8-gemm/gen/2x1c4-minmax-fp32-armsimd32.c
  src/qu8-gemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qu8-igemm/gen/1x1c4-minmax-fp32-armsimd32.c
  src/qu8-igemm/gen/1x2c4-minmax-fp32-armsimd32.c
  src/qu8-igemm/gen/2x1c4-minmax-fp32-armsimd32.c
  src/qu8-igemm/gen/2x2c4-minmax-fp32-armsimd32.c
  src/qu8-vcvt/gen/vcvt-armsimd32-x4.c
  src/qu8-vcvt/gen/vcvt-armsimd32-x8.c
  src/qu8-vlrelu/gen/vlrelu-armsimd32-x4.c
  src/qu8-vlrelu/gen/vlrelu-armsimd32-x8.c)

SET(ALL_FP16ARITH_MICROKERNEL_SRCS
  src/f16-vbinary/gen/vadd-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vadd-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vadd-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vaddc-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vaddc-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vaddc-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vdiv-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vdiv-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vdiv-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vdivc-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vdivc-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vdivc-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vmax-fp16arith-x1.c
  src/f16-vbinary/gen/vmax-fp16arith-x2.c
  src/f16-vbinary/gen/vmax-fp16arith-x4.c
  src/f16-vbinary/gen/vmaxc-fp16arith-x1.c
  src/f16-vbinary/gen/vmaxc-fp16arith-x2.c
  src/f16-vbinary/gen/vmaxc-fp16arith-x4.c
  src/f16-vbinary/gen/vmin-fp16arith-x1.c
  src/f16-vbinary/gen/vmin-fp16arith-x2.c
  src/f16-vbinary/gen/vmin-fp16arith-x4.c
  src/f16-vbinary/gen/vminc-fp16arith-x1.c
  src/f16-vbinary/gen/vminc-fp16arith-x2.c
  src/f16-vbinary/gen/vminc-fp16arith-x4.c
  src/f16-vbinary/gen/vmul-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vmul-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vmul-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vmulc-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vmulc-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vmulc-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vrdivc-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vrdivc-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vrdivc-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vrsubc-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vrsubc-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vrsubc-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vsqrdiff-fp16arith-x1.c
  src/f16-vbinary/gen/vsqrdiff-fp16arith-x2.c
  src/f16-vbinary/gen/vsqrdiff-fp16arith-x4.c
  src/f16-vbinary/gen/vsqrdiffc-fp16arith-x1.c
  src/f16-vbinary/gen/vsqrdiffc-fp16arith-x2.c
  src/f16-vbinary/gen/vsqrdiffc-fp16arith-x4.c
  src/f16-vbinary/gen/vsub-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vsub-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vsub-minmax-fp16arith-x4.c
  src/f16-vbinary/gen/vsubc-minmax-fp16arith-x1.c
  src/f16-vbinary/gen/vsubc-minmax-fp16arith-x2.c
  src/f16-vbinary/gen/vsubc-minmax-fp16arith-x4.c
  src/f16-vsqrt/gen/fp16arith-sqrt-x1.c
  src/f16-vsqrt/gen/fp16arith-sqrt-x2.c
  src/f16-vsqrt/gen/fp16arith-sqrt-x4.c)

SET(PROD_NEON_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-neon-int16-x16.c
  src/f32-argmaxpool/4x-neon-c4.c
  src/f32-argmaxpool/9p8x-neon-c4.c
  src/f32-argmaxpool/9x-neon-c4.c
  src/f32-avgpool/9p8x-minmax-neon-c4.c
  src/f32-avgpool/9x-minmax-neon-c4.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-neon-2x2.c
  src/f32-dwconv/gen/up8x3-minmax-neon.c
  src/f32-dwconv/gen/up8x4-minmax-neon.c
  src/f32-dwconv/gen/up8x9-minmax-neon.c
  src/f32-dwconv/gen/up8x25-minmax-neon-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-2x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4.c
  src/f32-f16-vcvt/gen/vcvt-neon-x8.c
  src/f32-gavgpool-cw/neon-x4.c
  src/f32-gavgpool/7p7x-minmax-neon-c4.c
  src/f32-gavgpool/7x-minmax-neon-c4.c
  src/f32-gemm/gen/1x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/4x2-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neon-lane-ld128.c
  src/f32-ibilinear-chw/gen/neon-p8.c
  src/f32-ibilinear/gen/neon-c8.c
  src/f32-igemm/gen/1x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x2-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neon-lane-ld128.c
  src/f32-maxpool/9p8x-minmax-neon-c4.c
  src/f32-pavgpool/9p8x-minmax-neon-c4.c
  src/f32-pavgpool/9x-minmax-neon-c4.c
  src/f32-prelu/gen/neon-2x8.c
  src/f32-qs8-vcvt/gen/vcvt-neon-x32.c
  src/f32-qu8-vcvt/gen/vcvt-neon-x32.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x8.c
  src/f32-rmax/neon.c
  src/f32-spmm/gen/32x1-minmax-neon.c
  src/f32-vbinary/gen/vadd-minmax-neon-x8.c
  src/f32-vbinary/gen/vaddc-minmax-neon-x8.c
  src/f32-vbinary/gen/vmax-neon-x8.c
  src/f32-vbinary/gen/vmaxc-neon-x8.c
  src/f32-vbinary/gen/vmin-neon-x8.c
  src/f32-vbinary/gen/vminc-neon-x8.c
  src/f32-vbinary/gen/vmul-minmax-neon-x8.c
  src/f32-vbinary/gen/vmulc-minmax-neon-x8.c
  src/f32-vbinary/gen/vrsubc-minmax-neon-x8.c
  src/f32-vbinary/gen/vsqrdiff-neon-x8.c
  src/f32-vbinary/gen/vsqrdiffc-neon-x8.c
  src/f32-vbinary/gen/vsub-minmax-neon-x8.c
  src/f32-vbinary/gen/vsubc-minmax-neon-x8.c
  src/f32-vclamp/gen/vclamp-neon-x8.c
  src/f32-velu/gen/velu-neon-rr2-lut16-p3-x8.c
  src/f32-vhswish/gen/vhswish-neon-x16.c
  src/f32-vlrelu/gen/vlrelu-neon-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-neon-2x.c
  src/f32-vrnd/gen/vrndd-neon-x8.c
  src/f32-vrnd/gen/vrndne-neon-x8.c
  src/f32-vrnd/gen/vrndu-neon-x8.c
  src/f32-vrnd/gen/vrndz-neon-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x8.c
  src/f32-vunary/gen/vabs-neon-x8.c
  src/f32-vunary/gen/vneg-neon-x8.c
  src/f32-vunary/gen/vsqr-neon-x8.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-neon-mla8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mla8-ld64.c
  src/qc8-gemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c
  src/qc8-gemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c
  src/qs8-dwconv/gen/up8x25-minmax-rndnu-neon-mla8-ld64.c
  src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mla8-ld64.c
  src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mla8-ld64.c
  src/qs8-f32-vcvt/gen/vcvt-neon-x32.c
  src/qs8-gavgpool/gen/7p7x-minmax-rndnu-neon-c8.c
  src/qs8-gavgpool/gen/7x-minmax-rndnu-neon-c8.c
  src/qs8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/1x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/2x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/1x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/2x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-vadd/gen/minmax-neon-ld64-x16.c
  src/qs8-vadd/gen/minmax-neon-ld64-x32.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x16.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x32.c
  src/qs8-vcvt/gen/vcvt-neon-x32.c
  src/qs8-vlrelu/gen/vlrelu-neon-x32.c
  src/qs8-vmul/gen/minmax-rndnu-neon-ld64-x16.c
  src/qs8-vmulc/gen/minmax-rndnu-neon-ld64-x16.c
  src/qu8-avgpool/9p8x-minmax-neon-c8.c
  src/qu8-avgpool/9x-minmax-neon-c8.c
  src/qu8-dwconv/gen/up8x25-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up16x9-minmax-rndnu-neon-mul8.c
  src/qu8-f32-vcvt/gen/vcvt-neon-x32.c
  src/qu8-gavgpool/gen/7p7x-minmax-rndnu-neon-c8.c
  src/qu8-gavgpool/gen/7x-minmax-rndnu-neon-c8.c
  src/qu8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-vadd/gen/minmax-neon-ld64-x16.c
  src/qu8-vadd/gen/minmax-neon-ld64-x32.c
  src/qu8-vaddc/gen/minmax-neon-ld64-x16.c
  src/qu8-vaddc/gen/minmax-neon-ld64-x32.c
  src/qu8-vcvt/gen/vcvt-neon-x32.c
  src/qu8-vlrelu/gen/vlrelu-neon-x32.c
  src/qu8-vmul/gen/minmax-rndnu-neon-ld64-x16.c
  src/qu8-vmulc/gen/minmax-rndnu-neon-ld64-x16.c
  src/s8-ibilinear/gen/neon-c8.c
  src/s8-ibilinear/gen/neon-c16.c
  src/s8-maxpool/9p8x-minmax-neon-c16.c
  src/s8-vclamp/neon-x64.c
  src/u8-ibilinear/gen/neon-c8.c
  src/u8-ibilinear/gen/neon-c16.c
  src/u8-maxpool/9p8x-minmax-neon-c16.c
  src/u8-rmax/neon.c
  src/u8-vclamp/neon-x64.c
  src/xx-fill/neon-x64.c
  src/xx-pad/neon.c
  src/x8-transposec/gen/16x16-reuse-dec-zip-neon.c
  src/x8-zip/xm-neon.c
  src/x8-zip/x2-neon.c
  src/x8-zip/x3-neon.c
  src/x8-zip/x4-neon.c
  src/x16-transposec/gen/8x8-reuse-dec-zip-neon.c
  src/x32-packx/x4-neon-st4.c
  src/x32-transposec/gen/4x4-reuse-dec-zip-neon.c
  src/x32-unpool/neon.c
  src/x32-zip/xm-neon.c
  src/x32-zip/x2-neon.c
  src/x32-zip/x3-neon.c
  src/x32-zip/x4-neon.c)

SET(ALL_NEON_MICROKERNEL_SRCS
  src/cs16-bfly4/neon-x1.c
  src/cs16-bfly4/neon-x4.c
  src/cs16-bfly4/samples1-neon.c
  src/cs16-bfly4/samples4-neon.c
  src/cs16-fftr/neon-x4.c
  src/cs16-vsquareabs/gen/neon-mlal-ld128-x4.c
  src/cs16-vsquareabs/gen/neon-mlal-ld128-x8.c
  src/cs16-vsquareabs/gen/neon-mlal-ld128-x12.c
  src/cs16-vsquareabs/gen/neon-mlal-ld128-x16.c
  src/f16-f32-vcvt/gen/vcvt-neon-int16-x8.c
  src/f16-f32-vcvt/gen/vcvt-neon-int16-x16.c
  src/f16-f32-vcvt/gen/vcvt-neon-int16-x24.c
  src/f16-f32-vcvt/gen/vcvt-neon-int16-x32.c
  src/f16-f32-vcvt/gen/vcvt-neon-int32-x8.c
  src/f16-f32-vcvt/gen/vcvt-neon-int32-x16.c
  src/f16-f32-vcvt/gen/vcvt-neon-int32-x24.c
  src/f16-f32-vcvt/gen/vcvt-neon-int32-x32.c
  src/f32-argmaxpool/4x-neon-c4.c
  src/f32-argmaxpool/9p8x-neon-c4.c
  src/f32-argmaxpool/9x-neon-c4.c
  src/f32-avgpool/9p8x-minmax-neon-c4.c
  src/f32-avgpool/9x-minmax-neon-c4.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x4-neon-2x1.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x4-neon-2x2.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x8-neon-2x1.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x8-neon-2x2.c
  src/f32-conv-hwc/gen/3x3s2p1c3x4-neon-2x1.c
  src/f32-conv-hwc/gen/3x3s2p1c3x4-neon-2x2.c
  src/f32-conv-hwc/gen/3x3s2p1c3x8-neon-2x1.c
  src/f32-conv-hwc/gen/3x3s2p1c3x8-neon-2x2.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-neon-2x2.c
  src/f32-dwconv/gen/up4x3-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x3-minmax-neon.c
  src/f32-dwconv/gen/up4x4-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-neon.c
  src/f32-dwconv/gen/up4x9-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-neon.c
  src/f32-dwconv/gen/up4x25-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-neon.c
  src/f32-dwconv/gen/up8x3-minmax-neon-acc2.c
  src/f32-dwconv/gen/up8x3-minmax-neon.c
  src/f32-dwconv/gen/up8x4-minmax-neon-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-neon.c
  src/f32-dwconv/gen/up8x9-minmax-neon-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-neon.c
  src/f32-dwconv/gen/up8x25-minmax-neon-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-neon.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-2x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-3x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-4x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-5x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-6x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-2x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-3x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-4x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc5.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-2x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-2x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-3x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-3x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-4x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-4x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-5x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc5.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-2x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-2x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-3x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-3x4.c
  src/f32-f16-vcvt/gen/vcvt-neon-x8.c
  src/f32-f16-vcvt/gen/vcvt-neon-x16.c
  src/f32-f16-vcvt/gen/vcvt-neon-x24.c
  src/f32-f16-vcvt/gen/vcvt-neon-x32.c
  src/f32-gavgpool-cw/neon-x4.c
  src/f32-gavgpool/7p7x-minmax-neon-c4.c
  src/f32-gavgpool/7x-minmax-neon-c4.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neon-dup-ld64.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/1x8s4inc-minmax-neon.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-dup-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-dup-ld128.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-lane-ld128.c
  src/f32-gemm/gen-inc/4x8s4inc-minmax-neon.c
  src/f32-gemm/gen-inc/5x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-dup-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-dup-ld128.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-lane-ld128.c
  src/f32-gemm/gen-inc/6x8s4inc-minmax-neon.c
  src/f32-gemm/gen-inc/8x8s4inc-minmax-neon.c
  src/f32-gemm/gen/1x8-minmax-neon-dup-ld64.c
  src/f32-gemm/gen/1x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/1x8s4-minmax-neon.c
  src/f32-gemm/gen/4x2-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neon-dup-ld64.c
  src/f32-gemm/gen/4x8-minmax-neon-dup-ld128.c
  src/f32-gemm/gen/4x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neon-lane-ld128.c
  src/f32-gemm/gen/4x8s4-minmax-neon.c
  src/f32-gemm/gen/5x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/6x2-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-dup-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-dup-ld128.c
  src/f32-gemm/gen/6x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-lane-ld128.c
  src/f32-gemm/gen/6x8s4-minmax-neon.c
  src/f32-gemm/gen/8x8s4-minmax-neon.c
  src/f32-ibilinear-chw/gen/neon-p4.c
  src/f32-ibilinear-chw/gen/neon-p8.c
  src/f32-ibilinear-chw/gen/neon-p16.c
  src/f32-ibilinear/gen/neon-c4.c
  src/f32-ibilinear/gen/neon-c8.c
  src/f32-igemm/gen/1x8-minmax-neon-dup-ld64.c
  src/f32-igemm/gen/1x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/1x8s4-minmax-neon.c
  src/f32-igemm/gen/4x2-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x4-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neon-dup-ld64.c
  src/f32-igemm/gen/4x8-minmax-neon-dup-ld128.c
  src/f32-igemm/gen/4x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neon-lane-ld128.c
  src/f32-igemm/gen/4x8s4-minmax-neon.c
  src/f32-igemm/gen/6x2-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neon-dup-ld64.c
  src/f32-igemm/gen/6x8-minmax-neon-dup-ld128.c
  src/f32-igemm/gen/6x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neon-lane-ld128.c
  src/f32-igemm/gen/6x8s4-minmax-neon.c
  src/f32-igemm/gen/8x8s4-minmax-neon.c
  src/f32-maxpool/9p8x-minmax-neon-c4.c
  src/f32-pavgpool/9p8x-minmax-neon-c4.c
  src/f32-pavgpool/9x-minmax-neon-c4.c
  src/f32-ppmm/gen/4x8-minmax-neon.c
  src/f32-ppmm/gen/8x8-minmax-neon.c
  src/f32-prelu/gen/neon-1x4.c
  src/f32-prelu/gen/neon-1x8.c
  src/f32-prelu/gen/neon-1x16.c
  src/f32-prelu/gen/neon-2x4.c
  src/f32-prelu/gen/neon-2x8.c
  src/f32-prelu/gen/neon-2x16.c
  src/f32-prelu/gen/neon-4x4.c
  src/f32-prelu/gen/neon-4x8.c
  src/f32-prelu/gen/neon-4x16.c
  src/f32-qs8-vcvt/gen/vcvt-neon-x8.c
  src/f32-qs8-vcvt/gen/vcvt-neon-x16.c
  src/f32-qs8-vcvt/gen/vcvt-neon-x24.c
  src/f32-qs8-vcvt/gen/vcvt-neon-x32.c
  src/f32-qu8-vcvt/gen/vcvt-neon-x8.c
  src/f32-qu8-vcvt/gen/vcvt-neon-x16.c
  src/f32-qu8-vcvt/gen/vcvt-neon-x24.c
  src/f32-qu8-vcvt/gen/vcvt-neon-x32.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x8.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x12.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x16.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-lut64-p2-x20.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neon-rr2-p5-x20.c
  src/f32-rmax/neon.c
  src/f32-spmm/gen/4x1-minmax-neon-pipelined.c
  src/f32-spmm/gen/4x1-minmax-neon-x2.c
  src/f32-spmm/gen/4x1-minmax-neon.c
  src/f32-spmm/gen/8x1-minmax-neon-pipelined.c
  src/f32-spmm/gen/8x1-minmax-neon-x2.c
  src/f32-spmm/gen/8x1-minmax-neon.c
  src/f32-spmm/gen/12x1-minmax-neon.c
  src/f32-spmm/gen/16x1-minmax-neon-pipelined.c
  src/f32-spmm/gen/16x1-minmax-neon-x2.c
  src/f32-spmm/gen/16x1-minmax-neon.c
  src/f32-spmm/gen/32x1-minmax-neon-pipelined.c
  src/f32-spmm/gen/32x1-minmax-neon-x2.c
  src/f32-spmm/gen/32x1-minmax-neon.c
  src/f32-vbinary/gen/vadd-minmax-neon-x4.c
  src/f32-vbinary/gen/vadd-minmax-neon-x8.c
  src/f32-vbinary/gen/vaddc-minmax-neon-x4.c
  src/f32-vbinary/gen/vaddc-minmax-neon-x8.c
  src/f32-vbinary/gen/vmax-neon-x4.c
  src/f32-vbinary/gen/vmax-neon-x8.c
  src/f32-vbinary/gen/vmaxc-neon-x4.c
  src/f32-vbinary/gen/vmaxc-neon-x8.c
  src/f32-vbinary/gen/vmin-neon-x4.c
  src/f32-vbinary/gen/vmin-neon-x8.c
  src/f32-vbinary/gen/vminc-neon-x4.c
  src/f32-vbinary/gen/vminc-neon-x8.c
  src/f32-vbinary/gen/vmul-minmax-neon-x4.c
  src/f32-vbinary/gen/vmul-minmax-neon-x8.c
  src/f32-vbinary/gen/vmulc-minmax-neon-x4.c
  src/f32-vbinary/gen/vmulc-minmax-neon-x8.c
  src/f32-vbinary/gen/vrsubc-minmax-neon-x4.c
  src/f32-vbinary/gen/vrsubc-minmax-neon-x8.c
  src/f32-vbinary/gen/vsqrdiff-neon-x4.c
  src/f32-vbinary/gen/vsqrdiff-neon-x8.c
  src/f32-vbinary/gen/vsqrdiffc-neon-x4.c
  src/f32-vbinary/gen/vsqrdiffc-neon-x8.c
  src/f32-vbinary/gen/vsub-minmax-neon-x4.c
  src/f32-vbinary/gen/vsub-minmax-neon-x8.c
  src/f32-vbinary/gen/vsubc-minmax-neon-x4.c
  src/f32-vbinary/gen/vsubc-minmax-neon-x8.c
  src/f32-vclamp/gen/vclamp-neon-x4.c
  src/f32-vclamp/gen/vclamp-neon-x8.c
  src/f32-velu/gen/velu-neon-rr2-lut16-p3-x4.c
  src/f32-velu/gen/velu-neon-rr2-lut16-p3-x8.c
  src/f32-velu/gen/velu-neon-rr2-lut16-p3-x12.c
  src/f32-velu/gen/velu-neon-rr2-lut16-p3-x16.c
  src/f32-velu/gen/velu-neon-rr2-lut16-p3-x20.c
  src/f32-velu/gen/velu-neon-rr2-lut16-p3-x24.c
  src/f32-velu/gen/velu-neon-rr2-p6-x4.c
  src/f32-velu/gen/velu-neon-rr2-p6-x8.c
  src/f32-velu/gen/velu-neon-rr2-p6-x12.c
  src/f32-velu/gen/velu-neon-rr2-p6-x16.c
  src/f32-velu/gen/velu-neon-rr2-p6-x20.c
  src/f32-velu/gen/velu-neon-rr2-p6-x24.c
  src/f32-vhswish/gen/vhswish-neon-x4.c
  src/f32-vhswish/gen/vhswish-neon-x8.c
  src/f32-vhswish/gen/vhswish-neon-x16.c
  src/f32-vlrelu/gen/vlrelu-neon-x4.c
  src/f32-vlrelu/gen/vlrelu-neon-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-neon-2x.c
  src/f32-vmulcaddc/gen/c8-minmax-neon-2x.c
  src/f32-vrelu/gen/vrelu-neon-x4.c
  src/f32-vrelu/gen/vrelu-neon-x8.c
  src/f32-vrnd/gen/vrndd-neon-x4.c
  src/f32-vrnd/gen/vrndd-neon-x8.c
  src/f32-vrnd/gen/vrndne-neon-x4.c
  src/f32-vrnd/gen/vrndne-neon-x8.c
  src/f32-vrnd/gen/vrndu-neon-x4.c
  src/f32-vrnd/gen/vrndu-neon-x8.c
  src/f32-vrnd/gen/vrndz-neon-x4.c
  src/f32-vrnd/gen/vrndz-neon-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x24.c
  src/f32-vunary/gen/vabs-neon-x4.c
  src/f32-vunary/gen/vabs-neon-x8.c
  src/f32-vunary/gen/vneg-neon-x4.c
  src/f32-vunary/gen/vneg-neon-x8.c
  src/f32-vunary/gen/vsqr-neon-x4.c
  src/f32-vunary/gen/vsqr-neon-x8.c
  src/math/cvt-f16-f32-neon-int16.c
  src/math/cvt-f16-f32-neon-int32.c
  src/math/cvt-f32-f16-neon.c
  src/math/cvt-f32-qs8-neon.c
  src/math/cvt-f32-qu8-neon.c
  src/math/expm1minus-f32-neon-rr2-lut16-p3.c
  src/math/expm1minus-f32-neon-rr2-p6.c
  src/math/roundd-neon-addsub.c
  src/math/roundd-neon-cvt.c
  src/math/roundne-neon-addsub.c
  src/math/roundu-neon-addsub.c
  src/math/roundu-neon-cvt.c
  src/math/roundz-neon-addsub.c
  src/math/roundz-neon-cvt.c
  src/math/sigmoid-f32-neon-rr2-lut64-p2-nr2recps.c
  src/math/sigmoid-f32-neon-rr2-lut2048-p1-nr2recps.c
  src/math/sigmoid-f32-neon-rr2-p5-nr2recps.c
  src/math/sqrt-neon-nr1rsqrts.c
  src/math/sqrt-neon-nr2rsqrts.c
  src/math/sqrt-neon-nr3rsqrts.c
  src/qc8-dwconv/gen/up8x3-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-neon-mul8-ld64.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-neon-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neon-mul8-ld64.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neon-mul16.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-neon-mla8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mla8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mul8-ld64.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mul8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mla8-ld64.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mla8-ld128.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mul8-ld64.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mul8-ld128.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-neon-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-neon-mul16.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-neon-mul16.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-neon-mul16.c
  src/qc8-gemm/gen/1x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/1x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-dup.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qc8-gemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-dup.c
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-gemm/gen/1x8c4s2-minmax-fp32-neon-mlal.c
  src/qc8-gemm/gen/1x8c8-minmax-fp32-neon-mlal.c
  src/qc8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/2x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/2x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-dup.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qc8-gemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c
  src/qc8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-dup.c
  src/qc8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-gemm/gen/2x8c4s2-minmax-fp32-neon-mlal.c
  src/qc8-gemm/gen/2x8c8-minmax-fp32-neon-mlal.c
  src/qc8-gemm/gen/2x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/2x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/3x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/3x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/3x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/3x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/4x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/4x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/6x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/6x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-gemm/gen/6x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-gemm/gen/6x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/1x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/1x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-dup.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qc8-igemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neon-mlal-dup.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-igemm/gen/1x8c4s2-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/1x8c8-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/1x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/2x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/2x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-dup.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qc8-igemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/2x8c4-minmax-fp32-neon-mlal-dup.c
  src/qc8-igemm/gen/2x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qc8-igemm/gen/2x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qc8-igemm/gen/2x8c4s2-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/2x8c8-minmax-fp32-neon-mlal.c
  src/qc8-igemm/gen/2x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/2x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/3x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/3x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/3x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/3x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/4x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/4x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/4x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/4x16-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/6x8-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/6x8-minmax-fp32-neon-mlal-lane.c
  src/qc8-igemm/gen/6x16-minmax-fp32-neon-mlal-lane-prfm.c
  src/qc8-igemm/gen/6x16-minmax-fp32-neon-mlal-lane.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-rndnu-neon-mla8-ld64.c
  src/qs8-dwconv/gen/up8x9-minmax-rndnu-neon-mul8-ld64.c
  src/qs8-dwconv/gen/up8x9-minmax-rndnu-neon-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-rndnu-neon-mla8-ld64.c
  src/qs8-dwconv/gen/up8x25-minmax-rndnu-neon-mul8-ld64.c
  src/qs8-dwconv/gen/up8x25-minmax-rndnu-neon-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mla8-ld64.c
  src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mla8-ld128.c
  src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mul8-ld64.c
  src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mul8-ld128.c
  src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mla8-ld64.c
  src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mla8-ld128.c
  src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mul8-ld64.c
  src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mul8-ld128.c
  src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-rndnu-neon-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-rndnu-neon-mul16.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up32x9-minmax-rndnu-neon-mul16.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-neon-mul16.c
  src/qs8-dwconv/gen/up32x25-minmax-rndnu-neon-mul16.c
  src/qs8-f32-vcvt/gen/vcvt-neon-x8.c
  src/qs8-f32-vcvt/gen/vcvt-neon-x16.c
  src/qs8-f32-vcvt/gen/vcvt-neon-x24.c
  src/qs8-f32-vcvt/gen/vcvt-neon-x32.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neon-c8.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neon-c16.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neon-c24.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neon-c32.c
  src/qs8-gavgpool/gen/7p7x-minmax-rndnu-neon-c8.c
  src/qs8-gavgpool/gen/7p7x-minmax-rndnu-neon-c16.c
  src/qs8-gavgpool/gen/7p7x-minmax-rndnu-neon-c24.c
  src/qs8-gavgpool/gen/7p7x-minmax-rndnu-neon-c32.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neon-c8.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neon-c16.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neon-c24.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neon-c32.c
  src/qs8-gavgpool/gen/7x-minmax-rndnu-neon-c8.c
  src/qs8-gavgpool/gen/7x-minmax-rndnu-neon-c16.c
  src/qs8-gavgpool/gen/7x-minmax-rndnu-neon-c24.c
  src/qs8-gavgpool/gen/7x-minmax-rndnu-neon-c32.c
  src/qs8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/1x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-dup.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c
  src/qs8-gemm/gen/1x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-dup.c
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/1x8c4s2-minmax-fp32-neon-mlal.c
  src/qs8-gemm/gen/1x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/1x8c8-minmax-fp32-neon-mlal.c
  src/qs8-gemm/gen/1x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x8c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/1x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
  src/qs8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/1x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/1x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/1x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/1x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/1x16c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/1x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/2x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/2x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-dup.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c
  src/qs8-gemm/gen/2x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-dup.c
  src/qs8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/2x8c4s2-minmax-fp32-neon-mlal.c
  src/qs8-gemm/gen/2x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/2x8c8-minmax-fp32-neon-mlal.c
  src/qs8-gemm/gen/2x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x8c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/2x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/2x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/2x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/2x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/2x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/2x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/2x16c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/2x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/3x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/3x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/3x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/3x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x8c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/3x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/3x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/3x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/3x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/3x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/3x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/3x16c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/3x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/4x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/4x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/4x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/4x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/4x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x8c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/4x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane.c
  src/qs8-gemm/gen/4x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/4x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-gemm/gen/4x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-gemm/gen/4x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/4x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/4x16c8-minmax-rndnu-neon-mull.c
  src/qs8-gemm/gen/4x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-gemm/gen/6x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/6x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-gemm/gen/6x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/6x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/1x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/1x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-dup.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/1x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c
  src/qs8-igemm/gen/1x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neon-mlal-dup.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/1x8c4s2-minmax-fp32-neon-mlal.c
  src/qs8-igemm/gen/1x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/1x8c8-minmax-fp32-neon-mlal.c
  src/qs8-igemm/gen/1x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x8c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/1x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
  src/qs8-igemm/gen/1x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/1x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/1x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/1x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/1x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/1x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/1x16c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/1x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/2x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/2x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-dup.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-ld4r.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/2x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c
  src/qs8-igemm/gen/2x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/2x8c4-minmax-fp32-neon-mlal-dup.c
  src/qs8-igemm/gen/2x8c4-minmax-fp32-neon-mlal-ld1r.c
  src/qs8-igemm/gen/2x8c4-minmax-fp32-neon-mlal-ld2r.c
  src/qs8-igemm/gen/2x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/2x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/2x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/2x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/2x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/2x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/2x8c4s2-minmax-fp32-neon-mlal.c
  src/qs8-igemm/gen/2x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/2x8c8-minmax-fp32-neon-mlal.c
  src/qs8-igemm/gen/2x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x8c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/2x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/2x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/2x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/2x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/2x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/2x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/2x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/2x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/2x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/2x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/2x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/2x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/2x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/2x16c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/2x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/3x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/3x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/3x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/3x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/3x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/3x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/3x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/3x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/3x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/3x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/3x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x8c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/3x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/3x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/3x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/3x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/3x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/3x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/3x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/3x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/3x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/3x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/3x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/3x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/3x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/3x16c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/3x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/4x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/4x8-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/4x8c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/4x8c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x8c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/4x8c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x8c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/4x8c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x8c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/4x8c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x16-minmax-fp32-neon-mlal-lane.c
  src/qs8-igemm/gen/4x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/4x16-minmax-rndnu-neon-mull-addw-dup.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld4r.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/4x16c2-minmax-rndnu-neon-mull-ld4r.c
  src/qs8-igemm/gen/4x16c2s4-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x16c2s4-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neon-mlal-dup.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neon-mlal-ld1r.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neon-mlal-ld2r.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neon-mull-dup.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neon-mull-ld1r.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neon-mull-ld2r.c
  src/qs8-igemm/gen/4x16c4s2-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x16c4s2-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/4x16c8-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/4x16c8-minmax-rndnu-neon-mull.c
  src/qs8-igemm/gen/4x16c16-minmax-rndnu-neon-mlal.c
  src/qs8-igemm/gen/6x8-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/6x8-minmax-rndnu-neon-mlal-lane.c
  src/qs8-igemm/gen/6x16-minmax-rndnu-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/6x16-minmax-rndnu-neon-mlal-lane.c
  src/qs8-requantization/fp32-neon.c
  src/qs8-requantization/gemmlowp-neon.c
  src/qs8-requantization/rndna-neon.c
  src/qs8-requantization/rndnu-neon-mull.c
  src/qs8-requantization/rndnu-neon-qdmulh.c
  src/qs8-vadd/gen/minmax-neon-ld64-x8.c
  src/qs8-vadd/gen/minmax-neon-ld64-x16.c
  src/qs8-vadd/gen/minmax-neon-ld64-x24.c
  src/qs8-vadd/gen/minmax-neon-ld64-x32.c
  src/qs8-vadd/gen/minmax-neon-ld128-x16.c
  src/qs8-vadd/gen/minmax-neon-ld128-x32.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x8.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x16.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x24.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x32.c
  src/qs8-vaddc/gen/minmax-neon-ld128-x16.c
  src/qs8-vaddc/gen/minmax-neon-ld128-x32.c
  src/qs8-vcvt/gen/vcvt-neon-x8.c
  src/qs8-vcvt/gen/vcvt-neon-x16.c
  src/qs8-vcvt/gen/vcvt-neon-x32.c
  src/qs8-vlrelu/gen/vlrelu-neon-x8.c
  src/qs8-vlrelu/gen/vlrelu-neon-x16.c
  src/qs8-vlrelu/gen/vlrelu-neon-x32.c
  src/qs8-vmul/gen/minmax-fp32-neon-ld64-x8.c
  src/qs8-vmul/gen/minmax-fp32-neon-ld64-x16.c
  src/qs8-vmul/gen/minmax-fp32-neon-ld128-x16.c
  src/qs8-vmul/gen/minmax-rndnu-neon-ld64-x8.c
  src/qs8-vmul/gen/minmax-rndnu-neon-ld64-x16.c
  src/qs8-vmul/gen/minmax-rndnu-neon-ld128-x16.c
  src/qs8-vmulc/gen/minmax-fp32-neon-ld64-x8.c
  src/qs8-vmulc/gen/minmax-fp32-neon-ld64-x16.c
  src/qs8-vmulc/gen/minmax-fp32-neon-ld128-x16.c
  src/qs8-vmulc/gen/minmax-rndnu-neon-ld64-x8.c
  src/qs8-vmulc/gen/minmax-rndnu-neon-ld64-x16.c
  src/qs8-vmulc/gen/minmax-rndnu-neon-ld128-x16.c
  src/qu8-avgpool/9p8x-minmax-neon-c8.c
  src/qu8-avgpool/9x-minmax-neon-c8.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up8x9-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up8x9-minmax-rndnu-neon-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up8x25-minmax-rndnu-neon-mul16.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up16x9-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up16x9-minmax-rndnu-neon-mul16.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up16x25-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up16x25-minmax-rndnu-neon-mul16.c
  src/qu8-dwconv/gen/up24x9-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up24x9-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up24x9-minmax-rndnu-neon-mul16.c
  src/qu8-dwconv/gen/up24x25-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up24x25-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up24x25-minmax-rndnu-neon-mul16.c
  src/qu8-dwconv/gen/up32x9-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up32x9-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up32x9-minmax-rndnu-neon-mul16.c
  src/qu8-dwconv/gen/up32x25-minmax-fp32-neon-mul16.c
  src/qu8-dwconv/gen/up32x25-minmax-rndnu-neon-mul8.c
  src/qu8-dwconv/gen/up32x25-minmax-rndnu-neon-mul16.c
  src/qu8-f32-vcvt/gen/vcvt-neon-x8.c
  src/qu8-f32-vcvt/gen/vcvt-neon-x16.c
  src/qu8-f32-vcvt/gen/vcvt-neon-x24.c
  src/qu8-f32-vcvt/gen/vcvt-neon-x32.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neon-c8.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neon-c16.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neon-c24.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neon-c32.c
  src/qu8-gavgpool/gen/7p7x-minmax-rndnu-neon-c8.c
  src/qu8-gavgpool/gen/7p7x-minmax-rndnu-neon-c16.c
  src/qu8-gavgpool/gen/7p7x-minmax-rndnu-neon-c24.c
  src/qu8-gavgpool/gen/7p7x-minmax-rndnu-neon-c32.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neon-c8.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neon-c16.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neon-c24.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neon-c32.c
  src/qu8-gavgpool/gen/7x-minmax-rndnu-neon-c8.c
  src/qu8-gavgpool/gen/7x-minmax-rndnu-neon-c16.c
  src/qu8-gavgpool/gen/7x-minmax-rndnu-neon-c24.c
  src/qu8-gavgpool/gen/7x-minmax-rndnu-neon-c32.c
  src/qu8-gemm/gen/1x8-minmax-fp32-neon-mlal-lane.c
  src/qu8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
  src/qu8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/2x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/2x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/3x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/4x8-minmax-fp32-neon-mlal-lane.c
  src/qu8-gemm/gen/4x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane.c
  src/qu8-gemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/6x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-gemm/gen/6x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/1x8-minmax-fp32-neon-mlal-lane.c
  src/qu8-igemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
  src/qu8-igemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/2x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/2x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/3x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/4x8-minmax-fp32-neon-mlal-lane.c
  src/qu8-igemm/gen/4x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/4x16-minmax-fp32-neon-mlal-lane.c
  src/qu8-igemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/6x8-minmax-rndnu-neon-mlal-lane.c
  src/qu8-igemm/gen/6x16-minmax-rndnu-neon-mlal-lane.c
  src/qu8-requantization/fp32-neon.c
  src/qu8-requantization/gemmlowp-neon.c
  src/qu8-requantization/rndna-neon.c
  src/qu8-vadd/gen/minmax-neon-ld64-x8.c
  src/qu8-vadd/gen/minmax-neon-ld64-x16.c
  src/qu8-vadd/gen/minmax-neon-ld64-x32.c
  src/qu8-vadd/gen/minmax-neon-ld128-x16.c
  src/qu8-vaddc/gen/minmax-neon-ld64-x8.c
  src/qu8-vaddc/gen/minmax-neon-ld64-x16.c
  src/qu8-vaddc/gen/minmax-neon-ld64-x32.c
  src/qu8-vaddc/gen/minmax-neon-ld128-x16.c
  src/qu8-vcvt/gen/vcvt-neon-x8.c
  src/qu8-vcvt/gen/vcvt-neon-x16.c
  src/qu8-vcvt/gen/vcvt-neon-x32.c
  src/qu8-vlrelu/gen/vlrelu-neon-x8.c
  src/qu8-vlrelu/gen/vlrelu-neon-x16.c
  src/qu8-vlrelu/gen/vlrelu-neon-x32.c
  src/qu8-vmul/gen/minmax-fp32-neon-ld64-x8.c
  src/qu8-vmul/gen/minmax-fp32-neon-ld64-x16.c
  src/qu8-vmul/gen/minmax-fp32-neon-ld128-x16.c
  src/qu8-vmul/gen/minmax-rndnu-neon-ld64-x8.c
  src/qu8-vmul/gen/minmax-rndnu-neon-ld64-x16.c
  src/qu8-vmul/gen/minmax-rndnu-neon-ld128-x16.c
  src/qu8-vmulc/gen/minmax-fp32-neon-ld64-x8.c
  src/qu8-vmulc/gen/minmax-fp32-neon-ld64-x16.c
  src/qu8-vmulc/gen/minmax-fp32-neon-ld128-x16.c
  src/qu8-vmulc/gen/minmax-rndnu-neon-ld64-x8.c
  src/qu8-vmulc/gen/minmax-rndnu-neon-ld64-x16.c
  src/qu8-vmulc/gen/minmax-rndnu-neon-ld128-x16.c
  src/s8-ibilinear/gen/neon-c8.c
  src/s8-ibilinear/gen/neon-c16.c
  src/s8-maxpool/2p2x-minmax-neon-c16.c
  src/s8-maxpool/4p3x-minmax-neon-c16.c
  src/s8-maxpool/9p8x-minmax-neon-c16.c
  src/s8-vclamp/neon-x64.c
  src/s16-rmaxabs/gen/neon-x8.c
  src/s16-rmaxabs/gen/neon-x16.c
  src/s16-rmaxabs/gen/neon-x24.c
  src/s16-rmaxabs/gen/neon-x32.c
  src/s16-vlshift/gen/neon-x8.c
  src/s16-vlshift/gen/neon-x16.c
  src/s16-vlshift/gen/neon-x24.c
  src/s16-vlshift/gen/neon-x32.c
  src/s16-window/gen/neon-x8.c
  src/s16-window/gen/neon-x16.c
  src/s16-window/gen/neon-x24.c
  src/s16-window/gen/neon-x32.c
  src/s16-window/gen/shift12-neon-x8.c
  src/s16-window/gen/shift12-neon-x16.c
  src/s16-window/gen/shift12-neon-x24.c
  src/s16-window/gen/shift12-neon-x32.c
  src/s16-window/gen/shift15-neon-x8.c
  src/s16-window/gen/shift15-neon-x16.c
  src/s16-window/gen/shift15-neon-x24.c
  src/s16-window/gen/shift15-neon-x32.c
  src/u8-ibilinear/gen/neon-c8.c
  src/u8-ibilinear/gen/neon-c16.c
  src/u8-maxpool/9p8x-minmax-neon-c16.c
  src/u8-rmax/neon.c
  src/u8-vclamp/neon-x64.c
  src/u32-filterbank-accumulate/gen/neon-x1.c
  src/u32-filterbank-accumulate/gen/neon-x2.c
  src/xx-fill/neon-x64.c
  src/xx-pad/neon.c
  src/x8-transposec/gen/8x8-multi-dec-zip-neon.c
  src/x8-transposec/gen/8x8-multi-mov-zip-neon.c
  src/x8-transposec/gen/8x8-multi-switch-zip-neon.c
  src/x8-transposec/gen/8x8-reuse-dec-zip-neon.c
  src/x8-transposec/gen/8x8-reuse-mov-zip-neon.c
  src/x8-transposec/gen/8x8-reuse-multi-zip-neon.c
  src/x8-transposec/gen/8x8-reuse-switch-zip-neon.c
  src/x8-transposec/gen/16x16-reuse-dec-zip-neon.c
  src/x8-transposec/gen/16x16-reuse-mov-zip-neon.c
  src/x8-transposec/gen/16x16-reuse-switch-zip-neon.c
  src/x8-zip/xm-neon.c
  src/x8-zip/x2-neon.c
  src/x8-zip/x3-neon.c
  src/x8-zip/x4-neon.c
  src/x16-transposec/gen/4x4-multi-dec-zip-neon.c
  src/x16-transposec/gen/4x4-multi-mov-zip-neon.c
  src/x16-transposec/gen/4x4-multi-multi-zip-neon.c
  src/x16-transposec/gen/4x4-multi-switch-zip-neon.c
  src/x16-transposec/gen/4x4-reuse-dec-zip-neon.c
  src/x16-transposec/gen/4x4-reuse-mov-zip-neon.c
  src/x16-transposec/gen/4x4-reuse-multi-zip-neon.c
  src/x16-transposec/gen/4x4-reuse-switch-zip-neon.c
  src/x16-transposec/gen/8x8-multi-dec-zip-neon.c
  src/x16-transposec/gen/8x8-multi-mov-zip-neon.c
  src/x16-transposec/gen/8x8-multi-switch-zip-neon.c
  src/x16-transposec/gen/8x8-reuse-dec-zip-neon.c
  src/x16-transposec/gen/8x8-reuse-mov-zip-neon.c
  src/x16-transposec/gen/8x8-reuse-multi-zip-neon.c
  src/x16-transposec/gen/8x8-reuse-switch-zip-neon.c
  src/x24-transposec/2x2-neon-tbl.c
  src/x32-packx/x4-neon-st4.c
  src/x32-transposec/gen/2x2-multi-dec-zip-neon.c
  src/x32-transposec/gen/2x2-multi-mov-zip-neon.c
  src/x32-transposec/gen/2x2-multi-multi-zip-neon.c
  src/x32-transposec/gen/2x2-multi-switch-zip-neon.c
  src/x32-transposec/gen/2x2-reuse-dec-zip-neon.c
  src/x32-transposec/gen/2x2-reuse-mov-zip-neon.c
  src/x32-transposec/gen/2x2-reuse-multi-zip-neon.c
  src/x32-transposec/gen/2x2-reuse-switch-zip-neon.c
  src/x32-transposec/gen/4x4-multi-dec-zip-neon.c
  src/x32-transposec/gen/4x4-multi-mov-zip-neon.c
  src/x32-transposec/gen/4x4-multi-multi-zip-neon.c
  src/x32-transposec/gen/4x4-multi-switch-zip-neon.c
  src/x32-transposec/gen/4x4-reuse-dec-zip-neon.c
  src/x32-transposec/gen/4x4-reuse-mov-zip-neon.c
  src/x32-transposec/gen/4x4-reuse-multi-zip-neon.c
  src/x32-transposec/gen/4x4-reuse-switch-zip-neon.c
  src/x32-unpool/neon.c
  src/x32-zip/xm-neon.c
  src/x32-zip/x2-neon.c
  src/x32-zip/x3-neon.c
  src/x32-zip/x4-neon.c)

SET(PROD_NEONFP16_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-neonfp16-x16.c
  src/f32-f16-vcvt/gen/vcvt-neonfp16-x16.c)

SET(ALL_NEONFP16_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-neonfp16-x8.c
  src/f16-f32-vcvt/gen/vcvt-neonfp16-x16.c
  src/f32-f16-vcvt/gen/vcvt-neonfp16-x8.c
  src/f32-f16-vcvt/gen/vcvt-neonfp16-x16.c
  src/math/cvt-f16-f32-neonfp16.c
  src/math/cvt-f32-f16-neonfp16.c)

SET(PROD_NEONFMA_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up8x3-minmax-neonfma.c
  src/f32-dwconv/gen/up8x4-minmax-neonfma.c
  src/f32-dwconv/gen/up8x9-minmax-neonfma.c
  src/f32-dwconv/gen/up8x25-minmax-neonfma-acc2.c
  src/f32-gemm/gen/1x8s4-minmax-neonfma.c
  src/f32-gemm/gen/6x8s4-minmax-neonfma.c
  src/f32-ibilinear-chw/gen/neonfma-p8.c
  src/f32-ibilinear/gen/neonfma-c8.c
  src/f32-igemm/gen/1x8s4-minmax-neonfma.c
  src/f32-igemm/gen/6x8s4-minmax-neonfma.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x16.c
  src/f32-spmm/gen/32x1-minmax-neonfma-pipelined.c
  src/f32-velu/gen/velu-neonfma-rr1-lut16-p3-x16.c
  src/f32-velu/gen/velu-neonfma-rr1-p6-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-neonfma-2x.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x16.c)

SET(ALL_NEONFMA_MICROKERNEL_SRCS
  src/bf16-gemm/gen/1x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/2x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/3x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/4x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/5x4c8-minmax-neonfma-shland.c
  src/f32-dwconv/gen/up4x3-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x3-minmax-neonfma.c
  src/f32-dwconv/gen/up4x4-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-neonfma.c
  src/f32-dwconv/gen/up4x9-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-neonfma.c
  src/f32-dwconv/gen/up4x25-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-neonfma.c
  src/f32-dwconv/gen/up8x3-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x3-minmax-neonfma.c
  src/f32-dwconv/gen/up8x4-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-neonfma.c
  src/f32-dwconv/gen/up8x9-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-neonfma.c
  src/f32-dwconv/gen/up8x25-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-neonfma.c
  src/f32-dwconv/gen/up16x3-minmax-neon-acc2.c
  src/f32-dwconv/gen/up16x3-minmax-neon.c
  src/f32-dwconv/gen/up16x3-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up16x3-minmax-neonfma.c
  src/f32-dwconv/gen/up16x4-minmax-neon-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-neon.c
  src/f32-dwconv/gen/up16x4-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-neonfma.c
  src/f32-dwconv/gen/up16x9-minmax-neon-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-neon.c
  src/f32-dwconv/gen/up16x9-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-neonfma.c
  src/f32-dwconv/gen/up16x25-minmax-neon-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-neon.c
  src/f32-dwconv/gen/up16x25-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-neonfma.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen-inc/1x8s4inc-minmax-neonfma.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen-inc/4x8s4inc-minmax-neonfma.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen-inc/6x8s4inc-minmax-neonfma.c
  src/f32-gemm/gen-inc/8x8s4inc-minmax-neonfma.c
  src/f32-gemm/gen/1x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/1x8s4-minmax-neonfma.c
  src/f32-gemm/gen/4x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/4x8-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen/4x8s4-minmax-neonfma.c
  src/f32-gemm/gen/6x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen/6x8s4-minmax-neonfma.c
  src/f32-gemm/gen/8x8s4-minmax-neonfma.c
  src/f32-ibilinear-chw/gen/neonfma-p4.c
  src/f32-ibilinear-chw/gen/neonfma-p8.c
  src/f32-ibilinear-chw/gen/neonfma-p16.c
  src/f32-ibilinear/gen/neonfma-c4.c
  src/f32-ibilinear/gen/neonfma-c8.c
  src/f32-igemm/gen/1x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/1x8s4-minmax-neonfma.c
  src/f32-igemm/gen/4x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/4x8-minmax-neonfma-dup-ld128.c
  src/f32-igemm/gen/4x8s4-minmax-neonfma.c
  src/f32-igemm/gen/6x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-dup-ld128.c
  src/f32-igemm/gen/6x8s4-minmax-neonfma.c
  src/f32-igemm/gen/8x8s4-minmax-neonfma.c
  src/f32-ppmm/gen/4x8-minmax-neonfma.c
  src/f32-ppmm/gen/8x8-minmax-neonfma.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x8.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x12.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x16.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-lut64-p2-x20.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neonfma-rr1-p5-x20.c
  src/f32-spmm/gen/4x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/4x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/4x1-minmax-neonfma.c
  src/f32-spmm/gen/8x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/8x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/8x1-minmax-neonfma.c
  src/f32-spmm/gen/12x1-minmax-neonfma.c
  src/f32-spmm/gen/16x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/16x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/16x1-minmax-neonfma.c
  src/f32-spmm/gen/32x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/32x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/32x1-minmax-neonfma.c
  src/f32-velu/gen/velu-neonfma-rr1-lut16-p3-x4.c
  src/f32-velu/gen/velu-neonfma-rr1-lut16-p3-x8.c
  src/f32-velu/gen/velu-neonfma-rr1-lut16-p3-x12.c
  src/f32-velu/gen/velu-neonfma-rr1-lut16-p3-x16.c
  src/f32-velu/gen/velu-neonfma-rr1-lut16-p3-x20.c
  src/f32-velu/gen/velu-neonfma-rr1-lut16-p3-x24.c
  src/f32-velu/gen/velu-neonfma-rr1-p6-x4.c
  src/f32-velu/gen/velu-neonfma-rr1-p6-x8.c
  src/f32-velu/gen/velu-neonfma-rr1-p6-x12.c
  src/f32-velu/gen/velu-neonfma-rr1-p6-x16.c
  src/f32-velu/gen/velu-neonfma-rr1-p6-x20.c
  src/f32-velu/gen/velu-neonfma-rr1-p6-x24.c
  src/f32-vmulcaddc/gen/c4-minmax-neonfma-2x.c
  src/f32-vmulcaddc/gen/c8-minmax-neonfma-2x.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x24.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x4.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x8.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x12.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x16.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x20.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x24.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x28.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x32.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x36.c
  src/f32-vsqrt/gen/neonfma-nr1rsqrts1fma1adj-x40.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x4.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x8.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x12.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x16.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x20.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x24.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x28.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x32.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x36.c
  src/f32-vsqrt/gen/neonfma-nr2fma1adj-x40.c
  src/math/exp-f32-neonfma-rr2-lut64-p2.c
  src/math/exp-f32-neonfma-rr2-p5.c
  src/math/expminus-f32-neonfma-rr2-lut64-p2.c
  src/math/expminus-f32-neonfma-rr2-lut2048-p1.c
  src/math/expminus-f32-neonfma-rr2-p5.c
  src/math/expm1minus-f32-neonfma-rr1-lut16-p3.c
  src/math/expm1minus-f32-neonfma-rr1-p6.c
  src/math/sigmoid-f32-neonfma-rr1-lut64-p2-nr1recps1fma.c
  src/math/sigmoid-f32-neonfma-rr1-lut64-p2-nr2fma.c
  src/math/sigmoid-f32-neonfma-rr1-lut64-p2-nr2recps.c
  src/math/sigmoid-f32-neonfma-rr1-lut2048-p1-nr1recps1fma.c
  src/math/sigmoid-f32-neonfma-rr1-lut2048-p1-nr2fma.c
  src/math/sigmoid-f32-neonfma-rr1-lut2048-p1-nr2recps.c
  src/math/sigmoid-f32-neonfma-rr1-p5-nr1recps1fma.c
  src/math/sigmoid-f32-neonfma-rr1-p5-nr2fma.c
  src/math/sigmoid-f32-neonfma-rr1-p5-nr2recps.c
  src/math/sigmoid-f32-neonfma-rr2-lut64-p2-nr1recps1fma.c
  src/math/sigmoid-f32-neonfma-rr2-lut64-p2-nr2fma.c
  src/math/sigmoid-f32-neonfma-rr2-lut64-p2-nr2recps.c
  src/math/sigmoid-f32-neonfma-rr2-lut2048-p1-nr1recps1fma.c
  src/math/sigmoid-f32-neonfma-rr2-lut2048-p1-nr2fma.c
  src/math/sigmoid-f32-neonfma-rr2-lut2048-p1-nr2recps.c
  src/math/sigmoid-f32-neonfma-rr2-p5-nr1recps1fma.c
  src/math/sigmoid-f32-neonfma-rr2-p5-nr2fma.c
  src/math/sigmoid-f32-neonfma-rr2-p5-nr2recps.c
  src/math/sqrt-neonfma-nr1fma.c
  src/math/sqrt-neonfma-nr1rsqrts1fma1adj.c
  src/math/sqrt-neonfma-nr2fma.c
  src/math/sqrt-neonfma-nr2fma1adj.c
  src/math/sqrt-neonfma-nr3fma.c)

SET(PROD_AARCH64_NEON_MICROKERNEL_SRCS
  src/f32-conv-hwc2chw/3x3s2p1c3x4-neonfma-2x2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-3x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-4x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc2.c
  src/f32-gemm/gen/1x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x2-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/1x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/6x2-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-lane-ld64.c
  src/f32-spmm/gen/32x2-minmax-neonfma.c
  src/f32-spmm/gen/32x4-minmax-neonfma.c
  src/f32-vbinary/gen/vdiv-minmax-neon-x8.c
  src/f32-vbinary/gen/vdivc-minmax-neon-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-neon-x8.c
  src/f32-vsqrt/gen/neon-sqrt-x4.c
  src/x8-lut/gen/lut-neon-tbx128x4-x64.c
  src/x32-transposec/4x4-aarch64-tbl.c)

SET(ALL_AARCH64_NEON_MICROKERNEL_SRCS
  src/bf16-gemm/gen/1x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/2x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/3x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/4x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/5x4c8-minmax-neonfma-zip.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x4-neonfma-2x1.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x4-neonfma-2x2.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x8-neonfma-2x1.c
  src/f32-conv-hwc/gen/3x3s2p0p1c3x8-neonfma-2x2.c
  src/f32-conv-hwc/gen/3x3s2p1c3x4-neonfma-2x1.c
  src/f32-conv-hwc/gen/3x3s2p1c3x4-neonfma-2x2.c
  src/f32-conv-hwc/gen/3x3s2p1c3x8-neonfma-2x1.c
  src/f32-conv-hwc/gen/3x3s2p1c3x8-neonfma-2x2.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-neonfma-2x2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-2x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-3x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-4x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-5x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-6x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-2x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-3x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-4x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc5.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-2x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-2x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-3x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-3x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-4x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-4x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-5x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc5.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-2x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-2x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-3x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-3x4.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-lane-ld128.c
  src/f32-gemm/gen-inc/5x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-lane-ld128.c
  src/f32-gemm/gen/1x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/4x2-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neonfma-lane-ld128.c
  src/f32-gemm/gen/5x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x2-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld128.c
  src/f32-igemm/gen/1x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x2-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x4-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neonfma-lane-ld128.c
  src/f32-igemm/gen/6x2-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-lane-ld128.c
  src/f32-spmm/gen/4x2-minmax-neonfma.c
  src/f32-spmm/gen/4x4-minmax-neonfma.c
  src/f32-spmm/gen/8x2-minmax-neonfma.c
  src/f32-spmm/gen/8x4-minmax-neonfma.c
  src/f32-spmm/gen/12x2-minmax-neonfma.c
  src/f32-spmm/gen/12x4-minmax-neonfma.c
  src/f32-spmm/gen/16x2-minmax-neonfma.c
  src/f32-spmm/gen/16x4-minmax-neonfma.c
  src/f32-spmm/gen/32x2-minmax-neonfma.c
  src/f32-spmm/gen/32x4-minmax-neonfma.c
  src/f32-vbinary/gen/vdiv-minmax-neon-x4.c
  src/f32-vbinary/gen/vdiv-minmax-neon-x8.c
  src/f32-vbinary/gen/vdivc-minmax-neon-x4.c
  src/f32-vbinary/gen/vdivc-minmax-neon-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-neon-x4.c
  src/f32-vbinary/gen/vrdivc-minmax-neon-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x24.c
  src/f32-vsqrt/gen/neon-sqrt-x4.c
  src/f32-vsqrt/gen/neon-sqrt-x8.c
  src/math/sigmoid-f32-neonfma-rr1-lut64-p2-div.c
  src/math/sigmoid-f32-neonfma-rr1-lut2048-p1-div.c
  src/math/sigmoid-f32-neonfma-rr1-p5-div.c
  src/math/sigmoid-f32-neonfma-rr2-lut64-p2-div.c
  src/math/sigmoid-f32-neonfma-rr2-lut2048-p1-div.c
  src/math/sigmoid-f32-neonfma-rr2-p5-div.c
  src/x8-lut/gen/lut-neon-tbx128x4-x16.c
  src/x8-lut/gen/lut-neon-tbx128x4-x32.c
  src/x8-lut/gen/lut-neon-tbx128x4-x48.c
  src/x8-lut/gen/lut-neon-tbx128x4-x64.c
  src/x24-transposec/4x4-aarch64-tbl.c
  src/x32-transposec/4x4-aarch64-tbl.c)

SET(PROD_NEONV8_MICROKERNEL_SRCS
  src/f32-qs8-vcvt/gen/vcvt-neonv8-x32.c
  src/f32-qu8-vcvt/gen/vcvt-neonv8-x32.c
  src/f32-vrnd/gen/vrndd-neonv8-x8.c
  src/f32-vrnd/gen/vrndne-neonv8-x8.c
  src/f32-vrnd/gen/vrndu-neonv8-x8.c
  src/f32-vrnd/gen/vrndz-neonv8-x8.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-neonv8-mla8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-gemm/gen/1x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/1x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/1x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/1x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c)

SET(ALL_NEONV8_MICROKERNEL_SRCS
  src/f32-qs8-vcvt/gen/vcvt-neonv8-x8.c
  src/f32-qs8-vcvt/gen/vcvt-neonv8-x16.c
  src/f32-qs8-vcvt/gen/vcvt-neonv8-x24.c
  src/f32-qs8-vcvt/gen/vcvt-neonv8-x32.c
  src/f32-qu8-vcvt/gen/vcvt-neonv8-x8.c
  src/f32-qu8-vcvt/gen/vcvt-neonv8-x16.c
  src/f32-qu8-vcvt/gen/vcvt-neonv8-x24.c
  src/f32-qu8-vcvt/gen/vcvt-neonv8-x32.c
  src/f32-vrnd/gen/vrndd-neonv8-x4.c
  src/f32-vrnd/gen/vrndd-neonv8-x8.c
  src/f32-vrnd/gen/vrndne-neonv8-x4.c
  src/f32-vrnd/gen/vrndne-neonv8-x8.c
  src/f32-vrnd/gen/vrndu-neonv8-x4.c
  src/f32-vrnd/gen/vrndu-neonv8-x8.c
  src/f32-vrnd/gen/vrndz-neonv8-x4.c
  src/f32-vrnd/gen/vrndz-neonv8-x8.c
  src/math/cvt-f32-qs8-neonv8.c
  src/math/cvt-f32-qu8-neonv8.c
  src/math/roundd-neonv8.c
  src/math/roundne-neonv8.c
  src/math/roundu-neonv8.c
  src/math/roundz-neonv8.c
  src/qc8-dwconv/gen/up8x3-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul8-ld64.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul8-ld64.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul16.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-neonv8-mla8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mla8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul8-ld64.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul8-ld128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mla8-ld64.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mla8-ld128.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul8-ld64.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul8-ld128.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-neonv8-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-neonv8-mul16.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-neonv8-mul16.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-neonv8-mul16.c
  src/qc8-gemm/gen/1x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/1x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qc8-gemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-gemm/gen/1x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/2x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/2x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qc8-gemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-gemm/gen/2x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-gemm/gen/2x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/2x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/3x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/3x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/3x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/3x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/4x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/4x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/6x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/6x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-gemm/gen/6x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-gemm/gen/6x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/1x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/1x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qc8-igemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-igemm/gen/1x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/1x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/2x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/2x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qc8-igemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/2x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qc8-igemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qc8-igemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qc8-igemm/gen/2x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c
  src/qc8-igemm/gen/2x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/2x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/3x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/3x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/3x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/3x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/4x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/4x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/4x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/6x8-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/6x8-minmax-fp32-neonv8-mlal-lane.c
  src/qc8-igemm/gen/6x16-minmax-fp32-neonv8-mlal-lane-prfm.c
  src/qc8-igemm/gen/6x16-minmax-fp32-neonv8-mlal-lane.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-neonv8-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-neonv8-mul16.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-neonv8-mul16.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-neonv8-mul16.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c8.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c16.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c24.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c32.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neonv8-c8.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neonv8-c16.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neonv8-c24.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-neonv8-c32.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qs8-gemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-gemm/gen/1x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qs8-gemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c
  src/qs8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qs8-gemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qs8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-gemm/gen/2x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qs8-gemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c
  src/qs8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qs8-igemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-igemm/gen/1x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qs8-igemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c
  src/qs8-igemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld4r.c
  src/qs8-igemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c
  src/qs8-igemm/gen/2x8c4-minmax-fp32-neonv8-mlal-dup.c
  src/qs8-igemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld1r.c
  src/qs8-igemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld2r.c
  src/qs8-igemm/gen/2x8c4s2-minmax-fp32-neonv8-mlal.c
  src/qs8-igemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c
  src/qs8-igemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c
  src/qs8-vmul/gen/minmax-fp32-neonv8-ld64-x8.c
  src/qs8-vmul/gen/minmax-fp32-neonv8-ld64-x16.c
  src/qs8-vmul/gen/minmax-fp32-neonv8-ld128-x16.c
  src/qs8-vmulc/gen/minmax-fp32-neonv8-ld64-x8.c
  src/qs8-vmulc/gen/minmax-fp32-neonv8-ld64-x16.c
  src/qs8-vmulc/gen/minmax-fp32-neonv8-ld128-x16.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul16.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul16.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul16.c
  src/qu8-dwconv/gen/up24x9-minmax-fp32-neonv8-mul16.c
  src/qu8-dwconv/gen/up24x25-minmax-fp32-neonv8-mul16.c
  src/qu8-dwconv/gen/up32x9-minmax-fp32-neonv8-mul16.c
  src/qu8-dwconv/gen/up32x25-minmax-fp32-neonv8-mul16.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c8.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c16.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c24.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-neonv8-c32.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neonv8-c8.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neonv8-c16.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neonv8-c24.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-neonv8-c32.c
  src/qu8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qu8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c
  src/qu8-igemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
  src/qu8-igemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c
  src/qu8-vmul/gen/minmax-fp32-neonv8-ld64-x8.c
  src/qu8-vmul/gen/minmax-fp32-neonv8-ld64-x16.c
  src/qu8-vmul/gen/minmax-fp32-neonv8-ld128-x16.c
  src/qu8-vmulc/gen/minmax-fp32-neonv8-ld64-x8.c
  src/qu8-vmulc/gen/minmax-fp32-neonv8-ld64-x16.c
  src/qu8-vmulc/gen/minmax-fp32-neonv8-ld128-x16.c)

SET(PROD_NEONFP16ARITH_MICROKERNEL_SRCS
  src/f16-avgpool/9p8x-minmax-neonfp16arith-c8.c
  src/f16-avgpool/9x-minmax-neonfp16arith-c8.c
  src/f16-dwconv/gen/up8x25-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x3-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x4-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x9-minmax-neonfp16arith.c
  src/f16-gavgpool-cw/neonfp16arith-x4.c
  src/f16-gavgpool/gen/7p7x-minmax-neonfp16arith-c8.c
  src/f16-gavgpool/gen/7x-minmax-neonfp16arith-c8.c
  src/f16-gemm/gen/1x16-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/6x16-minmax-neonfp16arith-ld64.c
  src/f16-ibilinear-chw/gen/neonfp16arith-p8.c
  src/f16-ibilinear/gen/neonfp16arith-c8.c
  src/f16-igemm/gen/1x16-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/6x16-minmax-neonfp16arith-ld64.c
  src/f16-maxpool/9p8x-minmax-neonfp16arith-c8.c
  src/f16-pavgpool/9p8x-minmax-neonfp16arith-c8.c
  src/f16-pavgpool/9x-minmax-neonfp16arith-c8.c
  src/f16-prelu/gen/neonfp16arith-2x16.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x40.c
  src/f16-rmax/neonfp16arith.c
  src/f16-spmm/gen/32x1-minmax-neonfp16arith.c
  src/f16-vbinary/gen/vadd-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vaddc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmaxc-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmin-neonfp16arith-x16.c
  src/f16-vbinary/gen/vminc-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmul-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmulc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vrsubc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsqrdiff-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsqrdiffc-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsub-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsubc-minmax-neonfp16arith-x16.c
  src/f16-vclamp/gen/vclamp-neonfp16arith-x16.c
  src/f16-velu/gen/velu-neonfp16arith-rr1-p3-x16.c
  src/f16-vhswish/gen/vhswish-neonfp16arith-x16.c
  src/f16-vlrelu/gen/vlrelu-neonfp16arith-x16.c
  src/f16-vmulcaddc/gen/c8-minmax-neonfp16arith-2x.c
  src/f16-vrnd/gen/vrndd-neonfp16arith-x16.c
  src/f16-vrnd/gen/vrndne-neonfp16arith-x16.c
  src/f16-vrnd/gen/vrndu-neonfp16arith-x16.c
  src/f16-vrnd/gen/vrndz-neonfp16arith-x16.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x40.c
  src/f16-vunary/gen/vabs-neonfp16arith-x16.c
  src/f16-vunary/gen/vneg-neonfp16arith-x16.c
  src/f16-vunary/gen/vsqr-neonfp16arith-x16.c)

SET(ALL_NEONFP16ARITH_MICROKERNEL_SRCS
  src/f16-avgpool/9p8x-minmax-neonfp16arith-c8.c
  src/f16-avgpool/9x-minmax-neonfp16arith-c8.c
  src/f16-dwconv/gen/up8x3-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up8x3-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up8x4-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up8x4-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up8x9-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up8x9-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up8x25-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up8x25-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x3-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x3-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x4-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x4-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x9-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x9-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x25-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x25-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up32x3-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up32x3-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up32x4-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up32x4-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up32x9-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up32x9-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up32x25-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up32x25-minmax-neonfp16arith.c
  src/f16-gavgpool-cw/neonfp16arith-x4.c
  src/f16-gavgpool-cw/neonfp16arith-x8.c
  src/f16-gavgpool/gen/7p7x-minmax-neonfp16arith-c8.c
  src/f16-gavgpool/gen/7p7x-minmax-neonfp16arith-c16.c
  src/f16-gavgpool/gen/7p7x-minmax-neonfp16arith-c24.c
  src/f16-gavgpool/gen/7p7x-minmax-neonfp16arith-c32.c
  src/f16-gavgpool/gen/7x-minmax-neonfp16arith-c8.c
  src/f16-gavgpool/gen/7x-minmax-neonfp16arith-c16.c
  src/f16-gavgpool/gen/7x-minmax-neonfp16arith-c24.c
  src/f16-gavgpool/gen/7x-minmax-neonfp16arith-c32.c
  src/f16-gemm/gen-inc/1x8inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/1x16inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/4x8inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/4x16inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/6x8inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/6x16inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/8x8inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/8x16inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/1x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/1x16-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/4x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/4x16-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/6x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/6x16-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/8x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/8x16-minmax-neonfp16arith-ld64.c
  src/f16-ibilinear-chw/gen/neonfp16arith-p4.c
  src/f16-ibilinear-chw/gen/neonfp16arith-p8.c
  src/f16-ibilinear-chw/gen/neonfp16arith-p16.c
  src/f16-ibilinear/gen/neonfp16arith-c8.c
  src/f16-ibilinear/gen/neonfp16arith-c16.c
  src/f16-igemm/gen/1x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/1x16-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/4x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/4x16-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/6x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/6x16-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/8x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/8x16-minmax-neonfp16arith-ld64.c
  src/f16-maxpool/9p8x-minmax-neonfp16arith-c8.c
  src/f16-pavgpool/9p8x-minmax-neonfp16arith-c8.c
  src/f16-pavgpool/9x-minmax-neonfp16arith-c8.c
  src/f16-prelu/gen/neonfp16arith-2x8.c
  src/f16-prelu/gen/neonfp16arith-2x16.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x32-acc2.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x32-acc4.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x32.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x40-acc2.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x40-acc5.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x40.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x48-acc2.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x48-acc3.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x48.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x64-acc2.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x64-acc4.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x64.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x72-acc3.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x72.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x80-acc2.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x80-acc5.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x80.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x96-acc2.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x96-acc3.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x96-acc6.c
  src/f16-raddstoreexpminusmax/gen/neonfp16arith-rr2-p2-x96.c
  src/f16-rmax/neonfp16arith.c
  src/f16-spmm/gen/8x1-minmax-neonfp16arith-x2.c
  src/f16-spmm/gen/8x1-minmax-neonfp16arith.c
  src/f16-spmm/gen/16x1-minmax-neonfp16arith-x2.c
  src/f16-spmm/gen/16x1-minmax-neonfp16arith.c
  src/f16-spmm/gen/24x1-minmax-neonfp16arith-x2.c
  src/f16-spmm/gen/24x1-minmax-neonfp16arith.c
  src/f16-spmm/gen/32x1-minmax-neonfp16arith-x2.c
  src/f16-spmm/gen/32x1-minmax-neonfp16arith.c
  src/f16-vbinary/gen/vadd-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vadd-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vaddc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vaddc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmaxc-neonfp16arith-x8.c
  src/f16-vbinary/gen/vmaxc-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmin-neonfp16arith-x8.c
  src/f16-vbinary/gen/vmin-neonfp16arith-x16.c
  src/f16-vbinary/gen/vminc-neonfp16arith-x8.c
  src/f16-vbinary/gen/vminc-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmul-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vmul-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vmulc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vmulc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vrsubc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vrsubc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsqrdiff-neonfp16arith-x8.c
  src/f16-vbinary/gen/vsqrdiff-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsqrdiffc-neonfp16arith-x8.c
  src/f16-vbinary/gen/vsqrdiffc-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsub-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vsub-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsubc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vsubc-minmax-neonfp16arith-x16.c
  src/f16-vclamp/gen/vclamp-neonfp16arith-x8.c
  src/f16-vclamp/gen/vclamp-neonfp16arith-x16.c
  src/f16-velu/gen/velu-neonfp16arith-rr1-p3-x8.c
  src/f16-velu/gen/velu-neonfp16arith-rr1-p3-x16.c
  src/f16-vhswish/gen/vhswish-neonfp16arith-x8.c
  src/f16-vhswish/gen/vhswish-neonfp16arith-x16.c
  src/f16-vlrelu/gen/vlrelu-neonfp16arith-x8.c
  src/f16-vlrelu/gen/vlrelu-neonfp16arith-x16.c
  src/f16-vmulcaddc/gen/c8-minmax-neonfp16arith-2x.c
  src/f16-vmulcaddc/gen/c16-minmax-neonfp16arith-2x.c
  src/f16-vrnd/gen/vrndd-neonfp16arith-x8.c
  src/f16-vrnd/gen/vrndd-neonfp16arith-x16.c
  src/f16-vrnd/gen/vrndne-neonfp16arith-x8.c
  src/f16-vrnd/gen/vrndne-neonfp16arith-x16.c
  src/f16-vrnd/gen/vrndu-neonfp16arith-x8.c
  src/f16-vrnd/gen/vrndu-neonfp16arith-x16.c
  src/f16-vrnd/gen/vrndz-neonfp16arith-x8.c
  src/f16-vrnd/gen/vrndz-neonfp16arith-x16.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x8.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x16.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x24.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x32.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x40.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x48.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x56.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1fma-x64.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x8.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x16.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x24.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x32.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x40.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x48.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x56.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-nr1recps-x64.c
  src/f16-vunary/gen/vabs-neonfp16arith-x8.c
  src/f16-vunary/gen/vabs-neonfp16arith-x16.c
  src/f16-vunary/gen/vneg-neonfp16arith-x8.c
  src/f16-vunary/gen/vneg-neonfp16arith-x16.c
  src/f16-vunary/gen/vsqr-neonfp16arith-x8.c
  src/f16-vunary/gen/vsqr-neonfp16arith-x16.c
  src/math/exp-f16-neonfp16arith-rr2-p3.c
  src/math/expminus-f16-neonfp16arith-rr1-p2.c
  src/math/expminus-f16-neonfp16arith-rr1-p3.c
  src/math/expminus-f16-neonfp16arith-rr2-p2.c
  src/math/expminus-f16-neonfp16arith-rr2-p3.c
  src/math/expm1minus-f16-neonfp16arith-rr1-p3.c
  src/math/expm1minus-f16-neonfp16arith-rr2-p3.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p2-nr1fma.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p2-nr1recps.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p2-recpe.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p3-nr1fma.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p3-nr1recps.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p3-recpe.c)

SET(PROD_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS
  src/f16-conv-hwc2chw/3x3s2p1c3x4-neonfp16arith-2x2.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-2x8.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-1x4.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-1x4.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-1x4.c
  src/f16-vbinary/gen/vdiv-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vdivc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vrdivc-minmax-neonfp16arith-x8.c
  src/f16-vsqrt/gen/neonfp16arith-sqrt-x8.c)

SET(ALL_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS
  src/f16-conv-hwc2chw/3x3s2p1c3x4-neonfp16arith-2x2.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-1x8-acc2.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-1x8-acc3.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-1x8-acc4.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-1x8.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-2x8-acc2.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-2x8.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-3x8.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-4x8.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-5x8.c
  src/f16-dwconv2d-chw/gen/3x3p1-minmax-neonfp16arith-6x8.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-1x4-acc2.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-1x4-acc3.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-1x4-acc4.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-1x4.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-2x4-acc2.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-2x4.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-3x4.c
  src/f16-dwconv2d-chw/gen/3x3s2p1-minmax-neonfp16arith-4x4.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-1x4-acc2.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-1x4-acc3.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-1x4-acc4.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-1x4-acc5.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-1x4.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-2x4-acc2.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-2x4-acc3.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-2x4.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-3x4-acc2.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-3x4.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-4x4-acc2.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-4x4.c
  src/f16-dwconv2d-chw/gen/5x5p2-minmax-neonfp16arith-5x4.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-1x4-acc2.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-1x4-acc3.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-1x4-acc4.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-1x4-acc5.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-1x4.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-2x4-acc2.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-2x4-acc3.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-2x4.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-3x4-acc2.c
  src/f16-dwconv2d-chw/gen/5x5s2p2-minmax-neonfp16arith-3x4.c
  src/f16-vbinary/gen/vdiv-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vdiv-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vdivc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vdivc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vrdivc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vrdivc-minmax-neonfp16arith-x16.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x8.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x16.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x24.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x32.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x40.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x48.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x56.c
  src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr2-p2-div-x64.c
  src/f16-vsqrt/gen/neonfp16arith-sqrt-x8.c
  src/f16-vsqrt/gen/neonfp16arith-sqrt-x16.c
  src/math/sigmoid-f16-neonfp16arith-rr1-p2-div.c
  src/math/sigmoid-f16-neonfp16arith-rr1-p3-div.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p2-div.c
  src/math/sigmoid-f16-neonfp16arith-rr2-p3-div.c)

SET(ALL_NEONBF16_MICROKERNEL_SRCS
  src/bf16-gemm/gen/1x4c8-minmax-neonbf16-bfdot.c
  src/bf16-gemm/gen/1x4c8-minmax-neonbf16-bfmlal.c
  src/bf16-gemm/gen/1x8c2-minmax-neonbf16-bfdot-lane-ld128.c
  src/bf16-gemm/gen/2x4c8-minmax-neonbf16-bfdot.c
  src/bf16-gemm/gen/2x4c8-minmax-neonbf16-bfmlal.c
  src/bf16-gemm/gen/3x4c8-minmax-neonbf16-bfdot.c
  src/bf16-gemm/gen/3x4c8-minmax-neonbf16-bfmlal.c
  src/bf16-gemm/gen/4x4c8-minmax-neonbf16-bfdot.c
  src/bf16-gemm/gen/4x4c8-minmax-neonbf16-bfmlal.c
  src/bf16-gemm/gen/4x8c2-minmax-neonbf16-bfdot-lane-ld128.c
  src/bf16-gemm/gen/5x4c8-minmax-neonbf16-bfdot.c
  src/bf16-gemm/gen/5x4c8-minmax-neonbf16-bfmlal.c
  src/bf16-gemm/gen/5x8c2-minmax-neonbf16-bfdot-lane-ld128.c
  src/bf16-gemm/gen/6x8c2-minmax-neonbf16-bfdot-lane-ld128.c)

SET(PROD_NEONDOT_MICROKERNEL_SRCS
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/4x8c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/4x8c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/4x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/4x16c4-minmax-rndnu-neondot.c)

SET(ALL_NEONDOT_MICROKERNEL_SRCS
  src/qc8-gemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/4x8c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/6x8c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/6x16c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/8x8c4-minmax-fp32-neondot.c
  src/qc8-gemm/gen/8x16c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/4x8c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/6x8c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/6x16c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/8x8c4-minmax-fp32-neondot.c
  src/qc8-igemm/gen/8x16c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/6x8c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/6x16c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/8x8c4-minmax-rndnu-neondot.c
  src/qs8-gemm/gen/8x16c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/6x8c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/6x16c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/8x8c4-minmax-rndnu-neondot.c
  src/qs8-igemm/gen/8x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qu8-gemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/1x32c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/2x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/2x16c4-minmax-fp32-neondot.c
  src/qu8-gemm/gen/2x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/2x32c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/3x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/3x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/3x32c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qu8-gemm/gen/4x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/5x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/5x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/6x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/6x16c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/8x8c4-minmax-rndnu-neondot.c
  src/qu8-gemm/gen/8x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/1x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qu8-igemm/gen/1x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/1x32c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/2x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/2x16c4-minmax-fp32-neondot.c
  src/qu8-igemm/gen/2x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/2x32c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/3x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/3x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/3x32c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/4x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qu8-igemm/gen/4x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/5x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/5x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/6x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/6x16c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/8x8c4-minmax-rndnu-neondot.c
  src/qu8-igemm/gen/8x16c4-minmax-rndnu-neondot.c)

SET(PROD_SSE_MICROKERNEL_SRCS
  src/f32-avgpool/9p8x-minmax-sse-c4.c
  src/f32-avgpool/9x-minmax-sse-c4.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-sse-2x2.c
  src/f32-dwconv/gen/up8x3-minmax-sse.c
  src/f32-dwconv/gen/up8x4-minmax-sse.c
  src/f32-dwconv/gen/up8x9-minmax-sse.c
  src/f32-dwconv/gen/up8x25-minmax-sse.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-4x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-2x4.c
  src/f32-gavgpool-cw/sse-x4.c
  src/f32-gavgpool/7p7x-minmax-sse-c4.c
  src/f32-gavgpool/7x-minmax-sse-c4.c
  src/f32-gemm/gen/1x8-minmax-sse-load1.c
  src/f32-gemm/gen/4x2c4-minmax-sse.c
  src/f32-gemm/gen/4x8-minmax-sse-load1.c
  src/f32-ibilinear-chw/gen/sse-p8.c
  src/f32-ibilinear/gen/sse-c8.c
  src/f32-igemm/gen/1x8-minmax-sse-load1.c
  src/f32-igemm/gen/4x2c4-minmax-sse.c
  src/f32-igemm/gen/4x8-minmax-sse-load1.c
  src/f32-maxpool/9p8x-minmax-sse-c4.c
  src/f32-pavgpool/9p8x-minmax-sse-c4.c
  src/f32-pavgpool/9x-minmax-sse-c4.c
  src/f32-rmax/sse.c
  src/f32-spmm/gen/32x1-minmax-sse.c
  src/f32-vbinary/gen/vadd-minmax-sse-x8.c
  src/f32-vbinary/gen/vaddc-minmax-sse-x8.c
  src/f32-vbinary/gen/vdiv-minmax-sse-x8.c
  src/f32-vbinary/gen/vdivc-minmax-sse-x8.c
  src/f32-vbinary/gen/vmax-sse-x8.c
  src/f32-vbinary/gen/vmaxc-sse-x8.c
  src/f32-vbinary/gen/vmin-sse-x8.c
  src/f32-vbinary/gen/vminc-sse-x8.c
  src/f32-vbinary/gen/vmul-minmax-sse-x8.c
  src/f32-vbinary/gen/vmulc-minmax-sse-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-sse-x8.c
  src/f32-vbinary/gen/vrsubc-minmax-sse-x8.c
  src/f32-vbinary/gen/vsqrdiff-sse-x8.c
  src/f32-vbinary/gen/vsqrdiffc-sse-x8.c
  src/f32-vbinary/gen/vsub-minmax-sse-x8.c
  src/f32-vbinary/gen/vsubc-minmax-sse-x8.c
  src/f32-vclamp/gen/vclamp-sse-x8.c
  src/f32-vhswish/gen/vhswish-sse-x8.c
  src/f32-vlrelu/gen/vlrelu-sse-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-sse-2x.c
  src/f32-vsqrt/gen/sse-sqrt-x4.c
  src/f32-vunary/gen/vabs-sse-x8.c
  src/f32-vunary/gen/vneg-sse-x8.c
  src/f32-vunary/gen/vsqr-sse-x8.c
  src/x32-packx/x4-sse.c
  src/x32-transposec/4x4-sse.c)

SET(ALL_SSE_MICROKERNEL_SRCS
  src/f32-avgpool/9p8x-minmax-sse-c4.c
  src/f32-avgpool/9x-minmax-sse-c4.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-sse-1x1.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-sse-2x2.c
  src/f32-dwconv/gen/up4x3-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x3-minmax-sse.c
  src/f32-dwconv/gen/up4x4-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-sse.c
  src/f32-dwconv/gen/up4x9-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-sse.c
  src/f32-dwconv/gen/up4x25-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-sse.c
  src/f32-dwconv/gen/up8x3-minmax-sse-acc2.c
  src/f32-dwconv/gen/up8x3-minmax-sse.c
  src/f32-dwconv/gen/up8x4-minmax-sse-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-sse.c
  src/f32-dwconv/gen/up8x9-minmax-sse-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-sse.c
  src/f32-dwconv/gen/up8x25-minmax-sse-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-sse.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-2x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-3x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-4x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-5x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-6x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-2x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-3x4.c
  src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-4x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc5.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-2x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-2x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-3x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-3x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-4x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-4x4.c
  src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-5x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc5.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-2x4-acc3.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-2x4.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-3x4-acc2.c
  src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-3x4.c
  src/f32-gavgpool-cw/sse-x4.c
  src/f32-gavgpool/7p7x-minmax-sse-c4.c
  src/f32-gavgpool/7x-minmax-sse-c4.c
  src/f32-gemm/gen-inc/1x8inc-minmax-sse-dup.c
  src/f32-gemm/gen-inc/1x8inc-minmax-sse-load1.c
  src/f32-gemm/gen-inc/1x8s4inc-minmax-sse.c
  src/f32-gemm/gen-inc/3x8inc-minmax-sse-dup.c
  src/f32-gemm/gen-inc/3x8inc-minmax-sse-load1.c
  src/f32-gemm/gen-inc/3x8s4inc-minmax-sse.c
  src/f32-gemm/gen-inc/4x8inc-minmax-sse-dup.c
  src/f32-gemm/gen-inc/4x8inc-minmax-sse-load1.c
  src/f32-gemm/gen-inc/4x8s4inc-minmax-sse.c
  src/f32-gemm/gen-inc/5x8inc-minmax-sse-dup.c
  src/f32-gemm/gen-inc/5x8inc-minmax-sse-load1.c
  src/f32-gemm/gen-inc/5x8s4inc-minmax-sse.c
  src/f32-gemm/gen/1x8-minmax-sse-dup.c
  src/f32-gemm/gen/1x8-minmax-sse-load1.c
  src/f32-gemm/gen/1x8s4-minmax-sse.c
  src/f32-gemm/gen/3x8-minmax-sse-dup.c
  src/f32-gemm/gen/3x8-minmax-sse-load1.c
  src/f32-gemm/gen/3x8s4-minmax-sse.c
  src/f32-gemm/gen/4x2c4-minmax-sse.c
  src/f32-gemm/gen/4x8-minmax-sse-dup.c
  src/f32-gemm/gen/4x8-minmax-sse-load1.c
  src/f32-gemm/gen/4x8s4-minmax-sse.c
  src/f32-gemm/gen/5x8-minmax-sse-dup.c
  src/f32-gemm/gen/5x8-minmax-sse-load1.c
  src/f32-gemm/gen/5x8s4-minmax-sse.c
  src/f32-ibilinear-chw/gen/sse-p4.c
  src/f32-ibilinear-chw/gen/sse-p8.c
  src/f32-ibilinear/gen/sse-c4.c
  src/f32-ibilinear/gen/sse-c8.c
  src/f32-igemm/gen/1x8-minmax-sse-dup.c
  src/f32-igemm/gen/1x8-minmax-sse-load1.c
  src/f32-igemm/gen/1x8s4-minmax-sse.c
  src/f32-igemm/gen/3x8-minmax-sse-dup.c
  src/f32-igemm/gen/3x8-minmax-sse-load1.c
  src/f32-igemm/gen/3x8s4-minmax-sse.c
  src/f32-igemm/gen/4x2c4-minmax-sse.c
  src/f32-igemm/gen/4x8-minmax-sse-dup.c
  src/f32-igemm/gen/4x8-minmax-sse-load1.c
  src/f32-igemm/gen/4x8s4-minmax-sse.c
  src/f32-igemm/gen/5x8-minmax-sse-dup.c
  src/f32-igemm/gen/5x8-minmax-sse-load1.c
  src/f32-igemm/gen/5x8s4-minmax-sse.c
  src/f32-maxpool/9p8x-minmax-sse-c4.c
  src/f32-pavgpool/9p8x-minmax-sse-c4.c
  src/f32-pavgpool/9x-minmax-sse-c4.c
  src/f32-ppmm/gen/4x8-minmax-sse.c
  src/f32-prelu/gen/sse-2x4.c
  src/f32-prelu/gen/sse-2x8.c
  src/f32-rmax/sse.c
  src/f32-spmm/gen/4x1-minmax-sse.c
  src/f32-spmm/gen/8x1-minmax-sse.c
  src/f32-spmm/gen/16x1-minmax-sse.c
  src/f32-spmm/gen/32x1-minmax-sse.c
  src/f32-vbinary/gen/vadd-minmax-sse-x4.c
  src/f32-vbinary/gen/vadd-minmax-sse-x8.c
  src/f32-vbinary/gen/vaddc-minmax-sse-x4.c
  src/f32-vbinary/gen/vaddc-minmax-sse-x8.c
  src/f32-vbinary/gen/vdiv-minmax-sse-x4.c
  src/f32-vbinary/gen/vdiv-minmax-sse-x8.c
  src/f32-vbinary/gen/vdivc-minmax-sse-x4.c
  src/f32-vbinary/gen/vdivc-minmax-sse-x8.c
  src/f32-vbinary/gen/vmax-sse-x4.c
  src/f32-vbinary/gen/vmax-sse-x8.c
  src/f32-vbinary/gen/vmaxc-sse-x4.c
  src/f32-vbinary/gen/vmaxc-sse-x8.c
  src/f32-vbinary/gen/vmin-sse-x4.c
  src/f32-vbinary/gen/vmin-sse-x8.c
  src/f32-vbinary/gen/vminc-sse-x4.c
  src/f32-vbinary/gen/vminc-sse-x8.c
  src/f32-vbinary/gen/vmul-minmax-sse-x4.c
  src/f32-vbinary/gen/vmul-minmax-sse-x8.c
  src/f32-vbinary/gen/vmulc-minmax-sse-x4.c
  src/f32-vbinary/gen/vmulc-minmax-sse-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-sse-x4.c
  src/f32-vbinary/gen/vrdivc-minmax-sse-x8.c
  src/f32-vbinary/gen/vrsubc-minmax-sse-x4.c
  src/f32-vbinary/gen/vrsubc-minmax-sse-x8.c
  src/f32-vbinary/gen/vsqrdiff-sse-x4.c
  src/f32-vbinary/gen/vsqrdiff-sse-x8.c
  src/f32-vbinary/gen/vsqrdiffc-sse-x4.c
  src/f32-vbinary/gen/vsqrdiffc-sse-x8.c
  src/f32-vbinary/gen/vsub-minmax-sse-x4.c
  src/f32-vbinary/gen/vsub-minmax-sse-x8.c
  src/f32-vbinary/gen/vsubc-minmax-sse-x4.c
  src/f32-vbinary/gen/vsubc-minmax-sse-x8.c
  src/f32-vclamp/gen/vclamp-sse-x4.c
  src/f32-vclamp/gen/vclamp-sse-x8.c
  src/f32-vhswish/gen/vhswish-sse-x4.c
  src/f32-vhswish/gen/vhswish-sse-x8.c
  src/f32-vlrelu/gen/vlrelu-sse-x4.c
  src/f32-vlrelu/gen/vlrelu-sse-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-sse-2x.c
  src/f32-vmulcaddc/gen/c8-minmax-sse-2x.c
  src/f32-vrelu/gen/vrelu-sse-x4.c
  src/f32-vrelu/gen/vrelu-sse-x8.c
  src/f32-vsqrt/gen/sse-sqrt-x4.c
  src/f32-vsqrt/gen/sse-sqrt-x8.c
  src/f32-vunary/gen/vabs-sse-x4.c
  src/f32-vunary/gen/vabs-sse-x8.c
  src/f32-vunary/gen/vneg-sse-x4.c
  src/f32-vunary/gen/vneg-sse-x8.c
  src/f32-vunary/gen/vsqr-sse-x4.c
  src/f32-vunary/gen/vsqr-sse-x8.c
  src/math/roundd-sse-addsub.c
  src/math/roundne-sse-addsub.c
  src/math/roundu-sse-addsub.c
  src/math/roundz-sse-addsub.c
  src/math/sqrt-sse-hh1mac.c
  src/math/sqrt-sse-nr1mac.c
  src/math/sqrt-sse-nr2mac.c
  src/x32-packx/x4-sse.c
  src/x32-transposec/4x4-sse.c)

SET(PROD_SSE2_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-sse2-int16-x32.c
  src/f16-vunary/gen/vabs-sse2-x16.c
  src/f16-vunary/gen/vneg-sse2-x16.c
  src/f32-argmaxpool/4x-sse2-c4.c
  src/f32-argmaxpool/9p8x-sse2-c4.c
  src/f32-argmaxpool/9x-sse2-c4.c
  src/f32-f16-vcvt/gen/vcvt-sse2-x16.c
  src/f32-prelu/gen/sse2-2x8.c
  src/f32-qs8-vcvt/gen/vcvt-sse2-x32.c
  src/f32-qu8-vcvt/gen/vcvt-sse2-x32.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x20-acc2.c
  src/f32-velu/gen/velu-sse2-rr2-lut16-p3-x12.c
  src/f32-vlrelu/gen/vlrelu-sse2-x8.c
  src/f32-vrnd/gen/vrndd-sse2-x8.c
  src/f32-vrnd/gen/vrndne-sse2-x8.c
  src/f32-vrnd/gen/vrndu-sse2-x8.c
  src/f32-vrnd/gen/vrndz-sse2-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-lut64-p2-div-x8.c
  src/qc8-dwconv/gen/up8x3-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16-add16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16-add16.c
  src/qs8-f32-vcvt/gen/vcvt-sse2-x32.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse2-c8.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse2-c8.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x8.c
  src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x8.c
  src/qs8-vcvt/gen/vcvt-sse2-x32.c
  src/qs8-vlrelu/gen/vlrelu-sse2-x32.c
  src/qs8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/qs8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/qu8-avgpool/9p8x-minmax-sse2-c8.c
  src/qu8-avgpool/9x-minmax-sse2-c8.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c
  src/qu8-f32-vcvt/gen/vcvt-sse2-x32.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse2-c8.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse2-c8.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-vadd/gen/minmax-sse2-mul16-ld64-x8.c
  src/qu8-vaddc/gen/minmax-sse2-mul16-ld64-x8.c
  src/qu8-vcvt/gen/vcvt-sse2-x32.c
  src/qu8-vlrelu/gen/vlrelu-sse2-x32.c
  src/qu8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/qu8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/s8-ibilinear/gen/sse2-c8.c
  src/s8-maxpool/9p8x-minmax-sse2-c16.c
  src/s8-vclamp/sse2-x64.c
  src/u8-ibilinear/gen/sse2-c8.c
  src/u8-maxpool/9p8x-minmax-sse2-c16.c
  src/u8-rmax/sse2.c
  src/u8-vclamp/sse2-x64.c
  src/xx-fill/sse2-x64.c
  src/xx-pad/sse2.c
  src/x8-transposec/gen/16x16-reuse-mov-sse2.c
  src/x8-zip/xm-sse2.c
  src/x8-zip/x2-sse2.c
  src/x8-zip/x3-sse2.c
  src/x8-zip/x4-sse2.c
  src/x16-transposec/gen/8x8-reuse-multi-sse2.c
  src/x32-unpool/sse2.c
  src/x32-zip/xm-sse2.c
  src/x32-zip/x2-sse2.c
  src/x32-zip/x3-sse2.c
  src/x32-zip/x4-sse2.c)

SET(ALL_SSE2_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-sse2-int16-x8.c
  src/f16-f32-vcvt/gen/vcvt-sse2-int16-x16.c
  src/f16-f32-vcvt/gen/vcvt-sse2-int16-x24.c
  src/f16-f32-vcvt/gen/vcvt-sse2-int16-x32.c
  src/f16-f32-vcvt/gen/vcvt-sse2-int32-x8.c
  src/f16-f32-vcvt/gen/vcvt-sse2-int32-x16.c
  src/f16-f32-vcvt/gen/vcvt-sse2-int32-x24.c
  src/f16-f32-vcvt/gen/vcvt-sse2-int32-x32.c
  src/f16-vunary/gen/vabs-sse2-x8.c
  src/f16-vunary/gen/vabs-sse2-x16.c
  src/f16-vunary/gen/vneg-sse2-x8.c
  src/f16-vunary/gen/vneg-sse2-x16.c
  src/f32-argmaxpool/4x-sse2-c4.c
  src/f32-argmaxpool/9p8x-sse2-c4.c
  src/f32-argmaxpool/9x-sse2-c4.c
  src/f32-f16-vcvt/gen/vcvt-sse2-x8.c
  src/f32-f16-vcvt/gen/vcvt-sse2-x16.c
  src/f32-f16-vcvt/gen/vcvt-sse2-x24.c
  src/f32-f16-vcvt/gen/vcvt-sse2-x32.c
  src/f32-gemm/gen-inc/1x8inc-minmax-sse2-dup.c
  src/f32-gemm/gen-inc/3x8inc-minmax-sse2-dup.c
  src/f32-gemm/gen-inc/4x8inc-minmax-sse2-dup.c
  src/f32-gemm/gen-inc/5x8inc-minmax-sse2-dup.c
  src/f32-gemm/gen/1x8-minmax-sse2-dup.c
  src/f32-gemm/gen/3x8-minmax-sse2-dup.c
  src/f32-gemm/gen/4x8-minmax-sse2-dup.c
  src/f32-gemm/gen/5x8-minmax-sse2-dup.c
  src/f32-igemm/gen/1x8-minmax-sse2-dup.c
  src/f32-igemm/gen/3x8-minmax-sse2-dup.c
  src/f32-igemm/gen/4x8-minmax-sse2-dup.c
  src/f32-igemm/gen/5x8-minmax-sse2-dup.c
  src/f32-prelu/gen/sse2-2x4.c
  src/f32-prelu/gen/sse2-2x8.c
  src/f32-qs8-vcvt/gen/vcvt-sse2-x8.c
  src/f32-qs8-vcvt/gen/vcvt-sse2-x16.c
  src/f32-qs8-vcvt/gen/vcvt-sse2-x24.c
  src/f32-qs8-vcvt/gen/vcvt-sse2-x32.c
  src/f32-qu8-vcvt/gen/vcvt-sse2-x8.c
  src/f32-qu8-vcvt/gen/vcvt-sse2-x16.c
  src/f32-qu8-vcvt/gen/vcvt-sse2-x24.c
  src/f32-qu8-vcvt/gen/vcvt-sse2-x32.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/sse2-rr2-p5-x20.c
  src/f32-velu/gen/velu-sse2-rr2-lut16-p3-x4.c
  src/f32-velu/gen/velu-sse2-rr2-lut16-p3-x8.c
  src/f32-velu/gen/velu-sse2-rr2-lut16-p3-x12.c
  src/f32-velu/gen/velu-sse2-rr2-lut16-p3-x16.c
  src/f32-velu/gen/velu-sse2-rr2-lut16-p3-x20.c
  src/f32-velu/gen/velu-sse2-rr2-lut16-p3-x24.c
  src/f32-velu/gen/velu-sse2-rr2-p6-x4.c
  src/f32-velu/gen/velu-sse2-rr2-p6-x8.c
  src/f32-velu/gen/velu-sse2-rr2-p6-x12.c
  src/f32-velu/gen/velu-sse2-rr2-p6-x16.c
  src/f32-velu/gen/velu-sse2-rr2-p6-x20.c
  src/f32-velu/gen/velu-sse2-rr2-p6-x24.c
  src/f32-vlrelu/gen/vlrelu-sse2-x4.c
  src/f32-vlrelu/gen/vlrelu-sse2-x8.c
  src/f32-vrnd/gen/vrndd-sse2-x4.c
  src/f32-vrnd/gen/vrndd-sse2-x8.c
  src/f32-vrnd/gen/vrndne-sse2-x4.c
  src/f32-vrnd/gen/vrndne-sse2-x8.c
  src/f32-vrnd/gen/vrndu-sse2-x4.c
  src/f32-vrnd/gen/vrndu-sse2-x8.c
  src/f32-vrnd/gen/vrndz-sse2-x4.c
  src/f32-vrnd/gen/vrndz-sse2-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-lut64-p2-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-lut64-p2-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-lut64-p2-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-lut64-p2-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-lut64-p2-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-lut64-p2-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-p5-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-p5-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-p5-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-p5-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-p5-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-rr2-p5-div-x24.c
  src/math/cvt-f16-f32-sse2-int16.c
  src/math/cvt-f16-f32-sse2-int32.c
  src/math/cvt-f32-f16-sse2.c
  src/math/exp-f32-sse2-rr2-lut64-p2.c
  src/math/exp-f32-sse2-rr2-p5.c
  src/math/expminus-f32-sse2-rr2-p5.c
  src/math/expm1minus-f32-sse2-rr2-lut16-p3.c
  src/math/expm1minus-f32-sse2-rr2-p6.c
  src/math/roundd-sse2-cvt.c
  src/math/roundne-sse2-cvt.c
  src/math/roundu-sse2-cvt.c
  src/math/roundz-sse2-cvt.c
  src/math/sigmoid-f32-sse2-rr2-lut64-p2-div.c
  src/math/sigmoid-f32-sse2-rr2-lut64-p2-nr1.c
  src/math/sigmoid-f32-sse2-rr2-lut64-p2-nr2.c
  src/math/sigmoid-f32-sse2-rr2-p5-div.c
  src/math/sigmoid-f32-sse2-rr2-p5-nr1.c
  src/math/sigmoid-f32-sse2-rr2-p5-nr2.c
  src/qc8-dwconv/gen/up8x3-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16-add16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16-add16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16-add16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-sse2-mul16.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16-add16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16-add16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16-add16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16-add16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-sse2-mul16.c
  src/qs8-f32-vcvt/gen/vcvt-sse2-x8.c
  src/qs8-f32-vcvt/gen/vcvt-sse2-x16.c
  src/qs8-f32-vcvt/gen/vcvt-sse2-x24.c
  src/qs8-f32-vcvt/gen/vcvt-sse2-x32.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse2-c8.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse2-c16.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse2-c24.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse2-c8.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse2-c16.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse2-c24.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-sse2.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-sse2.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c
  src/qs8-requantization/fp32-sse2.c
  src/qs8-requantization/gemmlowp-sse2.c
  src/qs8-requantization/rndna-sse2.c
  src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x8.c
  src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x16.c
  src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x24.c
  src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x32.c
  src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x8.c
  src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x16.c
  src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x24.c
  src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x32.c
  src/qs8-vcvt/gen/vcvt-sse2-x16.c
  src/qs8-vcvt/gen/vcvt-sse2-x32.c
  src/qs8-vlrelu/gen/vlrelu-sse2-x16.c
  src/qs8-vlrelu/gen/vlrelu-sse2-x32.c
  src/qs8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/qs8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x16.c
  src/qs8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/qs8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x16.c
  src/qu8-avgpool/9p8x-minmax-sse2-c8.c
  src/qu8-avgpool/9x-minmax-sse2-c8.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c
  src/qu8-f32-vcvt/gen/vcvt-sse2-x8.c
  src/qu8-f32-vcvt/gen/vcvt-sse2-x16.c
  src/qu8-f32-vcvt/gen/vcvt-sse2-x24.c
  src/qu8-f32-vcvt/gen/vcvt-sse2-x32.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse2-c8.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse2-c16.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse2-c24.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse2-c8.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse2-c16.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse2-c24.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c
  src/qu8-requantization/fp32-sse2.c
  src/qu8-requantization/gemmlowp-sse2.c
  src/qu8-requantization/rndna-sse2.c
  src/qu8-vadd/gen/minmax-sse2-mul16-ld64-x8.c
  src/qu8-vadd/gen/minmax-sse2-mul16-ld64-x16.c
  src/qu8-vaddc/gen/minmax-sse2-mul16-ld64-x8.c
  src/qu8-vaddc/gen/minmax-sse2-mul16-ld64-x16.c
  src/qu8-vcvt/gen/vcvt-sse2-x16.c
  src/qu8-vcvt/gen/vcvt-sse2-x32.c
  src/qu8-vlrelu/gen/vlrelu-sse2-x16.c
  src/qu8-vlrelu/gen/vlrelu-sse2-x32.c
  src/qu8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/qu8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x16.c
  src/qu8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x8.c
  src/qu8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x16.c
  src/s8-ibilinear/gen/sse2-c8.c
  src/s8-ibilinear/gen/sse2-c16.c
  src/s8-maxpool/9p8x-minmax-sse2-c16.c
  src/s8-vclamp/sse2-x64.c
  src/u8-ibilinear/gen/sse2-c8.c
  src/u8-ibilinear/gen/sse2-c16.c
  src/u8-maxpool/9p8x-minmax-sse2-c16.c
  src/u8-rmax/sse2.c
  src/u8-vclamp/sse2-x64.c
  src/xx-fill/sse2-x64.c
  src/xx-pad/sse2.c
  src/x8-transposec/gen/16x16-reuse-mov-sse2.c
  src/x8-transposec/gen/16x16-reuse-switch-sse2.c
  src/x8-zip/xm-sse2.c
  src/x8-zip/x2-sse2.c
  src/x8-zip/x3-sse2.c
  src/x8-zip/x4-sse2.c
  src/x16-transposec/gen/8x8-multi-mov-sse2.c
  src/x16-transposec/gen/8x8-multi-switch-sse2.c
  src/x16-transposec/gen/8x8-reuse-mov-sse2.c
  src/x16-transposec/gen/8x8-reuse-multi-sse2.c
  src/x16-transposec/gen/8x8-reuse-switch-sse2.c
  src/x16-transposec/4x8-sse2.c
  src/x32-transposec/gen/4x4-multi-mov-sse2.c
  src/x32-transposec/gen/4x4-multi-multi-sse2.c
  src/x32-transposec/gen/4x4-multi-switch-sse2.c
  src/x32-transposec/gen/4x4-reuse-mov-sse2.c
  src/x32-transposec/gen/4x4-reuse-multi-sse2.c
  src/x32-transposec/gen/4x4-reuse-switch-sse2.c
  src/x32-unpool/sse2.c
  src/x32-zip/xm-sse2.c
  src/x32-zip/x2-sse2.c
  src/x32-zip/x3-sse2.c
  src/x32-zip/x4-sse2.c
  src/x64-transposec/gen/2x2-multi-mov-sse2.c
  src/x64-transposec/gen/2x2-multi-multi-sse2.c
  src/x64-transposec/gen/2x2-multi-switch-sse2.c
  src/x64-transposec/gen/2x2-reuse-mov-sse2.c
  src/x64-transposec/gen/2x2-reuse-multi-sse2.c
  src/x64-transposec/gen/2x2-reuse-switch-sse2.c)

SET(PROD_SSSE3_MICROKERNEL_SRCS
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-2x4-acc2.c
  src/qs8-vcvt/gen/vcvt-ssse3-x32.c
  src/qs8-vlrelu/gen/vlrelu-ssse3-x32.c
  src/qu8-vcvt/gen/vcvt-ssse3-x32.c
  src/qu8-vlrelu/gen/vlrelu-ssse3-x32.c)

SET(ALL_SSSE3_MICROKERNEL_SRCS
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4-acc3.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4-acc4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-2x4-acc2.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-2x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-3x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-4x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-5x4.c
  src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-6x4.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-ssse3.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-ssse3.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-ssse3.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-requantization/gemmlowp-ssse3.c
  src/qs8-requantization/rndna-ssse3.c
  src/qs8-vcvt/gen/vcvt-ssse3-x16.c
  src/qs8-vcvt/gen/vcvt-ssse3-x32.c
  src/qs8-vlrelu/gen/vlrelu-ssse3-x16.c
  src/qs8-vlrelu/gen/vlrelu-ssse3-x32.c
  src/qu8-requantization/gemmlowp-ssse3.c
  src/qu8-requantization/rndna-ssse3.c
  src/qu8-vcvt/gen/vcvt-ssse3-x16.c
  src/qu8-vcvt/gen/vcvt-ssse3-x32.c
  src/qu8-vlrelu/gen/vlrelu-ssse3-x16.c
  src/qu8-vlrelu/gen/vlrelu-ssse3-x32.c
  src/x8-lut/gen/lut-ssse3-x16.c
  src/x8-lut/gen/lut-ssse3-x32.c
  src/x24-transposec/4x4-ssse3.c)

SET(PROD_SSE41_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-sse41-int16-x16.c
  src/f32-f16-vcvt/gen/vcvt-sse41-x8.c
  src/f32-prelu/gen/sse41-2x8.c
  src/f32-qs8-vcvt/gen/vcvt-sse41-x32.c
  src/f32-vlrelu/gen/vlrelu-sse41-x8.c
  src/f32-vrnd/gen/vrndd-sse41-x8.c
  src/f32-vrnd/gen/vrndne-sse41-x8.c
  src/f32-vrnd/gen/vrndu-sse41-x8.c
  src/f32-vrnd/gen/vrndz-sse41-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-lut64-p2-div-x8.c
  src/qc8-dwconv/gen/up8x3-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16-add16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16-add16.c
  src/qs8-f32-vcvt/gen/vcvt-sse41-x16.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse41-c8.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse41-c8.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x8.c
  src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x8.c
  src/qs8-vcvt/gen/vcvt-sse41-x32.c
  src/qs8-vlrelu/gen/vlrelu-sse41-x32.c
  src/qs8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/qs8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c
  src/qu8-f32-vcvt/gen/vcvt-sse41-x16.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse41-c8.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse41-c8.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-vadd/gen/minmax-sse41-mul16-ld64-x8.c
  src/qu8-vaddc/gen/minmax-sse41-mul16-ld64-x8.c
  src/qu8-vcvt/gen/vcvt-sse41-x32.c
  src/qu8-vlrelu/gen/vlrelu-sse41-x32.c
  src/qu8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/qu8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/s8-ibilinear/gen/sse41-c16.c
  src/s8-maxpool/9p8x-minmax-sse41-c16.c
  src/s8-vclamp/sse41-x64.c
  src/u8-ibilinear/gen/sse41-c16.c)

SET(ALL_SSE41_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-sse41-int16-x8.c
  src/f16-f32-vcvt/gen/vcvt-sse41-int16-x16.c
  src/f16-f32-vcvt/gen/vcvt-sse41-int16-x24.c
  src/f16-f32-vcvt/gen/vcvt-sse41-int16-x32.c
  src/f16-f32-vcvt/gen/vcvt-sse41-int32-x8.c
  src/f16-f32-vcvt/gen/vcvt-sse41-int32-x16.c
  src/f16-f32-vcvt/gen/vcvt-sse41-int32-x24.c
  src/f16-f32-vcvt/gen/vcvt-sse41-int32-x32.c
  src/f32-f16-vcvt/gen/vcvt-sse41-x8.c
  src/f32-f16-vcvt/gen/vcvt-sse41-x16.c
  src/f32-f16-vcvt/gen/vcvt-sse41-x24.c
  src/f32-f16-vcvt/gen/vcvt-sse41-x32.c
  src/f32-prelu/gen/sse41-2x4.c
  src/f32-prelu/gen/sse41-2x8.c
  src/f32-qs8-vcvt/gen/vcvt-sse41-x8.c
  src/f32-qs8-vcvt/gen/vcvt-sse41-x16.c
  src/f32-qs8-vcvt/gen/vcvt-sse41-x24.c
  src/f32-qs8-vcvt/gen/vcvt-sse41-x32.c
  src/f32-velu/gen/velu-sse41-rr2-lut16-p3-x4.c
  src/f32-velu/gen/velu-sse41-rr2-lut16-p3-x8.c
  src/f32-velu/gen/velu-sse41-rr2-lut16-p3-x12.c
  src/f32-velu/gen/velu-sse41-rr2-lut16-p3-x16.c
  src/f32-velu/gen/velu-sse41-rr2-lut16-p3-x20.c
  src/f32-velu/gen/velu-sse41-rr2-lut16-p3-x24.c
  src/f32-velu/gen/velu-sse41-rr2-p6-x4.c
  src/f32-velu/gen/velu-sse41-rr2-p6-x8.c
  src/f32-velu/gen/velu-sse41-rr2-p6-x12.c
  src/f32-velu/gen/velu-sse41-rr2-p6-x16.c
  src/f32-velu/gen/velu-sse41-rr2-p6-x20.c
  src/f32-velu/gen/velu-sse41-rr2-p6-x24.c
  src/f32-vlrelu/gen/vlrelu-sse41-x4.c
  src/f32-vlrelu/gen/vlrelu-sse41-x8.c
  src/f32-vrnd/gen/vrndd-sse41-x4.c
  src/f32-vrnd/gen/vrndd-sse41-x8.c
  src/f32-vrnd/gen/vrndne-sse41-x4.c
  src/f32-vrnd/gen/vrndne-sse41-x8.c
  src/f32-vrnd/gen/vrndu-sse41-x4.c
  src/f32-vrnd/gen/vrndu-sse41-x8.c
  src/f32-vrnd/gen/vrndz-sse41-x4.c
  src/f32-vrnd/gen/vrndz-sse41-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-lut64-p2-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-lut64-p2-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-lut64-p2-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-lut64-p2-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-lut64-p2-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-lut64-p2-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-p5-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-p5-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-p5-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-p5-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-p5-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-rr2-p5-div-x24.c
  src/math/cvt-f16-f32-sse41-int16.c
  src/math/cvt-f16-f32-sse41-int32.c
  src/math/cvt-f32-f16-sse41.c
  src/math/roundd-sse41.c
  src/math/roundne-sse41.c
  src/math/roundu-sse41.c
  src/math/roundz-sse41.c
  src/qc8-dwconv/gen/up8x3-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16-add16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16-add16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16-add16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-sse41-mul32.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16-add16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16-add16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16-add16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16-add16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-sse41-mul32.c
  src/qs8-f32-vcvt/gen/vcvt-sse41-x8.c
  src/qs8-f32-vcvt/gen/vcvt-sse41-x16.c
  src/qs8-f32-vcvt/gen/vcvt-sse41-x24.c
  src/qs8-f32-vcvt/gen/vcvt-sse41-x32.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse41-c8.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse41-c16.c
  src/qs8-gavgpool/gen/7p7x-minmax-fp32-sse41-c24.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse41-c8.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse41-c16.c
  src/qs8-gavgpool/gen/7x-minmax-fp32-sse41-c24.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-sse41.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-sse41.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c
  src/qs8-requantization/fp32-sse4.c
  src/qs8-requantization/gemmlowp-sse4.c
  src/qs8-requantization/rndna-sse4.c
  src/qs8-requantization/rndnu-sse4-sra.c
  src/qs8-requantization/rndnu-sse4-srl.c
  src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x8.c
  src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x16.c
  src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x24.c
  src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x32.c
  src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x8.c
  src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x16.c
  src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x24.c
  src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x32.c
  src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x8.c
  src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x16.c
  src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x24.c
  src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x32.c
  src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x8.c
  src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x16.c
  src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x24.c
  src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x32.c
  src/qs8-vcvt/gen/vcvt-sse41-x8.c
  src/qs8-vcvt/gen/vcvt-sse41-x16.c
  src/qs8-vcvt/gen/vcvt-sse41-x32.c
  src/qs8-vlrelu/gen/vlrelu-sse41-x8.c
  src/qs8-vlrelu/gen/vlrelu-sse41-x16.c
  src/qs8-vlrelu/gen/vlrelu-sse41-x32.c
  src/qs8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x8.c
  src/qs8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/qs8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x8.c
  src/qs8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c
  src/qu8-f32-vcvt/gen/vcvt-sse41-x8.c
  src/qu8-f32-vcvt/gen/vcvt-sse41-x16.c
  src/qu8-f32-vcvt/gen/vcvt-sse41-x24.c
  src/qu8-f32-vcvt/gen/vcvt-sse41-x32.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse41-c8.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse41-c16.c
  src/qu8-gavgpool/gen/7p7x-minmax-fp32-sse41-c24.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse41-c8.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse41-c16.c
  src/qu8-gavgpool/gen/7x-minmax-fp32-sse41-c24.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c
  src/qu8-requantization/gemmlowp-sse4.c
  src/qu8-requantization/rndna-sse4.c
  src/qu8-vadd/gen/minmax-sse41-mul16-ld64-x8.c
  src/qu8-vadd/gen/minmax-sse41-mul16-ld64-x16.c
  src/qu8-vadd/gen/minmax-sse41-mul32-ld32-x8.c
  src/qu8-vadd/gen/minmax-sse41-mul32-ld32-x16.c
  src/qu8-vaddc/gen/minmax-sse41-mul16-ld64-x8.c
  src/qu8-vaddc/gen/minmax-sse41-mul16-ld64-x16.c
  src/qu8-vaddc/gen/minmax-sse41-mul32-ld32-x8.c
  src/qu8-vaddc/gen/minmax-sse41-mul32-ld32-x16.c
  src/qu8-vcvt/gen/vcvt-sse41-x8.c
  src/qu8-vcvt/gen/vcvt-sse41-x16.c
  src/qu8-vcvt/gen/vcvt-sse41-x32.c
  src/qu8-vlrelu/gen/vlrelu-sse41-x8.c
  src/qu8-vlrelu/gen/vlrelu-sse41-x16.c
  src/qu8-vlrelu/gen/vlrelu-sse41-x32.c
  src/qu8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x8.c
  src/qu8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/qu8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x8.c
  src/qu8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x16.c
  src/s8-ibilinear/gen/sse41-c8.c
  src/s8-ibilinear/gen/sse41-c16.c
  src/s8-maxpool/9p8x-minmax-sse41-c16.c
  src/s8-vclamp/sse41-x64.c
  src/u8-ibilinear/gen/sse41-c8.c
  src/u8-ibilinear/gen/sse41-c16.c)

SET(PROD_AVX_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-avx-int16-x16.c
  src/f32-dwconv/gen/up8x25-minmax-avx.c
  src/f32-dwconv/gen/up16x3-minmax-avx.c
  src/f32-dwconv/gen/up16x4-minmax-avx.c
  src/f32-dwconv/gen/up16x9-minmax-avx.c
  src/f32-f16-vcvt/gen/vcvt-avx-x24.c
  src/f32-gemm/gen/1x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/5x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/1x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/5x16-minmax-avx-broadcast.c
  src/f32-prelu/gen/avx-2x16.c
  src/f32-qs8-vcvt/gen/vcvt-avx-x32.c
  src/f32-qu8-vcvt/gen/vcvt-avx-x32.c
  src/f32-vbinary/gen/vadd-minmax-avx-x16.c
  src/f32-vbinary/gen/vaddc-minmax-avx-x16.c
  src/f32-vbinary/gen/vdiv-minmax-avx-x16.c
  src/f32-vbinary/gen/vdivc-minmax-avx-x16.c
  src/f32-vbinary/gen/vmax-avx-x16.c
  src/f32-vbinary/gen/vmaxc-avx-x16.c
  src/f32-vbinary/gen/vmin-avx-x16.c
  src/f32-vbinary/gen/vminc-avx-x16.c
  src/f32-vbinary/gen/vmul-minmax-avx-x16.c
  src/f32-vbinary/gen/vmulc-minmax-avx-x16.c
  src/f32-vbinary/gen/vrdivc-minmax-avx-x16.c
  src/f32-vbinary/gen/vrsubc-minmax-avx-x16.c
  src/f32-vbinary/gen/vsqrdiff-avx-x16.c
  src/f32-vbinary/gen/vsqrdiffc-avx-x16.c
  src/f32-vbinary/gen/vsub-minmax-avx-x16.c
  src/f32-vbinary/gen/vsubc-minmax-avx-x16.c
  src/f32-vclamp/gen/vclamp-avx-x16.c
  src/f32-velu/gen/velu-avx-rr2-lut4-p4-perm-x32.c
  src/f32-vhswish/gen/vhswish-avx-x16.c
  src/f32-vlrelu/gen/vlrelu-avx-x16.c
  src/f32-vrnd/gen/vrndd-avx-x16.c
  src/f32-vrnd/gen/vrndne-avx-x16.c
  src/f32-vrnd/gen/vrndu-avx-x16.c
  src/f32-vrnd/gen/vrndz-avx-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x40.c
  src/f32-vsqrt/gen/avx-sqrt-x8.c
  src/f32-vunary/gen/vabs-avx-x16.c
  src/f32-vunary/gen/vneg-avx-x16.c
  src/f32-vunary/gen/vsqr-avx-x16.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-avx-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul16-add16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul16-add16.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul16-add16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul16-add16.c
  src/qs8-f32-vcvt/gen/vcvt-avx-x32.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qs8-vadd/gen/minmax-avx-mul32-ld32-x8.c
  src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x8.c
  src/qs8-vcvt/gen/vcvt-avx-x32.c
  src/qs8-vlrelu/gen/vlrelu-avx-x32.c
  src/qs8-vmul/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/qs8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c
  src/qu8-f32-vcvt/gen/vcvt-avx-x32.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qu8-vadd/gen/minmax-avx-mul32-ld32-x8.c
  src/qu8-vaddc/gen/minmax-avx-mul32-ld32-x8.c
  src/qu8-vcvt/gen/vcvt-avx-x32.c
  src/qu8-vlrelu/gen/vlrelu-avx-x32.c
  src/qu8-vmul/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/qu8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/x8-lut/gen/lut-avx-x64.c)

SET(ALL_AVX_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-avx-int16-x8.c
  src/f16-f32-vcvt/gen/vcvt-avx-int16-x16.c
  src/f16-f32-vcvt/gen/vcvt-avx-int16-x24.c
  src/f16-f32-vcvt/gen/vcvt-avx-int16-x32.c
  src/f16-f32-vcvt/gen/vcvt-avx-int32-x8.c
  src/f16-f32-vcvt/gen/vcvt-avx-int32-x16.c
  src/f16-f32-vcvt/gen/vcvt-avx-int32-x24.c
  src/f16-f32-vcvt/gen/vcvt-avx-int32-x32.c
  src/f32-dwconv/gen/up8x3-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x3-minmax-avx.c
  src/f32-dwconv/gen/up8x4-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-avx.c
  src/f32-dwconv/gen/up8x9-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-avx.c
  src/f32-dwconv/gen/up8x25-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-avx.c
  src/f32-dwconv/gen/up16x3-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x3-minmax-avx.c
  src/f32-dwconv/gen/up16x4-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-avx.c
  src/f32-dwconv/gen/up16x9-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-avx.c
  src/f32-dwconv/gen/up16x25-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-avx.c
  src/f32-f16-vcvt/gen/vcvt-avx-x8.c
  src/f32-f16-vcvt/gen/vcvt-avx-x16.c
  src/f32-f16-vcvt/gen/vcvt-avx-x24.c
  src/f32-f16-vcvt/gen/vcvt-avx-x32.c
  src/f32-gemm/gen-inc/1x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/1x16inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/3x16inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/4x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/4x16inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/5x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/5x16inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/6x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/7x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen/1x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/1x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/3x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/4x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/4x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/5x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/5x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/6x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/7x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/1x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/1x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/3x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/4x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/4x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/5x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/5x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/6x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/7x8-minmax-avx-broadcast.c
  src/f32-prelu/gen/avx-2x8.c
  src/f32-prelu/gen/avx-2x16.c
  src/f32-qs8-vcvt/gen/vcvt-avx-x8.c
  src/f32-qs8-vcvt/gen/vcvt-avx-x16.c
  src/f32-qs8-vcvt/gen/vcvt-avx-x24.c
  src/f32-qs8-vcvt/gen/vcvt-avx-x32.c
  src/f32-qu8-vcvt/gen/vcvt-avx-x8.c
  src/f32-qu8-vcvt/gen/vcvt-avx-x16.c
  src/f32-qu8-vcvt/gen/vcvt-avx-x24.c
  src/f32-qu8-vcvt/gen/vcvt-avx-x32.c
  src/f32-rmax/avx.c
  src/f32-vbinary/gen/vadd-minmax-avx-x8.c
  src/f32-vbinary/gen/vadd-minmax-avx-x16.c
  src/f32-vbinary/gen/vaddc-minmax-avx-x8.c
  src/f32-vbinary/gen/vaddc-minmax-avx-x16.c
  src/f32-vbinary/gen/vdiv-minmax-avx-x8.c
  src/f32-vbinary/gen/vdiv-minmax-avx-x16.c
  src/f32-vbinary/gen/vdivc-minmax-avx-x8.c
  src/f32-vbinary/gen/vdivc-minmax-avx-x16.c
  src/f32-vbinary/gen/vmax-avx-x8.c
  src/f32-vbinary/gen/vmax-avx-x16.c
  src/f32-vbinary/gen/vmaxc-avx-x8.c
  src/f32-vbinary/gen/vmaxc-avx-x16.c
  src/f32-vbinary/gen/vmin-avx-x8.c
  src/f32-vbinary/gen/vmin-avx-x16.c
  src/f32-vbinary/gen/vminc-avx-x8.c
  src/f32-vbinary/gen/vminc-avx-x16.c
  src/f32-vbinary/gen/vmul-minmax-avx-x8.c
  src/f32-vbinary/gen/vmul-minmax-avx-x16.c
  src/f32-vbinary/gen/vmulc-minmax-avx-x8.c
  src/f32-vbinary/gen/vmulc-minmax-avx-x16.c
  src/f32-vbinary/gen/vrdivc-minmax-avx-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-avx-x16.c
  src/f32-vbinary/gen/vrsubc-minmax-avx-x8.c
  src/f32-vbinary/gen/vrsubc-minmax-avx-x16.c
  src/f32-vbinary/gen/vsqrdiff-avx-x8.c
  src/f32-vbinary/gen/vsqrdiff-avx-x16.c
  src/f32-vbinary/gen/vsqrdiffc-avx-x8.c
  src/f32-vbinary/gen/vsqrdiffc-avx-x16.c
  src/f32-vbinary/gen/vsub-minmax-avx-x8.c
  src/f32-vbinary/gen/vsub-minmax-avx-x16.c
  src/f32-vbinary/gen/vsubc-minmax-avx-x8.c
  src/f32-vbinary/gen/vsubc-minmax-avx-x16.c
  src/f32-vclamp/gen/vclamp-avx-x8.c
  src/f32-vclamp/gen/vclamp-avx-x16.c
  src/f32-velu/gen/velu-avx-rr2-lut4-p4-perm-x8.c
  src/f32-velu/gen/velu-avx-rr2-lut4-p4-perm-x16.c
  src/f32-velu/gen/velu-avx-rr2-lut4-p4-perm-x24.c
  src/f32-velu/gen/velu-avx-rr2-lut4-p4-perm-x32.c
  src/f32-velu/gen/velu-avx-rr2-lut4-p4-perm-x40.c
  src/f32-velu/gen/velu-avx-rr2-lut4-p4-perm-x48.c
  src/f32-velu/gen/velu-avx-rr2-lut16-p3-x8.c
  src/f32-velu/gen/velu-avx-rr2-lut16-p3-x16.c
  src/f32-velu/gen/velu-avx-rr2-lut16-p3-x24.c
  src/f32-velu/gen/velu-avx-rr2-lut16-p3-x32.c
  src/f32-velu/gen/velu-avx-rr2-lut16-p3-x40.c
  src/f32-velu/gen/velu-avx-rr2-lut16-p3-x48.c
  src/f32-velu/gen/velu-avx-rr2-p6-x8.c
  src/f32-velu/gen/velu-avx-rr2-p6-x16.c
  src/f32-velu/gen/velu-avx-rr2-p6-x24.c
  src/f32-velu/gen/velu-avx-rr2-p6-x32.c
  src/f32-velu/gen/velu-avx-rr2-p6-x40.c
  src/f32-velu/gen/velu-avx-rr2-p6-x48.c
  src/f32-vhswish/gen/vhswish-avx-x8.c
  src/f32-vhswish/gen/vhswish-avx-x16.c
  src/f32-vlrelu/gen/vlrelu-avx-x8.c
  src/f32-vlrelu/gen/vlrelu-avx-x16.c
  src/f32-vrnd/gen/vrndd-avx-x8.c
  src/f32-vrnd/gen/vrndd-avx-x16.c
  src/f32-vrnd/gen/vrndne-avx-x8.c
  src/f32-vrnd/gen/vrndne-avx-x16.c
  src/f32-vrnd/gen/vrndu-avx-x8.c
  src/f32-vrnd/gen/vrndu-avx-x16.c
  src/f32-vrnd/gen/vrndz-avx-x8.c
  src/f32-vrnd/gen/vrndz-avx-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x40.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x56.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x72.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x8.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x24.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x40.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x56.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x72.c
  src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x80.c
  src/f32-vsqrt/gen/avx-sqrt-x8.c
  src/f32-vsqrt/gen/avx-sqrt-x16.c
  src/f32-vunary/gen/vabs-avx-x8.c
  src/f32-vunary/gen/vabs-avx-x16.c
  src/f32-vunary/gen/vneg-avx-x8.c
  src/f32-vunary/gen/vneg-avx-x16.c
  src/f32-vunary/gen/vsqr-avx-x8.c
  src/f32-vunary/gen/vsqr-avx-x16.c
  src/math/exp-f32-avx-rr2-p5.c
  src/math/expm1minus-f32-avx-rr2-lut4-p4-perm.c
  src/math/expm1minus-f32-avx-rr2-lut16-p3.c
  src/math/expm1minus-f32-avx-rr2-p6.c
  src/math/sigmoid-f32-avx-rr2-lut64-p2-div.c
  src/math/sigmoid-f32-avx-rr2-p5-div.c
  src/math/sigmoid-f32-avx-rr2-p5-nr1.c
  src/math/sigmoid-f32-avx-rr2-p5-nr2.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul16-add16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul16-add16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-avx-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul16-add16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-avx-mul32.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul16-add16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul16-add16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul16-add16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul16-add16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-avx-mul32.c
  src/qs8-f32-vcvt/gen/vcvt-avx-x8.c
  src/qs8-f32-vcvt/gen/vcvt-avx-x16.c
  src/qs8-f32-vcvt/gen/vcvt-avx-x24.c
  src/qs8-f32-vcvt/gen/vcvt-avx-x32.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-avx.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-avx.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c
  src/qs8-vadd/gen/minmax-avx-mul16-ld64-x8.c
  src/qs8-vadd/gen/minmax-avx-mul16-ld64-x16.c
  src/qs8-vadd/gen/minmax-avx-mul16-ld64-x24.c
  src/qs8-vadd/gen/minmax-avx-mul16-ld64-x32.c
  src/qs8-vadd/gen/minmax-avx-mul32-ld32-x8.c
  src/qs8-vadd/gen/minmax-avx-mul32-ld32-x16.c
  src/qs8-vadd/gen/minmax-avx-mul32-ld32-x24.c
  src/qs8-vadd/gen/minmax-avx-mul32-ld32-x32.c
  src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x8.c
  src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x16.c
  src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x24.c
  src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x32.c
  src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x8.c
  src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x16.c
  src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x24.c
  src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x32.c
  src/qs8-vcvt/gen/vcvt-avx-x8.c
  src/qs8-vcvt/gen/vcvt-avx-x16.c
  src/qs8-vcvt/gen/vcvt-avx-x32.c
  src/qs8-vlrelu/gen/vlrelu-avx-x8.c
  src/qs8-vlrelu/gen/vlrelu-avx-x16.c
  src/qs8-vlrelu/gen/vlrelu-avx-x32.c
  src/qs8-vmul/gen/minmax-fp32-avx-mul16-ld64-x8.c
  src/qs8-vmul/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/qs8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x8.c
  src/qs8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c
  src/qu8-f32-vcvt/gen/vcvt-avx-x8.c
  src/qu8-f32-vcvt/gen/vcvt-avx-x16.c
  src/qu8-f32-vcvt/gen/vcvt-avx-x24.c
  src/qu8-f32-vcvt/gen/vcvt-avx-x32.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c
  src/qu8-vadd/gen/minmax-avx-mul16-ld64-x8.c
  src/qu8-vadd/gen/minmax-avx-mul16-ld64-x16.c
  src/qu8-vadd/gen/minmax-avx-mul32-ld32-x8.c
  src/qu8-vadd/gen/minmax-avx-mul32-ld32-x16.c
  src/qu8-vaddc/gen/minmax-avx-mul16-ld64-x8.c
  src/qu8-vaddc/gen/minmax-avx-mul16-ld64-x16.c
  src/qu8-vaddc/gen/minmax-avx-mul32-ld32-x8.c
  src/qu8-vaddc/gen/minmax-avx-mul32-ld32-x16.c
  src/qu8-vcvt/gen/vcvt-avx-x8.c
  src/qu8-vcvt/gen/vcvt-avx-x16.c
  src/qu8-vcvt/gen/vcvt-avx-x32.c
  src/qu8-vlrelu/gen/vlrelu-avx-x8.c
  src/qu8-vlrelu/gen/vlrelu-avx-x16.c
  src/qu8-vlrelu/gen/vlrelu-avx-x32.c
  src/qu8-vmul/gen/minmax-fp32-avx-mul16-ld64-x8.c
  src/qu8-vmul/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/qu8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x8.c
  src/qu8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x16.c
  src/x8-lut/gen/lut-avx-x16.c
  src/x8-lut/gen/lut-avx-x32.c
  src/x8-lut/gen/lut-avx-x48.c
  src/x8-lut/gen/lut-avx-x64.c)

SET(PROD_F16C_MICROKERNEL_SRCS
  src/f16-avgpool/9p8x-minmax-f16c-c8.c
  src/f16-avgpool/9x-minmax-f16c-c8.c
  src/f16-f32-vcvt/gen/vcvt-f16c-x16.c
  src/f16-gavgpool/gen/7p7x-minmax-f16c-c8.c
  src/f16-gavgpool/gen/7x-minmax-f16c-c8.c
  src/f16-maxpool/9p8x-minmax-f16c-c8.c
  src/f16-prelu/gen/f16c-2x16.c
  src/f16-rmax/f16c.c
  src/f16-vbinary/gen/vadd-minmax-f16c-x16.c
  src/f16-vbinary/gen/vaddc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vdiv-minmax-f16c-x8.c
  src/f16-vbinary/gen/vdivc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vmax-f16c-x16.c
  src/f16-vbinary/gen/vmaxc-f16c-x16.c
  src/f16-vbinary/gen/vmin-f16c-x16.c
  src/f16-vbinary/gen/vminc-f16c-x16.c
  src/f16-vbinary/gen/vmul-minmax-f16c-x16.c
  src/f16-vbinary/gen/vmulc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vrdivc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vrsubc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vsqrdiff-f16c-x16.c
  src/f16-vbinary/gen/vsqrdiffc-f16c-x16.c
  src/f16-vbinary/gen/vsub-minmax-f16c-x16.c
  src/f16-vbinary/gen/vsubc-minmax-f16c-x16.c
  src/f16-vclamp/gen/vclamp-f16c-x16.c
  src/f16-vhswish/gen/vhswish-f16c-x16.c
  src/f16-vlrelu/gen/vlrelu-f16c-x16.c
  src/f16-vrnd/gen/vrndd-f16c-x16.c
  src/f16-vrnd/gen/vrndne-f16c-x16.c
  src/f16-vrnd/gen/vrndu-f16c-x16.c
  src/f16-vrnd/gen/vrndz-f16c-x16.c
  src/f16-vsqrt/gen/f16c-sqrt-x8.c
  src/f16-vunary/gen/vsqr-f16c-x16.c
  src/f32-f16-vcvt/gen/vcvt-f16c-x16.c)

SET(ALL_F16C_MICROKERNEL_SRCS
  src/f16-avgpool/9p8x-minmax-f16c-c8.c
  src/f16-avgpool/9x-minmax-f16c-c8.c
  src/f16-f32-vcvt/gen/vcvt-f16c-x8.c
  src/f16-f32-vcvt/gen/vcvt-f16c-x16.c
  src/f16-gavgpool/gen/7p7x-minmax-f16c-c8.c
  src/f16-gavgpool/gen/7p7x-minmax-f16c-c16.c
  src/f16-gavgpool/gen/7p7x-minmax-f16c-c24.c
  src/f16-gavgpool/gen/7p7x-minmax-f16c-c32.c
  src/f16-gavgpool/gen/7x-minmax-f16c-c8.c
  src/f16-gavgpool/gen/7x-minmax-f16c-c16.c
  src/f16-gavgpool/gen/7x-minmax-f16c-c24.c
  src/f16-gavgpool/gen/7x-minmax-f16c-c32.c
  src/f16-maxpool/9p8x-minmax-f16c-c8.c
  src/f16-prelu/gen/f16c-2x8.c
  src/f16-prelu/gen/f16c-2x16.c
  src/f16-rmax/f16c.c
  src/f16-vbinary/gen/vadd-minmax-f16c-x8.c
  src/f16-vbinary/gen/vadd-minmax-f16c-x16.c
  src/f16-vbinary/gen/vaddc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vaddc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vdiv-minmax-f16c-x8.c
  src/f16-vbinary/gen/vdiv-minmax-f16c-x16.c
  src/f16-vbinary/gen/vdivc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vdivc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vmax-f16c-x8.c
  src/f16-vbinary/gen/vmax-f16c-x16.c
  src/f16-vbinary/gen/vmaxc-f16c-x8.c
  src/f16-vbinary/gen/vmaxc-f16c-x16.c
  src/f16-vbinary/gen/vmin-f16c-x8.c
  src/f16-vbinary/gen/vmin-f16c-x16.c
  src/f16-vbinary/gen/vminc-f16c-x8.c
  src/f16-vbinary/gen/vminc-f16c-x16.c
  src/f16-vbinary/gen/vmul-minmax-f16c-x8.c
  src/f16-vbinary/gen/vmul-minmax-f16c-x16.c
  src/f16-vbinary/gen/vmulc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vmulc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vrdivc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vrdivc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vrsubc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vrsubc-minmax-f16c-x16.c
  src/f16-vbinary/gen/vsqrdiff-f16c-x8.c
  src/f16-vbinary/gen/vsqrdiff-f16c-x16.c
  src/f16-vbinary/gen/vsqrdiffc-f16c-x8.c
  src/f16-vbinary/gen/vsqrdiffc-f16c-x16.c
  src/f16-vbinary/gen/vsub-minmax-f16c-x8.c
  src/f16-vbinary/gen/vsub-minmax-f16c-x16.c
  src/f16-vbinary/gen/vsubc-minmax-f16c-x8.c
  src/f16-vbinary/gen/vsubc-minmax-f16c-x16.c
  src/f16-vclamp/gen/vclamp-f16c-x8.c
  src/f16-vclamp/gen/vclamp-f16c-x16.c
  src/f16-vhswish/gen/vhswish-f16c-x8.c
  src/f16-vhswish/gen/vhswish-f16c-x16.c
  src/f16-vlrelu/gen/vlrelu-f16c-x8.c
  src/f16-vlrelu/gen/vlrelu-f16c-x16.c
  src/f16-vrnd/gen/vrndd-f16c-x8.c
  src/f16-vrnd/gen/vrndd-f16c-x16.c
  src/f16-vrnd/gen/vrndne-f16c-x8.c
  src/f16-vrnd/gen/vrndne-f16c-x16.c
  src/f16-vrnd/gen/vrndu-f16c-x8.c
  src/f16-vrnd/gen/vrndu-f16c-x16.c
  src/f16-vrnd/gen/vrndz-f16c-x8.c
  src/f16-vrnd/gen/vrndz-f16c-x16.c
  src/f16-vsqrt/gen/f16c-sqrt-x8.c
  src/f16-vsqrt/gen/f16c-sqrt-x16.c
  src/f16-vunary/gen/vsqr-f16c-x8.c
  src/f16-vunary/gen/vsqr-f16c-x16.c
  src/f32-f16-vcvt/gen/vcvt-f16c-x8.c
  src/f32-f16-vcvt/gen/vcvt-f16c-x16.c
  src/math/cvt-f16-f32-f16c.c
  src/math/cvt-f32-f16-f16c.c)

SET(PROD_XOP_MICROKERNEL_SRCS
  src/qc8-dwconv/gen/up16x3-minmax-fp32-xop-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-xop-mul16-add16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-xop-mul16-add16.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-xop-mul16-add16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-xop-mul16-add16.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x8.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x8.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qu8-vadd/gen/minmax-xop-mul32-ld32-x8.c
  src/qu8-vaddc/gen/minmax-xop-mul32-ld32-x8.c)

SET(ALL_XOP_MICROKERNEL_SRCS
  src/qc8-dwconv/gen/up8x9-minmax-fp32-xop-mul16-add16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-xop-mul16-add16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-xop-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-xop-mul16-add16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-xop-mul16-add16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-xop-mul32.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-xop-mul16-add16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-xop-mul16-add16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-xop-mul16-add16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-xop-mul16-add16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-xop-mul32.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-xop.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-xop.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x8.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x16.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x24.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x32.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x8.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x16.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x24.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x32.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c
  src/qu8-igemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c
  src/qu8-vadd/gen/minmax-xop-mul32-ld32-x8.c
  src/qu8-vadd/gen/minmax-xop-mul32-ld32-x16.c
  src/qu8-vaddc/gen/minmax-xop-mul32-ld32-x8.c
  src/qu8-vaddc/gen/minmax-xop-mul32-ld32-x16.c)

SET(PROD_FMA3_MICROKERNEL_SRCS
  src/f16-dwconv/gen/up8x25-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up16x3-minmax-fma3.c
  src/f16-dwconv/gen/up16x4-minmax-fma3.c
  src/f16-dwconv/gen/up16x9-minmax-fma3.c
  src/f16-ibilinear/gen/fma3-c8.c
  src/f16-ibilinear/gen/fma3-c16.c
  src/f16-vmulcaddc/gen/c8-minmax-fma3-2x.c
  src/f32-dwconv/gen/up8x25-minmax-fma3.c
  src/f32-dwconv/gen/up16x3-minmax-fma3.c
  src/f32-dwconv/gen/up16x4-minmax-fma3.c
  src/f32-dwconv/gen/up16x9-minmax-fma3.c
  src/f32-gemm/gen/1x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/1x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/4x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/5x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/1x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/1x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/4x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/5x16-minmax-fma3-broadcast.c
  src/f32-vhswish/gen/vhswish-fma3-x16.c)

SET(ALL_FMA3_MICROKERNEL_SRCS
  src/f16-dwconv/gen/up8x3-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up8x3-minmax-fma3.c
  src/f16-dwconv/gen/up8x4-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up8x4-minmax-fma3.c
  src/f16-dwconv/gen/up8x9-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up8x9-minmax-fma3.c
  src/f16-dwconv/gen/up8x25-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up8x25-minmax-fma3.c
  src/f16-dwconv/gen/up16x3-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up16x3-minmax-fma3.c
  src/f16-dwconv/gen/up16x4-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up16x4-minmax-fma3.c
  src/f16-dwconv/gen/up16x9-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up16x9-minmax-fma3.c
  src/f16-dwconv/gen/up16x25-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up16x25-minmax-fma3.c
  src/f16-dwconv/gen/up32x3-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up32x3-minmax-fma3.c
  src/f16-dwconv/gen/up32x4-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up32x4-minmax-fma3.c
  src/f16-dwconv/gen/up32x9-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up32x9-minmax-fma3.c
  src/f16-dwconv/gen/up32x25-minmax-fma3-acc2.c
  src/f16-dwconv/gen/up32x25-minmax-fma3.c
  src/f16-ibilinear/gen/fma3-c8.c
  src/f16-ibilinear/gen/fma3-c16.c
  src/f16-vmulcaddc/gen/c8-minmax-fma3-2x.c
  src/f16-vmulcaddc/gen/c16-minmax-fma3-2x.c
  src/f32-dwconv/gen/up8x3-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x3-minmax-fma3.c
  src/f32-dwconv/gen/up8x4-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-fma3.c
  src/f32-dwconv/gen/up8x9-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-fma3.c
  src/f32-dwconv/gen/up8x25-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-fma3.c
  src/f32-dwconv/gen/up16x3-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up16x3-minmax-fma3.c
  src/f32-dwconv/gen/up16x4-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-fma3.c
  src/f32-dwconv/gen/up16x9-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-fma3.c
  src/f32-dwconv/gen/up16x25-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-fma3.c
  src/f32-gemm/gen-inc/1x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/1x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/1x16s4inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/3x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/3x16s4inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/4x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/4x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/4x16s4inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/5x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/5x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/5x16s4inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/6x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/7x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/8x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen/1x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/1x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/1x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/3x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/3x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/4x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/4x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/4x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/5x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/5x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/5x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/6x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/7x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/8x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/1x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/1x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/1x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/3x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/3x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/4x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/4x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/4x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/5x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/5x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/5x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/6x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/7x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/8x8-minmax-fma3-broadcast.c
  src/f32-vhswish/gen/vhswish-fma3-x8.c
  src/f32-vhswish/gen/vhswish-fma3-x16.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x8.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x16.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x24.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x32.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x40.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x48.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x56.c
  src/f32-vsqrt/gen/fma3-nr1fma1adj-x64.c
  src/math/sqrt-fma3-nr1fma.c
  src/math/sqrt-fma3-nr1fma1adj.c
  src/math/sqrt-fma3-nr2fma.c)

SET(PROD_AVX2_MICROKERNEL_SRCS
  src/f16-gemm/gen/1x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/4x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/1x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/4x16-minmax-avx2-broadcast.c
  src/f16-pavgpool/9p8x-minmax-avx2-c8.c
  src/f16-pavgpool/9x-minmax-avx2-c8.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x40.c
  src/f16-velu/gen/velu-avx2-rr1-p3-x16.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x32.c
  src/f32-qs8-vcvt/gen/vcvt-avx2-x64.c
  src/f32-qu8-vcvt/gen/vcvt-avx2-x64.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x56.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x40.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qc8-gemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qc8-gemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qc8-igemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qc8-igemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qs8-f32-vcvt/gen/vcvt-avx2-x16.c
  src/qs8-gemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qs8-gemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-igemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qs8-igemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x16.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x16.c
  src/qs8-vcvt/gen/vcvt-avx2-x32.c
  src/qs8-vlrelu/gen/vlrelu-avx2-x32.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qu8-f32-vcvt/gen/vcvt-avx2-x16.c
  src/qu8-gemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qu8-gemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qu8-vadd/gen/minmax-avx2-mul32-ld64-x16.c
  src/qu8-vaddc/gen/minmax-avx2-mul32-ld64-x16.c
  src/qu8-vcvt/gen/vcvt-avx2-x32.c
  src/qu8-vlrelu/gen/vlrelu-avx2-x32.c
  src/x8-lut/gen/lut-avx2-x128.c)

SET(ALL_AVX2_MICROKERNEL_SRCS
  src/f16-gemm/gen/1x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/1x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/3x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/4x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/4x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/5x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/5x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/6x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/7x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/1x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/1x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/3x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/4x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/4x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/5x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/5x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/6x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/7x8-minmax-avx2-broadcast.c
  src/f16-pavgpool/9p8x-minmax-avx2-c8.c
  src/f16-pavgpool/9x-minmax-avx2-c8.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x32-acc2.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x32-acc4.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x32.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x40-acc2.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x40-acc5.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x40.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x48-acc2.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x48-acc3.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x48.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x64-acc2.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x64-acc4.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x64.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x72-acc3.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x72.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x80-acc2.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x80-acc5.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x80.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x96-acc2.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x96-acc3.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x96-acc6.c
  src/f16-raddstoreexpminusmax/gen/avx2-rr1-p2-x96.c
  src/f16-velu/gen/velu-avx2-rr1-p3-x8.c
  src/f16-velu/gen/velu-avx2-rr1-p3-x16.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x8.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x16.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x24.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x32.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x40.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x48.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x56.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-div-x64.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x8.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x16.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x24.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x32.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x40.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x48.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x56.c
  src/f16-vsigmoid/gen/vsigmoid-avx2-rr1-p2-rcp-x64.c
  src/f32-qs8-vcvt/gen/vcvt-avx2-x16.c
  src/f32-qs8-vcvt/gen/vcvt-avx2-x32.c
  src/f32-qs8-vcvt/gen/vcvt-avx2-x48.c
  src/f32-qs8-vcvt/gen/vcvt-avx2-x64.c
  src/f32-qu8-vcvt/gen/vcvt-avx2-x16.c
  src/f32-qu8-vcvt/gen/vcvt-avx2-x32.c
  src/f32-qu8-vcvt/gen/vcvt-avx2-x48.c
  src/f32-qu8-vcvt/gen/vcvt-avx2-x64.c
  src/f32-raddexpminusmax/gen/avx2-p5-x64-acc2.c
  src/f32-raddexpminusmax/gen/avx2-p5-x64-acc4.c
  src/f32-raddexpminusmax/gen/avx2-p5-x64.c
  src/f32-raddexpminusmax/gen/avx2-p5-x72-acc3.c
  src/f32-raddexpminusmax/gen/avx2-p5-x72.c
  src/f32-raddexpminusmax/gen/avx2-p5-x80-acc2.c
  src/f32-raddexpminusmax/gen/avx2-p5-x80-acc5.c
  src/f32-raddexpminusmax/gen/avx2-p5-x80.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96-acc2.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96-acc3.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96-acc6.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96.c
  src/f32-raddextexp/gen/avx2-p5-x64-acc2.c
  src/f32-raddextexp/gen/avx2-p5-x64-acc4.c
  src/f32-raddextexp/gen/avx2-p5-x64.c
  src/f32-raddextexp/gen/avx2-p5-x72-acc3.c
  src/f32-raddextexp/gen/avx2-p5-x72.c
  src/f32-raddextexp/gen/avx2-p5-x80-acc2.c
  src/f32-raddextexp/gen/avx2-p5-x80-acc5.c
  src/f32-raddextexp/gen/avx2-p5-x80.c
  src/f32-raddextexp/gen/avx2-p5-x96-acc2.c
  src/f32-raddextexp/gen/avx2-p5-x96-acc3.c
  src/f32-raddextexp/gen/avx2-p5-x96-acc6.c
  src/f32-raddextexp/gen/avx2-p5-x96.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x64-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x64-acc4.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x64.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x72-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x72.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x80-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x80-acc5.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x80.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x96-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x96-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x96-acc6.c
  src/f32-raddstoreexpminusmax/gen/avx2-rr1-p5-x96.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x8.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x16.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x24.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x32.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x40.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x48.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x56.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x64.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x72.c
  src/f32-velu/gen/velu-avx2-rr1-lut4-p4-perm-x80.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x8.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x16.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x24.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x32.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x40.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x48.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x56.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x64.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x72.c
  src/f32-velu/gen/velu-avx2-rr1-lut8-p4-perm-x80.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x8.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x16.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x24.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x32.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x40.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x48.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x56.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x64.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x72.c
  src/f32-velu/gen/velu-avx2-rr1-lut16-p3-gather-x80.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x8.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x16.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x24.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x32.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x40.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x48.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x56.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x64.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x72.c
  src/f32-velu/gen/velu-avx2-rr1-p6-x80.c
  src/f32-vrelu/gen/vrelu-avx-x8.c
  src/f32-vrelu/gen/vrelu-avx-x16.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x8.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x16.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x24.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x32.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x40.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x48.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x56.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x64.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x72.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x80.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x88.c
  src/f32-vscaleexpminusmax/gen/avx2-p5-x96.c
  src/f32-vscaleextexp/gen/avx2-p5-x8.c
  src/f32-vscaleextexp/gen/avx2-p5-x16.c
  src/f32-vscaleextexp/gen/avx2-p5-x24.c
  src/f32-vscaleextexp/gen/avx2-p5-x32.c
  src/f32-vscaleextexp/gen/avx2-p5-x40.c
  src/f32-vscaleextexp/gen/avx2-p5-x48.c
  src/f32-vscaleextexp/gen/avx2-p5-x56.c
  src/f32-vscaleextexp/gen/avx2-p5-x64.c
  src/f32-vscaleextexp/gen/avx2-p5-x72.c
  src/f32-vscaleextexp/gen/avx2-p5-x80.c
  src/f32-vscaleextexp/gen/avx2-p5-x88.c
  src/f32-vscaleextexp/gen/avx2-p5-x96.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x40.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x56.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x72.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x40.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x56.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x72.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x8.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x24.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x40.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x56.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x72.c
  src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x80.c
  src/math/exp-f32-avx2-rr2-lut8-p3-perm.c
  src/math/exp-f32-avx2-rr2-lut8-p4-perm.c
  src/math/exp-f32-avx2-rr2-p5.c
  src/math/expminus-f16-avx2-rr1-p2.c
  src/math/expminus-f16-avx2-rr1-p3.c
  src/math/expminus-f32-avx2-rr1-p5.c
  src/math/expminus-f32-avx2-rr2-p5.c
  src/math/expm1minus-f16-avx2-rr1-p3.c
  src/math/expm1minus-f32-avx2-rr1-lut4-p4-perm.c
  src/math/expm1minus-f32-avx2-rr1-lut8-p4-perm.c
  src/math/expm1minus-f32-avx2-rr1-lut16-p3-gather.c
  src/math/expm1minus-f32-avx2-rr1-p6.c
  src/math/extexp-avx2-p5.c
  src/math/sigmoid-f16-avx2-rr1-p2-div.c
  src/math/sigmoid-f16-avx2-rr1-p2-rcp.c
  src/math/sigmoid-f16-avx2-rr1-p3-div.c
  src/math/sigmoid-f16-avx2-rr1-p3-rcp.c
  src/math/sigmoid-f32-avx2-rr1-lut64-p2-gather-div.c
  src/math/sigmoid-f32-avx2-rr1-lut64-p2-gather-nr1fma.c
  src/math/sigmoid-f32-avx2-rr1-lut64-p2-gather-nr2fma.c
  src/math/sigmoid-f32-avx2-rr1-lut64-p2-gather-nr2fma1adj.c
  src/math/sigmoid-f32-avx2-rr1-p5-div.c
  src/math/sigmoid-f32-avx2-rr1-p5-nr1fma.c
  src/math/sigmoid-f32-avx2-rr1-p5-nr2fma.c
  src/math/sigmoid-f32-avx2-rr2-lut64-p2-gather-div.c
  src/math/sigmoid-f32-avx2-rr2-lut64-p2-gather-nr1fma.c
  src/math/sigmoid-f32-avx2-rr2-lut64-p2-gather-nr2fma.c
  src/math/sigmoid-f32-avx2-rr2-lut64-p2-gather-nr2fma1adj.c
  src/math/sigmoid-f32-avx2-rr2-p5-div.c
  src/math/sigmoid-f32-avx2-rr2-p5-nr1fma.c
  src/math/sigmoid-f32-avx2-rr2-p5-nr2fma.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up16x3-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpunpck.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpunpck.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpunpck.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpunpck.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul32.c
  src/qc8-gemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qc8-gemm/gen/1x8c8-xw-minmax-fp32-avx2.c
  src/qc8-gemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qc8-gemm/gen/2x8c8-xw-minmax-fp32-avx2.c
  src/qc8-gemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qc8-gemm/gen/3x8c8-xw-minmax-fp32-avx2.c
  src/qc8-igemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qc8-igemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qc8-igemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul32.c
  src/qs8-f32-vcvt/gen/vcvt-avx2-x8.c
  src/qs8-f32-vcvt/gen/vcvt-avx2-x16.c
  src/qs8-f32-vcvt/gen/vcvt-avx2-x24.c
  src/qs8-f32-vcvt/gen/vcvt-avx2-x32.c
  src/qs8-gemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qs8-gemm/gen/1x8c8-xw-minmax-fp32-avx2.c
  src/qs8-gemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qs8-gemm/gen/2x8c8-xw-minmax-fp32-avx2.c
  src/qs8-gemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-gemm/gen/3x8c8-xw-minmax-fp32-avx2.c
  src/qs8-igemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qs8-igemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qs8-igemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x8.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x16.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x24.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x32.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x8.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x16.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x24.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x32.c
  src/qs8-vcvt/gen/vcvt-avx2-x16.c
  src/qs8-vcvt/gen/vcvt-avx2-x32.c
  src/qs8-vcvt/gen/vcvt-avx2-x64.c
  src/qs8-vlrelu/gen/vlrelu-avx2-x16.c
  src/qs8-vlrelu/gen/vlrelu-avx2-x32.c
  src/qs8-vlrelu/gen/vlrelu-avx2-x64.c
  src/qu8-dwconv/gen/up8x9-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/up32x25-minmax-fp32-avx2-mul32.c
  src/qu8-f32-vcvt/gen/vcvt-avx2-x8.c
  src/qu8-f32-vcvt/gen/vcvt-avx2-x16.c
  src/qu8-f32-vcvt/gen/vcvt-avx2-x24.c
  src/qu8-f32-vcvt/gen/vcvt-avx2-x32.c
  src/qu8-gemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qu8-gemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qu8-gemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qu8-vadd/gen/minmax-avx2-mul32-ld64-x8.c
  src/qu8-vadd/gen/minmax-avx2-mul32-ld64-x16.c
  src/qu8-vaddc/gen/minmax-avx2-mul32-ld64-x8.c
  src/qu8-vaddc/gen/minmax-avx2-mul32-ld64-x16.c
  src/qu8-vcvt/gen/vcvt-avx2-x16.c
  src/qu8-vcvt/gen/vcvt-avx2-x32.c
  src/qu8-vcvt/gen/vcvt-avx2-x64.c
  src/qu8-vlrelu/gen/vlrelu-avx2-x16.c
  src/qu8-vlrelu/gen/vlrelu-avx2-x32.c
  src/qu8-vlrelu/gen/vlrelu-avx2-x64.c
  src/x8-lut/gen/lut-avx2-x32.c
  src/x8-lut/gen/lut-avx2-x64.c
  src/x8-lut/gen/lut-avx2-x96.c
  src/x8-lut/gen/lut-avx2-x128.c)

SET(PROD_AVX512F_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up16x3-minmax-avx512f.c
  src/f32-dwconv/gen/up16x4-minmax-avx512f.c
  src/f32-dwconv/gen/up16x9-minmax-avx512f.c
  src/f32-dwconv/gen/up16x25-minmax-avx512f.c
  src/f32-gemm/gen/1x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/7x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/1x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/7x16-minmax-avx512f-broadcast.c
  src/f32-prelu/gen/avx512f-2x16.c
  src/f32-vbinary/gen/vadd-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vaddc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vdiv-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vdivc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vmax-avx512f-x32.c
  src/f32-vbinary/gen/vmaxc-avx512f-x32.c
  src/f32-vbinary/gen/vmin-avx512f-x32.c
  src/f32-vbinary/gen/vminc-avx512f-x32.c
  src/f32-vbinary/gen/vmul-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vmulc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vrdivc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vrsubc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vsqrdiff-avx512f-x32.c
  src/f32-vbinary/gen/vsqrdiffc-avx512f-x32.c
  src/f32-vbinary/gen/vsub-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vsubc-minmax-avx512f-x32.c
  src/f32-vclamp/gen/vclamp-avx512f-x16.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x64.c
  src/f32-vhswish/gen/vhswish-avx512f-x16.c
  src/f32-vlrelu/gen/vlrelu-avx512f-x16.c
  src/f32-vrnd/gen/vrndd-avx512f-x16.c
  src/f32-vrnd/gen/vrndne-avx512f-x16.c
  src/f32-vrnd/gen/vrndu-avx512f-x16.c
  src/f32-vrnd/gen/vrndz-avx512f-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x64.c
  src/f32-vunary/gen/vabs-avx512f-x16.c
  src/f32-vunary/gen/vneg-avx512f-x16.c
  src/f32-vunary/gen/vsqr-avx512f-x16.c)

SET(ALL_AVX512F_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up16x3-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x3-minmax-avx512f.c
  src/f32-dwconv/gen/up16x4-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-avx512f.c
  src/f32-dwconv/gen/up16x9-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-avx512f.c
  src/f32-dwconv/gen/up16x25-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-avx512f.c
  src/f32-dwconv/gen/up32x3-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up32x3-minmax-avx512f.c
  src/f32-dwconv/gen/up32x4-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up32x4-minmax-avx512f.c
  src/f32-dwconv/gen/up32x9-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up32x9-minmax-avx512f.c
  src/f32-dwconv/gen/up32x25-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up32x25-minmax-avx512f.c
  src/f32-gemm/gen-inc/1x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/4x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/5x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/6x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/7x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/8x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/1x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/4x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/5x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/6x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/7x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/8x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/1x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/4x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/5x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/6x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/7x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/8x16-minmax-avx512f-broadcast.c
  src/f32-prelu/gen/avx512f-2x16.c
  src/f32-prelu/gen/avx512f-2x32.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x128-acc2.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x128-acc4.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x128.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x144-acc3.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x144.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x160-acc2.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x160-acc5.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x160.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192-acc2.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192-acc3.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192-acc6.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x128-acc2.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x128-acc4.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x128.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x144-acc3.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x144.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x160-acc2.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x160-acc5.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x160.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc2.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc3.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc6.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x128-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x128-acc4.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x128.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x144-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x144.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x160-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x160-acc5.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x160.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x192-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x192-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x192-acc6.c
  src/f32-raddstoreexpminusmax/gen/avx512f-rr1-p5-scalef-x192.c
  src/f32-rmax/avx512f.c
  src/f32-vbinary/gen/vadd-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vadd-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vaddc-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vaddc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vdiv-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vdiv-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vdivc-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vdivc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vmax-avx512f-x16.c
  src/f32-vbinary/gen/vmax-avx512f-x32.c
  src/f32-vbinary/gen/vmaxc-avx512f-x16.c
  src/f32-vbinary/gen/vmaxc-avx512f-x32.c
  src/f32-vbinary/gen/vmin-avx512f-x16.c
  src/f32-vbinary/gen/vmin-avx512f-x32.c
  src/f32-vbinary/gen/vminc-avx512f-x16.c
  src/f32-vbinary/gen/vminc-avx512f-x32.c
  src/f32-vbinary/gen/vmul-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vmul-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vmulc-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vmulc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vrdivc-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vrdivc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vrsubc-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vrsubc-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vsqrdiff-avx512f-x16.c
  src/f32-vbinary/gen/vsqrdiff-avx512f-x32.c
  src/f32-vbinary/gen/vsqrdiffc-avx512f-x16.c
  src/f32-vbinary/gen/vsqrdiffc-avx512f-x32.c
  src/f32-vbinary/gen/vsub-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vsub-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vsubc-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vsubc-minmax-avx512f-x32.c
  src/f32-vclamp/gen/vclamp-avx512f-x16.c
  src/f32-vclamp/gen/vclamp-avx512f-x32.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x16.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x32.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x48.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x64.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x80.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x96.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x112.c
  src/f32-velu/gen/velu-avx512f-rr1-lut16-p3-perm-x128.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x16.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x32.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x48.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x64.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x80.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x96.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x112.c
  src/f32-velu/gen/velu-avx512f-rr1-p6-x128.c
  src/f32-vhswish/gen/vhswish-avx512f-x16.c
  src/f32-vhswish/gen/vhswish-avx512f-x32.c
  src/f32-vlrelu/gen/vlrelu-avx512f-x16.c
  src/f32-vlrelu/gen/vlrelu-avx512f-x32.c
  src/f32-vrelu/gen/vrelu-avx512f-x16.c
  src/f32-vrelu/gen/vrelu-avx512f-x32.c
  src/f32-vrnd/gen/vrndd-avx512f-x16.c
  src/f32-vrnd/gen/vrndd-avx512f-x32.c
  src/f32-vrnd/gen/vrndne-avx512f-x16.c
  src/f32-vrnd/gen/vrndne-avx512f-x32.c
  src/f32-vrnd/gen/vrndu-avx512f-x16.c
  src/f32-vrnd/gen/vrndu-avx512f-x32.c
  src/f32-vrnd/gen/vrndz-avx512f-x16.c
  src/f32-vrnd/gen/vrndz-avx512f-x32.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x16.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x32.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x48.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x64.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x80.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x96.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x112.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x128.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x144.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x160.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x176.c
  src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x192.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x16.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x32.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x48.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x64.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x80.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x96.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x112.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x128.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x144.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x160.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x176.c
  src/f32-vscaleextexp/gen/avx512f-p5-scalef-x192.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x96.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x112.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x128.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x96.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x112.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x128.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x96.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x112.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x128.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x96.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x112.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x128.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x96.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x112.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x128.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x16.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x32.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x48.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x64.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x80.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x96.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x112.c
  src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x128.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x16.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x32.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x48.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x64.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x80.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x96.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x112.c
  src/f32-vsqrt/gen/avx512f-nr1fma1adj-x128.c
  src/f32-vunary/gen/vabs-avx512f-x16.c
  src/f32-vunary/gen/vabs-avx512f-x32.c
  src/f32-vunary/gen/vneg-avx512f-x16.c
  src/f32-vunary/gen/vneg-avx512f-x32.c
  src/f32-vunary/gen/vsqr-avx512f-x16.c
  src/f32-vunary/gen/vsqr-avx512f-x32.c
  src/math/exp-f32-avx512f-rr2-lut16-p3-perm-scalef.c
  src/math/exp-f32-avx512f-rr2-lut16-p3-perm.c
  src/math/exp-f32-avx512f-rr2-lut32-p2-perm2-scalef.c
  src/math/exp-f32-avx512f-rr2-lut32-p2-perm2.c
  src/math/exp-f32-avx512f-rr2-p5-scalef.c
  src/math/exp-f32-avx512f-rr2-p5.c
  src/math/expm1minus-f32-avx512f-rr1-lut16-p3-perm.c
  src/math/expm1minus-f32-avx512f-rr1-p6.c
  src/math/extexp-avx512f-p5.c
  src/math/sigmoid-f32-avx512f-rr1-lut16-p3-perm-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr1-lut16-p3-perm-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr1-lut16-p3-perm-scalef-nr1fma1adj.c
  src/math/sigmoid-f32-avx512f-rr1-lut32-p2-perm2-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr1-lut32-p2-perm2-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr1-lut32-p2-perm2-scalef-nr1fma1adj.c
  src/math/sigmoid-f32-avx512f-rr1-lut64-p2-gather-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr1-lut64-p2-gather-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr1-lut64-p2-gather-scalef-nr1fma1adj.c
  src/math/sigmoid-f32-avx512f-rr1-p5-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr1-p5-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr1-p5-scalef-nr1fma1adj.c
  src/math/sigmoid-f32-avx512f-rr2-lut16-p3-perm-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr2-lut16-p3-perm-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr2-lut16-p3-perm-scalef-nr1fma1adj.c
  src/math/sigmoid-f32-avx512f-rr2-lut32-p2-perm2-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma1adj.c
  src/math/sigmoid-f32-avx512f-rr2-lut64-p2-gather-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr2-lut64-p2-gather-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr2-lut64-p2-gather-scalef-nr1fma1adj.c
  src/math/sigmoid-f32-avx512f-rr2-p5-scalef-div.c
  src/math/sigmoid-f32-avx512f-rr2-p5-scalef-nr1fma.c
  src/math/sigmoid-f32-avx512f-rr2-p5-scalef-nr1fma1adj.c
  src/math/sqrt-avx512f-nr1fma.c
  src/math/sqrt-avx512f-nr1fma1adj.c
  src/math/sqrt-avx512f-nr2fma.c)

SET(PROD_AVX512SKX_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-avx512skx-x16.c
  src/f32-f16-vcvt/gen/vcvt-avx512skx-x16.c
  src/f32-qs8-vcvt/gen/vcvt-avx512skx-x128.c
  src/f32-qu8-vcvt/gen/vcvt-avx512skx-x128.c
  src/qc8-dwconv/gen/up32x3-minmax-fp32-avx512skx-mul32.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c
  src/qc8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qc8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qc8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qc8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c
  src/qs8-f32-vcvt/gen/vcvt-avx512skx-x32.c
  src/qs8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-vadd/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/qs8-vaddc/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/qu8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c
  src/qu8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c
  src/qu8-f32-vcvt/gen/vcvt-avx512skx-x32.c
  src/qu8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qu8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qu8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qu8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qu8-vadd/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/qu8-vaddc/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/x8-lut/gen/lut-avx512skx-vpshufb-x64.c)

SET(ALL_AVX512SKX_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/vcvt-avx512skx-x16.c
  src/f16-f32-vcvt/gen/vcvt-avx512skx-x32.c
  src/f32-f16-vcvt/gen/vcvt-avx512skx-x16.c
  src/f32-f16-vcvt/gen/vcvt-avx512skx-x32.c
  src/f32-qs8-vcvt/gen/vcvt-avx512skx-x32.c
  src/f32-qs8-vcvt/gen/vcvt-avx512skx-x64.c
  src/f32-qs8-vcvt/gen/vcvt-avx512skx-x96.c
  src/f32-qs8-vcvt/gen/vcvt-avx512skx-x128.c
  src/f32-qu8-vcvt/gen/vcvt-avx512skx-x32.c
  src/f32-qu8-vcvt/gen/vcvt-avx512skx-x64.c
  src/f32-qu8-vcvt/gen/vcvt-avx512skx-x96.c
  src/f32-qu8-vcvt/gen/vcvt-avx512skx-x128.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx512skx-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c
  src/qc8-dwconv/gen/up32x3-minmax-fp32-avx512skx-mul32.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c
  src/qc8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qc8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qc8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qc8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qc8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qc8-igemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qc8-igemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qc8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx512skx-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c
  src/qs8-f32-vcvt/gen/vcvt-avx512skx-x16.c
  src/qs8-f32-vcvt/gen/vcvt-avx512skx-x32.c
  src/qs8-f32-vcvt/gen/vcvt-avx512skx-x48.c
  src/qs8-f32-vcvt/gen/vcvt-avx512skx-x64.c
  src/qs8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-vadd/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/qs8-vadd/gen/minmax-avx512skx-mul32-ld128-x32.c
  src/qs8-vaddc/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/qs8-vaddc/gen/minmax-avx512skx-mul32-ld128-x32.c
  src/qu8-dwconv/gen/up16x9-minmax-fp32-avx512skx-mul32.c
  src/qu8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c
  src/qu8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c
  src/qu8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c
  src/qu8-f32-vcvt/gen/vcvt-avx512skx-x16.c
  src/qu8-f32-vcvt/gen/vcvt-avx512skx-x32.c
  src/qu8-f32-vcvt/gen/vcvt-avx512skx-x48.c
  src/qu8-f32-vcvt/gen/vcvt-avx512skx-x64.c
  src/qu8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qu8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qu8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qu8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qu8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qu8-igemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qu8-igemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qu8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qu8-vadd/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/qu8-vadd/gen/minmax-avx512skx-mul32-ld128-x32.c
  src/qu8-vaddc/gen/minmax-avx512skx-mul32-ld128-x16.c
  src/qu8-vaddc/gen/minmax-avx512skx-mul32-ld128-x32.c
  src/x8-lut/gen/lut-avx512skx-vpshufb-x64.c
  src/x8-lut/gen/lut-avx512skx-vpshufb-x128.c
  src/x8-lut/gen/lut-avx512skx-vpshufb-x192.c
  src/x8-lut/gen/lut-avx512skx-vpshufb-x256.c)

SET(AARCH32_ASM_MICROKERNEL_SRCS
  src/cs16-bfly4/samples1-aarch32-neon-x1.S
  src/cs16-bfly4/samples1-aarch32-neon-x2.S
  src/cs16-bfly4/samples1-aarch32-neon-x4.S
  src/cs16-fftr/aarch32-neon-x1.S
  src/cs16-fftr/aarch32-neon-x4.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a7.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a75.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-ld64.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-prfm-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-prfm-cortex-a75.S
  src/f32-gemm/4x4-aarch32-vfp-ld64.S
  src/f32-gemm/4x4-minmax-aarch32-vfp-ld64.S
  src/f32-gemm/4x8-minmax-aarch32-neon-cortex-a55.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-cortex-a7.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-cortex-a53.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-cortex-a75.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-ld64.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-prfm-cortex-a53.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-prfm-cortex-a75.S
  src/f32-igemm/4x8-minmax-aarch32-neon-cortex-a55.S
  src/qc8-dwconv/up8x3-minmax-fp32-aarch32-neonv8-mla8-cortex-a35.S
  src/qc8-dwconv/up16x3-minmax-fp32-aarch32-neonv8-mla8-cortex-a35.S
  src/qc8-gemm/gen/1x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a7.S
  src/qc8-gemm/gen/1x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qc8-gemm/gen/1x8-minmax-fp32-aarch32-neonv8-mlal-lane-cortex-a35.S
  src/qc8-gemm/gen/1x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-cortex-a35.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a7.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a53.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-ld64.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a53.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-ld64.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-cortex-a35.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-cortex-a53.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-ld64.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-cortex-a35.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-cortex-a53.S
  src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-ld64.S
  src/qc8-gemm/gen/4x8c4-minmax-fp32-aarch32-neondot-cortex-a55.S
  src/qc8-gemm/gen/4x8c4-minmax-fp32-aarch32-neondot-ld64.S
  src/qc8-igemm/gen/1x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a7.S
  src/qc8-igemm/gen/1x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qc8-igemm/gen/1x8-minmax-fp32-aarch32-neonv8-mlal-lane-cortex-a35.S
  src/qc8-igemm/gen/1x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-cortex-a35.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a7.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a53.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-ld64.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a53.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-ld64.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-cortex-a35.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-cortex-a53.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-ld64.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-cortex-a35.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-cortex-a53.S
  src/qc8-igemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-ld64.S
  src/qc8-igemm/gen/4x8c4-minmax-fp32-aarch32-neondot-cortex-a55.S
  src/qc8-igemm/gen/4x8c4-minmax-fp32-aarch32-neondot-ld64.S
  src/qs8-gemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qs8-gemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a53.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-ld64.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-ld64.S
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-aarch32-neondot-cortex-a55.S
  src/qs8-gemm/gen/4x8c4-minmax-rndnu-aarch32-neondot-ld64.S
  src/qs8-igemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qs8-igemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a53.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-ld64.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-ld64.S
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-aarch32-neondot-cortex-a55.S
  src/qs8-igemm/gen/4x8c4-minmax-rndnu-aarch32-neondot-ld64.S
  src/qu8-gemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qu8-gemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a53.S
  src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-ld64.S
  src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a53.S
  src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-ld64.S
  src/qu8-igemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qu8-igemm/gen/1x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qu8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S
  src/qu8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a53.S
  src/qu8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-ld64.S
  src/qu8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S
  src/qu8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a53.S
  src/qu8-igemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-ld64.S
  src/u32-filterbank-accumulate/aarch32-arm-x1.S
  src/u32-filterbank-accumulate/aarch32-neon-x1.S
  src/u32-filterbank-accumulate/aarch32-neon-x2.S)

SET(AARCH64_ASM_MICROKERNEL_SRCS
  src/f16-gemm/gen-inc/1x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen-inc/1x16inc-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen-inc/4x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen-inc/4x16inc-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen-inc/6x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen-inc/6x16inc-minmax-aarch64-neonfp16arith-cortex-a55.S
  src/f16-gemm/gen-inc/6x16inc-minmax-aarch64-neonfp16arith-cortex-a75.S
  src/f16-gemm/gen-inc/6x16inc-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen-inc/8x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/1x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/1x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/1x16-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/4x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/4x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/4x16-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/6x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-cortex-a55.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-cortex-a55r0.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-cortex-a75.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/8x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-igemm/1x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-igemm/1x16-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-igemm/4x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-igemm/4x16-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-igemm/6x16-minmax-aarch64-neonfp16arith-cortex-a55.S
  src/f16-igemm/6x16-minmax-aarch64-neonfp16arith-cortex-a55r0.S
  src/f16-igemm/6x16-minmax-aarch64-neonfp16arith-cortex-a75.S
  src/f16-igemm/6x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-igemm/6x16-minmax-aarch64-neonfp16arith-ld64.S
  src/f32-dwconv/up4x9-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-dwconv/up4x9-minmax-aarch64-neonfma.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen-inc/1x12inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen-inc/4x12inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/5x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/5x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a73.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a53.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen/1x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/4x2-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/4x2-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/4x2-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen/4x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/5x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/5x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a73.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a53.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a53.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/4x2-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/4x2-minmax-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/4x2-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-ld128.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a53.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/5x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/5x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-ld128.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a53.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/1x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/4x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/4x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a73.S
  src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qc8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld32.S
  src/qc8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld64.S
  src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mull.S
  src/qc8-gemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal.S
  src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S
  src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-ld64.S
  src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
  src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld32.S
  src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S
  src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S
  src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qc8-igemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal.S
  src/qc8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S
  src/qc8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-ld64.S
  src/qc8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qc8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qc8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
  src/qc8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S
  src/qc8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S
  src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-cortex-a53.S
  src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-prfm.S
  src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal.S
  src/qs8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld32.S
  src/qs8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld64.S
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-aarch64-neondot-ld32.S
  src/qs8-gemm/gen/1x16c4-minmax-rndnu-aarch64-neondot-ld64.S
  src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mull.S
  src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-cortex-a53.S
  src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-prfm.S
  src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal.S
  src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mull.S
  src/qs8-gemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal.S
  src/qs8-gemm/gen/2x8c16-minmax-rndnu-aarch64-neon-mlal.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S
  src/qs8-gemm/gen/4x8-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-ld64.S
  src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S
  src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
  src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld32.S
  src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S
  src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-cortex-a55.S
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld32.S
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld64.S
  src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld128.S
  src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qs8-igemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-cortex-a53.S
  src/qs8-igemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-igemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-prfm.S
  src/qs8-igemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal.S
  src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S
  src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S
  src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal.S
  src/qs8-igemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-cortex-a53.S
  src/qs8-igemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-prfm-cortex-a53.S
  src/qs8-igemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-prfm.S
  src/qs8-igemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal.S
  src/qs8-igemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal.S
  src/qs8-igemm/gen/2x8c16-minmax-rndnu-aarch64-neon-mlal.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S
  src/qs8-igemm/gen/4x8-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qs8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-ld64.S
  src/qs8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qs8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S
  src/qs8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qs8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
  src/qs8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S
  src/qs8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-cortex-a55.S
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld64.S
  src/qs8-igemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld128.S
  src/qu8-gemm/gen/4x8c4-minmax-rndnu-aarch64-neondot-cortex-a55.S
  src/qu8-gemm/gen/4x8c4-minmax-rndnu-aarch64-neondot-ld128.S
  src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a53.S
  src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a75.S
  src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S
  src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a75.S
  src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qu8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
  src/qu8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S
  src/qu8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-cortex-a55.S
  src/qu8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld128.S
  src/qu8-igemm/gen/4x8c4-minmax-rndnu-aarch64-neondot-cortex-a55.S
  src/qu8-igemm/gen/4x8c4-minmax-rndnu-aarch64-neondot-ld128.S
  src/qu8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a53.S
  src/qu8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a75.S
  src/qu8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S
  src/qu8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qu8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a75.S
  src/qu8-igemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S
  src/qu8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
  src/qu8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S
  src/qu8-igemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-cortex-a55.S
  src/qu8-igemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld128.S)

SET(PROD_MICROKERNEL_SRCS ${PROD_SCALAR_PORTABLE_MICROKERNEL_SRCS})
SET(ALL_MICROKERNEL_SRCS ${ALL_SCALAR_MICROKERNEL_SRCS})
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_SCALAR_AARCH32_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_ARMSIMD32_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEON_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONFP16_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONV8_MICROKERNEL_SRCS})
  IF(XNNPACK_ENABLE_ARM_FP16)
    LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONFP16ARITH_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ARM_DOTPROD)
    LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONDOT_MICROKERNEL_SRCS})
  ENDIF()
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_ARMSIMD32_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEON_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONFP16_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONV8_MICROKERNEL_SRCS})
  IF(XNNPACK_ENABLE_ARM_FP16)
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONFP16ARITH_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ARM_BF16)
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONBF16_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ARM_DOTPROD)
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONDOT_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ASSEMBLY)
    LIST(APPEND PROD_MICROKERNEL_SRCS ${AARCH32_ASM_MICROKERNEL_SRCS})
    LIST(APPEND ALL_MICROKERNEL_SRCS ${AARCH32_ASM_MICROKERNEL_SRCS})
  ENDIF()
  LIST(APPEND JIT_SRCS ${JIT_AARCH32_SRCS})
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(aarch64|arm64)$" OR IOS_ARCH MATCHES "^arm64.*")
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEON_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONFP16_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONV8_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_AARCH64_NEON_MICROKERNEL_SRCS})
  IF(XNNPACK_ENABLE_ARM_FP16)
    LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONFP16ARITH_MICROKERNEL_SRCS})
    LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ARM_DOTPROD)
    LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_NEONDOT_MICROKERNEL_SRCS})
  ENDIF()
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEON_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONFP16_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONV8_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_AARCH64_NEON_MICROKERNEL_SRCS})
  IF(XNNPACK_ENABLE_ARM_FP16)
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_FP16ARITH_MICROKERNEL_SRCS})
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONFP16ARITH_MICROKERNEL_SRCS})
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ARM_BF16)
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONBF16_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ARM_DOTPROD)
    LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_NEONDOT_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ASSEMBLY)
    LIST(APPEND PROD_MICROKERNEL_SRCS ${AARCH64_ASM_MICROKERNEL_SRCS})
    LIST(APPEND ALL_MICROKERNEL_SRCS ${AARCH64_ASM_MICROKERNEL_SRCS})
  ENDIF()
  LIST(APPEND JIT_SRCS ${JIT_AARCH64_SRCS})
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(i[3-6]86|x86_64|AMD64)$" OR IOS_ARCH MATCHES "^(i386|x86_64|AMD64)$")
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_SSE_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_SSE2_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_SSSE3_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_SSE41_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_AVX_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_F16C_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_XOP_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_FMA3_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_AVX2_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_AVX512F_MICROKERNEL_SRCS})
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_AVX512SKX_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_SSE_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_SSE2_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_SSSE3_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_SSE41_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_AVX_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_F16C_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_XOP_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_FMA3_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_AVX2_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_AVX512F_MICROKERNEL_SRCS})
  LIST(APPEND ALL_MICROKERNEL_SRCS ${ALL_AVX512SKX_MICROKERNEL_SRCS})
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^riscv(32|64)$")
  LIST(APPEND PROD_MICROKERNEL_SRCS ${PROD_SCALAR_RISCV_MICROKERNEL_SRCS})
ENDIF()
ADD_LIBRARY(all_microkernels OBJECT ${ALL_MICROKERNEL_SRCS} ${TABLE_SRCS})
ADD_LIBRARY(allocator OBJECT ${ALLOCATOR_SRCS})
# Need C_EXTENSIONS to get constants for mmap (MAP_ANONYMOUS).
SET_TARGET_PROPERTIES(allocator PROPERTIES C_EXTENSIONS YES)
ADD_LIBRARY(indirection OBJECT src/indirection.c)
ADD_LIBRARY(microparams_init OBJECT src/microparams-init.c)
ADD_LIBRARY(normalization OBJECT src/normalization.c)
ADD_LIBRARY(packing OBJECT src/packing.c)
ADD_LIBRARY(jit OBJECT ${JIT_SRCS})
ADD_LIBRARY(logging OBJECT ${LOGGING_SRCS})
ADD_LIBRARY(cache OBJECT src/cache.c)
ADD_LIBRARY(subgraph OBJECT ${SUBGRAPH_SRCS})
ADD_LIBRARY(operators OBJECT ${OPERATOR_SRCS})
ADD_LIBRARY(mutex OBJECT src/mutex.c)
ADD_LIBRARY(convolution-test-helpers OBJECT test/convolution-test-helpers.cc)
ADD_LIBRARY(post-operation OBJECT src/operators/post-operation.c)
IF(XNNPACK_LIBRARY_TYPE STREQUAL "default")
  ADD_LIBRARY(XNNPACK ${COLD_SRCS} ${HOT_SRCS} ${TABLE_SRCS} ${JIT_SRCS} ${PROD_MICROKERNEL_SRCS})
ELSEIF(XNNPACK_LIBRARY_TYPE STREQUAL "shared")
  ADD_LIBRARY(XNNPACK SHARED ${COLD_SRCS} ${HOT_SRCS} ${TABLE_SRCS} ${JIT_SRCS} ${PROD_MICROKERNEL_SRCS})
ELSEIF(XNNPACK_LIBRARY_TYPE STREQUAL "static")
  ADD_LIBRARY(XNNPACK STATIC ${COLD_SRCS} ${HOT_SRCS} ${TABLE_SRCS} ${JIT_SRCS} ${PROD_MICROKERNEL_SRCS})
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported XNNPACK library type \"${XNNPACK_LIBRARY_TYPE}\". Must be \"static\", \"shared\", or \"default\"")
ENDIF()
SET_TARGET_PROPERTIES(XNNPACK PROPERTIES
  C_EXTENSIONS YES)
IF(NOT MSVC)
  SET_PROPERTY(SOURCE ${PROD_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -fno-math-errno ")
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  SET_PROPERTY(SOURCE ${ALL_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -marm ")
  SET_PROPERTY(SOURCE ${ALL_ARMSIMD32_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv6 -mfpu=vfp -munaligned-access ")
  SET_PROPERTY(SOURCE ${ALL_NEON_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon ")
  SET_PROPERTY(SOURCE ${ALL_NEONFP16_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon-fp16 ")
  # GCC requires -mfp16-format=ieee to define __fp16 type, but Clang doesn't support this option at all.
  IF(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    SET_PROPERTY(SOURCE ${ALL_NEONFP16_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfp16-format=ieee ")
    SET_PROPERTY(SOURCE ${ALL_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfp16-format=ieee ")
  ENDIF()
  SET_PROPERTY(SOURCE ${ALL_NEONFMA_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon-vfpv4 ")
  SET_PROPERTY(SOURCE ${ALL_NEONV8_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8-a -mfpu=neon-fp-armv8 ")
  SET_PROPERTY(SOURCE ${ALL_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8 ")
  SET_PROPERTY(SOURCE ${ALL_NEONBF16_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+bf16 -mfpu=neon-fp-armv8 ")
  SET_PROPERTY(SOURCE ${ALL_NEONDOT_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+dotprod -mfpu=neon-fp-armv8 ")
  SET_PROPERTY(SOURCE ${AARCH32_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+dotprod -mfpu=neon-fp-armv8 ")
  # Workground the neon detection bug in ARM v8
  # Related links:
  #   https://github.com/android/ndk/issues/910
  #   https://reviews.llvm.org/D58477
  IF(ANDROID_NDK_MAJOR AND ANDROID_NDK_MAJOR LESS 21)
    SET_PROPERTY(SOURCE ${ALL_NEONV8_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
    SET_PROPERTY(SOURCE ${ALL_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
    SET_PROPERTY(SOURCE ${ALL_NEONBF16_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
    SET_PROPERTY(SOURCE ${ALL_NEONDOT_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
    SET_PROPERTY(SOURCE ${AARCH32_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
  ENDIF()
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(aarch64|arm64)$" OR IOS_ARCH MATCHES "^arm64.*")
  SET_PROPERTY(SOURCE ${ALL_FP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  SET_PROPERTY(SOURCE ${ALL_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  SET_PROPERTY(SOURCE ${ALL_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  SET_PROPERTY(SOURCE ${ALL_NEONBF16_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+bf16 ")
  SET_PROPERTY(SOURCE ${ALL_NEONDOT_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+dotprod ")
  SET_PROPERTY(SOURCE ${AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16+dotprod ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
  IF(XNNPACK_TARGET_PROCESSOR STREQUAL "arm64" AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET_PROPERTY(SOURCE ${AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch arm64 ")
  ENDIF()
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(i[3-6]86|x86|x86_64|AMD64)$" OR IOS_ARCH MATCHES "^(i386|x86_64|AMD64)$")
  IF(MSVC)
    IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86" OR CMAKE_SIZEOF_VOID_P EQUAL 4)
      SET_PROPERTY(SOURCE ${ALL_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE ")
      SET_PROPERTY(SOURCE ${ALL_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
      SET_PROPERTY(SOURCE ${ALL_SSSE3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
      SET_PROPERTY(SOURCE ${ALL_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
    ENDIF()
    SET_PROPERTY(SOURCE ${ALL_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${ALL_F16C_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${ALL_XOP_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${ALL_FMA3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${ALL_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX2 ")
    SET_PROPERTY(SOURCE ${ALL_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX512 ")
    SET_PROPERTY(SOURCE ${ALL_AVX512SKX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX512 ")
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      SET_PROPERTY(SOURCE ${ALL_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-msse ")
      SET_PROPERTY(SOURCE ${ALL_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-msse2 ")
      SET_PROPERTY(SOURCE ${ALL_SSSE3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-mssse3 ")
      SET_PROPERTY(SOURCE ${ALL_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-msse4.1 ")
      SET_PROPERTY(SOURCE ${ALL_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-mavx ")
      SET_PROPERTY(SOURCE ${ALL_F16C_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-mf16c ")
      SET_PROPERTY(SOURCE ${ALL_XOP_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-mxop ")
      SET_PROPERTY(SOURCE ${ALL_FMA3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-mf16c -clang:-mfma ")
      SET_PROPERTY(SOURCE ${ALL_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-mf16c -clang:-mfma -clang:-mavx2 ")
      SET_PROPERTY(SOURCE ${ALL_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -clang:-mavx512f ")
      SET_PROPERTY(SOURCE ${ALL_AVX512SKX_MICROKERNEL_SRCS} APPEND_STRIDE PROPERTY COMPILE_FLAGS " -clang:-mavx512f -clang:-mavx512cd -clang:-mavx512bw -clang:-mavx512dq -clang:-mavx512vl ")
    ENDIF()
  ELSE()
    SET_PROPERTY(SOURCE ${ALL_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse ")
    SET_PROPERTY(SOURCE ${ALL_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse2 ")
    SET_PROPERTY(SOURCE ${ALL_SSSE3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mssse3 ")
    SET_PROPERTY(SOURCE ${ALL_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse4.1 ")
    SET_PROPERTY(SOURCE ${ALL_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx ")
    SET_PROPERTY(SOURCE ${ALL_F16C_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mf16c ")
    SET_PROPERTY(SOURCE ${ALL_XOP_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mxop ")
    SET_PROPERTY(SOURCE ${ALL_FMA3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mf16c -mfma ")
    SET_PROPERTY(SOURCE ${ALL_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mf16c -mfma -mavx2 ")
    SET_PROPERTY(SOURCE ${ALL_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx512f ")
    SET_PROPERTY(SOURCE ${ALL_AVX512SKX_MICROKERNEL_SRCS} APPEND_STRIDE PROPERTY COMPILE_FLAGS " -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl ")
    IF(MINGW OR CMAKE_SYSTEM_NAME MATCHES "^(CYGWIN|MSYS)$")
      # Work-around for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782
      SET_PROPERTY(SOURCE ${ALL_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -fno-asynchronous-unwind-tables ")
      SET_PROPERTY(SOURCE ${ALL_AVX512SKX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -fno-asynchronous-unwind-tables ")
    ENDIF()
  ENDIF()
ENDIF()
TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
TARGET_COMPILE_DEFINITIONS(allocator PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
TARGET_COMPILE_DEFINITIONS(jit PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
TARGET_COMPILE_DEFINITIONS(cache PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
TARGET_COMPILE_DEFINITIONS(logging PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
TARGET_COMPILE_DEFINITIONS(subgraph PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
TARGET_COMPILE_DEFINITIONS(operators PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
TARGET_COMPILE_DEFINITIONS(mutex PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
IF(MSVC)
  # Even though MSVC has __restrict, it can't be used in all the same contexts as the C99 restrict keyword
  TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(all_microkernels PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(packing PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(indirection PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(jit PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(cache PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(mutex PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(subgraph PRIVATE "restrict=")
  TARGET_COMPILE_DEFINITIONS(operators PRIVATE "restrict=")
  SET_PROPERTY(SOURCE ${ALL_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O2 >")
  SET_PROPERTY(SOURCE ${HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O2 >")
  SET_PROPERTY(SOURCE ${COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O1 >")
ELSE()
  SET_PROPERTY(SOURCE ${ALL_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O2 >")
  SET_PROPERTY(SOURCE ${HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O2 >")
  SET_PROPERTY(SOURCE ${COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -Os >")
ENDIF()

TARGET_INCLUDE_DIRECTORIES(XNNPACK PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(XNNPACK PRIVATE src)
TARGET_INCLUDE_DIRECTORIES(allocator PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(all_microkernels PRIVATE src)
TARGET_INCLUDE_DIRECTORIES(packing PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(microparams_init PRIVATE src)
TARGET_INCLUDE_DIRECTORIES(normalization PRIVATE src)
TARGET_INCLUDE_DIRECTORIES(indirection PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(jit PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(logging PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(cache PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(subgraph PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(operators PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(mutex PRIVATE include src)
TARGET_INCLUDE_DIRECTORIES(post-operation PUBLIC include src)
IF(WIN32)
  # Target Windows 7+ API
  TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE _WIN32_WINNT=0x0601)
  TARGET_COMPILE_DEFINITIONS(mutex PRIVATE _WIN32_WINNT=0x0601)
ENDIF()
SET_TARGET_PROPERTIES(XNNPACK PROPERTIES PUBLIC_HEADER include/xnnpack.h)

# ---[ Find libm
FIND_LIBRARY(LIBM m)
IF(LIBM)
  TARGET_LINK_LIBRARIES(XNNPACK PRIVATE ${LIBM})
  TARGET_LINK_LIBRARIES(all_microkernels PRIVATE ${LIBM})
  TARGET_LINK_LIBRARIES(indirection PRIVATE ${LIBM})
ENDIF()

# ---[ Configure clog
IF(NOT TARGET clog)
  IF(NOT XNNPACK_USE_SYSTEM_LIBS)
    SET(CLOG_BUILD_TESTS OFF CACHE BOOL "")
    SET(CLOG_RUNTIME_TYPE "${CPUINFO_RUNTIME_TYPE}" CACHE STRING "")
    ADD_SUBDIRECTORY(
      "${CLOG_SOURCE_DIR}/deps/clog"
      "${CMAKE_BINARY_DIR}/clog")
    # We build static version of clog but a dynamic library may indirectly depend on it
    SET_PROPERTY(TARGET clog PROPERTY POSITION_INDEPENDENT_CODE ON)
  ELSE()
    ADD_LIBRARY(clog STATIC IMPORTED)
    FIND_LIBRARY(CLOG_LIBRARY clog)
    IF(NOT CLOG_LIBRARY)
      MESSAGE(FATAL_ERROR "Cannot find clog")
    ENDIF()
    SET_PROPERTY(TARGET clog PROPERTY IMPORTED_LOCATION "${CLOG_LIBRARY}")
  ENDIF()
ENDIF()

# ---[ Configure cpuinfo
IF(NOT TARGET cpuinfo)
  IF(NOT XNNPACK_USE_SYSTEM_LIBS)
    SET(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
    SET(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
    SET(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
    SET(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${CPUINFO_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/cpuinfo")
  ELSE()
    ADD_LIBRARY(cpuinfo SHARED IMPORTED)
    FIND_LIBRARY(CPUINFO_LIBRARY cpuinfo)
    IF(NOT CPUINFO_LIBRARY)
      MESSAGE(FATAL_ERROR "Cannot find cpuinfo")
    ENDIF()
    SET_PROPERTY(TARGET cpuinfo PROPERTY IMPORTED_LOCATION "${CPUINFO_LIBRARY}")
    SET_PROPERTY(TARGET cpuinfo PROPERTY IMPORTED_IMPLIB "${CPUINFO_LIBRARY}")
  ENDIF()
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE cpuinfo)

# ---[ Configure pthreadpool
IF(NOT TARGET pthreadpool)
  IF(NOT XNNPACK_USE_SYSTEM_LIBS)
    SET(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
    SET(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
    SET(PTHREADPOOL_ALLOW_DEPRECATED_API OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${PTHREADPOOL_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/pthreadpool")
  ELSE()
    ADD_LIBRARY(pthreadpool SHARED IMPORTED)
    FIND_LIBRARY(PTHREADPOOL_LIBRARY pthreadpool)
    IF(NOT PTHREADPOOL_LIBRARY)
      MESSAGE(FATAL_ERROR "Cannot find pthreadpool")
    ENDIF()
    SET_PROPERTY(TARGET pthreadpool PROPERTY IMPORTED_LOCATION "${PTHREADPOOL_LIBRARY}")
    SET_PROPERTY(TARGET pthreadpool PROPERTY IMPORTED_IMPLIB "${PTHREADPOOL_LIBRARY}")
  ENDIF()
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PUBLIC pthreadpool allocator)
TARGET_LINK_LIBRARIES(allocator PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(all_microkernels PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(cache PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(indirection PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(jit PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(logging PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(packing PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(microparams_init PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(subgraph PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(operators PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(mutex PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(post-operation PUBLIC pthreadpool allocator)

# ---[ Configure FXdiv
IF(NOT TARGET fxdiv)
  IF(NOT XNNPACK_USE_SYSTEM_LIBS)
    SET(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
    SET(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${FXDIV_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/FXdiv")
  ELSE()
    FIND_FILE(FXDIV_HDR fxdiv.h PATH_SUFFIXES include)
    IF(NOT FXDIV_HDR)
      MESSAGE(FATAL_ERROR "Cannot find fxdiv")
    ENDIF()
    ADD_LIBRARY(fxdiv STATIC "${FXDIV_HDR}")
    SET_PROPERTY(TARGET fxdiv PROPERTY LINKER_LANGUAGE C)
  ENDIF()
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE fxdiv)
TARGET_LINK_LIBRARIES(all_microkernels PRIVATE fxdiv)
TARGET_LINK_LIBRARIES(indirection PRIVATE fxdiv)

# ---[ Configure FP16
IF(NOT TARGET fp16)
  IF(NOT XNNPACK_USE_SYSTEM_LIBS)
    SET(FP16_BUILD_TESTS OFF CACHE BOOL "")
    SET(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${FP16_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/FP16")
  ELSE()
    FIND_FILE(FP16_HDR fp16.h PATH_SUFFIXES include)
    IF(NOT FP16_HDR)
      MESSAGE(FATAL_ERROR "Cannot find fp16")
    ENDIF()
    ADD_LIBRARY(fp16 STATIC "${FP16_HDR}")
    SET_PROPERTY(TARGET fp16 PROPERTY LINKER_LANGUAGE C)
  ENDIF()
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE
  cache
  fp16
  logging
  mutex
  normalization
  operators
  microparams_init
  subgraph)
TARGET_LINK_LIBRARIES(all_microkernels PRIVATE fp16)
TARGET_LINK_LIBRARIES(packing PRIVATE fp16)
TARGET_LINK_LIBRARIES(microparams_init PRIVATE fp16)
TARGET_LINK_LIBRARIES(indirection PRIVATE fp16)
TARGET_LINK_LIBRARIES(subgraph PRIVATE fp16)
TARGET_LINK_LIBRARIES(operators PRIVATE fp16)

INSTALL(TARGETS XNNPACK
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---[ XNNPACK unit tests
IF(XNNPACK_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googletest")
  ENDIF()

  # Helper libraries
  ADD_LIBRARY(gemm-microkernel-tester STATIC test/gemm-microkernel-tester.cc)
  TARGET_INCLUDE_DIRECTORIES(gemm-microkernel-tester PRIVATE . include src test)
  TARGET_LINK_LIBRARIES(gemm-microkernel-tester PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest jit packing)

  # ---[ Build size tests
  ADD_EXECUTABLE(operator-size-test test/operator-size.c)
  TARGET_LINK_LIBRARIES(operator-size-test PRIVATE XNNPACK cache microparams_init logging operators)

  ADD_EXECUTABLE(subgraph-size-test test/subgraph-size.c)
  TARGET_LINK_LIBRARIES(subgraph-size-test PRIVATE XNNPACK cache microparams_init logging subgraph operators)

  # ---[ Build operator-level unit tests
  ADD_EXECUTABLE(abs-nc-test test/abs-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(abs-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(abs-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME abs-nc-test COMMAND abs-nc-test)

  ADD_EXECUTABLE(add-nd-test test/add-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(add-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add-nd-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME add-nd-test COMMAND add-nd-test)

  ADD_EXECUTABLE(add-nd-eager-test test/add-nd-eager.cc)
  TARGET_INCLUDE_DIRECTORIES(add-nd-eager-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add-nd-eager-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME add-nd-eager-test COMMAND add-nd-eager-test)

  ADD_EXECUTABLE(argmax-pooling-nhwc-test test/argmax-pooling-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(argmax-pooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(argmax-pooling-nhwc-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME argmax-pooling-nhwc-test COMMAND argmax-pooling-nhwc-test)

  ADD_EXECUTABLE(average-pooling-nhwc-test test/average-pooling-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(average-pooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(average-pooling-nhwc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME average-pooling-nhwc-test COMMAND average-pooling-nhwc-test)

  ADD_EXECUTABLE(bankers-rounding-nc-test test/bankers-rounding-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(bankers-rounding-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(bankers-rounding-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME bankers-rounding-nc-test COMMAND bankers-rounding-nc-test)

  ADD_EXECUTABLE(ceiling-nc-test test/ceiling-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(ceiling-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(ceiling-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME ceiling-nc-test COMMAND ceiling-nc-test)

  ADD_EXECUTABLE(channel-shuffle-nc-test test/channel-shuffle-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(channel-shuffle-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(channel-shuffle-nc-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME channel-shuffle-nc-test COMMAND channel-shuffle-nc-test)

  ADD_EXECUTABLE(clamp-nc-test test/clamp-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(clamp-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(clamp-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME clamp-nc-test COMMAND clamp-nc-test)

  ADD_EXECUTABLE(constant-pad-nd-test test/constant-pad-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(constant-pad-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(constant-pad-nd-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME constant-pad-nd-test COMMAND constant-pad-nd-test)

  ADD_EXECUTABLE(convert-nc-test test/convert-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(convert-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convert-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME convert-nc-test COMMAND convert-nc-test)

  ADD_EXECUTABLE(convolution-nhwc-test test/convolution-nhwc.cc)
  SET_TARGET_PROPERTIES(convolution-nhwc-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(convolution-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-nhwc-test PRIVATE XNNPACK cache fp16 gtest gtest_main microparams_init logging operators
    convolution-test-helpers)
  ADD_TEST(NAME convolution-nhwc-test COMMAND convolution-nhwc-test)

  ADD_EXECUTABLE(convolution-nchw-test test/convolution-nchw.cc)
  SET_TARGET_PROPERTIES(convolution-nchw-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(convolution-nchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-nchw-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME convolution-nchw-test COMMAND convolution-nchw-test)

  ADD_EXECUTABLE(copy-nc-test test/copy-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(copy-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(copy-nc-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME copy-nc-test COMMAND copy-nc-test)

  ADD_EXECUTABLE(deconvolution-nhwc-test test/deconvolution-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(deconvolution-nhwc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME deconvolution-nhwc-test COMMAND deconvolution-nhwc-test)

  ADD_EXECUTABLE(depth-to-space-nchw2nhwc-test test/depth-to-space-nchw2nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(depth-to-space-nchw2nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(depth-to-space-nchw2nhwc-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME depth-to-space-nchw2nhwc-test COMMAND depth-to-space-nchw2nhwc-test)

  ADD_EXECUTABLE(depth-to-space-nhwc-test test/depth-to-space-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(depth-to-space-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(depth-to-space-nhwc-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME depth-to-space-nhwc-test COMMAND depth-to-space-nhwc-test)

  ADD_EXECUTABLE(divide-nd-test test/divide-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(divide-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(divide-nd-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME divide-nd-test COMMAND divide-nd-test)

  ADD_EXECUTABLE(elu-nc-test test/elu-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(elu-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(elu-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME elu-nc-test COMMAND elu-nc-test)

  ADD_EXECUTABLE(fully-connected-nc-test test/fully-connected-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(fully-connected-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME fully-connected-nc-test COMMAND fully-connected-nc-test)

  ADD_EXECUTABLE(floor-nc-test test/floor-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(floor-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(floor-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME floor-nc-test COMMAND floor-nc-test)

  ADD_EXECUTABLE(global-average-pooling-nwc-test test/global-average-pooling-nwc.cc)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-nwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-nwc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME global-average-pooling-nwc-test COMMAND global-average-pooling-nwc-test)

  ADD_EXECUTABLE(global-average-pooling-ncw-test test/global-average-pooling-ncw.cc)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-ncw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-ncw-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME global-average-pooling-ncw-test COMMAND global-average-pooling-ncw-test)

  ADD_EXECUTABLE(hardswish-nc-test test/hardswish-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(hardswish-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(hardswish-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME hardswish-nc-test COMMAND hardswish-nc-test)

  ADD_EXECUTABLE(leaky-relu-nc-test test/leaky-relu-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(leaky-relu-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(leaky-relu-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME leaky-relu-nc-test COMMAND leaky-relu-nc-test)

  ADD_EXECUTABLE(max-pooling-nhwc-test test/max-pooling-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(max-pooling-nhwc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME max-pooling-nhwc-test COMMAND max-pooling-nhwc-test)

  ADD_EXECUTABLE(maximum-nd-test test/maximum-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(maximum-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(maximum-nd-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME maximum-nd-test COMMAND maximum-nd-test)

  ADD_EXECUTABLE(minimum-nd-test test/minimum-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(minimum-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(minimum-nd-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME minimum-nd-test COMMAND minimum-nd-test)

  ADD_EXECUTABLE(multiply-nd-test test/multiply-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(multiply-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(multiply-nd-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME multiply-nd-test COMMAND multiply-nd-test)

  ADD_EXECUTABLE(negate-nc-test test/negate-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(negate-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(negate-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME negate-nc-test COMMAND negate-nc-test)

  ADD_EXECUTABLE(transpose-normalization-test test/transpose-normalization.cc)
  TARGET_INCLUDE_DIRECTORIES(transpose-normalization-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(transpose-normalization-test PRIVATE XNNPACK gtest gtest_main normalization)
  ADD_TEST(NAME transpose-normalization-test COMMAND transpose-normalization-test)

  ADD_EXECUTABLE(prelu-nc-test test/prelu-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(prelu-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(prelu-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME prelu-nc-test COMMAND prelu-nc-test)

  ADD_EXECUTABLE(resize-bilinear-nhwc-test test/resize-bilinear-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(resize-bilinear-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(resize-bilinear-nhwc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME resize-bilinear-nhwc-test COMMAND resize-bilinear-nhwc-test)

  ADD_EXECUTABLE(resize-bilinear-nchw-test test/resize-bilinear-nchw.cc)
  TARGET_INCLUDE_DIRECTORIES(resize-bilinear-nchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(resize-bilinear-nchw-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME resize-bilinear-nchw-test COMMAND resize-bilinear-nchw-test)

  ADD_EXECUTABLE(sigmoid-nc-test test/sigmoid-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(sigmoid-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(sigmoid-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME sigmoid-nc-test COMMAND sigmoid-nc-test)

  ADD_EXECUTABLE(softmax-nc-test test/softmax-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(softmax-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(softmax-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME softmax-nc-test COMMAND softmax-nc-test)

  ADD_EXECUTABLE(space-to-depth-nhwc-test test/space-to-depth-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(space-to-depth-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(space-to-depth-nhwc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME space-to-depth-nhwc-test COMMAND space-to-depth-nhwc-test)

  ADD_EXECUTABLE(square-nc-test test/square-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(square-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(square-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME square-nc-test COMMAND square-nc-test)

  ADD_EXECUTABLE(square-root-nc-test test/square-root-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(square-root-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(square-root-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME square-root-nc-test COMMAND square-root-nc-test)

  ADD_EXECUTABLE(squared-difference-nd-test test/squared-difference-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(squared-difference-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(squared-difference-nd-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME squared-difference-nd-test COMMAND squared-difference-nd-test)

  ADD_EXECUTABLE(subtract-nd-test test/subtract-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(subtract-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(subtract-nd-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME subtract-nd-test COMMAND subtract-nd-test)

  ADD_EXECUTABLE(tanh-nc-test test/tanh-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(tanh-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(tanh-nc-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME tanh-nc-test COMMAND tanh-nc-test)

  ADD_EXECUTABLE(transpose-nd-test test/transpose-nd.cc)
  TARGET_INCLUDE_DIRECTORIES(transpose-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(transpose-nd-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(NAME transpose-nd-test COMMAND transpose-nd-test)

  ADD_EXECUTABLE(transpose-nd-eager-test test/transpose-nd-eager.cc)
  TARGET_INCLUDE_DIRECTORIES(transpose-nd-eager-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(transpose-nd-eager-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(NAME transpose-nd-eager-test COMMAND transpose-nd-eager-test)

  ADD_EXECUTABLE(truncation-nc-test test/truncation-nc.cc)
  TARGET_INCLUDE_DIRECTORIES(truncation-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(truncation-nc-test PRIVATE XNNPACK fp16 gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME truncation-nc-test COMMAND truncation-nc-test)

  ADD_EXECUTABLE(unpooling-nhwc-test test/unpooling-nhwc.cc)
  TARGET_INCLUDE_DIRECTORIES(unpooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(unpooling-nhwc-test PRIVATE XNNPACK gtest gtest_main microparams_init logging operators)
  ADD_TEST(NAME unpooling-nhwc-test COMMAND unpooling-nhwc-test)

  ADD_EXECUTABLE(memory-planner-test test/memory-planner-test.cc)
  TARGET_INCLUDE_DIRECTORIES(memory-planner-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(memory-planner-test PRIVATE XNNPACK gtest gtest_main microparams_init subgraph logging operators)
  ADD_TEST(NAME memory-planner-test COMMAND memory-planner-test)

  ADD_EXECUTABLE(subgraph-nchw-test test/subgraph-nchw.cc)
  TARGET_INCLUDE_DIRECTORIES(subgraph-nchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(subgraph-nchw-test PRIVATE XNNPACK cache gtest gtest_main microparams_init logging subgraph operators)
  ADD_TEST(NAME subgraph-nchw-test COMMAND subgraph-nchw-test)

  ADD_EXECUTABLE(subgraph-fp16-test test/subgraph-fp16.cc)
  TARGET_INCLUDE_DIRECTORIES(subgraph-fp16-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(subgraph-fp16-test PRIVATE XNNPACK cache gtest gtest_main microparams_init logging subgraph operators)
  ADD_TEST(NAME subgraph-fp16-test COMMAND subgraph-fp16-test)

  # ---[ Build subgraph-level unit tests
  ADD_EXECUTABLE(workspace-test test/workspace.cc)
  TARGET_INCLUDE_DIRECTORIES(workspace-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(workspace-test PRIVATE XNNPACK fp16 gmock gtest gtest_main subgraph)
  ADD_TEST(NAME workspace-test COMMAND workspace-test)

  ADD_EXECUTABLE(abs-test test/abs.cc)
  TARGET_INCLUDE_DIRECTORIES(abs-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(abs-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME abs-test COMMAND abs-test)

  ADD_EXECUTABLE(add2-test test/add2.cc)
  TARGET_INCLUDE_DIRECTORIES(add2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add2-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME add2-test COMMAND add2-test)

  ADD_EXECUTABLE(argmax-pooling-2d-test test/argmax-pooling-2d.cc)
  TARGET_INCLUDE_DIRECTORIES(argmax-pooling-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(argmax-pooling-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME argmax-pooling-2d-test COMMAND argmax-pooling-2d-test)

  ADD_EXECUTABLE(average-pooling-2d-test test/average-pooling-2d.cc)
  TARGET_INCLUDE_DIRECTORIES(average-pooling-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(average-pooling-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME average-pooling-2d-test COMMAND average-pooling-2d-test)

  ADD_EXECUTABLE(bankers-rounding-test test/bankers-rounding.cc)
  TARGET_INCLUDE_DIRECTORIES(bankers-rounding-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(bankers-rounding-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME bankers-rounding-test COMMAND bankers-rounding-test)

  ADD_EXECUTABLE(ceiling-test test/ceiling.cc)
  TARGET_INCLUDE_DIRECTORIES(ceiling-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(ceiling-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME ceiling-test COMMAND ceiling-test)

  ADD_EXECUTABLE(clamp-test test/clamp.cc)
  TARGET_INCLUDE_DIRECTORIES(clamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(clamp-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME clamp-test COMMAND clamp-test)

  ADD_EXECUTABLE(concatenate2-test test/concatenate2.cc)
  TARGET_INCLUDE_DIRECTORIES(concatenate2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(concatenate2-test PRIVATE XNNPACK gtest gtest_main subgraph)
  ADD_TEST(NAME concatenate2-test COMMAND concatenate2-test)

  ADD_EXECUTABLE(concatenate3-test test/concatenate3.cc)
  TARGET_INCLUDE_DIRECTORIES(concatenate3-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(concatenate3-test PRIVATE XNNPACK gtest gtest_main subgraph)
  ADD_TEST(NAME concatenate3-test COMMAND concatenate3-test)

  ADD_EXECUTABLE(concatenate4-test test/concatenate4.cc)
  TARGET_INCLUDE_DIRECTORIES(concatenate4-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(concatenate4-test PRIVATE XNNPACK gtest gtest_main subgraph)
  ADD_TEST(NAME concatenate4-test COMMAND concatenate4-test)

  ADD_EXECUTABLE(convert-test test/convert.cc)
  TARGET_INCLUDE_DIRECTORIES(convert-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convert-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME convert-test COMMAND convert-test)

  ADD_EXECUTABLE(convolution-2d-test test/convolution-2d.cc)
  SET_TARGET_PROPERTIES(convolution-2d-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(convolution-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph convolution-test-helpers)
  ADD_TEST(NAME convolution-2d-test COMMAND convolution-2d-test)

  ADD_EXECUTABLE(copy-test test/copy.cc)
  TARGET_INCLUDE_DIRECTORIES(copy-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(copy-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME copy-test COMMAND copy-test)

  ADD_EXECUTABLE(deconvolution-2d-test test/deconvolution-2d.cc)
  SET_TARGET_PROPERTIES(deconvolution-2d-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(deconvolution-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph convolution-test-helpers)
  ADD_TEST(NAME deconvolution-2d-test COMMAND deconvolution-2d-test)

  ADD_EXECUTABLE(depth-to-space-test test/depth-to-space.cc)
  TARGET_INCLUDE_DIRECTORIES(depth-to-space-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(depth-to-space-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME depth-to-space-test COMMAND depth-to-space-test)

  ADD_EXECUTABLE(depthwise-convolution-2d-test test/depthwise-convolution-2d.cc)
  SET_TARGET_PROPERTIES(depthwise-convolution-2d-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(depthwise-convolution-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(depthwise-convolution-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph convolution-test-helpers)
  ADD_TEST(NAME depthwise-convolution-2d-test COMMAND depthwise-convolution-2d-test)

  ADD_EXECUTABLE(divide2-test test/divide2.cc)
  TARGET_INCLUDE_DIRECTORIES(divide2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(divide2-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME divide2-test COMMAND divide2-test)

  ADD_EXECUTABLE(elu-test test/elu.cc)
  TARGET_INCLUDE_DIRECTORIES(elu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(elu-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME elu-test COMMAND elu-test)

  ADD_EXECUTABLE(even-split2-test test/even-split2.cc)
  TARGET_INCLUDE_DIRECTORIES(even-split2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(even-split2-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME even-split2-test COMMAND even-split2-test)

  ADD_EXECUTABLE(even-split3-test test/even-split3.cc)
  TARGET_INCLUDE_DIRECTORIES(even-split3-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(even-split3-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME even-split3-test COMMAND even-split3-test)

  ADD_EXECUTABLE(even-split4-test test/even-split4.cc)
  TARGET_INCLUDE_DIRECTORIES(even-split4-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(even-split4-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME even-split4-test COMMAND even-split4-test)

  ADD_EXECUTABLE(floor-test test/floor.cc)
  TARGET_INCLUDE_DIRECTORIES(floor-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(floor-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME floor-test COMMAND floor-test)

  ADD_EXECUTABLE(fully-connected-test test/fully-connected.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(fully-connected-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME fully-connected-test COMMAND fully-connected-test)

  ADD_EXECUTABLE(global-average-pooling-1d-test test/global-average-pooling-1d.cc)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-1d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-1d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME global-average-pooling-1d-test COMMAND global-average-pooling-1d-test)

  ADD_EXECUTABLE(global-average-pooling-2d-test test/global-average-pooling-2d.cc)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME global-average-pooling-2d-test COMMAND global-average-pooling-2d-test)

  ADD_EXECUTABLE(hardswish-test test/hardswish.cc)
  TARGET_INCLUDE_DIRECTORIES(hardswish-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(hardswish-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME hardswish-test COMMAND hardswish-test)

  ADD_EXECUTABLE(leaky-relu-test test/leaky-relu.cc)
  TARGET_INCLUDE_DIRECTORIES(leaky-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(leaky-relu-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME leaky-relu-test COMMAND leaky-relu-test)

  ADD_EXECUTABLE(max-pooling-2d-test test/max-pooling-2d.cc)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(max-pooling-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME max-pooling-2d-test COMMAND max-pooling-2d-test)

  ADD_EXECUTABLE(maximum2-test test/maximum2.cc)
  TARGET_INCLUDE_DIRECTORIES(maximum2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(maximum2-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME maximum2-test COMMAND maximum2-test)

  ADD_EXECUTABLE(minimum2-test test/minimum2.cc)
  TARGET_INCLUDE_DIRECTORIES(minimum2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(minimum2-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME minimum2-test COMMAND minimum2-test)

  ADD_EXECUTABLE(multiply2-test test/multiply2.cc)
  TARGET_INCLUDE_DIRECTORIES(multiply2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(multiply2-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME multiply2-test COMMAND multiply2-test)

  ADD_EXECUTABLE(negate-test test/negate.cc)
  TARGET_INCLUDE_DIRECTORIES(negate-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(negate-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME negate-test COMMAND negate-test)

  ADD_EXECUTABLE(prelu-test test/prelu.cc)
  TARGET_INCLUDE_DIRECTORIES(prelu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(prelu-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME prelu-test COMMAND prelu-test)

  ADD_EXECUTABLE(sigmoid-test test/sigmoid.cc)
  SET_TARGET_PROPERTIES(sigmoid-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(sigmoid-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(sigmoid-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME sigmoid-test COMMAND sigmoid-test)

  ADD_EXECUTABLE(softmax-test test/softmax.cc)
  TARGET_INCLUDE_DIRECTORIES(softmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(softmax-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME softmax-test COMMAND softmax-test)

  ADD_EXECUTABLE(square-test test/square.cc)
  TARGET_INCLUDE_DIRECTORIES(square-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(square-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME square-test COMMAND square-test)

  ADD_EXECUTABLE(square-root-test test/square.cc)
  TARGET_INCLUDE_DIRECTORIES(square-root-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(square-root-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME square-root-test COMMAND square-root-test)

  ADD_EXECUTABLE(squared-difference-test test/squared-difference.cc)
  TARGET_INCLUDE_DIRECTORIES(squared-difference-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(squared-difference-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME squared-difference-test COMMAND squared-difference-test)

  ADD_EXECUTABLE(static-constant-pad-test test/static-constant-pad.cc)
  TARGET_INCLUDE_DIRECTORIES(static-constant-pad-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(static-constant-pad-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME static-constant-pad-test COMMAND static-constant-pad-test)

  ADD_EXECUTABLE(static-reshape-test test/static-reshape.cc)
  TARGET_INCLUDE_DIRECTORIES(static-reshape-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(static-reshape-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME static-reshape-test COMMAND static-reshape-test)

  ADD_EXECUTABLE(static-resize-bilinear-2d-test test/static-resize-bilinear-2d.cc)
  TARGET_INCLUDE_DIRECTORIES(static-resize-bilinear-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(static-resize-bilinear-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME static-resize-bilinear-2d-test COMMAND static-resize-bilinear-2d-test)

  ADD_EXECUTABLE(static-transpose-test test/static-transpose.cc)
  TARGET_INCLUDE_DIRECTORIES(static-transpose-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(static-transpose-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME static-transpose-test COMMAND static-transpose-test)

  ADD_EXECUTABLE(subtract2-test test/subtract2.cc)
  TARGET_INCLUDE_DIRECTORIES(subtract2-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(subtract2-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME subtract2-test COMMAND subtract2-test)

  ADD_EXECUTABLE(unpooling-2d-test test/unpooling-2d.cc)
  TARGET_INCLUDE_DIRECTORIES(unpooling-2d-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(unpooling-2d-test PRIVATE XNNPACK fp16 gtest gtest_main subgraph)
  ADD_TEST(NAME unpooling-2d-test COMMAND unpooling-2d-test)

  ADD_EXECUTABLE(fusion-test test/fusion.cc)
  TARGET_INCLUDE_DIRECTORIES(fusion-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(fusion-test PRIVATE XNNPACK fp16 gmock gtest gtest_main subgraph)
  ADD_TEST(NAME fusion-test COMMAND fusion-test)

  # ---[ Build microkernel-level unit tests
  ADD_EXECUTABLE(bf16-gemm-minmax-test test/bf16-gemm-minmax.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(bf16-gemm-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(bf16-gemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main jit gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME bf16-gemm-minmax-test COMMAND bf16-gemm-minmax-test)

  ADD_EXECUTABLE(f16-f32-vcvt-test test/f16-f32-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-f32-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-f32-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-f32-vcvt-test COMMAND f16-f32-vcvt-test)

  ADD_EXECUTABLE(f16-gavgpool-cw-test test/f16-gavgpool-cw.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-gavgpool-cw-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-gavgpool-cw-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-gavgpool-cw-test COMMAND f16-gavgpool-cw-test)

  ADD_EXECUTABLE(f16-avgpool-minmax-test test/f16-avgpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-avgpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-avgpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-avgpool-minmax-test COMMAND f16-avgpool-minmax-test)

  ADD_EXECUTABLE(f16-dwconv-minmax-test test/f16-dwconv-minmax.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-dwconv-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-dwconv-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-dwconv-minmax-test COMMAND f16-dwconv-minmax-test)

  ADD_EXECUTABLE(f16-dwconv2d-chw-test test/f16-dwconv2d-chw.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-dwconv2d-chw-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-dwconv2d-chw-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-dwconv2d-chw-test COMMAND f16-dwconv2d-chw-test)

  ADD_EXECUTABLE(f16-gavgpool-minmax-test test/f16-gavgpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-gavgpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-gavgpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-gavgpool-minmax-test COMMAND f16-gavgpool-minmax-test)

  ADD_EXECUTABLE(f16-gemm-minmax-test test/f16-gemm-minmax.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-gemm-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f16-gemm-minmax-test COMMAND f16-gemm-minmax-test)

  ADD_EXECUTABLE(f16-ibilinear-chw-test test/f16-ibilinear-chw.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-ibilinear-chw-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-ibilinear-chw-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME f16-ibilinear-chw-test COMMAND f16-ibilinear-chw-test)

  ADD_EXECUTABLE(f16-ibilinear-test test/f16-ibilinear.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-ibilinear-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-ibilinear-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME f16-ibilinear-test COMMAND f16-ibilinear-test)

  ADD_EXECUTABLE(f16-igemm-minmax-test test/f16-igemm-minmax.cc $<TARGET_OBJECTS:all_microkernels>  $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-igemm-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f16-igemm-minmax-test COMMAND f16-igemm-minmax-test)

  ADD_EXECUTABLE(f16-maxpool-minmax-test test/f16-maxpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-maxpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-maxpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-maxpool-minmax-test COMMAND f16-maxpool-minmax-test)

  ADD_EXECUTABLE(f16-spmm-minmax-test test/f16-spmm-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-spmm-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-spmm-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-spmm-minmax-test COMMAND f16-spmm-minmax-test)

  ADD_EXECUTABLE(f16-vabs-test test/f16-vabs.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vabs-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vabs-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vabs-test COMMAND f16-vabs-test)

  ADD_EXECUTABLE(f16-vadd-minmax-test test/f16-vadd-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vadd-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vadd-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vadd-minmax-test COMMAND f16-vadd-minmax-test)

  ADD_EXECUTABLE(f16-vaddc-minmax-test test/f16-vaddc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vaddc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vaddc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vaddc-minmax-test COMMAND f16-vaddc-minmax-test)

  ADD_EXECUTABLE(f16-vclamp-test test/f16-vclamp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vclamp-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vclamp-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vclamp-test COMMAND f16-vclamp-test)

  ADD_EXECUTABLE(f16-vdiv-minmax-test test/f16-vdiv-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vdiv-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vdiv-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vdiv-minmax-test COMMAND f16-vdiv-minmax-test)

  ADD_EXECUTABLE(f16-vdivc-minmax-test test/f16-vdivc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vdivc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vdivc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vdivc-minmax-test COMMAND f16-vdivc-minmax-test)

  ADD_EXECUTABLE(f16-vrdivc-minmax-test test/f16-vrdivc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vrdivc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vrdivc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vrdivc-minmax-test COMMAND f16-vrdivc-minmax-test)

  ADD_EXECUTABLE(f16-velu-test test/f16-velu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-velu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-velu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-velu-test COMMAND f16-velu-test)

  ADD_EXECUTABLE(f16-vhswish-test test/f16-vhswish.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vhswish-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vhswish-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vhswish-test COMMAND f16-vhswish-test)

  ADD_EXECUTABLE(f16-vlrelu-test test/f16-vlrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vlrelu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vlrelu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vlrelu-test COMMAND f16-vlrelu-test)

  ADD_EXECUTABLE(f16-vmax-test test/f16-vmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vmax-test COMMAND f16-vmax-test)

  ADD_EXECUTABLE(f16-vmaxc-test test/f16-vmaxc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vmaxc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vmaxc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vmaxc-test COMMAND f16-vmaxc-test)

  ADD_EXECUTABLE(f16-vmin-test test/f16-vmin.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vmin-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vmin-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vmin-test COMMAND f16-vmin-test)

  ADD_EXECUTABLE(f16-vminc-test test/f16-vminc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vminc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vminc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vminc-test COMMAND f16-vminc-test)

  ADD_EXECUTABLE(f16-vmul-minmax-test test/f16-vmul-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vmul-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vmul-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vmul-minmax-test COMMAND f16-vmul-minmax-test)

  ADD_EXECUTABLE(f16-vmulc-minmax-test test/f16-vmulc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vmulc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vmulc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vmulc-minmax-test COMMAND f16-vmulc-minmax-test)

  ADD_EXECUTABLE(f16-vmulcaddc-minmax-test test/f16-vmulcaddc-minmax.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-vmulcaddc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vmulcaddc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vmulcaddc-minmax-test COMMAND f16-vmulcaddc-minmax-test)

  ADD_EXECUTABLE(f16-vneg-test test/f16-vneg.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vneg-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vneg-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vneg-test COMMAND f16-vneg-test)

  ADD_EXECUTABLE(f16-pavgpool-minmax-test test/f16-pavgpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-pavgpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-pavgpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-pavgpool-minmax-test COMMAND f16-pavgpool-minmax-test)

  ADD_EXECUTABLE(f16-prelu-test test/f16-prelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-prelu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-prelu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-prelu-test COMMAND f16-prelu-test)

  ADD_EXECUTABLE(f16-raddstoreexpminusmax-test test/f16-raddstoreexpminusmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-raddstoreexpminusmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-raddstoreexpminusmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-raddstoreexpminusmax-test COMMAND f16-raddstoreexpminusmax-test)

  ADD_EXECUTABLE(f16-vrndne-test test/f16-vrndne.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vrndne-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vrndne-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vrndne-test COMMAND f16-vrndne-test)

  ADD_EXECUTABLE(f16-vrndz-test test/f16-vrndz.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vrndz-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vrndz-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vrndz-test COMMAND f16-vrndz-test)

  ADD_EXECUTABLE(f16-vrndu-test test/f16-vrndu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vrndu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vrndu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vrndu-test COMMAND f16-vrndu-test)

  ADD_EXECUTABLE(f16-vrndd-test test/f16-vrndd.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vrndd-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vrndd-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vrndd-test COMMAND f16-vrndd-test)

  ADD_EXECUTABLE(f16-vsigmoid-test test/f16-vsigmoid.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vsigmoid-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vsigmoid-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vsigmoid-test COMMAND f16-vsigmoid-test)

  ADD_EXECUTABLE(f16-vsqr-test test/f16-vsqr.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vsqr-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vsqr-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vsqr-test COMMAND f16-vsqr-test)

  ADD_EXECUTABLE(f16-vsqrt-test test/f16-vsqrt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vsqrt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vsqrt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vsqrt-test COMMAND f16-vsqrt-test)

  ADD_EXECUTABLE(f16-vsub-minmax-test test/f16-vsub-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vsub-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vsub-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vsub-minmax-test COMMAND f16-vsub-minmax-test)

  ADD_EXECUTABLE(f16-vsubc-minmax-test test/f16-vsubc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vsubc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vsubc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vsubc-minmax-test COMMAND f16-vsubc-minmax-test)

  ADD_EXECUTABLE(f16-vrsubc-minmax-test test/f16-vrsubc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vrsubc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f16-vrsubc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f16-vrsubc-minmax-test COMMAND f16-vrsubc-minmax-test)

  ADD_EXECUTABLE(f32-argmaxpool-test test/f32-argmaxpool.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-argmaxpool-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-argmaxpool-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-argmaxpool-test COMMAND f32-argmaxpool-test)

  ADD_EXECUTABLE(f32-avgpool-minmax-test test/f32-avgpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-avgpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-avgpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-avgpool-minmax-test COMMAND f32-avgpool-minmax-test)

  ADD_EXECUTABLE(f32-conv-hwc-test test/f32-conv-hwc.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-conv-hwc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-conv-hwc-test COMMAND f32-conv-hwc-test)

  ADD_EXECUTABLE(f32-conv-hwc2chw-test test/f32-conv-hwc2chw.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc2chw-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-conv-hwc2chw-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-conv-hwc2chw-test COMMAND f32-conv-hwc2chw-test)

  ADD_EXECUTABLE(f32-dwconv-test test/f32-dwconv.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-dwconv-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-dwconv-test COMMAND f32-dwconv-test)

  ADD_EXECUTABLE(f32-dwconv2d-chw-test test/f32-dwconv2d-chw.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv2d-chw-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-dwconv2d-chw-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-dwconv2d-chw-test COMMAND f32-dwconv2d-chw-test)

  ADD_EXECUTABLE(f32-dwconv-minmax-test test/f32-dwconv-minmax.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-dwconv-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-dwconv-minmax-test COMMAND f32-dwconv-minmax-test)

  ADD_EXECUTABLE(f32-f16-vcvt-test test/f32-f16-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-f16-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-f16-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-f16-vcvt-test COMMAND f32-f16-vcvt-test)

  ADD_EXECUTABLE(f32-qs8-vcvt-test test/f32-qs8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-qs8-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-qs8-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-qs8-vcvt-test COMMAND f32-qs8-vcvt-test)

  ADD_EXECUTABLE(f32-qu8-vcvt-test test/f32-qu8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-qu8-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-qu8-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-qu8-vcvt-test COMMAND f32-qu8-vcvt-test)

  ADD_EXECUTABLE(f32-gavgpool-cw-test test/f32-gavgpool-cw.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-gavgpool-cw-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-gavgpool-cw-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-gavgpool-cw-test COMMAND f32-gavgpool-cw-test)

  ADD_EXECUTABLE(f32-gavgpool-minmax-test test/f32-gavgpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-gavgpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-gavgpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-gavgpool-minmax-test COMMAND f32-gavgpool-minmax-test)

  ADD_EXECUTABLE(f32-gemm-test test/f32-gemm.cc test/f32-gemm-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-gemm-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f32-gemm-test COMMAND f32-gemm-test)

  ADD_EXECUTABLE(f32-gemm-relu-test test/f32-gemm-relu.cc test/f32-gemm-relu-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-gemm-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f32-gemm-relu-test COMMAND f32-gemm-relu-test)

  ADD_EXECUTABLE(f32-gemm-minmax-test test/f32-gemm-minmax.cc test/f32-gemm-minmax-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-gemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main jit gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME f32-gemm-minmax-test COMMAND f32-gemm-minmax-test)

  ADD_EXECUTABLE(f32-gemminc-minmax-test test/f32-gemminc-minmax.cc test/f32-gemminc-minmax-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-gemminc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-gemminc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f32-gemminc-minmax-test COMMAND f32-gemminc-minmax-test)

  ADD_EXECUTABLE(f32-ibilinear-test test/f32-ibilinear.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-ibilinear-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-ibilinear-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME f32-ibilinear-test COMMAND f32-ibilinear-test)

  ADD_EXECUTABLE(f32-ibilinear-chw-test test/f32-ibilinear-chw.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-ibilinear-chw-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-ibilinear-chw-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME f32-ibilinear-chw-test COMMAND f32-ibilinear-chw-test)

  ADD_EXECUTABLE(f32-igemm-test test/f32-igemm.cc test/f32-igemm-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-igemm-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f32-igemm-test COMMAND f32-igemm-test)

  ADD_EXECUTABLE(f32-igemm-relu-test test/f32-igemm-relu.cc test/f32-igemm-relu-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-igemm-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f32-igemm-relu-test COMMAND f32-igemm-relu-test)

  ADD_EXECUTABLE(f32-igemm-minmax-test test/f32-igemm-minmax.cc test/f32-igemm-minmax-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-minmax-test PRIVATE include src test)
  IF(MINGW)
    # Work-around for "too many sections" error
    TARGET_COMPILE_OPTIONS(f32-igemm-minmax-test PRIVATE "$<$<NOT:$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>>:-Wa,-mbig-obj>")
  ENDIF()
  TARGET_LINK_LIBRARIES(f32-igemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main jit gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME f32-igemm-minmax-test COMMAND f32-igemm-minmax-test)

  ADD_EXECUTABLE(f32-maxpool-minmax-test test/f32-maxpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-maxpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-maxpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-maxpool-minmax-test COMMAND f32-maxpool-minmax-test)

  ADD_EXECUTABLE(f32-pavgpool-minmax-test test/f32-pavgpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-pavgpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-pavgpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-pavgpool-minmax-test COMMAND f32-pavgpool-minmax-test)

  ADD_EXECUTABLE(f32-ppmm-minmax-test test/f32-ppmm-minmax.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-ppmm-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-ppmm-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME f32-ppmm-minmax-test COMMAND f32-ppmm-minmax-test)

  ADD_EXECUTABLE(f32-prelu-test test/f32-prelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-prelu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-prelu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-prelu-test COMMAND f32-prelu-test)

  ADD_EXECUTABLE(f32-raddexpminusmax-test test/f32-raddexpminusmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-raddexpminusmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-raddexpminusmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-raddexpminusmax-test COMMAND f32-raddexpminusmax-test)

  ADD_EXECUTABLE(f32-raddextexp-test test/f32-raddextexp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-raddextexp-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-raddextexp-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-raddextexp-test COMMAND f32-raddextexp-test)

  ADD_EXECUTABLE(f32-raddstoreexpminusmax-test test/f32-raddstoreexpminusmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-raddstoreexpminusmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-raddstoreexpminusmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-raddstoreexpminusmax-test COMMAND f32-raddstoreexpminusmax-test)

  ADD_EXECUTABLE(f32-rmax-test test/f32-rmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-rmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-rmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-rmax-test COMMAND f32-rmax-test)

  ADD_EXECUTABLE(f32-spmm-minmax-test test/f32-spmm-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-spmm-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-spmm-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-spmm-minmax-test COMMAND f32-spmm-minmax-test)

  ADD_EXECUTABLE(f32-vabs-test test/f32-vabs.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vabs-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vabs-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vabs-test COMMAND f32-vabs-test)

  ADD_EXECUTABLE(f32-vadd-test test/f32-vadd.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vadd-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vadd-test COMMAND f32-vadd-test)

  ADD_EXECUTABLE(f32-vadd-minmax-test test/f32-vadd-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vadd-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vadd-minmax-test COMMAND f32-vadd-minmax-test)

  ADD_EXECUTABLE(f32-vadd-relu-test test/f32-vadd-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vadd-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vadd-relu-test COMMAND f32-vadd-relu-test)

  ADD_EXECUTABLE(f32-vaddc-test test/f32-vaddc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vaddc-test COMMAND f32-vaddc-test)

  ADD_EXECUTABLE(f32-vaddc-minmax-test test/f32-vaddc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vaddc-minmax-test COMMAND f32-vaddc-minmax-test)

  ADD_EXECUTABLE(f32-vaddc-relu-test test/f32-vaddc-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vaddc-relu-test COMMAND f32-vaddc-relu-test)

  ADD_EXECUTABLE(f32-vclamp-test test/f32-vclamp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vclamp-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vclamp-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vclamp-test COMMAND f32-vclamp-test)

  ADD_EXECUTABLE(f32-vhswish-test test/f32-vhswish.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vhswish-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vhswish-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vhswish-test COMMAND f32-vhswish-test)

  ADD_EXECUTABLE(f32-vdiv-test test/f32-vdiv.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vdiv-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vdiv-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vdiv-test COMMAND f32-vdiv-test)

  ADD_EXECUTABLE(f32-vdiv-minmax-test test/f32-vdiv-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vdiv-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vdiv-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vdiv-minmax-test COMMAND f32-vdiv-minmax-test)

  ADD_EXECUTABLE(f32-vdiv-relu-test test/f32-vdiv-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vdiv-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vdiv-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vdiv-relu-test COMMAND f32-vdiv-relu-test)

  ADD_EXECUTABLE(f32-vdivc-test test/f32-vdivc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vdivc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vdivc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vdivc-test COMMAND f32-vdivc-test)

  ADD_EXECUTABLE(f32-vdivc-minmax-test test/f32-vdivc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vdivc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vdivc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vdivc-minmax-test COMMAND f32-vdivc-minmax-test)

  ADD_EXECUTABLE(f32-vdivc-relu-test test/f32-vdivc-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vdivc-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vdivc-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vdivc-relu-test COMMAND f32-vdivc-relu-test)

  ADD_EXECUTABLE(f32-vrdivc-test test/f32-vrdivc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrdivc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrdivc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrdivc-test COMMAND f32-vrdivc-test)

  ADD_EXECUTABLE(f32-vrdivc-minmax-test test/f32-vrdivc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrdivc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrdivc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrdivc-minmax-test COMMAND f32-vrdivc-minmax-test)

  ADD_EXECUTABLE(f32-vrdivc-relu-test test/f32-vrdivc-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrdivc-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrdivc-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrdivc-relu-test COMMAND f32-vrdivc-relu-test)

  ADD_EXECUTABLE(f32-velu-test test/f32-velu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-velu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-velu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-velu-test COMMAND f32-velu-test)

  ADD_EXECUTABLE(f32-vlrelu-test test/f32-vlrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vlrelu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vlrelu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vlrelu-test COMMAND f32-vlrelu-test)

  ADD_EXECUTABLE(f32-vmax-test test/f32-vmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmax-test COMMAND f32-vmax-test)

  ADD_EXECUTABLE(f32-vmaxc-test test/f32-vmaxc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmaxc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmaxc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmaxc-test COMMAND f32-vmaxc-test)

  ADD_EXECUTABLE(f32-vmin-test test/f32-vmin.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmin-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmin-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmin-test COMMAND f32-vmin-test)

  ADD_EXECUTABLE(f32-vminc-test test/f32-vminc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vminc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vminc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vminc-test COMMAND f32-vminc-test)

  ADD_EXECUTABLE(f32-vmul-test test/f32-vmul.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmul-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmul-test COMMAND f32-vmul-test)

  ADD_EXECUTABLE(f32-vmul-minmax-test test/f32-vmul-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmul-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmul-minmax-test COMMAND f32-vmul-minmax-test)

  ADD_EXECUTABLE(f32-vmul-relu-test test/f32-vmul-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmul-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmul-relu-test COMMAND f32-vmul-relu-test)

  ADD_EXECUTABLE(f32-vmulc-test test/f32-vmulc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmulc-test COMMAND f32-vmulc-test)

  ADD_EXECUTABLE(f32-vmulc-minmax-test test/f32-vmulc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmulc-minmax-test COMMAND f32-vmulc-minmax-test)

  ADD_EXECUTABLE(f32-vmulc-relu-test test/f32-vmulc-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmulc-relu-test COMMAND f32-vmulc-relu-test)

  ADD_EXECUTABLE(f32-vmulcaddc-minmax-test test/f32-vmulcaddc-minmax.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulcaddc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vmulcaddc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vmulcaddc-minmax-test COMMAND f32-vmulcaddc-minmax-test)

  ADD_EXECUTABLE(f32-vneg-test test/f32-vneg.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vneg-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vneg-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vneg-test COMMAND f32-vneg-test)

  ADD_EXECUTABLE(f32-vrelu-test test/f32-vrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrelu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrelu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrelu-test COMMAND f32-vrelu-test)

  ADD_EXECUTABLE(f32-vrndne-test test/f32-vrndne.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndne-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrndne-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrndne-test COMMAND f32-vrndne-test)

  ADD_EXECUTABLE(f32-vrndz-test test/f32-vrndz.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndz-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrndz-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrndz-test COMMAND f32-vrndz-test)

  ADD_EXECUTABLE(f32-vrndu-test test/f32-vrndu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrndu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrndu-test COMMAND f32-vrndu-test)

  ADD_EXECUTABLE(f32-vrndd-test test/f32-vrndd.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndd-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrndd-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrndd-test COMMAND f32-vrndd-test)

  ADD_EXECUTABLE(f32-vscaleexpminusmax-test test/f32-vscaleexpminusmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vscaleexpminusmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vscaleexpminusmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vscaleexpminusmax-test COMMAND f32-vscaleexpminusmax-test)

  ADD_EXECUTABLE(f32-vscaleextexp-test test/f32-vscaleextexp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vscaleextexp-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vscaleextexp-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vscaleextexp-test COMMAND f32-vscaleextexp-test)

  ADD_EXECUTABLE(f32-vsigmoid-test test/f32-vsigmoid.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsigmoid-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsigmoid-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsigmoid-test COMMAND f32-vsigmoid-test)

  ADD_EXECUTABLE(f32-vsqr-test test/f32-vsqr.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqr-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsqr-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsqr-test COMMAND f32-vsqr-test)

  ADD_EXECUTABLE(f32-vsqrdiff-test test/f32-vsqrdiff.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrdiff-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsqrdiff-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsqrdiff-test COMMAND f32-vsqrdiff-test)

  ADD_EXECUTABLE(f32-vsqrdiffc-test test/f32-vsqrdiffc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrdiffc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsqrdiffc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsqrdiffc-test COMMAND f32-vsqrdiffc-test)

  ADD_EXECUTABLE(f32-vsqrt-test test/f32-vsqrt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsqrt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsqrt-test COMMAND f32-vsqrt-test)

  ADD_EXECUTABLE(f32-vsub-test test/f32-vsub.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsub-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsub-test COMMAND f32-vsub-test)

  ADD_EXECUTABLE(f32-vsub-minmax-test test/f32-vsub-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsub-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsub-minmax-test COMMAND f32-vsub-minmax-test)

  ADD_EXECUTABLE(f32-vsub-relu-test test/f32-vsub-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsub-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsub-relu-test COMMAND f32-vsub-relu-test)

  ADD_EXECUTABLE(f32-vsubc-test test/f32-vsubc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsubc-test COMMAND f32-vsubc-test)

  ADD_EXECUTABLE(f32-vsubc-minmax-test test/f32-vsubc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsubc-minmax-test COMMAND f32-vsubc-minmax-test)

  ADD_EXECUTABLE(f32-vsubc-relu-test test/f32-vsubc-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vsubc-relu-test COMMAND f32-vsubc-relu-test)

  ADD_EXECUTABLE(f32-vrsubc-test test/f32-vrsubc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrsubc-test COMMAND f32-vrsubc-test)

  ADD_EXECUTABLE(f32-vrsubc-minmax-test test/f32-vrsubc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrsubc-minmax-test COMMAND f32-vrsubc-minmax-test)

  ADD_EXECUTABLE(f32-vrsubc-relu-test test/f32-vrsubc-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-relu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-relu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME f32-vrsubc-relu-test COMMAND f32-vrsubc-relu-test)

  ADD_EXECUTABLE(qc8-dwconv-minmax-fp32-test test/qc8-dwconv-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qc8-dwconv-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qc8-dwconv-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qc8-dwconv-minmax-fp32-test COMMAND qc8-dwconv-minmax-fp32-test)

  ADD_EXECUTABLE(qc8-gemm-minmax-fp32-test test/qc8-gemm-minmax-fp32.cc test/qc8-gemm-minmax-fp32-2.cc test/qc8-gemm-minmax-fp32-3.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qc8-gemm-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qc8-gemm-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main jit gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME qc8-gemm-minmax-fp32-test COMMAND qc8-gemm-minmax-fp32-test)

  ADD_EXECUTABLE(qc8-igemm-minmax-fp32-test test/qc8-igemm-minmax-fp32.cc test/qc8-igemm-minmax-fp32-2.cc test/qc8-igemm-minmax-fp32-3.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qc8-igemm-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qc8-igemm-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main jit gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME qc8-igemm-minmax-fp32-test COMMAND qc8-igemm-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-dwconv-minmax-fp32-test test/qs8-dwconv-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-dwconv-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-dwconv-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-dwconv-minmax-fp32-test COMMAND qs8-dwconv-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-dwconv-minmax-rndnu-test test/qs8-dwconv-minmax-rndnu.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-dwconv-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-dwconv-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-dwconv-minmax-rndnu-test COMMAND qs8-dwconv-minmax-rndnu-test)

  ADD_EXECUTABLE(qs8-f32-vcvt-test test/f32-f16-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-f32-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-f32-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-f32-vcvt-test COMMAND qs8-f32-vcvt-test)

  ADD_EXECUTABLE(qs8-gavgpool-minmax-fp32-test test/qs8-gavgpool-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-gavgpool-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-gavgpool-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-gavgpool-minmax-fp32-test COMMAND qs8-gavgpool-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-gavgpool-minmax-rndnu-test test/qs8-gavgpool-minmax-rndnu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-gavgpool-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-gavgpool-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-gavgpool-minmax-rndnu-test COMMAND qs8-gavgpool-minmax-rndnu-test)

  ADD_EXECUTABLE(qs8-gemm-minmax-fp32-test test/qs8-gemm-minmax-fp32.cc test/qs8-gemm-minmax-fp32-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-gemm-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME qs8-gemm-minmax-fp32-test COMMAND qs8-gemm-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-gemm-minmax-rndnu-test test/qs8-gemm-minmax-rndnu.cc test/qs8-gemm-minmax-rndnu-2.cc test/qs8-gemm-minmax-rndnu-3.cc test/qs8-gemm-minmax-rndnu-4.cc test/qs8-gemm-minmax-rndnu-5.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-gemm-minmax-rndnu-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main jit gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME qs8-gemm-minmax-rndnu-test COMMAND qs8-gemm-minmax-rndnu-test)

  ADD_EXECUTABLE(qs8-igemm-minmax-fp32-test test/qs8-igemm-minmax-fp32.cc test/qs8-igemm-minmax-fp32-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-igemm-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-igemm-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME qs8-igemm-minmax-fp32-test COMMAND qs8-igemm-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-igemm-minmax-rndnu-test test/qs8-igemm-minmax-rndnu.cc test/qs8-igemm-minmax-rndnu-2.cc test/qs8-igemm-minmax-rndnu-3.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-igemm-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-igemm-minmax-rndnu-test PRIVATE XNNPACK cpuinfo fp16 pthreadpool gtest gtest_main jit gemm-microkernel-tester microparams_init allocator)
  ADD_TEST(NAME qs8-igemm-minmax-rndnu-test COMMAND qs8-igemm-minmax-rndnu-test)

  ADD_EXECUTABLE(qs8-requantization-test test/qs8-requantization.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qs8-requantization-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-requantization-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-requantization-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME qs8-requantization-test COMMAND qs8-requantization-test)

  ADD_EXECUTABLE(qs8-vadd-minmax-test test/qs8-vadd-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vadd-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-vadd-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-vadd-minmax-test COMMAND qs8-vadd-minmax-test)

  ADD_EXECUTABLE(qs8-vaddc-minmax-test test/qs8-vaddc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vaddc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-vaddc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-vaddc-minmax-test COMMAND qs8-vaddc-minmax-test)

  ADD_EXECUTABLE(qs8-vcvt-test test/qs8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-vcvt-test COMMAND qs8-vcvt-test)

  ADD_EXECUTABLE(qs8-vlrelu-test test/qs8-vlrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vlrelu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-vlrelu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-vlrelu-test COMMAND qs8-vlrelu-test)

  ADD_EXECUTABLE(qs8-vmul-minmax-fp32-test test/qs8-vmul-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qs8-vmul-minmax-fp32-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-vmul-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-vmul-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-vmul-minmax-fp32-test COMMAND qs8-vmul-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-vmulc-minmax-fp32-test test/qs8-vmulc-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qs8-vmulc-minmax-fp32-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-vmulc-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qs8-vmulc-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qs8-vmulc-minmax-fp32-test COMMAND qs8-vmulc-minmax-fp32-test)

  ADD_EXECUTABLE(qu8-avgpool-minmax-test test/qu8-avgpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-avgpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-avgpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-avgpool-minmax-test COMMAND qu8-avgpool-minmax-test)

  ADD_EXECUTABLE(qu8-dwconv-minmax-fp32-test test/qu8-dwconv-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qu8-dwconv-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-dwconv-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-dwconv-minmax-fp32-test COMMAND qu8-dwconv-minmax-fp32-test)

  ADD_EXECUTABLE(qu8-dwconv-minmax-rndnu-test test/qu8-dwconv-minmax-rndnu.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qu8-dwconv-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-dwconv-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-dwconv-minmax-rndnu-test COMMAND qu8-dwconv-minmax-rndnu-test)

  ADD_EXECUTABLE(qu8-f32-vcvt-test test/f32-f16-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-f32-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-f32-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-f32-vcvt-test COMMAND qu8-f32-vcvt-test)

  ADD_EXECUTABLE(qu8-gavgpool-minmax-fp32-test test/qu8-gavgpool-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-gavgpool-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-gavgpool-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-gavgpool-minmax-fp32-test COMMAND qu8-gavgpool-minmax-fp32-test)

  ADD_EXECUTABLE(qu8-gavgpool-minmax-rndnu-test test/qu8-gavgpool-minmax-rndnu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-gavgpool-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-gavgpool-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-gavgpool-minmax-rndnu-test COMMAND qu8-gavgpool-minmax-rndnu-test)

  ADD_EXECUTABLE(qu8-gemm-minmax-fp32-test test/qu8-gemm-minmax-fp32.cc test/qu8-gemm-minmax-fp32-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qu8-gemm-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-gemm-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME qu8-gemm-minmax-fp32-test COMMAND qu8-gemm-minmax-fp32-test)

  ADD_EXECUTABLE(qu8-gemm-minmax-rndnu-test test/qu8-gemm-minmax-rndnu.cc test/qu8-gemm-minmax-rndnu-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qu8-gemm-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-gemm-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME qu8-gemm-minmax-rndnu-test COMMAND qu8-gemm-minmax-rndnu-test)

  ADD_EXECUTABLE(qu8-igemm-minmax-fp32-test test/qu8-igemm-minmax-fp32.cc test/qu8-igemm-minmax-fp32-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qu8-igemm-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-igemm-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME qu8-igemm-minmax-fp32-test COMMAND qu8-igemm-minmax-fp32-test)

  ADD_EXECUTABLE(qu8-igemm-minmax-rndnu-test test/qu8-igemm-minmax-rndnu.cc test/qu8-igemm-minmax-rndnu-2.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qu8-igemm-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-igemm-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main gemm-microkernel-tester microparams_init)
  ADD_TEST(NAME qu8-igemm-minmax-rndnu-test COMMAND qu8-igemm-minmax-rndnu-test)

  ADD_EXECUTABLE(qu8-requantization-test test/qu8-requantization.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qu8-requantization-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-requantization-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-requantization-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME qu8-requantization-test COMMAND qu8-requantization-test)

  ADD_EXECUTABLE(qu8-vadd-minmax-test test/qu8-vadd-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vadd-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vadd-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vadd-minmax-test COMMAND qu8-vadd-minmax-test)

  ADD_EXECUTABLE(qu8-vaddc-minmax-test test/qu8-vaddc-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vaddc-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vaddc-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vaddc-minmax-test COMMAND qu8-vaddc-minmax-test)

  ADD_EXECUTABLE(qu8-vcvt-test test/qu8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vcvt-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vcvt-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vcvt-test COMMAND qu8-vcvt-test)

  ADD_EXECUTABLE(qu8-vlrelu-test test/qu8-vlrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vlrelu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vlrelu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vlrelu-test COMMAND qu8-vlrelu-test)

  ADD_EXECUTABLE(qu8-vmul-minmax-fp32-test test/qu8-vmul-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qu8-vmul-minmax-fp32-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-vmul-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vmul-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vmul-minmax-fp32-test COMMAND qu8-vmul-minmax-fp32-test)

  ADD_EXECUTABLE(qu8-vmul-minmax-rndnu-test test/qu8-vmul-minmax-rndnu.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qu8-vmul-minmax-rndnu-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-vmul-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vmul-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vmul-minmax-rndnu-test COMMAND qu8-vmul-minmax-rndnu-test)

  ADD_EXECUTABLE(qu8-vmulc-minmax-fp32-test test/qu8-vmulc-minmax-fp32.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qu8-vmulc-minmax-fp32-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-vmulc-minmax-fp32-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vmulc-minmax-fp32-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vmulc-minmax-fp32-test COMMAND qu8-vmulc-minmax-fp32-test)

  ADD_EXECUTABLE(qu8-vmulc-minmax-rndnu-test test/qu8-vmulc-minmax-rndnu.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qu8-vmulc-minmax-rndnu-test PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-vmulc-minmax-rndnu-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(qu8-vmulc-minmax-rndnu-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME qu8-vmulc-minmax-rndnu-test COMMAND qu8-vmulc-minmax-rndnu-test)

  ADD_EXECUTABLE(s16-rmaxabs-test test/s16-rmaxabs.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s16-rmaxabs-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(s16-rmaxabs-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME s16-rmaxabs-test COMMAND s16-rmaxabs-test)

  ADD_EXECUTABLE(s16-window-test test/s16-window.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s16-window-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(s16-window-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME s16-window-test COMMAND s16-window-test)

  ADD_EXECUTABLE(u32-filterbank-accumulate-test test/u32-filterbank-accumulate.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u32-filterbank-accumulate-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u32-filterbank-accumulate-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME u32-filterbank-accumulate-test COMMAND u32-filterbank-accumulate-test)

  ADD_EXECUTABLE(u32-filterbank-subtract-test test/u32-filterbank-subtract.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u32-filterbank-subtract-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u32-filterbank-subtract-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME u32-filterbank-subtract-test COMMAND u32-filterbank-subtract-test)

  ADD_EXECUTABLE(u32-vlog-test test/u32-vlog.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u32-vlog-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u32-vlog-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME u32-vlog-test COMMAND u32-vlog-test)

  ADD_EXECUTABLE(s16-vlshift-test test/s16-vlshift.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s16-vlshift-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(s16-vlshift-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME s16-vlshift-test COMMAND s16-vlshift-test)

  ADD_EXECUTABLE(cs16-vsquareabs-test test/cs16-vsquareabs.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(cs16-vsquareabs-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(cs16-vsquareabs-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME cs16-vsquareabs-test COMMAND cs16-vsquareabs-test)

  ADD_EXECUTABLE(cs16-bfly4-test test/cs16-bfly4.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(cs16-bfly4-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(cs16-bfly4-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME cs16-bfly4-test COMMAND cs16-bfly4-test)

  ADD_EXECUTABLE(cs16-fftr-test test/cs16-fftr.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(cs16-fftr-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(cs16-fftr-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME cs16-fftr-test COMMAND cs16-fftr-test)

  ADD_EXECUTABLE(s8-ibilinear-test test/s8-ibilinear.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s8-ibilinear-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(s8-ibilinear-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME s8-ibilinear-test COMMAND s8-ibilinear-test)

  ADD_EXECUTABLE(s8-maxpool-minmax-test test/s8-maxpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s8-maxpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(s8-maxpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME s8-maxpool-minmax-test COMMAND s8-maxpool-minmax-test)

  ADD_EXECUTABLE(s8-vclamp-test test/s8-vclamp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s8-vclamp-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(s8-vclamp-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME s8-vclamp-test COMMAND s8-vclamp-test)

  ADD_EXECUTABLE(u8-lut32norm-test test/u8-lut32norm.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u8-lut32norm-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u8-lut32norm-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME u8-lut32norm-test COMMAND u8-lut32norm-test)

  ADD_EXECUTABLE(u8-ibilinear-test test/u8-ibilinear.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u8-ibilinear-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u8-ibilinear-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME u8-ibilinear-test COMMAND u8-ibilinear-test)

  ADD_EXECUTABLE(u8-maxpool-minmax-test test/u8-maxpool-minmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u8-maxpool-minmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u8-maxpool-minmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME u8-maxpool-minmax-test COMMAND u8-maxpool-minmax-test)

  ADD_EXECUTABLE(u8-rmax-test test/u8-rmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u8-rmax-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u8-rmax-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME u8-rmax-test COMMAND u8-rmax-test)

  ADD_EXECUTABLE(u8-vclamp-test test/u8-vclamp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u8-vclamp-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(u8-vclamp-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main microparams_init)
  ADD_TEST(NAME u8-vclamp-test COMMAND u8-vclamp-test)

  ADD_EXECUTABLE(x8-transpose-test test/x8-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x8-transpose-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x8-transpose-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x8-transpose-test COMMAND x8-transpose-test)

  ADD_EXECUTABLE(x16-transpose-test test/x16-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x16-transpose-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x16-transpose-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x16-transpose-test COMMAND x16-transpose-test)

  ADD_EXECUTABLE(x24-transpose-test test/x24-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x24-transpose-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x24-transpose-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x24-transpose-test COMMAND x24-transpose-test)

  ADD_EXECUTABLE(x32-packx-test test/x32-packx.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x32-packx-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x32-packx-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x32-packx-test COMMAND x32-packx-test)

  ADD_EXECUTABLE(x32-unpool-test test/x32-unpool.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x32-unpool-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x32-unpool-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x32-unpool-test COMMAND x32-unpool-test)

  ADD_EXECUTABLE(x32-transpose-test test/x32-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x32-transpose-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x32-transpose-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x32-transpose-test COMMAND x32-transpose-test)

  ADD_EXECUTABLE(x32-zip-test test/x32-zip.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x32-zip-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x32-zip-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x32-zip-test COMMAND x32-zip-test)

  ADD_EXECUTABLE(x64-transpose-test test/x64-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x64-transpose-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x64-transpose-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x64-transpose-test COMMAND x64-transpose-test)

  ADD_EXECUTABLE(x8-lut-test test/x8-lut.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x8-lut-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x8-lut-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x8-lut-test COMMAND x8-lut-test)

  ADD_EXECUTABLE(x8-zip-test test/x8-zip.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x8-zip-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(x8-zip-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME x8-zip-test COMMAND x8-zip-test)

  ADD_EXECUTABLE(xx-fill-test test/xx-fill.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(xx-fill-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(xx-fill-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME xx-fill-test COMMAND xx-fill-test)

  ADD_EXECUTABLE(xx-pad-test test/xx-pad.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(xx-pad-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(xx-pad-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME xx-pad-test COMMAND xx-pad-test)

  ADD_EXECUTABLE(xx-transpose-test test/xx-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(xx-transpose-test PRIVATE include src test)
  TARGET_LINK_LIBRARIES(xx-transpose-test PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)
  ADD_TEST(NAME xx-transpose-test COMMAND xx-transpose-test)

  ADD_EXECUTABLE(jit-test test/jit.cc)
  TARGET_INCLUDE_DIRECTORIES(jit-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(jit-test PRIVATE XNNPACK jit pthreadpool gtest gtest_main allocator)

  ADD_EXECUTABLE(aarch32-assembler-test test/aarch32-assembler.cc)
  TARGET_INCLUDE_DIRECTORIES(aarch32-assembler-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(aarch32-assembler-test PRIVATE XNNPACK jit pthreadpool gtest gtest_main allocator)

  ADD_EXECUTABLE(aarch64-assembler-test test/aarch64-assembler.cc)
  TARGET_INCLUDE_DIRECTORIES(aarch64-assembler-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(aarch64-assembler-test PRIVATE XNNPACK jit pthreadpool gtest gtest_main allocator)

  ADD_EXECUTABLE(code-cache-test test/code-cache.cc)
  TARGET_INCLUDE_DIRECTORIES(code-cache-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(code-cache-test PRIVATE XNNPACK cache jit pthreadpool gtest gtest_main microparams_init)

  ADD_EXECUTABLE(weights-cache-test test/weights-cache.cc)
  TARGET_INCLUDE_DIRECTORIES(weights-cache-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(weights-cache-test PRIVATE XNNPACK jit pthreadpool gtest gtest_main cache microparams_init)

  ADD_EXECUTABLE(mutex-test test/mutex.cc)
  TARGET_INCLUDE_DIRECTORIES(mutex-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(mutex-test PRIVATE gtest gtest_main mutex pthreadpool)

  ADD_EXECUTABLE(operator-utils-test test/operator-utils.cc)
  TARGET_INCLUDE_DIRECTORIES(operator-utils-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(operator-utils-test PRIVATE XNNPACK gtest gtest_main operators pthreadpool)

  ADD_EXECUTABLE(packing-test test/packing.cc)
  TARGET_INCLUDE_DIRECTORIES(packing-test PRIVATE include src)
  TARGET_LINK_LIBRARIES(packing-test PRIVATE XNNPACK fp16 gtest gtest_main operators pthreadpool packing)
ENDIF()

# ---[ XNNPACK microbenchmarks
IF(XNNPACK_BUILD_BENCHMARKS)
  # ---[ Build google benchmark
  IF(NOT TARGET benchmark)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark")
  ENDIF()

  ADD_LIBRARY(bench-utils bench/utils.cc)
  TARGET_INCLUDE_DIRECTORIES(bench-utils PRIVATE .)
  TARGET_INCLUDE_DIRECTORIES(bench-utils PUBLIC include src)
  TARGET_LINK_LIBRARIES(bench-utils PRIVATE XNNPACK benchmark cpuinfo jit)

  # ---[ Build accuracy microbenchmarks
  ADD_EXECUTABLE(f16-exp-ulp-eval eval/f16-exp-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-exp-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f16-exp-ulp-eval PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f16-expminus-ulp-eval eval/f16-expminus-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-expminus-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f16-expminus-ulp-eval PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f16-expm1minus-ulp-eval eval/f16-expm1minus-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-expm1minus-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f16-expm1minus-ulp-eval PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f16-sigmoid-ulp-eval eval/f16-sigmoid-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-sigmoid-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f16-sigmoid-ulp-eval PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-exp-ulp-eval eval/f32-exp-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-exp-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-exp-ulp-eval PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-expminus-ulp-eval eval/f32-expminus-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-expminus-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-expminus-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-expm1minus-ulp-eval eval/f32-expm1minus-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-expm1minus-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-expm1minus-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-extexp-ulp-eval eval/f32-extexp-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(f32-extexp-ulp-eval PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-extexp-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-extexp-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-sigmoid-ulp-eval eval/f32-sigmoid-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-sigmoid-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-sigmoid-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-sqrt-ulp-eval eval/f32-sqrt-ulp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-sqrt-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-sqrt-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  # ---[ Build accuracy tests
  ADD_EXECUTABLE(f16-f32-cvt-eval eval/f16-f32-cvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-f32-cvt-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f16-f32-cvt-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-f16-cvt-eval eval/f32-f16-cvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-f16-cvt-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-f16-cvt-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-qs8-cvt-eval eval/f32-qs8-cvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-qs8-cvt-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-qs8-cvt-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-qu8-cvt-eval eval/f32-qu8-cvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-qu8-cvt-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-qu8-cvt-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-exp-eval eval/f32-exp.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-exp-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-exp-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-expm1minus-eval eval/f32-expm1minus.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-expm1minus-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-expm1minus-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-expminus-eval eval/f32-expminus.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-expminus-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-expminus-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-roundne-eval eval/f32-roundne.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-roundne-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-roundne-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-roundd-eval eval/f32-roundd.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-roundd-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-roundd-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-roundu-eval eval/f32-roundu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-roundu-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-roundu-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(f32-roundz-eval eval/f32-roundz.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-roundz-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(f32-roundz-eval PRIVATE cpuinfo fp16 pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(u32-sqrt-eval eval/u32-sqrt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u32-sqrt-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(u32-sqrt-eval PRIVATE cpuinfo pthreadpool gtest gtest_main)

  ADD_EXECUTABLE(u64-sqrt-eval eval/u64-sqrt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u64-sqrt-eval PRIVATE include src)
  TARGET_LINK_LIBRARIES(u64-sqrt-eval PRIVATE cpuinfo pthreadpool gtest gtest_main)

  # ---[ Build end-to-end microbenchmarks
  ADD_LIBRARY(bench-models STATIC
    models/fp16-mobilenet-v1.cc
    models/fp32-mobilenet-v1.cc
    models/qc8-mobilenet-v1.cc
    models/qc8-mobilenet-v2.cc
    models/qs8-mobilenet-v1.cc
    models/qs8-mobilenet-v2.cc
    models/qu8-mobilenet-v1.cc
    models/qu8-mobilenet-v2.cc
    models/fp16-mobilenet-v2.cc
    models/fp32-mobilenet-v2.cc
    models/fp16-mobilenet-v3-large.cc
    models/fp32-mobilenet-v3-large.cc
    models/fp16-mobilenet-v3-small.cc
    models/fp32-mobilenet-v3-small.cc
    models/fp32-sparse-mobilenet-v1.cc
    models/fp32-sparse-mobilenet-v2.cc
    models/fp32-sparse-mobilenet-v3-large.cc
    models/fp32-sparse-mobilenet-v3-small.cc)
  SET_TARGET_PROPERTIES(bench-models PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(bench-models PRIVATE .)
  TARGET_LINK_LIBRARIES(bench-models PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(end2end-bench bench/end2end.cc)
  TARGET_INCLUDE_DIRECTORIES(end2end-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(end2end-bench PRIVATE XNNPACK benchmark bench-models bench-utils cache microparams_init logging operators)

  ADD_EXECUTABLE(f32-dwconv-e2e-bench bench/f32-dwconv-e2e.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-e2e-bench PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-dwconv-e2e-bench PRIVATE XNNPACK benchmark bench-models bench-utils cache microparams_init logging operators)

  ADD_EXECUTABLE(f32-gemm-e2e-bench bench/f32-gemm-e2e.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-e2e-bench PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-gemm-e2e-bench PRIVATE XNNPACK fp16 benchmark bench-models bench-utils cache jit microparams_init logging operators)

  ADD_EXECUTABLE(qs8-dwconv-e2e-bench bench/qs8-dwconv-e2e.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-dwconv-e2e-bench PRIVATE . src)
  TARGET_LINK_LIBRARIES(qs8-dwconv-e2e-bench PRIVATE XNNPACK fp16 benchmark bench-models bench-utils cache microparams_init logging operators)

  ADD_EXECUTABLE(qs8-gemm-e2e-bench bench/qs8-gemm-e2e.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-e2e-bench PRIVATE . src)
  TARGET_LINK_LIBRARIES(qs8-gemm-e2e-bench PRIVATE XNNPACK fp16 benchmark bench-models bench-utils cache microparams_init logging operators)

  ADD_EXECUTABLE(qu8-gemm-e2e-bench bench/qu8-gemm-e2e.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-gemm-e2e-bench PRIVATE . src)
  TARGET_LINK_LIBRARIES(qu8-gemm-e2e-bench PRIVATE XNNPACK fp16 benchmark bench-models bench-utils cache microparams_init logging operators)

  ADD_EXECUTABLE(qu8-dwconv-e2e-bench bench/qu8-dwconv-e2e.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-dwconv-e2e-bench PRIVATE . src)
  TARGET_LINK_LIBRARIES(qu8-dwconv-e2e-bench PRIVATE XNNPACK fp16 benchmark bench-models bench-utils cache microparams_init logging operators)

  # ---[ Build operator-level microbenchmarks
  ADD_EXECUTABLE(abs-bench bench/abs.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(abs-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(abs-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(average-pooling-bench bench/average-pooling.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(average-pooling-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(average-pooling-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(bankers-rounding-bench bench/bankers-rounding.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(bankers-rounding-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(bankers-rounding-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(ceiling-bench bench/ceiling.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(ceiling-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(ceiling-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(channel-shuffle-bench bench/channel-shuffle.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(channel-shuffle-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(channel-shuffle-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(convert-bench bench/convert.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(convert-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(convert-bench PRIVATE XNNPACK fp16 benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(convolution-bench bench/convolution.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(convolution-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(convolution-bench PRIVATE XNNPACK cache fp16 benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(deconvolution-bench bench/deconvolution.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(deconvolution-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(elu-bench bench/elu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(elu-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(elu-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(floor-bench bench/floor.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(floor-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(floor-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(global-average-pooling-bench bench/global-average-pooling.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(global-average-pooling-bench PRIVATE XNNPACK fp16 benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(hardswish-bench bench/hardswish.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(hardswish-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(hardswish-bench PRIVATE XNNPACK fp16 benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(leaky-relu-bench bench/leaky-relu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(leaky-relu-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(leaky-relu-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(max-pooling-bench bench/max-pooling.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(max-pooling-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(negate-bench bench/negate.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(negate-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(negate-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(prelu-bench bench/prelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(prelu-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(prelu-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(sigmoid-bench bench/sigmoid.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(sigmoid-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(sigmoid-bench PRIVATE XNNPACK fp16 benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(softmax-bench bench/softmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(softmax-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(softmax-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(square-bench bench/square.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(square-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(square-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging logging operators)

  ADD_EXECUTABLE(square-root-bench bench/square-root.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(square-root-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(square-root-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  ADD_EXECUTABLE(truncation-bench bench/truncation.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(truncation-bench PRIVATE .)
  TARGET_LINK_LIBRARIES(truncation-bench PRIVATE XNNPACK benchmark bench-utils microparams_init logging operators)

  # ---[ Build microkernel-level microbenchmarks
  ADD_EXECUTABLE(bf16-gemm-bench bench/bf16-gemm.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(bf16-gemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(bf16-gemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-dwconv-bench bench/f16-dwconv.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:indirection> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-dwconv-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-dwconv-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-dwconv2d-chw-bench bench/f16-dwconv2d-chw.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-dwconv2d-chw-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-dwconv2d-chw-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-gemm-bench bench/f16-gemm.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:indirection> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-gemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-igemm-bench bench/f16-igemm.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:indirection> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-igemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-raddstoreexpminusmax-bench bench/f16-raddstoreexpminusmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-raddstoreexpminusmax-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-raddstoreexpminusmax-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-velu-bench bench/f16-velu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-velu-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-velu-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-vsigmoid-bench bench/f16-vsigmoid.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-vsigmoid-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-vsigmoid-bench PRIVATE benchmark bench-utils fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f16-f32-vcvt-bench bench/f16-f32-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f16-f32-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f16-f32-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-conv-hwc-bench bench/f32-conv-hwc.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-conv-hwc-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-dwconv-bench bench/f32-dwconv.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:indirection> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-dwconv-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-dwconv2d-chw-bench bench/f32-dwconv2d-chw.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv2d-chw-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-dwconv2d-chw-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-f16-vcvt-bench bench/f32-f16-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-f16-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-f16-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-gemm-bench bench/f32-gemm.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-gemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-igemm-bench bench/f32-igemm.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:indirection> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-igemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-im2col-gemm-bench bench/f32-im2col-gemm.cc src/im2col.c $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(f32-im2col-gemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-im2col-gemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-qs8-vcvt-bench bench/f32-qs8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-qs8-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-qs8-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-qu8-vcvt-bench bench/f32-qu8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-qu8-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-qu8-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-raddstoreexpminusmax-bench bench/f32-raddstoreexpminusmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-raddstoreexpminusmax-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-raddstoreexpminusmax-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-rmax-bench bench/f32-rmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-rmax-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-rmax-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-spmm-bench bench/f32-spmm.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-spmm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-spmm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-softmax-bench bench/f32-softmax.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-softmax-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-softmax-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-velu-bench bench/f32-velu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-velu-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-velu-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-vhswish-bench bench/f32-vhswish.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vhswish-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-vhswish-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-vlrelu-bench bench/f32-vlrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vlrelu-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-vlrelu-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-vrelu-bench bench/f32-vrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vrelu-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-vrelu-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-vsigmoid-bench bench/f32-vsigmoid.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsigmoid-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-vsigmoid-bench PRIVATE benchmark bench-utils fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(f32-vsqrt-bench bench/f32-vsqrt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(f32-vsqrt-bench PRIVATE benchmark bench-utils fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-dwconv-bench bench/qs8-dwconv.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:indirection> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-dwconv-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-dwconv-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-f32-vcvt-bench bench/qs8-f32-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-f32-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-f32-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-gemm-bench bench/qs8-gemm.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-gemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool jit microparams_init)

  ADD_EXECUTABLE(qs8-requantization-bench bench/qs8-requantization.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qs8-requantization-bench PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-requantization-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-requantization-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(qs8-vadd-bench bench/qs8-vadd.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vadd-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-vadd-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-vaddc-bench bench/qs8-vaddc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vaddc-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-vaddc-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-vcvt-bench bench/qs8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-vlrelu-bench bench/qs8-vlrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vlrelu-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-vlrelu-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-vmul-bench bench/qs8-vmul.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vmul-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-vmul-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qs8-vmulc-bench bench/qs8-vmulc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qs8-vmulc-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qs8-vmulc-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-f32-vcvt-bench bench/qu8-f32-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-f32-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-f32-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-gemm-bench bench/qu8-gemm.cc $<TARGET_OBJECTS:all_microkernels> $<TARGET_OBJECTS:packing>)
  TARGET_INCLUDE_DIRECTORIES(qu8-gemm-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-gemm-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-requantization-bench bench/qu8-requantization.cc $<TARGET_OBJECTS:all_microkernels>)
  SET_TARGET_PROPERTIES(qu8-requantization-bench PROPERTIES CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-requantization-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-requantization-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(qu8-vadd-bench bench/qu8-vadd.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vadd-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-vadd-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-vaddc-bench bench/qu8-vaddc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vaddc-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-vaddc-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-vcvt-bench bench/qu8-vcvt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vcvt-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-vcvt-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-vlrelu-bench bench/qu8-vlrelu.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vlrelu-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-vlrelu-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-vmul-bench bench/qu8-vmul.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vmul-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-vmul-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(qu8-vmulc-bench bench/qu8-vmulc.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(qu8-vmulc-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(qu8-vmulc-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(rounding-bench bench/rounding.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(rounding-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(rounding-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(s16-rmaxabs-bench bench/s16-rmaxabs.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s16-rmaxabs-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(s16-rmaxabs-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(s16-window-bench bench/s16-window.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s16-window-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(s16-window-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(u32-filterbank-accumulate-bench bench/u32-filterbank-accumulate.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u32-filterbank-accumulate-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(u32-filterbank-accumulate-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(u32-filterbank-subtract-bench bench/u32-filterbank-subtract.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u32-filterbank-subtract-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(u32-filterbank-subtract-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(u32-vlog-bench bench/u32-vlog.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u32-vlog-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(u32-vlog-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(u64-u32-vsqrtshift-bench bench/f32-vsqrt.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(u64-u32-vsqrtshift-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(u64-u32-vsqrtshift-bench PRIVATE benchmark bench-utils fp16 pthreadpool microparams_init)

  ADD_EXECUTABLE(s16-vlshift-bench bench/s16-vlshift.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(s16-vlshift-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(s16-vlshift-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(cs16-vsquareabs-bench bench/cs16-vsquareabs.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(cs16-vsquareabs-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(cs16-vsquareabs-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(cs16-bfly4-bench bench/cs16-bfly4.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(cs16-bfly4-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(cs16-bfly4-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(cs16-fftr-bench bench/cs16-fftr.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(cs16-fftr-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(cs16-fftr-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(xx-transpose-bench bench/x32-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(xx-transpose-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(xx-transpose-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(x8-lut-bench bench/x8-lut.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x8-lut-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(x8-lut-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(x8-transpose-bench bench/x32-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x8-transpose-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(x8-transpose-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(x16-transpose-bench bench/x16-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x16-transpose-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(x16-transpose-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(x24-transpose-bench bench/x16-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x24-transpose-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(x24-transpose-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(x32-transpose-bench bench/x32-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x32-transpose-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(x32-transpose-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(x64-transpose-bench bench/x32-transpose.cc $<TARGET_OBJECTS:all_microkernels>)
  TARGET_INCLUDE_DIRECTORIES(x64-transpose-bench PRIVATE . include src)
  TARGET_LINK_LIBRARIES(x64-transpose-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)
ENDIF()
