# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

# ---[ Project and semantic versioning.
PROJECT(XNNPACK C CXX ASM)

# ---[ Options.
SET(XNNPACK_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE XNNPACK_LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(XNNPACK_ENABLE_ASSEMBLY "Build XNNPACK with assembly micro-kernels" ON)
OPTION(XNNPACK_ENABLE_MEMOPT "Build XNNPACK with optimized memory allocation scheme" ON)
OPTION(XNNPACK_ENABLE_SPARSE "Build XNNPACK with graph rewriting for sparse inference" ON)
OPTION(XNNPACK_BUILD_TESTS "Build XNNPACK unit tests" ON)
OPTION(XNNPACK_BUILD_BENCHMARKS "Build XNNPACK benchmarks" ON)
OPTION(XNNPACK_USE_SYSTEM_LIBS "Use system-provided dependency libraries" OFF)

# --- [ Determine target processor
SET(XNNPACK_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
IF(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|arm64|arm64e)$")
  SET(XNNPACK_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
ENDIF()

# ---[ CMake options
INCLUDE(GNUInstallDirs)

IF(XNNPACK_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

IF(XNNPACK_ENABLE_ASSEMBLY)
  ADD_DEFINITIONS(-DXNN_ENABLE_ASSEMBLY=1)
ELSE()
  ADD_DEFINITIONS(-DXNN_ENABLE_ASSEMBLY=0)
ENDIF()

IF(XNNPACK_ENABLE_MEMOPT)
  ADD_DEFINITIONS(-DXNN_ENABLE_MEMOPT=1)
ELSE()
  ADD_DEFINITIONS(-DXNN_ENABLE_MEMOPT=0)
ENDIF()

IF(XNNPACK_ENABLE_SPARSE)
  ADD_DEFINITIONS(-DXNN_ENABLE_SPARSE=1)
ELSE()
  ADD_DEFINITIONS(-DXNN_ENABLE_SPARSE=0)
ENDIF()

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
SET(XNNPACK_OPERATOR_SRCS
  src/operators/argmax-pooling-nhwc.c
  src/operators/average-pooling-nhwc.c
  src/operators/binary-elementwise-nd.c
  src/operators/channel-shuffle-nc.c
  src/operators/constant-pad-nd.c
  src/operators/convolution-nchw.c
  src/operators/convolution-nhwc.c
  src/operators/deconvolution-nhwc.c
  src/operators/depth-to-space-nchw2nhwc.c
  src/operators/depth-to-space-nhwc.c
  src/operators/fully-connected-nc.c
  src/operators/global-average-pooling-ncw.c
  src/operators/global-average-pooling-nwc.c
  src/operators/leaky-relu-nc.c
  src/operators/max-pooling-nhwc.c
  src/operators/prelu-nc.c
  src/operators/resize-bilinear-nchw.c
  src/operators/resize-bilinear-nhwc.c
  src/operators/sigmoid-nc.c
  src/operators/softmax-nc.c
  src/operators/unary-elementwise-nc.c
  src/operators/unpooling-nhwc.c)

SET(XNNPACK_SUBGRAPH_SRCS
  src/subgraph/abs.c
  src/subgraph/add2.c
  src/subgraph/argmax-pooling-2d.c
  src/subgraph/average-pooling-2d.c
  src/subgraph/bankers-rounding.c
  src/subgraph/ceiling.c
  src/subgraph/clamp.c
  src/subgraph/convolution-2d.c
  src/subgraph/deconvolution-2d.c
  src/subgraph/depth-to-space.c
  src/subgraph/depthwise-convolution-2d.c
  src/subgraph/divide.c
  src/subgraph/elu.c
  src/subgraph/floor.c
  src/subgraph/fully-connected.c
  src/subgraph/global-average-pooling-2d.c
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
  src/subgraph/subtract.c
  src/subgraph/unpooling-2d.c)

SET(XNNPACK_LOGGING_SRCS
  src/datatype-strings.c
  src/operator-strings.c
  src/subgraph-strings.c)

SET(XNNPACK_COLD_SRCS ${XNNPACK_OPERATOR_SRCS} ${XNNPACK_SUBGRAPH_SRCS} ${XNNPACK_LOGGING_SRCS})
LIST(APPEND XNNPACK_COLD_SRCS
  src/allocator.c
  src/init.c
  src/memory-planner.c
  src/operator-delete.c
  src/runtime.c
  src/subgraph.c
  src/tensor.c)

SET(XNNPACK_HOT_SRCS
  src/indirection.c
  src/operator-run.c
  src/packing.c)

SET(XNNPACK_TABLE_SRCS
  src/tables/exp2-k-over-64.c
  src/tables/exp2-k-over-2048.c
  src/tables/exp2minus-k-over-4.c
  src/tables/exp2minus-k-over-8.c
  src/tables/exp2minus-k-over-16.c
  src/tables/exp2minus-k-over-64.c
  src/tables/exp2minus-k-over-2048.c)

SET(XNNPACK_SCALAR_MICROKERNEL_SRCS
  src/f32-argmaxpool/4x-scalar-c1.c
  src/f32-argmaxpool/9p8x-scalar-c1.c
  src/f32-argmaxpool/9x-scalar-c1.c
  src/f32-avgpool/9p8x-minmax-scalar-c1.c
  src/f32-avgpool/9x-minmax-scalar-c1.c
  src/f32-conv-hwc/3x3s2p0p1c3x4-scalar-1x1.c
  src/f32-conv-hwc/3x3s2p1c3x4-scalar-1x1.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-scalar-1x1.c
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
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x1.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x2-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x2.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4-acc4.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x1.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x2-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x2.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x4-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x4-acc4.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x4.c
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
  src/f32-vsigmoid/gen/vsigmoid-scalar-lut64-p2-div-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-lut64-p2-div-x2.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-lut64-p2-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-lut2048-p1-div-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-lut2048-p1-div-x2.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-lut2048-p1-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-p5-div-x1.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-p5-div-x2.c
  src/f32-vsigmoid/gen/vsigmoid-scalar-p5-div-x4.c
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
  src/math/expm1minus-scalar-rr2-lut4-p4.c
  src/math/expm1minus-scalar-rr2-lut8-p3.c
  src/math/expm1minus-scalar-rr2-lut8-p4.c
  src/math/expm1minus-scalar-rr2-lut16-p3.c
  src/math/expm1minus-scalar-rr2-lut16-p4.c
  src/math/expm1minus-scalar-rr2-p5.c
  src/math/expm1minus-scalar-rr2-p6.c
  src/math/expminus-scalar-rr2-lut64-p2.c
  src/math/expminus-scalar-rr2-lut2048-p1.c
  src/math/expminus-scalar-rr2-p5.c
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
  src/math/sigmoid-scalar-rr2-lut64-p2-div.c
  src/math/sigmoid-scalar-rr2-lut2048-p1-div.c
  src/math/sigmoid-scalar-rr2-p5-div.c
  src/qs8-dwconv/gen/up1x9-minmax-gemmlowp-scalar.c
  src/qs8-dwconv/gen/up1x25-minmax-gemmlowp-scalar.c
  src/qs8-dwconv/gen/up2x9-minmax-gemmlowp-scalar.c
  src/qs8-dwconv/gen/up2x25-minmax-gemmlowp-scalar.c
  src/qs8-dwconv/gen/up4x9-minmax-gemmlowp-scalar.c
  src/qs8-dwconv/gen/up4x25-minmax-gemmlowp-scalar.c
  src/qs8-gavgpool/gen/7p7x-minmax-scalar-c1.c
  src/qs8-gavgpool/gen/7p7x-minmax-scalar-c2.c
  src/qs8-gavgpool/gen/7p7x-minmax-scalar-c4.c
  src/qs8-gavgpool/gen/7x-minmax-scalar-c1.c
  src/qs8-gavgpool/gen/7x-minmax-scalar-c2.c
  src/qs8-gavgpool/gen/7x-minmax-scalar-c4.c
  src/qs8-gemm/gen/1x2-minmax-gemmlowp-scalar.c
  src/qs8-gemm/gen/1x4-minmax-gemmlowp-scalar.c
  src/qs8-gemm/gen/2x2-minmax-gemmlowp-scalar.c
  src/qs8-gemm/gen/2x4-minmax-gemmlowp-scalar.c
  src/qs8-gemm/gen/3x2-minmax-gemmlowp-scalar.c
  src/qs8-gemm/gen/3x4-minmax-gemmlowp-scalar.c
  src/qs8-gemm/gen/4x2-minmax-gemmlowp-scalar.c
  src/qs8-gemm/gen/4x4-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/1x2-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/1x4-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/2x2-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/2x4-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/3x2-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/3x4-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/4x2-minmax-gemmlowp-scalar.c
  src/qs8-igemm/gen/4x4-minmax-gemmlowp-scalar.c
  src/qs8-requantization/fp32-scalar-lrintf.c
  src/qs8-requantization/fp32-scalar-magic.c
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
  src/qu8-avgpool/9p8x-minmax-scalar-c1.c
  src/qu8-avgpool/9x-minmax-scalar-c1.c
  src/qu8-dwconv/up1x9-minmax-scalar.c
  src/qu8-gavgpool/7p7x-minmax-scalar-c1.c
  src/qu8-gavgpool/7x-minmax-scalar-c1.c
  src/qu8-gemm/2x2-minmax-scalar.c
  src/qu8-igemm/2x2-minmax-scalar.c
  src/qu8-requantization/fp32-scalar-lrintf.c
  src/qu8-requantization/fp32-scalar-magic.c
  src/qu8-requantization/gemmlowp-scalar.c
  src/qu8-requantization/rndna-scalar-signed64.c
  src/qu8-requantization/rndna-scalar-unsigned32.c
  src/qu8-requantization/rndna-scalar-unsigned64.c
  src/qu8-vadd/minmax-scalar.c
  src/u8-lut32norm/scalar.c
  src/u8-maxpool/9p8x-minmax-scalar-c1.c
  src/u8-rmax/scalar.c
  src/u8-vclamp/scalar-x4.c
  src/x8-lut/scalar.c
  src/x8-zip/x2-scalar.c
  src/x8-zip/x3-scalar.c
  src/x8-zip/x4-scalar.c
  src/x8-zip/xm-scalar.c
  src/x32-depthtospace2d-chw2hwc/scalar.c
  src/x32-fill/scalar-float.c
  src/x32-fill/scalar-int.c
  src/x32-packx/x2-scalar.c
  src/x32-packx/x3-scalar.c
  src/x32-packx/x4-scalar.c
  src/x32-pad/scalar-float.c
  src/x32-pad/scalar-int.c
  src/x32-unpool/scalar.c
  src/x32-zip/x2-scalar.c
  src/x32-zip/x3-scalar.c
  src/x32-zip/x4-scalar.c
  src/x32-zip/xm-scalar.c
  src/xx-copy/memcpy.c)

SET(XNNPACK_NEON_MICROKERNEL_SRCS
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
  src/f32-dwconv/gen/up4x4-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-neon.c
  src/f32-dwconv/gen/up4x9-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-neon.c
  src/f32-dwconv/gen/up4x25-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-neon.c
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
  src/f32-gemm/gen/6x8-minmax-neon-dup-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-dup-ld128.c
  src/f32-gemm/gen/6x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-lane-ld128.c
  src/f32-gemm/gen/6x8s4-minmax-neon.c
  src/f32-gemm/gen/8x8s4-minmax-neon.c
  src/f32-ibilinear-chw/gen/neon-p4.c
  src/f32-ibilinear-chw/gen/neon-p8.c
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
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x8.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x20.c
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
  src/math/expm1minus-neon-rr2-lut16-p3.c
  src/math/expm1minus-neon-rr2-p6.c
  src/math/roundd-neon-addsub.c
  src/math/roundd-neon-cvt.c
  src/math/roundne-neon-addsub.c
  src/math/roundu-neon-addsub.c
  src/math/roundu-neon-cvt.c
  src/math/roundz-neon-addsub.c
  src/math/roundz-neon-cvt.c
  src/math/sigmoid-neon-rr2-lut64-p2-nr2recps.c
  src/math/sigmoid-neon-rr2-lut2048-p1-nr2recps.c
  src/math/sigmoid-neon-rr2-p5-nr2recps.c
  src/math/sqrt-neon-nr1rsqrts.c
  src/math/sqrt-neon-nr2rsqrts.c
  src/math/sqrt-neon-nr3rsqrts.c
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-neon-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-neon-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-neon-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-neon-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-neon-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-neon-mul16.c
  src/qs8-dwconv/gen/up32x9-minmax-gemmlowp-neon-mul16.c
  src/qs8-dwconv/gen/up32x25-minmax-gemmlowp-neon-mul16.c
  src/qs8-gavgpool/gen/7p7x-minmax-neon-c8-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-neon-c16-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-neon-c24-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-neon-c32-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-neon-c8-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-neon-c16-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-neon-c24-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-neon-c32-acc2.c
  src/qs8-gemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/1x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/1x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/1x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/1x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/1x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/1x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/1x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/1x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/1x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/1x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/2x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/2x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/2x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/2x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/2x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/2x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/2x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/2x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/2x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/2x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/3x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/3x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/3x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/3x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/3x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/3x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/3x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/3x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/3x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/3x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/3x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/3x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/4x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/4x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/4x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/4x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/4x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/4x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/4x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-gemm/gen/4x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-gemm/gen/4x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-gemm/gen/4x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/4x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-gemm/gen/4x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-gemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-gemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-gemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/1x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/1x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/1x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/1x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/1x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/1x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/1x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/1x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/1x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/1x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/2x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/2x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/2x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/2x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/2x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/2x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/2x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/2x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/2x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/2x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/3x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/3x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/3x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/3x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/3x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/3x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/3x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/3x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/3x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/3x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/3x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/3x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/4x8-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/4x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/4x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/4x8c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/4x8c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/4x8c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/4x16-minmax-gemmlowp-neon-mull-addw-dup.c
  src/qs8-igemm/gen/4x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
  src/qs8-igemm/gen/4x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
  src/qs8-igemm/gen/4x16c8-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/4x16c8-minmax-gemmlowp-neon-mull-padal.c
  src/qs8-igemm/gen/4x16c16-minmax-gemmlowp-neon-mlal-padal.c
  src/qs8-igemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-igemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
  src/qs8-igemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane.c
  src/qs8-requantization/fp32-neon.c
  src/qs8-requantization/gemmlowp-neon.c
  src/qs8-requantization/rndna-neon.c
  src/qs8-requantization/rndnu-neon.c
  src/qs8-vadd/gen/minmax-neon-ld64-x8.c
  src/qs8-vadd/gen/minmax-neon-ld64-x16.c
  src/qs8-vadd/gen/minmax-neon-ld64-x24.c
  src/qs8-vadd/gen/minmax-neon-ld64-x32.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x8.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x16.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x24.c
  src/qs8-vaddc/gen/minmax-neon-ld64-x32.c
  src/qu8-avgpool/9p8x-minmax-neon-c8.c
  src/qu8-avgpool/9x-minmax-neon-c8.c
  src/qu8-dwconv/up8x9-minmax-neon.c
  src/qu8-gavgpool/7p7x-minmax-neon-c8.c
  src/qu8-gavgpool/7x-minmax-neon-c8.c
  src/qu8-gemm/4x8-minmax-neon.c
  src/qu8-gemm/8x8-minmax-neon.c
  src/qu8-igemm/4x8-minmax-neon.c
  src/qu8-igemm/8x8-minmax-neon.c
  src/qu8-requantization/fp32-neon.c
  src/qu8-requantization/gemmlowp-neon.c
  src/qu8-requantization/rndna-neon.c
  src/qu8-vadd/minmax-neon.c
  src/u8-maxpool/9p8x-minmax-neon-c16.c
  src/u8-rmax/neon.c
  src/u8-vclamp/neon-x64.c
  src/x8-zip/x2-neon.c
  src/x8-zip/x3-neon.c
  src/x8-zip/x4-neon.c
  src/x8-zip/xm-neon.c
  src/x32-fill/neon.c
  src/x32-packx/x4-neon-st4.c
  src/x32-pad/neon.c
  src/x32-unpool/neon.c
  src/x32-zip/x2-neon.c
  src/x32-zip/x3-neon.c
  src/x32-zip/x4-neon.c
  src/x32-zip/xm-neon.c)

SET(XNNPACK_NEONFMA_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up4x4-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-neonfma.c
  src/f32-dwconv/gen/up4x9-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-neonfma.c
  src/f32-dwconv/gen/up4x25-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-neonfma.c
  src/f32-dwconv/gen/up8x4-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-neonfma.c
  src/f32-dwconv/gen/up8x9-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-neonfma.c
  src/f32-dwconv/gen/up8x25-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-neonfma.c
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
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x8.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20.c
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
  src/math/exp-neonfma-rr2-lut64-p2.c
  src/math/exp-neonfma-rr2-p5.c
  src/math/expm1minus-neonfma-rr1-lut16-p3.c
  src/math/expm1minus-neonfma-rr1-p6.c
  src/math/expminus-neonfma-rr2-lut64-p2.c
  src/math/expminus-neonfma-rr2-lut2048-p1.c
  src/math/expminus-neonfma-rr2-p5.c
  src/math/sigmoid-neonfma-rr1-lut64-p2-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr1-lut64-p2-nr2fma.c
  src/math/sigmoid-neonfma-rr1-lut64-p2-nr2recps.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-nr2fma.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-nr2recps.c
  src/math/sigmoid-neonfma-rr1-p5-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr1-p5-nr2fma.c
  src/math/sigmoid-neonfma-rr1-p5-nr2recps.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-nr2fma.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-nr2recps.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-nr2fma.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-nr2recps.c
  src/math/sigmoid-neonfma-rr2-p5-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr2-p5-nr2fma.c
  src/math/sigmoid-neonfma-rr2-p5-nr2recps.c
  src/math/sqrt-neonfma-nr1fma.c
  src/math/sqrt-neonfma-nr1rsqrts1fma1adj.c
  src/math/sqrt-neonfma-nr2fma.c
  src/math/sqrt-neonfma-nr2fma1adj.c
  src/math/sqrt-neonfma-nr3fma.c)

SET(XNNPACK_NEONV8_MICROKERNEL_SRCS
  src/f32-vrnd/gen/vrndd-neonv8-x4.c
  src/f32-vrnd/gen/vrndd-neonv8-x8.c
  src/f32-vrnd/gen/vrndne-neonv8-x4.c
  src/f32-vrnd/gen/vrndne-neonv8-x8.c
  src/f32-vrnd/gen/vrndu-neonv8-x4.c
  src/f32-vrnd/gen/vrndu-neonv8-x8.c
  src/f32-vrnd/gen/vrndz-neonv8-x4.c
  src/f32-vrnd/gen/vrndz-neonv8-x8.c
  src/math/roundd-neonv8.c
  src/math/roundne-neonv8.c
  src/math/roundu-neonv8.c
  src/math/roundz-neonv8.c)

SET(XNNPACK_AARCH64_NEONFMA_MICROKERNEL_SRCS
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
  src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld128.c
  src/f32-igemm/gen/1x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x2-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x4-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neonfma-lane-ld128.c
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
  src/math/sigmoid-neonfma-rr1-lut64-p2-div.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-div.c
  src/math/sigmoid-neonfma-rr1-p5-div.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-div.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-div.c
  src/math/sigmoid-neonfma-rr2-p5-div.c)

SET(XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS
  src/f16-dwconv/gen/up8x4-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up8x4-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up8x9-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up8x9-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up8x25-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up8x25-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x4-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x4-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x9-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x9-minmax-neonfp16arith.c
  src/f16-dwconv/gen/up16x25-minmax-neonfp16arith-acc2.c
  src/f16-dwconv/gen/up16x25-minmax-neonfp16arith.c
  src/f16-gavgpool/7p7x-minmax-neonfp16arith-c8.c
  src/f16-gavgpool/7x-minmax-neonfp16arith-c8.c
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
  src/f16-igemm/gen/1x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/1x16-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/4x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/4x16-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/6x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/6x16-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/8x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/8x16-minmax-neonfp16arith-ld64.c
  src/f16-prelu/gen/neonfp16arith-2x8.c
  src/f16-prelu/gen/neonfp16arith-2x16.c
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
  src/f16-vbinary/gen/vdiv-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vdiv-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vdivc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vdivc-minmax-neonfp16arith-x16.c
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
  src/f16-vbinary/gen/vrdivc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vrdivc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vrsubc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vrsubc-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsub-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vsub-minmax-neonfp16arith-x16.c
  src/f16-vbinary/gen/vsubc-minmax-neonfp16arith-x8.c
  src/f16-vbinary/gen/vsubc-minmax-neonfp16arith-x16.c
  src/f16-vclamp/gen/vclamp-neonfp16arith-x8.c
  src/f16-vclamp/gen/vclamp-neonfp16arith-x16.c
  src/f16-vhswish/gen/vhswish-neonfp16arith-x8.c
  src/f16-vhswish/gen/vhswish-neonfp16arith-x16.c
  src/f16-vmulcaddc/gen/c8-minmax-neonfp16arith-2x.c
  src/f16-vmulcaddc/gen/c16-minmax-neonfp16arith-2x.c
  src/f16-vrelu/gen/vrelu-neonfp16arith-x8.c
  src/f16-vrelu/gen/vrelu-neonfp16arith-x16.c)

SET(XNNPACK_NEONDOT_MICROKERNEL_SRCS
  src/qs8-gemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/1x8c4-minmax-gemmlowp-neondot.c
  src/qs8-gemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/1x16c4-minmax-gemmlowp-neondot.c
  src/qs8-gemm/gen/4x8c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/4x8c4-minmax-gemmlowp-neondot.c
  src/qs8-gemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-neondot.c
  src/qs8-gemm/gen/6x8c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/6x8c4-minmax-gemmlowp-neondot.c
  src/qs8-gemm/gen/6x16c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/6x16c4-minmax-gemmlowp-neondot.c
  src/qs8-gemm/gen/8x8c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/8x8c4-minmax-gemmlowp-neondot.c
  src/qs8-gemm/gen/8x16c4-minmax-fp32-neondot.c
  src/qs8-gemm/gen/8x16c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/1x8c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/1x8c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/1x16c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/1x16c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/4x8c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/4x8c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/4x16c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/6x8c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/6x8c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/6x16c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/6x16c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/8x8c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/8x8c4-minmax-gemmlowp-neondot.c
  src/qs8-igemm/gen/8x16c4-minmax-fp32-neondot.c
  src/qs8-igemm/gen/8x16c4-minmax-gemmlowp-neondot.c)

SET(XNNPACK_SSE_MICROKERNEL_SRCS
  src/f32-avgpool/9p8x-minmax-sse-c4.c
  src/f32-avgpool/9x-minmax-sse-c4.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-sse-1x1.c
  src/f32-conv-hwc2chw/3x3s2p1c3x4-sse-2x2.c
  src/f32-dwconv/gen/up4x4-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-sse.c
  src/f32-dwconv/gen/up4x9-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-sse.c
  src/f32-dwconv/gen/up4x25-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-sse.c
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
  src/x32-fill/sse.c
  src/x32-packx/x4-sse.c
  src/x32-pad/sse.c)

SET(XNNPACK_SSE2_MICROKERNEL_SRCS
  src/f32-argmaxpool/4x-sse2-c4.c
  src/f32-argmaxpool/9p8x-sse2-c4.c
  src/f32-argmaxpool/9x-sse2-c4.c
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
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x20.c
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
  src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x24.c
  src/math/exp-sse2-rr2-lut64-p2.c
  src/math/exp-sse2-rr2-p5.c
  src/math/expm1minus-sse2-rr2-lut16-p3.c
  src/math/expm1minus-sse2-rr2-p6.c
  src/math/expminus-sse2-rr2-p5.c
  src/math/roundd-sse2-cvt.c
  src/math/roundne-sse2-cvt.c
  src/math/roundu-sse2-cvt.c
  src/math/roundz-sse2-cvt.c
  src/math/sigmoid-sse2-rr2-lut64-p2-div.c
  src/math/sigmoid-sse2-rr2-lut64-p2-nr1.c
  src/math/sigmoid-sse2-rr2-lut64-p2-nr2.c
  src/math/sigmoid-sse2-rr2-p5-div.c
  src/math/sigmoid-sse2-rr2-p5-nr1.c
  src/math/sigmoid-sse2-rr2-p5-nr2.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-sse2-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-sse2-mul16.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-sse2-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-sse2-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-sse2-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-sse2-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-sse2-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-sse2-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-sse2-mul16.c
  src/qs8-gavgpool/gen/7p7x-minmax-sse2-c8-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-sse2-c16-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-sse2-c24-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-sse2-c8-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-sse2-c16-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-sse2-c24-acc2.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse2-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-sse2.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse2-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse2-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-sse2.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse2-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-sse2.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse2-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse2-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-sse2.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse2-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-sse2.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse2-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse2-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-sse2.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse2-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-sse2.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse2-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse2-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse2-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse2-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse2-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse2-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse2-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse2-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse2-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse2-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse2-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse2-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse2-ld128.c
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
  src/qu8-avgpool/9p8x-minmax-sse2-c8.c
  src/qu8-avgpool/9x-minmax-sse2-c8.c
  src/qu8-dwconv/up8x9-minmax-sse2.c
  src/qu8-gavgpool/7p7x-minmax-sse2-c8.c
  src/qu8-gavgpool/7x-minmax-sse2-c8.c
  src/qu8-gemm/2x4c8-minmax-sse2.c
  src/qu8-gemm/4x4c2-minmax-sse2.c
  src/qu8-igemm/4x4c2-minmax-sse2.c
  src/qu8-requantization/fp32-sse2.c
  src/qu8-requantization/gemmlowp-sse2.c
  src/qu8-requantization/rndna-sse2.c
  src/qu8-vadd/minmax-sse2.c
  src/u8-maxpool/9p8x-minmax-sse2-c16.c
  src/u8-rmax/sse2.c
  src/u8-vclamp/sse2-x64.c
  src/x8-zip/x2-sse2.c
  src/x8-zip/x3-sse2.c
  src/x8-zip/x4-sse2.c
  src/x8-zip/xm-sse2.c
  src/x32-unpool/sse2.c
  src/x32-zip/x2-sse2.c
  src/x32-zip/x3-sse2.c
  src/x32-zip/x4-sse2.c
  src/x32-zip/xm-sse2.c)

SET(XNNPACK_SSSE3_MICROKERNEL_SRCS
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
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-ssse3-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-ssse3-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-ssse3-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-ssse3-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-ssse3-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-ssse3-mul16.c
  src/qs8-gavgpool/gen/7p7x-minmax-ssse3-c8-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-ssse3-c16-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-ssse3-c24-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-ssse3-c8-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-ssse3-c16-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-ssse3-c24-acc2.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-ssse3.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-ssse3.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-ssse3.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-ssse3.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-ssse3.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-ssse3.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-ssse3.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-ssse3-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-ssse3-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld128.c
  src/qs8-requantization/gemmlowp-ssse3.c
  src/qs8-requantization/rndna-ssse3.c
  src/qu8-requantization/gemmlowp-ssse3.c
  src/qu8-requantization/rndna-ssse3.c)

SET(XNNPACK_SSE41_MICROKERNEL_SRCS
  src/f32-prelu/gen/sse41-2x4.c
  src/f32-prelu/gen/sse41-2x8.c
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
  src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x24.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x4.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x8.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x12.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x16.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x20.c
  src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x24.c
  src/math/roundd-sse41.c
  src/math/roundne-sse41.c
  src/math/roundu-sse41.c
  src/math/roundz-sse41.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-sse41-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-sse41-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-sse41-mul32.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-sse41-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-sse41-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-sse41-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-sse41-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-sse41-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-sse41-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-sse41-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-sse41-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-sse41-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-sse41-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-sse41-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-sse41-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-sse41-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-sse41-mul32.c
  src/qs8-gavgpool/gen/7p7x-minmax-sse41-c8-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-sse41-c16-acc2.c
  src/qs8-gavgpool/gen/7p7x-minmax-sse41-c24-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-sse41-c8-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-sse41-c16-acc2.c
  src/qs8-gavgpool/gen/7x-minmax-sse41-c24-acc2.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse41-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-sse41.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse41-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse41-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-sse41.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse41-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-sse41.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse41-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse41-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-sse41.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse41-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-sse41.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse41-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse41-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-sse41.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse41-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-sse41.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse41-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse41-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse41-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse41-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse41-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse41-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse41-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse41-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse41-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse41-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-sse41-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse41-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse41-ld128.c
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
  src/qu8-requantization/gemmlowp-sse4.c
  src/qu8-requantization/rndna-sse4.c)

SET(XNNPACK_AVX_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up8x4-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-avx.c
  src/f32-dwconv/gen/up8x9-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-avx.c
  src/f32-dwconv/gen/up8x25-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-avx.c
  src/f32-dwconv/gen/up16x4-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-avx.c
  src/f32-dwconv/gen/up16x9-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-avx.c
  src/f32-dwconv/gen/up16x25-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-avx.c
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
  src/f32-vscale/avx-x32.c
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
  src/math/exp-avx-rr2-p5.c
  src/math/expm1minus-avx-rr2-lut4-p4-perm.c
  src/math/expm1minus-avx-rr2-lut16-p3.c
  src/math/expm1minus-avx-rr2-p6.c
  src/math/sigmoid-avx-rr2-lut64-p2-div.c
  src/math/sigmoid-avx-rr2-p5-div.c
  src/math/sigmoid-avx-rr2-p5-nr1.c
  src/math/sigmoid-avx-rr2-p5-nr2.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-avx-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-avx-mul16.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-avx-mul32.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-avx-mul16.c
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-avx-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-avx-mul16.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-avx-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-avx-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-avx-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-avx-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-avx-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-avx-mul16.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-avx-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-avx-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-avx-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-avx-mul16.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-avx-mul32.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-avx-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-avx.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-avx-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-avx-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-avx.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-avx-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-avx.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-avx-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-avx-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-avx.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-avx-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-avx.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-avx-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-avx-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-avx.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-avx-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-avx.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-avx-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-avx-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-avx-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-avx-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-avx-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-avx-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-avx-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-avx-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-avx-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-avx-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-avx-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-avx-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-avx-ld128.c
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
  src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x32.c)

SET(XNNPACK_XOP_MICROKERNEL_SRCS
  src/qc8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-xop-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-xop-mul32.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qc8-igemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qs8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-xop-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-xop-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-xop-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-xop-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-xop-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-xop-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-xop-mul32.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-xop.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-xop-ld64.c
  src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-xop-ld128.c
  src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-xop.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-xop.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-xop-ld64.c
  src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-xop-ld128.c
  src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-xop.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-xop.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-xop-ld64.c
  src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-xop-ld128.c
  src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-xop.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-xop.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-xop-ld64.c
  src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-xop-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-xop-ld64.c
  src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-xop-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-xop-ld64.c
  src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-xop-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-xop-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-fp32-xop-ld128.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-xop-ld64.c
  src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-xop-ld128.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x8.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x16.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x24.c
  src/qs8-vadd/gen/minmax-xop-mul32-ld32-x32.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x8.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x16.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x24.c
  src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x32.c)

SET(XNNPACK_FMA3_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up8x4-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-fma3.c
  src/f32-dwconv/gen/up8x9-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-fma3.c
  src/f32-dwconv/gen/up8x25-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-fma3.c
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

SET(XNNPACK_AVX2_MICROKERNEL_SRCS
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
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x64-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x64-acc4.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x64.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x72-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x72.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x80-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x80-acc5.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x80.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc6.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96.c
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
  src/math/exp-avx2-rr2-lut8-p3-perm.c
  src/math/exp-avx2-rr2-lut8-p4-perm.c
  src/math/exp-avx2-rr2-p5.c
  src/math/expm1minus-avx2-rr1-lut4-p4-perm.c
  src/math/expm1minus-avx2-rr1-lut8-p4-perm.c
  src/math/expm1minus-avx2-rr1-lut16-p3-gather.c
  src/math/expm1minus-avx2-rr1-p6.c
  src/math/expminus-avx2-rr2-p5.c
  src/math/extexp-avx2-p5.c
  src/math/sigmoid-avx2-rr1-lut64-p2-gather-div.c
  src/math/sigmoid-avx2-rr1-lut64-p2-gather-nr1fma.c
  src/math/sigmoid-avx2-rr1-lut64-p2-gather-nr2fma.c
  src/math/sigmoid-avx2-rr1-lut64-p2-gather-nr2fma1adj.c
  src/math/sigmoid-avx2-rr1-p5-div.c
  src/math/sigmoid-avx2-rr1-p5-nr1fma.c
  src/math/sigmoid-avx2-rr1-p5-nr2fma.c
  src/math/sigmoid-avx2-rr2-lut64-p2-gather-div.c
  src/math/sigmoid-avx2-rr2-lut64-p2-gather-nr1fma.c
  src/math/sigmoid-avx2-rr2-lut64-p2-gather-nr2fma.c
  src/math/sigmoid-avx2-rr2-lut64-p2-gather-nr2fma1adj.c
  src/math/sigmoid-avx2-rr2-p5-div.c
  src/math/sigmoid-avx2-rr2-p5-nr1fma.c
  src/math/sigmoid-avx2-rr2-p5-nr2fma.c
  src/qc8-dwconv/gen/up8x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16.c
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up24x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up24x25-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16.c
  src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c
  src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16.c
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
  src/qs8-dwconv/gen/up8x9-minmax-gemmlowp-avx2-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up8x25-minmax-gemmlowp-avx2-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-avx2-mul16.c
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-avx2-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-avx2-mul16.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-avx2-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up24x9-minmax-gemmlowp-avx2-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up24x25-minmax-gemmlowp-avx2-mul32.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up32x9-minmax-gemmlowp-avx2-mul16.c
  src/qs8-dwconv/gen/up32x9-minmax-gemmlowp-avx2-mul32.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/up32x25-minmax-gemmlowp-avx2-mul16.c
  src/qs8-dwconv/gen/up32x25-minmax-gemmlowp-avx2-mul32.c
  src/qs8-gemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-avx2.c
  src/qs8-gemm/gen/1x8c8-xw-minmax-fp32-avx2.c
  src/qs8-gemm/gen/1x8c8-xw-minmax-gemmlowp-avx2.c
  src/qs8-gemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-avx2.c
  src/qs8-gemm/gen/2x8c8-xw-minmax-fp32-avx2.c
  src/qs8-gemm/gen/2x8c8-xw-minmax-gemmlowp-avx2.c
  src/qs8-gemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-gemm/gen/3x8c8-minmax-gemmlowp-avx2.c
  src/qs8-gemm/gen/3x8c8-xw-minmax-fp32-avx2.c
  src/qs8-gemm/gen/3x8c8-xw-minmax-gemmlowp-avx2.c
  src/qs8-igemm/gen/1x8c8-minmax-fp32-avx2.c
  src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-avx2.c
  src/qs8-igemm/gen/2x8c8-minmax-fp32-avx2.c
  src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-avx2.c
  src/qs8-igemm/gen/3x8c8-minmax-fp32-avx2.c
  src/qs8-igemm/gen/3x8c8-minmax-gemmlowp-avx2.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x8.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x16.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x24.c
  src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x32.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x8.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x16.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x24.c
  src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x32.c)

SET(XNNPACK_AVX512F_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up16x4-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-avx512f.c
  src/f32-dwconv/gen/up16x9-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-avx512f.c
  src/f32-dwconv/gen/up16x25-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-avx512f.c
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
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128-acc4.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x144-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x144.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160-acc5.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc6.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192.c
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
  src/f32-vscale/avx512f-x64.c
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
  src/math/exp-avx512f-rr2-lut16-p3-perm-scalef.c
  src/math/exp-avx512f-rr2-lut16-p3-perm.c
  src/math/exp-avx512f-rr2-lut32-p2-perm2-scalef.c
  src/math/exp-avx512f-rr2-lut32-p2-perm2.c
  src/math/exp-avx512f-rr2-p5-scalef.c
  src/math/exp-avx512f-rr2-p5.c
  src/math/expm1minus-avx512f-rr1-lut16-p3-perm.c
  src/math/expm1minus-avx512f-rr1-p6.c
  src/math/extexp-avx512f-p5.c
  src/math/sigmoid-avx512f-rr1-lut16-p3-perm-scalef-div.c
  src/math/sigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma1adj.c
  src/math/sigmoid-avx512f-rr1-lut32-p2-perm2-scalef-div.c
  src/math/sigmoid-avx512f-rr1-lut32-p2-perm2-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr1-lut32-p2-perm2-scalef-nr1fma1adj.c
  src/math/sigmoid-avx512f-rr1-lut64-p2-gather-scalef-div.c
  src/math/sigmoid-avx512f-rr1-lut64-p2-gather-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr1-lut64-p2-gather-scalef-nr1fma1adj.c
  src/math/sigmoid-avx512f-rr1-p5-scalef-div.c
  src/math/sigmoid-avx512f-rr1-p5-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr1-p5-scalef-nr1fma1adj.c
  src/math/sigmoid-avx512f-rr2-lut16-p3-perm-scalef-div.c
  src/math/sigmoid-avx512f-rr2-lut16-p3-perm-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr2-lut16-p3-perm-scalef-nr1fma1adj.c
  src/math/sigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div.c
  src/math/sigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma1adj.c
  src/math/sigmoid-avx512f-rr2-lut64-p2-gather-scalef-div.c
  src/math/sigmoid-avx512f-rr2-lut64-p2-gather-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr2-lut64-p2-gather-scalef-nr1fma1adj.c
  src/math/sigmoid-avx512f-rr2-p5-scalef-div.c
  src/math/sigmoid-avx512f-rr2-p5-scalef-nr1fma.c
  src/math/sigmoid-avx512f-rr2-p5-scalef-nr1fma1adj.c
  src/math/sqrt-avx512f-nr1fma.c
  src/math/sqrt-avx512f-nr1fma1adj.c
  src/math/sqrt-avx512f-nr2fma.c)

SET(XNNPACK_AVX512SKX_MICROKERNEL_SRCS
  src/qc8-dwconv/gen/up16x9-minmax-fp32-avx512skx-mul32.c
  src/qc8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c
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
  src/qs8-dwconv/gen/up16x9-minmax-gemmlowp-avx512skx-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c
  src/qs8-dwconv/gen/up16x25-minmax-gemmlowp-avx512skx-mul32.c
  src/qs8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c
  src/qs8-dwconv/gen/up32x9-minmax-gemmlowp-avx512skx-mul32.c
  src/qs8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c
  src/qs8-dwconv/gen/up32x25-minmax-gemmlowp-avx512skx-mul32.c
  src/qs8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/1x16c8-minmax-gemmlowp-avx512skx.c
  src/qs8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/2x16c8-minmax-gemmlowp-avx512skx.c
  src/qs8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/3x16c8-minmax-gemmlowp-avx512skx.c
  src/qs8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-gemm/gen/4x16c8-minmax-gemmlowp-avx512skx.c
  src/qs8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/1x16c8-minmax-gemmlowp-avx512skx.c
  src/qs8-igemm/gen/2x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/2x16c8-minmax-gemmlowp-avx512skx.c
  src/qs8-igemm/gen/3x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/3x16c8-minmax-gemmlowp-avx512skx.c
  src/qs8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c
  src/qs8-igemm/gen/4x16c8-minmax-gemmlowp-avx512skx.c)

SET(XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS
  src/f32-gemm/4x4-aarch32-vfp-ld64.S
  src/f32-gemm/4x4-minmax-aarch32-vfp-ld64.S
  src/f32-gemm/4x8-minmax-aarch32-neon-cortex-a53.S
  src/f32-gemm/4x8-minmax-aarch32-neon-cortex-a55.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a7.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a75.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-ld64.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-pld-cortex-a75.S
  src/f32-igemm/4x8-minmax-aarch32-neon-cortex-a53.S
  src/f32-igemm/4x8-minmax-aarch32-neon-cortex-a55.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-cortex-a7.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-cortex-a75.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-ld64.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-pld-cortex-a75.S)

SET(XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS
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
  src/f16-gemm/gen/4x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/4x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/6x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-cortex-a55.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-cortex-a75.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/8x8-minmax-aarch64-neonfp16arith-ld64.S
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
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-gemm/gen/1x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld128.S
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
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/1x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/1x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/4x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/4x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/4x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a73.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-ld128.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/5x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/5x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-ld128.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
  src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
  src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
  src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S
  src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
  src/qs8-gemm/gen/1x16c4-minmax-gemmlowp-aarch64-neondot-ld32.S
  src/qs8-gemm/gen/1x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
  src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mull-padal.S
  src/qs8-gemm/gen/2x8c16-minmax-gemmlowp-aarch64-neon-mlal-padal.S
  src/qs8-gemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-gemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-cortex-a55.S
  src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld32.S
  src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S
  src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld128.S
  src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
  src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
  src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S
  src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
  src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
  src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
  src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S
  src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
  src/qs8-igemm/gen/2x8c16-minmax-gemmlowp-aarch64-neon-mlal-padal.S
  src/qs8-igemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-igemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-prfm-cortex-a53.S
  src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-cortex-a55.S
  src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S
  src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld128.S)

SET(XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SCALAR_MICROKERNEL_SRCS})
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEON_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONV8_MICROKERNEL_SRCS})
  IF(NOT IOS)
    LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONDOT_MICROKERNEL_SRCS})
  ENDIF()
  IF(XNNPACK_ENABLE_ASSEMBLY)
    LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS})
  ENDIF()
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(aarch64|arm64)$" OR IOS_ARCH MATCHES "^arm64.*")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEON_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONV8_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONDOT_MICROKERNEL_SRCS})
  IF(XNNPACK_ENABLE_ASSEMBLY)
    LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS})
  ENDIF()
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(i[3-6]86|x86_64|AMD64)$" OR IOS_ARCH MATCHES "^(i386|x86_64|AMD64)$")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE2_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSSE3_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE41_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_XOP_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_FMA3_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX2_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX512F_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX512SKX_MICROKERNEL_SRCS})
ENDIF()

IF(XNNPACK_LIBRARY_TYPE STREQUAL "default")
  ADD_LIBRARY(XNNPACK ${XNNPACK_COLD_SRCS} ${XNNPACK_HOT_SRCS} ${XNNPACK_TABLE_SRCS} ${XNNPACK_MICROKERNEL_SRCS})
ELSEIF(XNNPACK_LIBRARY_TYPE STREQUAL "shared")
  ADD_LIBRARY(XNNPACK SHARED ${XNNPACK_COLD_SRCS} ${XNNPACK_HOT_SRCS} ${XNNPACK_TABLE_SRCS} ${XNNPACK_MICROKERNEL_SRCS})
ELSEIF(XNNPACK_LIBRARY_TYPE STREQUAL "static")
  ADD_LIBRARY(XNNPACK STATIC ${XNNPACK_COLD_SRCS} ${XNNPACK_HOT_SRCS} ${XNNPACK_TABLE_SRCS} ${XNNPACK_MICROKERNEL_SRCS})
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported XNNPACK library type \"${XNNPACK_LIBRARY_TYPE}\". Must be \"static\", \"shared\", or \"default\"")
ENDIF()
SET_TARGET_PROPERTIES(XNNPACK PROPERTIES
  C_STANDARD 99
  C_EXTENSIONS YES)
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -marm ")
  SET_PROPERTY(SOURCE ${XNNPACK_NEON_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon ")
  SET_PROPERTY(SOURCE ${XNNPACK_NEONFMA_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon-vfpv4 ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${XNNPACK_NEONV8_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mcpu=cyclone -mtune=generic ")
    SET_PROPERTY(SOURCE ${XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ELSE()
    SET_PROPERTY(SOURCE ${XNNPACK_NEONV8_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8-a -mfpu=neon-fp-armv8 ")
    SET_PROPERTY(SOURCE ${XNNPACK_NEONDOT_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+dotprod -mfpu=neon-fp-armv8 ")
  ENDIF()
  # Workground the neon detection bug in ARM v8
  # Related links:
  #   https://github.com/android/ndk/issues/910
  #   https://reviews.llvm.org/D58477
  IF(ANDROID_NDK_MAJOR AND ANDROID_NDK_MAJOR LESS 21)
    SET_PROPERTY(SOURCE ${XNNPACK_NEONV8_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
    SET_PROPERTY(SOURCE ${XNNPACK_NEONDOT_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
  ENDIF()
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(aarch64|arm64)$" OR IOS_ARCH MATCHES "^arm64.*")
  SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  SET_PROPERTY(SOURCE ${XNNPACK_NEONDOT_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+dotprod ")
  SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16+dotprod ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
  IF(XNNPACK_TARGET_PROCESSOR STREQUAL "arm64" AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch arm64 ")
  ENDIF()
ENDIF()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^(i[3-6]86|x86|x86_64|AMD64)$" OR IOS_ARCH MATCHES "^(i386|x86_64|AMD64)$")
  IF(MSVC)
    IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86" OR CMAKE_SIZEOF_VOID_P EQUAL 4)
      SET_PROPERTY(SOURCE ${XNNPACK_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE ")
      SET_PROPERTY(SOURCE ${XNNPACK_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_SSSE3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
    ENDIF()
    SET_PROPERTY(SOURCE ${XNNPACK_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${XNNPACK_XOP_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${XNNPACK_FMA3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX512 ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX512SKX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX512 ")
  ELSE()
    SET_PROPERTY(SOURCE ${XNNPACK_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse ")
    SET_PROPERTY(SOURCE ${XNNPACK_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_SSSE3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mssse3 ")
    SET_PROPERTY(SOURCE ${XNNPACK_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse4.1 ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx ")
    SET_PROPERTY(SOURCE ${XNNPACK_XOP_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mxop ")
    SET_PROPERTY(SOURCE ${XNNPACK_FMA3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfma ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfma -mavx2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx512f ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX512SKX_MICROKERNEL_SRCS} APPEND_STRIDE PROPERTY COMPILE_FLAGS " -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl ")
    IF(MINGW OR CMAKE_SYSTEM_NAME MATCHES "^(CYGWIN|MSYS)$")
      # Work-around for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782
      SET_PROPERTY(SOURCE ${XNNPACK_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -fno-asynchronous-unwind-tables ")
      SET_PROPERTY(SOURCE ${XNNPACK_AVX512SKX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -fno-asynchronous-unwind-tables ")
    ENDIF()
  ENDIF()
ENDIF()
TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE "XNN_LOG_LEVEL=$<$<CONFIG:Debug>:5>$<$<NOT:$<CONFIG:Debug>>:0>")
IF(MSVC)
  # Even though MSVC has __restrict, it can't be used in all the same contexts as the C99 restrict keyword
  TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE "restrict=")
  IF(${CMAKE_VERSION} VERSION_LESS "3.8.0")
    IF(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /O2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /O2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /O1 ")
    ENDIF()
  ELSE()
    SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O2 >")
    SET_PROPERTY(SOURCE ${XNNPACK_HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O2 >")
    SET_PROPERTY(SOURCE ${XNNPACK_COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O1 >")
  ENDIF()
ELSE()
  IF(${CMAKE_VERSION} VERSION_LESS "3.8.0")
    IF(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -Os ")
    ENDIF()
  ELSE()
    SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O2 >")
    SET_PROPERTY(SOURCE ${XNNPACK_HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O2 >")
    SET_PROPERTY(SOURCE ${XNNPACK_COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -Os >")
  ENDIF()
ENDIF()

TARGET_INCLUDE_DIRECTORIES(XNNPACK PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(XNNPACK PRIVATE src)
IF(WIN32)
  # Target Windows 7+ API
  TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE _WIN32_WINNT=0x0601)
ENDIF()
SET_TARGET_PROPERTIES(XNNPACK PROPERTIES PUBLIC_HEADER include/xnnpack.h)

# ---[ Find libm
FIND_LIBRARY(LIBM m)
IF(LIBM)
  TARGET_LINK_LIBRARIES(XNNPACK PRIVATE ${LIBM})
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
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE clog)

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
  ENDIF()
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PUBLIC pthreadpool)

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
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE fp16)

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

  # ---[ Build size tests

  ADD_EXECUTABLE(operator-size-test test/operator-size.c)
  SET_TARGET_PROPERTIES(operator-size-test PROPERTIES
    C_STANDARD 99
    C_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(operator-size-test PRIVATE XNNPACK)

  ADD_EXECUTABLE(subgraph-size-test test/subgraph-size.c)
  SET_TARGET_PROPERTIES(subgraph-size-test PROPERTIES
    C_STANDARD 99
    C_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(subgraph-size-test PRIVATE XNNPACK)

  # ---[ Build operator-level unit tests
  ADD_EXECUTABLE(abs-nc-test test/abs-nc.cc)
  SET_TARGET_PROPERTIES(abs-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(abs-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(abs-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(abs-nc-test abs-nc-test)

  ADD_EXECUTABLE(add-nd-test test/add-nd.cc)
  SET_TARGET_PROPERTIES(add-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(add-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add-nd-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(add-nd-test add-nd-test)

  ADD_EXECUTABLE(argmax-pooling-nhwc-test test/argmax-pooling-nhwc.cc)
  SET_TARGET_PROPERTIES(argmax-pooling-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(argmax-pooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(argmax-pooling-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(argmax-pooling-nhwc-test argmax-pooling-nhwc-test)

  ADD_EXECUTABLE(average-pooling-nhwc-test test/average-pooling-nhwc.cc)
  SET_TARGET_PROPERTIES(average-pooling-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(average-pooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(average-pooling-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(average-pooling-nhwc-test average-pooling-nhwc-test)

  ADD_EXECUTABLE(bankers-rounding-nc-test test/bankers-rounding-nc.cc)
  SET_TARGET_PROPERTIES(bankers-rounding-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(bankers-rounding-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(bankers-rounding-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(bankers-rounding-nc-test bankers-rounding-nc-test)

  ADD_EXECUTABLE(ceiling-nc-test test/ceiling-nc.cc)
  SET_TARGET_PROPERTIES(ceiling-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(ceiling-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(ceiling-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(ceiling-nc-test ceiling-nc-test)

  ADD_EXECUTABLE(channel-shuffle-nc-test test/channel-shuffle-nc.cc)
  SET_TARGET_PROPERTIES(channel-shuffle-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(channel-shuffle-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(channel-shuffle-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(channel-shuffle-nc-test channel-shuffle-nc-test)

  ADD_EXECUTABLE(clamp-nc-test test/clamp-nc.cc)
  SET_TARGET_PROPERTIES(clamp-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(clamp-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(clamp-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(clamp-nc-test clamp-nc-test)

  ADD_EXECUTABLE(constant-pad-nd-test test/constant-pad-nd.cc)
  SET_TARGET_PROPERTIES(constant-pad-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(constant-pad-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(constant-pad-nd-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(constant-pad-nd-test constant-pad-nd-test)

  ADD_EXECUTABLE(convolution-nhwc-test test/convolution-nhwc.cc)
  SET_TARGET_PROPERTIES(convolution-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(convolution-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-nhwc-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(convolution-nhwc-test convolution-nhwc-test)

  ADD_EXECUTABLE(convolution-nchw-test test/convolution-nchw.cc)
  SET_TARGET_PROPERTIES(convolution-nchw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(convolution-nchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-nchw-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(convolution-nchw-test convolution-nchw-test)

  ADD_EXECUTABLE(copy-nc-test test/copy-nc.cc)
  SET_TARGET_PROPERTIES(copy-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(copy-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(copy-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(copy-nc-test copy-nc-test)

  ADD_EXECUTABLE(deconvolution-nhwc-test test/deconvolution-nhwc.cc)
  SET_TARGET_PROPERTIES(deconvolution-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(deconvolution-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(deconvolution-nhwc-test deconvolution-nhwc-test)

  ADD_EXECUTABLE(depth-to-space-nchw2nhwc-test test/depth-to-space-nchw2nhwc.cc)
  SET_TARGET_PROPERTIES(depth-to-space-nchw2nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(depth-to-space-nchw2nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(depth-to-space-nchw2nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(depth-to-space-nchw2nhwc-test depth-to-space-nchw2nhwc-test)

  ADD_EXECUTABLE(depth-to-space-nhwc-test test/depth-to-space-nhwc.cc)
  SET_TARGET_PROPERTIES(depth-to-space-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(depth-to-space-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(depth-to-space-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(depth-to-space-nhwc-test depth-to-space-nhwc-test)

  ADD_EXECUTABLE(divide-nd-test test/divide-nd.cc)
  SET_TARGET_PROPERTIES(divide-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(divide-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(divide-nd-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(divide-nd-test divide-nd-test)

  ADD_EXECUTABLE(elu-nc-test test/elu-nc.cc)
  SET_TARGET_PROPERTIES(elu-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(elu-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(elu-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(elu-nc-test elu-nc-test)

  ADD_EXECUTABLE(fully-connected-nc-test test/fully-connected-nc.cc)
  SET_TARGET_PROPERTIES(fully-connected-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(fully-connected-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(fully-connected-nc-test fully-connected-nc-test)

  ADD_EXECUTABLE(floor-nc-test test/floor-nc.cc)
  SET_TARGET_PROPERTIES(floor-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(floor-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(floor-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(floor-nc-test floor-nc-test)

  ADD_EXECUTABLE(global-average-pooling-nwc-test test/global-average-pooling-nwc.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-nwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-nwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-nwc-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(global-average-pooling-nwc-test global-average-pooling-nwc-test)

  ADD_EXECUTABLE(global-average-pooling-ncw-test test/global-average-pooling-ncw.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-ncw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-ncw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-ncw-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(global-average-pooling-ncw-test global-average-pooling-ncw-test)

  ADD_EXECUTABLE(hardswish-nc-test test/hardswish-nc.cc)
  SET_TARGET_PROPERTIES(hardswish-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(hardswish-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(hardswish-nc-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(hardswish-nc-test hardswish-nc-test)

  ADD_EXECUTABLE(leaky-relu-nc-test test/leaky-relu-nc.cc)
  SET_TARGET_PROPERTIES(leaky-relu-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(leaky-relu-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(leaky-relu-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(leaky-relu-nc-test leaky-relu-nc-test)

  ADD_EXECUTABLE(max-pooling-nhwc-test test/max-pooling-nhwc.cc)
  SET_TARGET_PROPERTIES(max-pooling-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(max-pooling-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(max-pooling-nhwc-test max-pooling-nhwc-test)

  ADD_EXECUTABLE(maximum-nd-test test/maximum-nd.cc)
  SET_TARGET_PROPERTIES(maximum-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(maximum-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(maximum-nd-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(maximum-nd-test maximum-nd-test)

  ADD_EXECUTABLE(minimum-nd-test test/minimum-nd.cc)
  SET_TARGET_PROPERTIES(minimum-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(minimum-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(minimum-nd-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(minimum-nd-test minimum-nd-test)

  ADD_EXECUTABLE(multiply-nd-test test/multiply-nd.cc)
  SET_TARGET_PROPERTIES(multiply-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(multiply-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(multiply-nd-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(multiply-nd-test multiply-nd-test)

  ADD_EXECUTABLE(negate-nc-test test/negate-nc.cc)
  SET_TARGET_PROPERTIES(negate-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(negate-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(negate-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(negate-nc-test negate-nc-test)

  ADD_EXECUTABLE(prelu-nc-test test/prelu-nc.cc)
  SET_TARGET_PROPERTIES(prelu-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(prelu-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(prelu-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(prelu-nc-test prelu-nc-test)

  ADD_EXECUTABLE(resize-bilinear-nhwc-test test/resize-bilinear-nhwc.cc)
  SET_TARGET_PROPERTIES(resize-bilinear-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(resize-bilinear-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(resize-bilinear-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(resize-bilinear-nhwc-test resize-bilinear-nhwc-test)

  ADD_EXECUTABLE(resize-bilinear-nchw-test test/resize-bilinear-nchw.cc)
  SET_TARGET_PROPERTIES(resize-bilinear-nchw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(resize-bilinear-nchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(resize-bilinear-nchw-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(resize-bilinear-nchw-test resize-bilinear-nchw-test)

  ADD_EXECUTABLE(sigmoid-nc-test test/sigmoid-nc.cc)
  SET_TARGET_PROPERTIES(sigmoid-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(sigmoid-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(sigmoid-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(sigmoid-nc-test sigmoid-nc-test)

  ADD_EXECUTABLE(softmax-nc-test test/softmax-nc.cc)
  SET_TARGET_PROPERTIES(softmax-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(softmax-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(softmax-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(softmax-nc-test softmax-nc-test)

  ADD_EXECUTABLE(square-nc-test test/square-nc.cc)
  SET_TARGET_PROPERTIES(square-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(square-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(square-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(square-nc-test square-nc-test)

  ADD_EXECUTABLE(square-root-nc-test test/square-root-nc.cc)
  SET_TARGET_PROPERTIES(square-root-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(square-root-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(square-root-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(square-root-nc-test square-root-nc-test)

  ADD_EXECUTABLE(squared-difference-nd-test test/squared-difference-nd.cc)
  SET_TARGET_PROPERTIES(squared-difference-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(squared-difference-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(squared-difference-nd-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(squared-difference-nd-test squared-difference-nd-test)

  ADD_EXECUTABLE(subtract-nd-test test/subtract-nd.cc)
  SET_TARGET_PROPERTIES(subtract-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(subtract-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(subtract-nd-test PRIVATE XNNPACK fp16 gtest gtest_main)
  ADD_TEST(subtract-nd-test subtract-nd-test)

  ADD_EXECUTABLE(truncation-nc-test test/truncation-nc.cc)
  SET_TARGET_PROPERTIES(truncation-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(truncation-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(truncation-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(truncation-nc-test truncation-nc-test)

  ADD_EXECUTABLE(unpooling-nhwc-test test/unpooling-nhwc.cc)
  SET_TARGET_PROPERTIES(unpooling-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(unpooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(unpooling-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(unpooling-nhwc-test unpooling-nhwc-test)

  ADD_EXECUTABLE(memory-planner-test test/memory-planner-test.cc)
  SET_TARGET_PROPERTIES(memory-planner-test PROPERTIES
    C_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    C_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(memory-planner-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(memory-planner-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(memory-planner-test memory-planner-test)

  ADD_EXECUTABLE(subgraph-nchw-test test/subgraph-nchw.cc)
  SET_TARGET_PROPERTIES(subgraph-nchw-test PROPERTIES
    C_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    C_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(subgraph-nchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(subgraph-nchw-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(subgraph-nchw-test subgraph-nchw-test)

  # ---[ Build microkernel-level unit tests
  ADD_EXECUTABLE(f16-dwconv-minmax-test test/f16-dwconv-minmax.cc)
  SET_TARGET_PROPERTIES(f16-dwconv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-dwconv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-dwconv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-dwconv-minmax-test f16-dwconv-minmax-test)

  ADD_EXECUTABLE(f16-gavgpool-minmax-test test/f16-gavgpool-minmax.cc)
  SET_TARGET_PROPERTIES(f16-gavgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-gavgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-gavgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-gavgpool-minmax-test f16-gavgpool-minmax-test)

  ADD_EXECUTABLE(f16-gemm-minmax-test test/f16-gemm-minmax.cc)
  SET_TARGET_PROPERTIES(f16-gemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-gemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-gemm-minmax-test f16-gemm-minmax-test)

  ADD_EXECUTABLE(f16-igemm-minmax-test test/f16-igemm-minmax.cc)
  SET_TARGET_PROPERTIES(f16-igemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-igemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-igemm-minmax-test f16-igemm-minmax-test)

  ADD_EXECUTABLE(f16-spmm-minmax-test test/f16-spmm-minmax.cc)
  SET_TARGET_PROPERTIES(f16-spmm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-spmm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-spmm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-spmm-minmax-test f16-spmm-minmax-test)

  ADD_EXECUTABLE(f16-vadd-minmax-test test/f16-vadd-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vadd-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vadd-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vadd-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vadd-minmax-test f16-vadd-minmax-test)

  ADD_EXECUTABLE(f16-vaddc-minmax-test test/f16-vaddc-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vaddc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vaddc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vaddc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vaddc-minmax-test f16-vaddc-minmax-test)

  ADD_EXECUTABLE(f16-vclamp-test test/f16-vclamp.cc)
  SET_TARGET_PROPERTIES(f16-vclamp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vclamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vclamp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vclamp-test f16-vclamp-test)

  ADD_EXECUTABLE(f16-vdiv-minmax-test test/f16-vdiv-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vdiv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vdiv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vdiv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vdiv-minmax-test f16-vdiv-minmax-test)

  ADD_EXECUTABLE(f16-vdivc-minmax-test test/f16-vdivc-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vdivc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vdivc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vdivc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vdivc-minmax-test f16-vdivc-minmax-test)

  ADD_EXECUTABLE(f16-vrdivc-minmax-test test/f16-vrdivc-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vrdivc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vrdivc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vrdivc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vrdivc-minmax-test f16-vrdivc-minmax-test)

  ADD_EXECUTABLE(f16-vhswish-test test/f16-vhswish.cc)
  SET_TARGET_PROPERTIES(f16-vhswish-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vhswish-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vhswish-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vhswish-test f16-vhswish-test)

  ADD_EXECUTABLE(f16-vmax-test test/f16-vmax.cc)
  SET_TARGET_PROPERTIES(f16-vmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vmax-test f16-vmax-test)

  ADD_EXECUTABLE(f16-vmaxc-test test/f16-vmaxc.cc)
  SET_TARGET_PROPERTIES(f16-vmaxc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vmaxc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vmaxc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vmaxc-test f16-vmaxc-test)

  ADD_EXECUTABLE(f16-vmin-test test/f16-vmin.cc)
  SET_TARGET_PROPERTIES(f16-vmin-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vmin-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vmin-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vmin-test f16-vmin-test)

  ADD_EXECUTABLE(f16-vminc-test test/f16-vminc.cc)
  SET_TARGET_PROPERTIES(f16-vminc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vminc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vminc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vminc-test f16-vminc-test)

  ADD_EXECUTABLE(f16-vmul-minmax-test test/f16-vmul-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vmul-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vmul-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vmul-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vmul-minmax-test f16-vmul-minmax-test)

  ADD_EXECUTABLE(f16-vmulc-minmax-test test/f16-vmulc-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vmulc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vmulc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vmulc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vmulc-minmax-test f16-vmulc-minmax-test)

  ADD_EXECUTABLE(f16-vmulcaddc-minmax-test test/f16-vmulcaddc-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vmulcaddc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vmulcaddc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vmulcaddc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vmulcaddc-minmax-test f16-vmulcaddc-minmax-test)

  ADD_EXECUTABLE(f16-prelu-test test/f16-prelu.cc)
  SET_TARGET_PROPERTIES(f16-prelu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-prelu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-prelu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-prelu-test f16-prelu-test)

  ADD_EXECUTABLE(f16-vsub-minmax-test test/f16-vsub-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vsub-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vsub-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vsub-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vsub-minmax-test f16-vsub-minmax-test)

  ADD_EXECUTABLE(f16-vsubc-minmax-test test/f16-vsubc-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vsubc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vsubc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vsubc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vsubc-minmax-test f16-vsubc-minmax-test)

  ADD_EXECUTABLE(f16-vrsubc-minmax-test test/f16-vrsubc-minmax.cc)
  SET_TARGET_PROPERTIES(f16-vrsubc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vrsubc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-vrsubc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-vrsubc-minmax-test f16-vrsubc-minmax-test)

  ADD_EXECUTABLE(f32-argmaxpool-test test/f32-argmaxpool.cc)
  SET_TARGET_PROPERTIES(f32-argmaxpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-argmaxpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-argmaxpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-argmaxpool-test f32-argmaxpool-test)

  ADD_EXECUTABLE(f32-avgpool-minmax-test test/f32-avgpool-minmax.cc)
  SET_TARGET_PROPERTIES(f32-avgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-avgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-avgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-avgpool-minmax-test f32-avgpool-minmax-test)

  ADD_EXECUTABLE(f32-conv-hwc-test test/f32-conv-hwc.cc)
  SET_TARGET_PROPERTIES(f32-conv-hwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-conv-hwc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-conv-hwc-test f32-conv-hwc-test)

  ADD_EXECUTABLE(f32-conv-hwc2chw-test test/f32-conv-hwc2chw.cc)
  SET_TARGET_PROPERTIES(f32-conv-hwc2chw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc2chw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-conv-hwc2chw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-conv-hwc2chw-test f32-conv-hwc2chw-test)

  ADD_EXECUTABLE(f32-dwconv-test test/f32-dwconv.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-dwconv-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-dwconv-test f32-dwconv-test)

  ADD_EXECUTABLE(f32-dwconv2d-chw-test test/f32-dwconv2d-chw.cc)
  SET_TARGET_PROPERTIES(f32-dwconv2d-chw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv2d-chw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-dwconv2d-chw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-dwconv2d-chw-test f32-dwconv2d-chw-test)

  ADD_EXECUTABLE(f32-dwconv-minmax-test test/f32-dwconv-minmax.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-dwconv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-dwconv-minmax-test f32-dwconv-minmax-test)

  ADD_EXECUTABLE(f32-gavgpool-cw-test test/f32-gavgpool-cw.cc)
  SET_TARGET_PROPERTIES(f32-gavgpool-cw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gavgpool-cw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gavgpool-cw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gavgpool-cw-test f32-gavgpool-cw-test)

  ADD_EXECUTABLE(f32-gavgpool-minmax-test test/f32-gavgpool-minmax.cc)
  SET_TARGET_PROPERTIES(f32-gavgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gavgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gavgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gavgpool-minmax-test f32-gavgpool-minmax-test)

  ADD_EXECUTABLE(f32-gemm-test test/f32-gemm.cc)
  SET_TARGET_PROPERTIES(f32-gemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gemm-test f32-gemm-test)

  ADD_EXECUTABLE(f32-gemm-relu-test test/f32-gemm-relu.cc)
  SET_TARGET_PROPERTIES(f32-gemm-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gemm-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gemm-relu-test f32-gemm-relu-test)

  ADD_EXECUTABLE(f32-gemm-minmax-test test/f32-gemm-minmax.cc)
  SET_TARGET_PROPERTIES(f32-gemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gemm-minmax-test f32-gemm-minmax-test)

  ADD_EXECUTABLE(f32-gemminc-minmax-test test/f32-gemminc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-gemminc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemminc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gemminc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gemminc-minmax-test f32-gemminc-minmax-test)

  ADD_EXECUTABLE(f32-ibilinear-test test/f32-ibilinear.cc)
  SET_TARGET_PROPERTIES(f32-ibilinear-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-ibilinear-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-ibilinear-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-ibilinear-test f32-ibilinear-test)

  ADD_EXECUTABLE(f32-ibilinear-chw-test test/f32-ibilinear-chw.cc)
  SET_TARGET_PROPERTIES(f32-ibilinear-chw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-ibilinear-chw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-ibilinear-chw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-ibilinear-chw-test f32-ibilinear-chw-test)

  ADD_EXECUTABLE(f32-igemm-test test/f32-igemm.cc)
  SET_TARGET_PROPERTIES(f32-igemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-igemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-igemm-test f32-igemm-test)

  ADD_EXECUTABLE(f32-igemm-relu-test test/f32-igemm-relu.cc)
  SET_TARGET_PROPERTIES(f32-igemm-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-igemm-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-igemm-relu-test f32-igemm-relu-test)

  ADD_EXECUTABLE(f32-igemm-minmax-test test/f32-igemm-minmax.cc)
  SET_TARGET_PROPERTIES(f32-igemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-minmax-test PRIVATE src test)
  IF(MINGW)
    # Work-around for "too many sections" error
    TARGET_COMPILE_OPTIONS(f32-igemm-minmax-test PRIVATE "$<$<NOT:$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>>:-Wa,-mbig-obj>")
  ENDIF()
  TARGET_LINK_LIBRARIES(f32-igemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-igemm-minmax-test f32-igemm-minmax-test)

  ADD_EXECUTABLE(f32-maxpool-minmax-test test/f32-maxpool-minmax.cc)
  SET_TARGET_PROPERTIES(f32-maxpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-maxpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-maxpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-maxpool-minmax-test f32-maxpool-minmax-test)

  ADD_EXECUTABLE(f32-pavgpool-minmax-test test/f32-pavgpool-minmax.cc)
  SET_TARGET_PROPERTIES(f32-pavgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-pavgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-pavgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-pavgpool-minmax-test f32-pavgpool-minmax-test)

  ADD_EXECUTABLE(f32-ppmm-minmax-test test/f32-ppmm-minmax.cc)
  SET_TARGET_PROPERTIES(f32-ppmm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-ppmm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-ppmm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-ppmm-minmax-test f32-ppmm-minmax-test)

  ADD_EXECUTABLE(f32-prelu-test test/f32-prelu.cc)
  SET_TARGET_PROPERTIES(f32-prelu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-prelu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-prelu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-prelu-test f32-prelu-test)

  ADD_EXECUTABLE(f32-raddexpminusmax-test test/f32-raddexpminusmax.cc)
  SET_TARGET_PROPERTIES(f32-raddexpminusmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-raddexpminusmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-raddexpminusmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-raddexpminusmax-test f32-raddexpminusmax-test)

  ADD_EXECUTABLE(f32-raddextexp-test test/f32-raddextexp.cc)
  SET_TARGET_PROPERTIES(f32-raddextexp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-raddextexp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-raddextexp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-raddextexp-test f32-raddextexp-test)

  ADD_EXECUTABLE(f32-raddstoreexpminusmax-test test/f32-raddstoreexpminusmax.cc)
  SET_TARGET_PROPERTIES(f32-raddstoreexpminusmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-raddstoreexpminusmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-raddstoreexpminusmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-raddstoreexpminusmax-test f32-raddstoreexpminusmax-test)

  ADD_EXECUTABLE(f32-rmax-test test/f32-rmax.cc)
  SET_TARGET_PROPERTIES(f32-rmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-rmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-rmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-rmax-test f32-rmax-test)

  ADD_EXECUTABLE(f32-spmm-minmax-test test/f32-spmm-minmax.cc)
  SET_TARGET_PROPERTIES(f32-spmm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-spmm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-spmm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-spmm-minmax-test f32-spmm-minmax-test)

  ADD_EXECUTABLE(f32-vabs-test test/f32-vabs.cc)
  SET_TARGET_PROPERTIES(f32-vabs-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vabs-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vabs-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vabs-test f32-vabs-test)

  ADD_EXECUTABLE(f32-vadd-test test/f32-vadd.cc)
  SET_TARGET_PROPERTIES(f32-vadd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vadd-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vadd-test f32-vadd-test)

  ADD_EXECUTABLE(f32-vadd-minmax-test test/f32-vadd-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vadd-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vadd-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vadd-minmax-test f32-vadd-minmax-test)

  ADD_EXECUTABLE(f32-vadd-relu-test test/f32-vadd-relu.cc)
  SET_TARGET_PROPERTIES(f32-vadd-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vadd-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vadd-relu-test f32-vadd-relu-test)

  ADD_EXECUTABLE(f32-vaddc-test test/f32-vaddc.cc)
  SET_TARGET_PROPERTIES(f32-vaddc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vaddc-test f32-vaddc-test)

  ADD_EXECUTABLE(f32-vaddc-minmax-test test/f32-vaddc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vaddc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vaddc-minmax-test f32-vaddc-minmax-test)

  ADD_EXECUTABLE(f32-vaddc-relu-test test/f32-vaddc-relu.cc)
  SET_TARGET_PROPERTIES(f32-vaddc-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vaddc-relu-test f32-vaddc-relu-test)

  ADD_EXECUTABLE(f32-vclamp-test test/f32-vclamp.cc)
  SET_TARGET_PROPERTIES(f32-vclamp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vclamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vclamp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vclamp-test f32-vclamp-test)

  ADD_EXECUTABLE(f32-vhswish-test test/f32-vhswish.cc)
  SET_TARGET_PROPERTIES(f32-vhswish-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vhswish-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vhswish-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vhswish-test f32-vhswish-test)

  ADD_EXECUTABLE(f32-vdiv-test test/f32-vdiv.cc)
  SET_TARGET_PROPERTIES(f32-vdiv-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdiv-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdiv-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdiv-test f32-vdiv-test)

  ADD_EXECUTABLE(f32-vdiv-minmax-test test/f32-vdiv-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vdiv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdiv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdiv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdiv-minmax-test f32-vdiv-minmax-test)

  ADD_EXECUTABLE(f32-vdiv-relu-test test/f32-vdiv-relu.cc)
  SET_TARGET_PROPERTIES(f32-vdiv-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdiv-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdiv-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdiv-relu-test f32-vdiv-relu-test)

  ADD_EXECUTABLE(f32-vdivc-test test/f32-vdivc.cc)
  SET_TARGET_PROPERTIES(f32-vdivc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdivc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdivc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdivc-test f32-vdivc-test)

  ADD_EXECUTABLE(f32-vdivc-minmax-test test/f32-vdivc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vdivc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdivc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdivc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdivc-minmax-test f32-vdivc-minmax-test)

  ADD_EXECUTABLE(f32-vdivc-relu-test test/f32-vdivc-relu.cc)
  SET_TARGET_PROPERTIES(f32-vdivc-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdivc-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdivc-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdivc-relu-test f32-vdivc-relu-test)

  ADD_EXECUTABLE(f32-vrdivc-test test/f32-vrdivc.cc)
  SET_TARGET_PROPERTIES(f32-vrdivc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrdivc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrdivc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrdivc-test f32-vrdivc-test)

  ADD_EXECUTABLE(f32-vrdivc-minmax-test test/f32-vrdivc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vrdivc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrdivc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrdivc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrdivc-minmax-test f32-vrdivc-minmax-test)

  ADD_EXECUTABLE(f32-vrdivc-relu-test test/f32-vrdivc-relu.cc)
  SET_TARGET_PROPERTIES(f32-vrdivc-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrdivc-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrdivc-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrdivc-relu-test f32-vrdivc-relu-test)

  ADD_EXECUTABLE(f32-velu-test test/f32-velu.cc)
  SET_TARGET_PROPERTIES(f32-velu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-velu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-velu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-velu-test f32-velu-test)

  ADD_EXECUTABLE(f32-vlrelu-test test/f32-vlrelu.cc)
  SET_TARGET_PROPERTIES(f32-vlrelu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vlrelu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vlrelu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vlrelu-test f32-vlrelu-test)

  ADD_EXECUTABLE(f32-vmax-test test/f32-vmax.cc)
  SET_TARGET_PROPERTIES(f32-vmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmax-test f32-vmax-test)

  ADD_EXECUTABLE(f32-vmaxc-test test/f32-vmaxc.cc)
  SET_TARGET_PROPERTIES(f32-vmaxc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmaxc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmaxc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmaxc-test f32-vmaxc-test)

  ADD_EXECUTABLE(f32-vmin-test test/f32-vmin.cc)
  SET_TARGET_PROPERTIES(f32-vmin-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmin-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmin-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmin-test f32-vmin-test)

  ADD_EXECUTABLE(f32-vminc-test test/f32-vminc.cc)
  SET_TARGET_PROPERTIES(f32-vminc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vminc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vminc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vminc-test f32-vminc-test)

  ADD_EXECUTABLE(f32-vmul-test test/f32-vmul.cc)
  SET_TARGET_PROPERTIES(f32-vmul-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmul-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmul-test f32-vmul-test)

  ADD_EXECUTABLE(f32-vmul-minmax-test test/f32-vmul-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vmul-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmul-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmul-minmax-test f32-vmul-minmax-test)

  ADD_EXECUTABLE(f32-vmul-relu-test test/f32-vmul-relu.cc)
  SET_TARGET_PROPERTIES(f32-vmul-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmul-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmul-relu-test f32-vmul-relu-test)

  ADD_EXECUTABLE(f32-vmulc-test test/f32-vmulc.cc)
  SET_TARGET_PROPERTIES(f32-vmulc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulc-test f32-vmulc-test)

  ADD_EXECUTABLE(f32-vmulc-minmax-test test/f32-vmulc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vmulc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulc-minmax-test f32-vmulc-minmax-test)

  ADD_EXECUTABLE(f32-vmulc-relu-test test/f32-vmulc-relu.cc)
  SET_TARGET_PROPERTIES(f32-vmulc-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulc-relu-test f32-vmulc-relu-test)

  ADD_EXECUTABLE(f32-vmulcaddc-minmax-test test/f32-vmulcaddc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vmulcaddc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulcaddc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulcaddc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulcaddc-minmax-test f32-vmulcaddc-minmax-test)

  ADD_EXECUTABLE(f32-vneg-test test/f32-vneg.cc)
  SET_TARGET_PROPERTIES(f32-vneg-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vneg-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vneg-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vneg-test f32-vneg-test)

  ADD_EXECUTABLE(f32-vrelu-test test/f32-vrelu.cc)
  SET_TARGET_PROPERTIES(f32-vrelu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrelu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrelu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrelu-test f32-vrelu-test)

  ADD_EXECUTABLE(f32-vrndne-test test/f32-vrndne.cc)
  SET_TARGET_PROPERTIES(f32-vrndne-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndne-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrndne-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrndne-test f32-vrndne-test)

  ADD_EXECUTABLE(f32-vrndz-test test/f32-vrndz.cc)
  SET_TARGET_PROPERTIES(f32-vrndz-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndz-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrndz-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrndz-test f32-vrndz-test)

  ADD_EXECUTABLE(f32-vrndu-test test/f32-vrndu.cc)
  SET_TARGET_PROPERTIES(f32-vrndu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrndu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrndu-test f32-vrndu-test)

  ADD_EXECUTABLE(f32-vrndd-test test/f32-vrndd.cc)
  SET_TARGET_PROPERTIES(f32-vrndd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrndd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrndd-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrndd-test f32-vrndd-test)

  ADD_EXECUTABLE(f32-vscaleexpminusmax-test test/f32-vscaleexpminusmax.cc)
  SET_TARGET_PROPERTIES(f32-vscaleexpminusmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vscaleexpminusmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vscaleexpminusmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vscaleexpminusmax-test f32-vscaleexpminusmax-test)

  ADD_EXECUTABLE(f32-vscaleextexp-test test/f32-vscaleextexp.cc)
  SET_TARGET_PROPERTIES(f32-vscaleextexp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vscaleextexp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vscaleextexp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vscaleextexp-test f32-vscaleextexp-test)

  ADD_EXECUTABLE(f32-vsigmoid-test test/f32-vsigmoid.cc)
  SET_TARGET_PROPERTIES(f32-vsigmoid-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsigmoid-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsigmoid-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsigmoid-test f32-vsigmoid-test)

  ADD_EXECUTABLE(f32-vsqr-test test/f32-vsqr.cc)
  SET_TARGET_PROPERTIES(f32-vsqr-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqr-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsqr-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsqr-test f32-vsqr-test)

  ADD_EXECUTABLE(f32-vsqrdiff-test test/f32-vsqrdiff.cc)
  SET_TARGET_PROPERTIES(f32-vsqrdiff-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrdiff-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsqrdiff-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsqrdiff-test f32-vsqrdiff-test)

  ADD_EXECUTABLE(f32-vsqrdiffc-test test/f32-vsqrdiffc.cc)
  SET_TARGET_PROPERTIES(f32-vsqrdiffc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrdiffc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsqrdiffc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsqrdiffc-test f32-vsqrdiffc-test)

  ADD_EXECUTABLE(f32-vsqrt-test test/f32-vsqrt.cc)
  SET_TARGET_PROPERTIES(f32-vsqrt-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrt-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsqrt-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsqrt-test f32-vsqrt-test)

  ADD_EXECUTABLE(f32-vsub-test test/f32-vsub.cc)
  SET_TARGET_PROPERTIES(f32-vsub-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsub-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsub-test f32-vsub-test)

  ADD_EXECUTABLE(f32-vsub-minmax-test test/f32-vsub-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vsub-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsub-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsub-minmax-test f32-vsub-minmax-test)

  ADD_EXECUTABLE(f32-vsub-relu-test test/f32-vsub-relu.cc)
  SET_TARGET_PROPERTIES(f32-vsub-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsub-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsub-relu-test f32-vsub-relu-test)

  ADD_EXECUTABLE(f32-vsubc-test test/f32-vsubc.cc)
  SET_TARGET_PROPERTIES(f32-vsubc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsubc-test f32-vsubc-test)

  ADD_EXECUTABLE(f32-vsubc-minmax-test test/f32-vsubc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vsubc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsubc-minmax-test f32-vsubc-minmax-test)

  ADD_EXECUTABLE(f32-vsubc-relu-test test/f32-vsubc-relu.cc)
  SET_TARGET_PROPERTIES(f32-vsubc-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsubc-relu-test f32-vsubc-relu-test)

  ADD_EXECUTABLE(f32-vrsubc-test test/f32-vrsubc.cc)
  SET_TARGET_PROPERTIES(f32-vrsubc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrsubc-test f32-vrsubc-test)

  ADD_EXECUTABLE(f32-vrsubc-minmax-test test/f32-vrsubc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vrsubc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrsubc-minmax-test f32-vrsubc-minmax-test)

  ADD_EXECUTABLE(f32-vrsubc-relu-test test/f32-vrsubc-relu.cc)
  SET_TARGET_PROPERTIES(f32-vrsubc-relu-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-relu-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrsubc-relu-test f32-vrsubc-relu-test)

  ADD_EXECUTABLE(qc8-dwconv-minmax-fp32-test test/qc8-dwconv-minmax-fp32.cc)
  SET_TARGET_PROPERTIES(qc8-dwconv-minmax-fp32-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qc8-dwconv-minmax-fp32-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qc8-dwconv-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qc8-dwconv-minmax-fp32-test qc8-dwconv-minmax-fp32-test)

  ADD_EXECUTABLE(qc8-gemm-minmax-fp32-test test/qc8-gemm-minmax-fp32.cc)
  SET_TARGET_PROPERTIES(qc8-gemm-minmax-fp32-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qc8-gemm-minmax-fp32-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qc8-gemm-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qc8-gemm-minmax-fp32-test qc8-gemm-minmax-fp32-test)

  ADD_EXECUTABLE(qc8-igemm-minmax-fp32-test test/qc8-igemm-minmax-fp32.cc)
  SET_TARGET_PROPERTIES(qc8-igemm-minmax-fp32-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qc8-igemm-minmax-fp32-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qc8-igemm-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qc8-igemm-minmax-fp32-test qc8-igemm-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-dwconv-minmax-gemmlowp-test test/qs8-dwconv-minmax-gemmlowp.cc)
  SET_TARGET_PROPERTIES(qs8-dwconv-minmax-gemmlowp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-dwconv-minmax-gemmlowp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-dwconv-minmax-gemmlowp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-dwconv-minmax-gemmlowp-test qs8-dwconv-minmax-gemmlowp-test)

  ADD_EXECUTABLE(qs8-dwconv-minmax-fp32-test test/qs8-dwconv-minmax-fp32.cc)
  SET_TARGET_PROPERTIES(qs8-dwconv-minmax-fp32-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-dwconv-minmax-fp32-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-dwconv-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-dwconv-minmax-fp32-test qs8-dwconv-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-gavgpool-minmax-test test/qs8-gavgpool-minmax.cc)
  SET_TARGET_PROPERTIES(qs8-gavgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-gavgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-gavgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-gavgpool-test qs8-gavgpool-minmax-test)

  ADD_EXECUTABLE(qs8-gemm-minmax-gemmlowp-test test/qs8-gemm-minmax-gemmlowp.cc)
  SET_TARGET_PROPERTIES(qs8-gemm-minmax-gemmlowp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-minmax-gemmlowp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-gemm-minmax-gemmlowp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-gemm-minmax-gemmlowp-test qs8-gemm-minmax-gemmlowp-test)

  ADD_EXECUTABLE(qs8-gemm-minmax-fp32-test test/qs8-gemm-minmax-fp32.cc)
  SET_TARGET_PROPERTIES(qs8-gemm-minmax-fp32-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-minmax-fp32-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-gemm-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-gemm-minmax-fp32-test qs8-gemm-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-igemm-minmax-gemmlowp-test test/qs8-igemm-minmax-gemmlowp.cc)
  SET_TARGET_PROPERTIES(qs8-igemm-minmax-gemmlowp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-igemm-minmax-gemmlowp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-igemm-minmax-gemmlowp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-igemm-minmax-gemmlowp-test qs8-igemm-minmax-gemmlowp-test)

  ADD_EXECUTABLE(qs8-igemm-minmax-fp32-test test/qs8-igemm-minmax-fp32.cc)
  SET_TARGET_PROPERTIES(qs8-igemm-minmax-fp32-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-igemm-minmax-fp32-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-igemm-minmax-fp32-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-igemm-minmax-fp32-test qs8-igemm-minmax-fp32-test)

  ADD_EXECUTABLE(qs8-vadd-minmax-test test/qs8-vadd-minmax.cc)
  SET_TARGET_PROPERTIES(qs8-vadd-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-vadd-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-vadd-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-vadd-minmax-test qs8-vadd-minmax-test)

  ADD_EXECUTABLE(qs8-vaddc-minmax-test test/qs8-vaddc-minmax.cc)
  SET_TARGET_PROPERTIES(qs8-vaddc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-vaddc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qs8-vaddc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qs8-vaddc-minmax-test qs8-vaddc-minmax-test)

  ADD_EXECUTABLE(qu8-avgpool-minmax-test test/qu8-avgpool-minmax.cc)
  SET_TARGET_PROPERTIES(qu8-avgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-avgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qu8-avgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qu8-avgpool-minmax-test qu8-avgpool-minmax-test)

  ADD_EXECUTABLE(qu8-dwconv-minmax-test test/qu8-dwconv-minmax.cc)
  SET_TARGET_PROPERTIES(qu8-dwconv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-dwconv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qu8-dwconv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qu8-dwconv-minmax-test qu8-dwconv-minmax-test)

  ADD_EXECUTABLE(qu8-gavgpool-minmax-test test/qu8-gavgpool-minmax.cc)
  SET_TARGET_PROPERTIES(qu8-gavgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-gavgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qu8-gavgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qu8-gavgpool-test qu8-gavgpool-minmax-test)

  ADD_EXECUTABLE(qu8-gemm-minmax-test test/qu8-gemm-minmax.cc)
  SET_TARGET_PROPERTIES(qu8-gemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-gemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qu8-gemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qu8-gemm-minmax-test qu8-gemm-minmax-test)

  ADD_EXECUTABLE(qu8-igemm-minmax-test test/qu8-igemm-minmax.cc)
  SET_TARGET_PROPERTIES(qu8-igemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-igemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qu8-igemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qu8-igemm-minmax-test qu8-igemm-minmax-test)

  ADD_EXECUTABLE(qu8-requantization-test test/qu8-requantization.cc)
  SET_TARGET_PROPERTIES(qu8-requantization-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-requantization-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qu8-requantization-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qu8-requantization-test qu8-requantization-test)

  ADD_EXECUTABLE(qu8-vadd-minmax-test test/qu8-vadd-minmax.cc)
  SET_TARGET_PROPERTIES(qu8-vadd-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-vadd-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(qu8-vadd-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(qu8-vadd-minmax-test qu8-vadd-minmax-test)

  ADD_EXECUTABLE(u8-lut32norm-test test/u8-lut32norm.cc)
  SET_TARGET_PROPERTIES(u8-lut32norm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-lut32norm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-lut32norm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-lut32norm-test u8-lut32norm-test)

  ADD_EXECUTABLE(u8-maxpool-minmax-test test/u8-maxpool-minmax.cc)
  SET_TARGET_PROPERTIES(u8-maxpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-maxpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-maxpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-maxpool-minmax-test u8-maxpool-minmax-test)

  ADD_EXECUTABLE(u8-rmax-test test/u8-rmax.cc)
  SET_TARGET_PROPERTIES(u8-rmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-rmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-rmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-rmax-test u8-rmax-test)

  ADD_EXECUTABLE(u8-vclamp-test test/u8-vclamp.cc)
  SET_TARGET_PROPERTIES(u8-vclamp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-vclamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-vclamp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-vclamp-test u8-vclamp-test)

  ADD_EXECUTABLE(x32-fill-test test/x32-fill.cc)
  SET_TARGET_PROPERTIES(x32-fill-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x32-fill-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x32-fill-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x32-fill-test x32-fill-test)

  ADD_EXECUTABLE(x32-packx-test test/x32-packx.cc)
  SET_TARGET_PROPERTIES(x32-packx-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x32-packx-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x32-packx-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x32-packx-test x32-packx-test)

  ADD_EXECUTABLE(x32-pad-test test/x32-pad.cc)
  SET_TARGET_PROPERTIES(x32-pad-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x32-pad-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x32-pad-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x32-pad-test x32-pad-test)

  ADD_EXECUTABLE(x32-unpool-test test/x32-unpool.cc)
  SET_TARGET_PROPERTIES(x32-unpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x32-unpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x32-unpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x32-unpool-test x32-unpool-test)

  ADD_EXECUTABLE(x32-depthtospace2d-chw2hwc-test test/x32-depthtospace2d-chw2hwc.cc)
  SET_TARGET_PROPERTIES(x32-depthtospace2d-chw2hwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x32-depthtospace2d-chw2hwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x32-depthtospace2d-chw2hwc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x32-depthtospace2d-chw2hwc-test x32-depthtospace2d-chw2hwc-test)

  ADD_EXECUTABLE(x32-zip-test test/x32-zip.cc)
  SET_TARGET_PROPERTIES(x32-zip-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x32-zip-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x32-zip-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x32-zip-test x32-zip-test)

  ADD_EXECUTABLE(x8-lut-test test/x8-lut.cc)
  SET_TARGET_PROPERTIES(x8-lut-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x8-lut-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x8-lut-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x8-lut-test x8-lut-test)

  ADD_EXECUTABLE(x8-zip-test test/x8-zip.cc)
  SET_TARGET_PROPERTIES(x8-zip-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(x8-zip-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x8-zip-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x8-zip-test x8-zip-test)
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
  SET_TARGET_PROPERTIES(bench-utils PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(bench-utils PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(bench-utils PRIVATE benchmark cpuinfo)

  # ---[ Build accuracy microbenchmarks
  ADD_EXECUTABLE(f32-exp-ulp-eval eval/f32-exp-ulp.cc)
  SET_TARGET_PROPERTIES(f32-exp-ulp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-exp-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-exp-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-expminus-ulp-eval eval/f32-expminus-ulp.cc)
  SET_TARGET_PROPERTIES(f32-expminus-ulp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-expminus-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-expminus-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-expm1minus-ulp-eval eval/f32-expm1minus-ulp.cc)
  SET_TARGET_PROPERTIES(f32-expm1minus-ulp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-expm1minus-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-expm1minus-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-extexp-ulp-eval eval/f32-extexp-ulp.cc)
  SET_TARGET_PROPERTIES(f32-extexp-ulp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-extexp-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-extexp-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-sigmoid-ulp-eval eval/f32-sigmoid-ulp.cc)
  SET_TARGET_PROPERTIES(f32-sigmoid-ulp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-sigmoid-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-sigmoid-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  ADD_EXECUTABLE(f32-sqrt-ulp-eval eval/f32-sqrt-ulp.cc)
  SET_TARGET_PROPERTIES(f32-sqrt-ulp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-sqrt-ulp-eval PRIVATE . src)
  TARGET_LINK_LIBRARIES(f32-sqrt-ulp-eval PRIVATE XNNPACK benchmark bench-utils cpuinfo fp16 pthreadpool)

  # ---[ Build accuracy tests
  ADD_EXECUTABLE(f32-exp-eval eval/f32-exp.cc)
  SET_TARGET_PROPERTIES(f32-exp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-exp-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-exp-eval PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)

  ADD_EXECUTABLE(f32-expm1minus-eval eval/f32-expm1minus.cc)
  SET_TARGET_PROPERTIES(f32-expm1minus-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-expm1minus-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-expm1minus-eval PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)

  ADD_EXECUTABLE(f32-expminus-eval eval/f32-expminus.cc)
  SET_TARGET_PROPERTIES(f32-expminus-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-expminus-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-expminus-eval PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)

  ADD_EXECUTABLE(f32-roundne-eval eval/f32-roundne.cc)
  SET_TARGET_PROPERTIES(f32-roundne-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-roundne-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-roundne-eval PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)

  ADD_EXECUTABLE(f32-roundd-eval eval/f32-roundd.cc)
  SET_TARGET_PROPERTIES(f32-roundd-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-roundd-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-roundd-eval PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)

  ADD_EXECUTABLE(f32-roundu-eval eval/f32-roundu.cc)
  SET_TARGET_PROPERTIES(f32-roundu-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-roundu-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-roundu-eval PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)

  ADD_EXECUTABLE(f32-roundz-eval eval/f32-roundz.cc)
  SET_TARGET_PROPERTIES(f32-roundz-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-roundz-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-roundz-eval PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)

  # ---[ Build end-to-end microbenchmarks
  ADD_LIBRARY(bench-models STATIC
    models/fp16-mobilenet-v1.cc
    models/fp32-mobilenet-v1.cc
    models/qs8-mobilenet-v1.cc
    models/qs8-mobilenet-v2.cc
    models/qu8-mobilenet-v1.cc
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
  SET_TARGET_PROPERTIES(bench-models PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(bench-models PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(bench-models PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(end2end-bench bench/end2end.cc)
  SET_TARGET_PROPERTIES(end2end-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(end2end-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(end2end-bench PRIVATE XNNPACK benchmark bench-models bench-utils)

  ADD_EXECUTABLE(f32-dwconv-e2e-bench bench/f32-dwconv-e2e.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-e2e-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-e2e-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" src)
  TARGET_LINK_LIBRARIES(f32-dwconv-e2e-bench PRIVATE XNNPACK benchmark bench-models bench-utils)

  ADD_EXECUTABLE(f32-gemm-e2e-bench bench/f32-gemm-e2e.cc)
  SET_TARGET_PROPERTIES(f32-gemm-e2e-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-e2e-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" src)
  TARGET_LINK_LIBRARIES(f32-gemm-e2e-bench PRIVATE XNNPACK fp16 benchmark bench-models bench-utils)

  ADD_EXECUTABLE(qs8-gemm-e2e-bench bench/qs8-gemm-e2e.cc)
  SET_TARGET_PROPERTIES(qs8-gemm-e2e-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-e2e-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" src)
  TARGET_LINK_LIBRARIES(qs8-gemm-e2e-bench PRIVATE XNNPACK fp16 benchmark bench-models bench-utils)

  # ---[ Build operator-level microbenchmarks
  ADD_EXECUTABLE(average-pooling-bench bench/average-pooling.cc)
  SET_TARGET_PROPERTIES(average-pooling-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(average-pooling-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(average-pooling-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(bankers-rounding-bench bench/bankers-rounding.cc)
  SET_TARGET_PROPERTIES(bankers-rounding-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(bankers-rounding-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(bankers-rounding-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(ceiling-bench bench/ceiling.cc)
  SET_TARGET_PROPERTIES(ceiling-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(ceiling-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(ceiling-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(channel-shuffle-bench bench/channel-shuffle.cc)
  SET_TARGET_PROPERTIES(channel-shuffle-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(channel-shuffle-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(channel-shuffle-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(convolution-bench bench/convolution.cc)
  SET_TARGET_PROPERTIES(convolution-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(convolution-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(convolution-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(deconvolution-bench bench/deconvolution.cc)
  SET_TARGET_PROPERTIES(deconvolution-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(deconvolution-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(elu-bench bench/elu.cc)
  SET_TARGET_PROPERTIES(elu-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(elu-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(elu-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(floor-bench bench/floor.cc)
  SET_TARGET_PROPERTIES(floor-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(floor-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(floor-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(global-average-pooling-bench bench/global-average-pooling.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(global-average-pooling-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(hardswish-bench bench/hardswish.cc)
  SET_TARGET_PROPERTIES(hardswish-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(hardswish-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(hardswish-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(max-pooling-bench bench/max-pooling.cc)
  SET_TARGET_PROPERTIES(max-pooling-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(max-pooling-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(prelu-bench bench/prelu.cc)
  SET_TARGET_PROPERTIES(prelu-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(prelu-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(prelu-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(sigmoid-bench bench/sigmoid.cc)
  SET_TARGET_PROPERTIES(sigmoid-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(sigmoid-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(sigmoid-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(softmax-bench bench/softmax.cc)
  SET_TARGET_PROPERTIES(softmax-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(softmax-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(softmax-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(square-root-bench bench/square-root.cc)
  SET_TARGET_PROPERTIES(square-root-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(square-root-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(square-root-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(truncation-bench bench/truncation.cc)
  SET_TARGET_PROPERTIES(truncation-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(truncation-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(truncation-bench PRIVATE XNNPACK benchmark bench-utils)

  # ---[ Build microkernel-level microbenchmarks
  ADD_EXECUTABLE(f16-dwconv-bench bench/f16-dwconv.cc)
  SET_TARGET_PROPERTIES(f16-dwconv-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-dwconv-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f16-dwconv-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f16-dwconv-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f16-gemm-bench bench/f16-gemm.cc)
  SET_TARGET_PROPERTIES(f16-gemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f16-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f16-igemm-bench bench/f16-igemm.cc)
  SET_TARGET_PROPERTIES(f16-igemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f16-igemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f16-vrelu-bench bench/f16-vrelu.cc)
  SET_TARGET_PROPERTIES(f16-vrelu-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-vrelu-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f16-vrelu-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f16-vrelu-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-conv-hwc-bench bench/f32-conv-hwc.cc)
  SET_TARGET_PROPERTIES(f32-conv-hwc-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-conv-hwc-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-dwconv-bench bench/f32-dwconv.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-dwconv-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-dwconv2d-chw-bench bench/f32-dwconv2d-chw.cc)
  SET_TARGET_PROPERTIES(f32-dwconv2d-chw-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv2d-chw-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv2d-chw-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-dwconv2d-chw-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-gemm-bench bench/f32-gemm.cc)
  SET_TARGET_PROPERTIES(f32-gemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-igemm-bench bench/f32-igemm.cc)
  SET_TARGET_PROPERTIES(f32-igemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-igemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-im2col-gemm-bench bench/f32-im2col-gemm.cc src/im2col.c)
  SET_TARGET_PROPERTIES(f32-im2col-gemm-bench PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-im2col-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-im2col-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-im2col-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-rmax-bench bench/f32-rmax.cc)
  SET_TARGET_PROPERTIES(f32-rmax-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-rmax-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-rmax-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-rmax-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-spmm-bench bench/f32-spmm.cc)
  SET_TARGET_PROPERTIES(f32-spmm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-spmm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-spmm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-spmm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-softmax-bench bench/f32-softmax.cc)
  SET_TARGET_PROPERTIES(f32-softmax-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-softmax-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-softmax-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-softmax-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-velu-bench bench/f32-velu.cc)
  SET_TARGET_PROPERTIES(f32-velu-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-velu-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-velu-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-velu-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-vhswish-bench bench/f32-vhswish.cc)
  SET_TARGET_PROPERTIES(f32-vhswish-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vhswish-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-vhswish-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-vhswish-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-vrelu-bench bench/f32-vrelu.cc)
  SET_TARGET_PROPERTIES(f32-vrelu-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrelu-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-vrelu-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-vrelu-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-vsigmoid-bench bench/f32-vsigmoid.cc)
  SET_TARGET_PROPERTIES(f32-vsigmoid-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsigmoid-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-vsigmoid-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-vsigmoid-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(f32-vsqrt-bench bench/f32-vsqrt.cc)
  SET_TARGET_PROPERTIES(f32-vsqrt-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrt-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-vsqrt-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-vsqrt-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(qs8-gemm-bench bench/qs8-gemm.cc)
  SET_TARGET_PROPERTIES(qs8-gemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(qs8-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(qs8-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(qs8-requantization-bench bench/qs8-requantization.cc)
  SET_TARGET_PROPERTIES(qs8-requantization-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qs8-requantization-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(qs8-requantization-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(qs8-requantization-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(qu8-gemm-bench bench/qu8-gemm.cc)
  SET_TARGET_PROPERTIES(qu8-gemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(qu8-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(qu8-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(qu8-requantization-bench bench/qu8-requantization.cc)
  SET_TARGET_PROPERTIES(qu8-requantization-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(qu8-requantization-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(qu8-requantization-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(qu8-requantization-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(rounding-bench bench/rounding.cc)
  SET_TARGET_PROPERTIES(rounding-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(rounding-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(rounding-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(rounding-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)
ENDIF()
