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
OPTION(XNNPACK_BUILD_TESTS "Build XNNPACK unit tests" ON)
OPTION(XNNPACK_BUILD_BENCHMARKS "Build XNNPACK benchmarks" ON)
OPTION(XNNPACK_USE_SYSTEM_LIBS "Use system-provided dependency libraries" OFF)

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

IF(MSVC)
  # Disable "unary minus operator applied to unsigned type, result still unsigned" warning
  ADD_COMPILE_OPTIONS("/wd4146")
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
ELSEIF(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86|x86_64|AMD64|armv[5-8].*|aarch64)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
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

  IF(NOT DEFINED PSIMD_SOURCE_DIR)
    MESSAGE(STATUS "Downloading PSimd to ${CMAKE_BINARY_DIR}/psimd-source (define PSIMD_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadPSimd.cmake "${CMAKE_BINARY_DIR}/psimd-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
    SET(PSIMD_SOURCE_DIR "${CMAKE_BINARY_DIR}/psimd-source" CACHE STRING "PSimd source directory")
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
  src/operators/add-nc.c
  src/operators/argmax-pooling-nhwc.c
  src/operators/average-pooling-nhwc.c
  src/operators/binary-elementwise-nd.c
  src/operators/channel-pad-nc.c
  src/operators/channel-shuffle-nc.c
  src/operators/clamp-nc.c
  src/operators/convolution-nchw.c
  src/operators/convolution-nhwc.c
  src/operators/deconvolution-nhwc.c
  src/operators/fully-connected-nc.c
  src/operators/global-average-pooling-ncw.c
  src/operators/global-average-pooling-nwc.c
  src/operators/hardswish-nc.c
  src/operators/leaky-relu-nc.c
  src/operators/max-pooling-nhwc.c
  src/operators/prelu-nc.c
  src/operators/resize-bilinear-nhwc.c
  src/operators/sigmoid-nc.c
  src/operators/softmax-nc.c
  src/operators/unpooling-nhwc.c)

SET(XNNPACK_COLD_SRCS ${XNNPACK_OPERATOR_SRCS})
LIST(APPEND XNNPACK_COLD_SRCS
  src/init.c
  src/memory.c
  src/operator-delete.c
  src/runtime.c
  src/subgraph.c
  src/tensor.c)

SET(XNNPACK_HOT_SRCS
  src/indirection.c
  src/operator-run.c)

SET(XNNPACK_TABLE_SRCS
  src/tables/exp2-k-over-64.c
  src/tables/exp2-k-over-2048.c)

SET(XNNPACK_SCALAR_MICROKERNEL_SRCS
  src/f32-argmaxpool/4x-scalar-c1.c
  src/f32-argmaxpool/9p8x-scalar-c1.c
  src/f32-argmaxpool/9x-scalar-c1.c
  src/f32-avgpool/9p8x-minmax-scalar-c1.c
  src/f32-avgpool/9x-minmax-scalar-c1.c
  src/f32-clamp/gen/scalar-x1.c
  src/f32-clamp/gen/scalar-x2.c
  src/f32-clamp/gen/scalar-x4.c
  src/f32-conv-hwc/3x3s2p1c3x4-scalar-1x1.c
  src/f32-conv-hwc/3x3s2p0p1c3x4-scalar-1x1.c
  src/f32-conv-hwc2spchw/3x3s2p1c3x4-scalar-1x1.c
  src/f32-dwconv-spchw/3x3p1-scalar.c
  src/f32-dwconv-spchw/3x3s2p1-scalar.c
  src/f32-dwconv-spchw/5x5p2-scalar.c
  src/f32-dwconv-spchw/5x5s2p2-scalar.c
  src/f32-dwconv/gen/up1x4-scalar.c
  src/f32-dwconv/gen/up1x4-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-scalar.c
  src/f32-dwconv/gen/up1x9-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-scalar.c
  src/f32-dwconv/gen/up1x25-scalar-acc2.c
  src/f32-dwconv/gen/up2x4-scalar.c
  src/f32-dwconv/gen/up2x4-scalar-acc2.c
  src/f32-dwconv/gen/up2x9-scalar.c
  src/f32-dwconv/gen/up2x9-scalar-acc2.c
  src/f32-dwconv/gen/up2x25-scalar.c
  src/f32-dwconv/gen/up2x25-scalar-acc2.c
  src/f32-dwconv/gen/up1x4-minmax-scalar.c
  src/f32-dwconv/gen/up1x4-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x9-minmax-scalar.c
  src/f32-dwconv/gen/up1x9-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up1x25-minmax-scalar.c
  src/f32-dwconv/gen/up1x25-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up2x4-minmax-scalar.c
  src/f32-dwconv/gen/up2x4-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up2x9-minmax-scalar.c
  src/f32-dwconv/gen/up2x9-minmax-scalar-acc2.c
  src/f32-dwconv/gen/up2x25-minmax-scalar.c
  src/f32-dwconv/gen/up2x25-minmax-scalar-acc2.c
  src/f32-gavgpool-spchw/scalar-x1.c
  src/f32-gavgpool/7p7x-minmax-scalar-c1.c
  src/f32-gavgpool/7x-minmax-scalar-c1.c
  src/f32-gemm/gen/1x4-scalar.c
  src/f32-gemm/gen/2x4-scalar.c
  src/f32-gemm/gen/4x2-scalar.c
  src/f32-gemm/gen/4x4-scalar.c
  src/f32-gemm/gen/1x4-minmax-scalar.c
  src/f32-gemm/gen/2x4-minmax-scalar.c
  src/f32-gemm/gen/4x2-minmax-scalar.c
  src/f32-gemm/gen/4x4-minmax-scalar.c
  src/f32-gemm/gen-inc/1x4inc-minmax-scalar.c
  src/f32-gemm/gen-inc/2x4inc-minmax-scalar.c
  src/f32-gemm/gen-inc/4x4inc-minmax-scalar.c
  src/f32-hswish/gen/scalar-x1.c
  src/f32-hswish/gen/scalar-x2.c
  src/f32-hswish/gen/scalar-x4.c
  src/f32-ibilinear/gen/scalar-c1.c
  src/f32-ibilinear/gen/scalar-c2.c
  src/f32-ibilinear/gen/scalar-c4.c
  src/f32-igemm/gen/1x4-scalar.c
  src/f32-igemm/gen/2x4-scalar.c
  src/f32-igemm/gen/4x2-scalar.c
  src/f32-igemm/gen/4x4-scalar.c
  src/f32-igemm/gen/1x4-minmax-scalar.c
  src/f32-igemm/gen/2x4-minmax-scalar.c
  src/f32-igemm/gen/4x2-minmax-scalar.c
  src/f32-igemm/gen/4x4-minmax-scalar.c
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
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x2.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x2-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4-acc4.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x1.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x2.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x2-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x4-acc2.c
  src/f32-raddstoreexpminusmax/gen/scalar-p5-x4-acc4.c
  src/f32-rmax/scalar.c
  src/f32-sigmoid/gen/scalar-lut2048-p1-div-x1.c
  src/f32-sigmoid/gen/scalar-lut2048-p1-div-x2.c
  src/f32-sigmoid/gen/scalar-lut2048-p1-div-x4.c
  src/f32-sigmoid/gen/scalar-lut64-p2-div-x1.c
  src/f32-sigmoid/gen/scalar-lut64-p2-div-x2.c
  src/f32-sigmoid/gen/scalar-lut64-p2-div-x4.c
  src/f32-sigmoid/gen/scalar-p5-div-x1.c
  src/f32-sigmoid/gen/scalar-p5-div-x2.c
  src/f32-sigmoid/gen/scalar-p5-div-x4.c
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
  src/f32-vbinary/gen/vaddc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vaddc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x1.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x2.c
  src/f32-vbinary/gen/vdiv-minmax-scalar-x4.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vdivc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vmax-scalar-x1.c
  src/f32-vbinary/gen/vmax-scalar-x2.c
  src/f32-vbinary/gen/vmax-scalar-x4.c
  src/f32-vbinary/gen/vmaxc-scalar-x1.c
  src/f32-vbinary/gen/vmaxc-scalar-x2.c
  src/f32-vbinary/gen/vmaxc-scalar-x4.c
  src/f32-vbinary/gen/vmin-scalar-x1.c
  src/f32-vbinary/gen/vmin-scalar-x2.c
  src/f32-vbinary/gen/vmin-scalar-x4.c
  src/f32-vbinary/gen/vminc-scalar-x1.c
  src/f32-vbinary/gen/vminc-scalar-x2.c
  src/f32-vbinary/gen/vminc-scalar-x4.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x1.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x2.c
  src/f32-vbinary/gen/vmul-minmax-scalar-x4.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vmulc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vrdivc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vrsubc-minmax-scalar-x4.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x1.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x2.c
  src/f32-vbinary/gen/vsub-minmax-scalar-x4.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x1.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x2.c
  src/f32-vbinary/gen/vsubc-minmax-scalar-x4.c
  src/f32-vmulcaddc/gen/c1-minmax-scalar-2x.c
  src/f32-vmulcaddc/gen/c2-minmax-scalar-2x.c
  src/f32-vmulcaddc/gen/c4-minmax-scalar-2x.c
  src/math/expminus-scalar-lut2048-p1.c
  src/math/expminus-scalar-lut64-p2.c
  src/math/expminus-scalar-p5.c
  src/math/sigmoid-scalar-lut2048-p1-div.c
  src/math/sigmoid-scalar-lut64-p2-div.c
  src/math/sigmoid-scalar-p5-div.c
  src/q8-avgpool/9p8x-minmax-scalar-c1.c
  src/q8-avgpool/9x-minmax-scalar-c1.c
  src/q8-dwconv/up1x9-minmax-scalar.c
  src/q8-gavgpool/7p7x-minmax-scalar-c1.c
  src/q8-gavgpool/7x-minmax-scalar-c1.c
  src/q8-gemm/2x2-minmax-scalar.c
  src/q8-igemm/2x2-minmax-scalar.c
  src/q8-vadd/minmax-scalar.c
  src/u8-clamp/scalar-x4.c
  src/u8-lut32norm/scalar.c
  src/u8-maxpool/9p8x-minmax-scalar-c1.c
  src/u8-rmax/scalar.c
  src/x32-packx/x2-scalar.c
  src/x32-packx/x3-scalar.c
  src/x32-packx/x4-scalar.c
  src/x32-pad/x2-scalar.c
  src/x32-unpool/scalar.c
  src/x32-zip/x2-scalar.c
  src/x32-zip/x3-scalar.c
  src/x32-zip/x4-scalar.c
  src/x32-zip/xm-scalar.c
  src/x8-lut/scalar.c
  src/x8-zip/x2-scalar.c
  src/x8-zip/x3-scalar.c
  src/x8-zip/x4-scalar.c
  src/x8-zip/xm-scalar.c
  src/requantization/precise-scalar.c
  src/requantization/fp32-scalar.c
  src/requantization/q31-scalar.c
  src/requantization/gemmlowp-scalar.c)

SET(XNNPACK_PSIMD_FASTMATH_MICROKERNEL_SRCS
  src/f32-argmaxpool/4x-psimd-c4.c
  src/f32-argmaxpool/9p8x-psimd-c4.c
  src/f32-argmaxpool/9x-psimd-c4.c
  src/f32-avgpool/9p8x-minmax-psimd-c4.c
  src/f32-avgpool/9x-minmax-psimd-c4.c
  src/f32-clamp/gen/psimd-x4.c
  src/f32-clamp/gen/psimd-x8.c
  src/f32-dwconv/gen/up4x25-minmax-psimd-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-psimd.c
  src/f32-dwconv/gen/up4x4-minmax-psimd-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-psimd.c
  src/f32-dwconv/gen/up4x9-minmax-psimd-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-psimd.c
  src/f32-dwconv/gen/up8x25-minmax-psimd-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-psimd.c
  src/f32-dwconv/gen/up8x4-minmax-psimd-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-psimd.c
  src/f32-dwconv/gen/up8x9-minmax-psimd-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-psimd.c
  src/f32-gavgpool/7p7x-minmax-psimd-c4.c
  src/f32-gavgpool/7x-minmax-psimd-c4.c
  src/f32-gemm/gen/1x8-minmax-psimd-loadsplat.c
  src/f32-gemm/gen/1x8-minmax-psimd-splat.c
  src/f32-gemm/gen/1x8s4-minmax-psimd.c
  src/f32-gemm/gen/4x2c4-minmax-psimd.c
  src/f32-gemm/gen/4x8-minmax-psimd-loadsplat.c
  src/f32-gemm/gen/4x8-minmax-psimd-splat.c
  src/f32-gemm/gen/4x8s4-minmax-psimd.c
  src/f32-gemm/gen/6x8-minmax-psimd-loadsplat.c
  src/f32-gemm/gen/6x8-minmax-psimd-splat.c
  src/f32-gemm/gen/6x8s4-minmax-psimd.c
  src/f32-gemm/gen-inc/1x8inc-minmax-psimd-loadsplat.c
  src/f32-gemm/gen-inc/1x8inc-minmax-psimd-splat.c
  src/f32-gemm/gen-inc/1x8s4inc-minmax-psimd.c
  src/f32-gemm/gen-inc/4x8inc-minmax-psimd-loadsplat.c
  src/f32-gemm/gen-inc/4x8inc-minmax-psimd-splat.c
  src/f32-gemm/gen-inc/4x8s4inc-minmax-psimd.c
  src/f32-gemm/gen-inc/6x8inc-minmax-psimd-loadsplat.c
  src/f32-gemm/gen-inc/6x8inc-minmax-psimd-splat.c
  src/f32-gemm/gen-inc/6x8s4inc-minmax-psimd.c
  src/f32-hswish/gen/psimd-x4.c
  src/f32-hswish/gen/psimd-x8.c
  src/f32-ibilinear/gen/psimd-c4.c
  src/f32-ibilinear/gen/psimd-c8.c
  src/f32-igemm/gen/1x8-minmax-psimd-loadsplat.c
  src/f32-igemm/gen/1x8-minmax-psimd-splat.c
  src/f32-igemm/gen/1x8s4-minmax-psimd.c
  src/f32-igemm/gen/4x2c4-minmax-psimd.c
  src/f32-igemm/gen/4x8-minmax-psimd-loadsplat.c
  src/f32-igemm/gen/4x8-minmax-psimd-splat.c
  src/f32-igemm/gen/4x8s4-minmax-psimd.c
  src/f32-igemm/gen/6x8-minmax-psimd-loadsplat.c
  src/f32-igemm/gen/6x8-minmax-psimd-splat.c
  src/f32-igemm/gen/6x8s4-minmax-psimd.c
  src/f32-maxpool/9p8x-minmax-psimd-c4.c
  src/f32-pavgpool/9p8x-minmax-psimd-c4.c
  src/f32-pavgpool/9x-minmax-psimd-c4.c
  src/f32-ppmm/gen/4x8-minmax-psimd.c
  src/f32-prelu/gen/psimd-2x4.c
  src/f32-prelu/gen/psimd-2x8.c
  src/f32-rmax/psimd.c
  src/f32-vbinary/gen/vadd-minmax-psimd-x4.c
  src/f32-vbinary/gen/vadd-minmax-psimd-x8.c
  src/f32-vbinary/gen/vaddc-minmax-psimd-x4.c
  src/f32-vbinary/gen/vaddc-minmax-psimd-x8.c
  src/f32-vbinary/gen/vdiv-minmax-psimd-x4.c
  src/f32-vbinary/gen/vdiv-minmax-psimd-x8.c
  src/f32-vbinary/gen/vdivc-minmax-psimd-x4.c
  src/f32-vbinary/gen/vdivc-minmax-psimd-x8.c
  src/f32-vbinary/gen/vmax-psimd-x4.c
  src/f32-vbinary/gen/vmax-psimd-x8.c
  src/f32-vbinary/gen/vmaxc-psimd-x4.c
  src/f32-vbinary/gen/vmaxc-psimd-x8.c
  src/f32-vbinary/gen/vmin-psimd-x4.c
  src/f32-vbinary/gen/vmin-psimd-x8.c
  src/f32-vbinary/gen/vminc-psimd-x4.c
  src/f32-vbinary/gen/vminc-psimd-x8.c
  src/f32-vbinary/gen/vmul-minmax-psimd-x4.c
  src/f32-vbinary/gen/vmul-minmax-psimd-x8.c
  src/f32-vbinary/gen/vmulc-minmax-psimd-x4.c
  src/f32-vbinary/gen/vmulc-minmax-psimd-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-psimd-x4.c
  src/f32-vbinary/gen/vrdivc-minmax-psimd-x8.c
  src/f32-vbinary/gen/vrsubc-minmax-psimd-x4.c
  src/f32-vbinary/gen/vrsubc-minmax-psimd-x8.c
  src/f32-vbinary/gen/vsub-minmax-psimd-x4.c
  src/f32-vbinary/gen/vsub-minmax-psimd-x8.c
  src/f32-vbinary/gen/vsubc-minmax-psimd-x4.c
  src/f32-vbinary/gen/vsubc-minmax-psimd-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-psimd-2x.c
  src/f32-vmulcaddc/gen/c8-minmax-psimd-2x.c
  src/x32-packx/x4-psimd.c
  src/x32-pad/x2-psimd.c
  src/x32-unpool/psimd.c
  src/x32-zip/x2-psimd.c
  src/x32-zip/x3-psimd.c
  src/x32-zip/x4-psimd.c
  src/x32-zip/xm-psimd.c
  src/requantization/precise-psimd.c
  src/requantization/fp32-psimd.c)

SET(XNNPACK_PSIMD_ACCMATH_MICROKERNEL_SRCS
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x20.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/psimd-p5-x20-acc5.c
  src/f32-sigmoid/gen/psimd-p5-div-x4.c
  src/f32-sigmoid/gen/psimd-p5-div-x8.c
  src/f32-sigmoid/gen/psimd-p5-div-x12.c
  src/f32-sigmoid/gen/psimd-p5-div-x16.c
  src/f32-sigmoid/gen/psimd-p5-div-x20.c
  src/f32-sigmoid/gen/psimd-p5-div-x24.c
  src/math/sigmoid-psimd-p5-div.c)

SET(XNNPACK_NEON_MICROKERNEL_SRCS
  src/f32-avgpool/9p8x-minmax-neon-c4.c
  src/f32-avgpool/9x-minmax-neon-c4.c
  src/f32-clamp/gen/neon-x4.c
  src/f32-clamp/gen/neon-x8.c
  src/f32-dwconv/gen/up4x4-minmax-neon.c
  src/f32-dwconv/gen/up4x4-minmax-neon-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-neon.c
  src/f32-dwconv/gen/up8x4-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-neon.c
  src/f32-dwconv/gen/up4x9-minmax-neon-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-neon.c
  src/f32-dwconv/gen/up8x9-minmax-neon-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-neon.c
  src/f32-dwconv/gen/up4x25-minmax-neon-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-neon.c
  src/f32-dwconv/gen/up8x25-minmax-neon-acc2.c
  src/f32-gavgpool-spchw/neon-x4.c
  src/f32-gavgpool/7p7x-minmax-neon-c4.c
  src/f32-gavgpool/7x-minmax-neon-c4.c
  src/f32-gemm/gen/1x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/4x2-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neon-lane-ld128.c
  src/f32-gemm/gen/4x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/5x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-lane-ld128.c
  src/f32-gemm/gen/1x8-minmax-neon-dup-ld64.c
  src/f32-gemm/gen/4x8-minmax-neon-dup-ld128.c
  src/f32-gemm/gen/4x8-minmax-neon-dup-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-dup-ld64.c
  src/f32-gemm/gen/6x8-minmax-neon-dup-ld128.c
  src/f32-gemm/gen/1x8s4-minmax-neon.c
  src/f32-gemm/gen/4x8s4-minmax-neon.c
  src/f32-gemm/gen/6x8s4-minmax-neon.c
  src/f32-gemm/gen/8x8s4-minmax-neon.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-lane-ld128.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/5x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-lane-ld128.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neon-dup-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-dup-ld128.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neon-dup-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-dup-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neon-dup-ld128.c
  src/f32-gemm/gen-inc/1x8s4inc-minmax-neon.c
  src/f32-gemm/gen-inc/4x8s4inc-minmax-neon.c
  src/f32-gemm/gen-inc/6x8s4inc-minmax-neon.c
  src/f32-gemm/gen-inc/8x8s4inc-minmax-neon.c
  src/f32-hswish/gen/neon-x4.c
  src/f32-hswish/gen/neon-x8.c
  src/f32-ibilinear/gen/neon-c4.c
  src/f32-ibilinear/gen/neon-c8.c
  src/f32-igemm/gen/1x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x2-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x4-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neon-lane-ld128.c
  src/f32-igemm/gen/4x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neon-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neon-lane-ld128.c
  src/f32-igemm/gen/1x8-minmax-neon-dup-ld64.c
  src/f32-igemm/gen/4x8-minmax-neon-dup-ld128.c
  src/f32-igemm/gen/4x8-minmax-neon-dup-ld64.c
  src/f32-igemm/gen/6x8-minmax-neon-dup-ld64.c
  src/f32-igemm/gen/6x8-minmax-neon-dup-ld128.c
  src/f32-igemm/gen/1x8s4-minmax-neon.c
  src/f32-igemm/gen/4x8s4-minmax-neon.c
  src/f32-igemm/gen/6x8s4-minmax-neon.c
  src/f32-igemm/gen/8x8s4-minmax-neon.c
  src/f32-maxpool/9p8x-minmax-neon-c4.c
  src/f32-pavgpool/9p8x-minmax-neon-c4.c
  src/f32-pavgpool/9x-minmax-neon-c4.c
  src/f32-ppmm/gen/4x8-minmax-neon.c
  src/f32-ppmm/gen/8x8-minmax-neon.c
  src/f32-prelu/gen/neon-2x4.c
  src/f32-prelu/gen/neon-2x8.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x20.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x8.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20-acc5.c
  src/f32-rmax/neon.c
  src/f32-sigmoid/gen/neon-frac-p9-p10-nr1recps-x16.c
  src/f32-sigmoid/gen/neon-rr2-p5-nr2recps-x4.c
  src/f32-sigmoid/gen/neon-rr2-p5-nr2recps-x8.c
  src/f32-sigmoid/gen/neon-rr2-p5-nr2recps-x12.c
  src/f32-sigmoid/gen/neon-rr2-p5-nr2recps-x16.c
  src/f32-sigmoid/gen/neon-rr2-p5-nr2recps-x20.c
  src/f32-sigmoid/gen/neon-rr2-p5-nr2recps-x24.c
  src/f32-sigmoid/gen/neon-rr2-lut64-p2-nr2recps-x4.c
  src/f32-sigmoid/gen/neon-rr2-lut64-p2-nr2recps-x8.c
  src/f32-sigmoid/gen/neon-rr2-lut64-p2-nr2recps-x12.c
  src/f32-sigmoid/gen/neon-rr2-lut64-p2-nr2recps-x16.c
  src/f32-sigmoid/gen/neon-rr2-lut64-p2-nr2recps-x20.c
  src/f32-sigmoid/gen/neon-rr2-lut64-p2-nr2recps-x24.c
  src/f32-sigmoid/gen/neon-rr2-lut2048-p1-nr2recps-x4.c
  src/f32-sigmoid/gen/neon-rr2-lut2048-p1-nr2recps-x8.c
  src/f32-sigmoid/gen/neon-rr2-lut2048-p1-nr2recps-x12.c
  src/f32-sigmoid/gen/neon-rr2-lut2048-p1-nr2recps-x16.c
  src/f32-sigmoid/gen/neon-rr2-lut2048-p1-nr2recps-x20.c
  src/f32-sigmoid/gen/neon-rr2-lut2048-p1-nr2recps-x24.c
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
  src/f32-vbinary/gen/vsub-minmax-neon-x4.c
  src/f32-vbinary/gen/vsub-minmax-neon-x8.c
  src/f32-vbinary/gen/vsubc-minmax-neon-x4.c
  src/f32-vbinary/gen/vsubc-minmax-neon-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-neon-2x.c
  src/f32-vmulcaddc/gen/c8-minmax-neon-2x.c
  src/q8-avgpool/9p8x-minmax-neon-c8.c
  src/q8-avgpool/9x-minmax-neon-c8.c
  src/q8-dwconv/up8x9-minmax-neon.c
  src/q8-gavgpool/7p7x-minmax-neon-c8.c
  src/q8-gavgpool/7x-minmax-neon-c8.c
  src/q8-gemm/4x8-minmax-neon.c
  src/q8-gemm/8x8-minmax-neon.c
  src/q8-igemm/4x8-minmax-neon.c
  src/q8-igemm/8x8-minmax-neon.c
  src/q8-vadd/minmax-neon.c
  src/u8-clamp/neon-x64.c
  src/u8-maxpool/9p8x-minmax-neon-c16.c
  src/u8-rmax/neon.c
  src/x32-packx/x4-neon-st4.c
  src/x32-pad/x2-neon.c
  src/x32-unpool/neon.c
  src/x32-zip/x2-neon.c
  src/x32-zip/x3-neon.c
  src/x32-zip/x4-neon.c
  src/x32-zip/xm-neon.c
  src/x8-zip/x2-neon.c
  src/x8-zip/x3-neon.c
  src/x8-zip/x4-neon.c
  src/x8-zip/xm-neon.c
  src/math/sigmoid-neon-frac-p9-p10-nr1recps.c
  src/math/sigmoid-neon-rr1-lut2048-p1-nr2recps.c
  src/math/sigmoid-neon-rr1-lut64-p2-nr2recps.c
  src/math/sigmoid-neon-rr1-p5-nr2recps.c
  src/math/sigmoid-neon-rr2-lut2048-p1-nr2recps.c
  src/math/sigmoid-neon-rr2-lut64-p2-nr2recps.c
  src/math/sigmoid-neon-rr2-p5-nr2recps.c
  src/requantization/precise-neon.c
  src/requantization/fp32-neon.c
  src/requantization/q31-neon.c
  src/requantization/gemmlowp-neon.c)

SET(XNNPACK_NEONFMA_MICROKERNEL_SRCS
  src/f32-ibilinear/gen/neonfma-c4.c
  src/f32-ibilinear/gen/neonfma-c8.c
  src/f32-igemm/gen/1x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/4x8-minmax-neonfma-dup-ld128.c
  src/f32-igemm/gen/4x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-dup-ld128.c
  src/f32-igemm/gen/1x8s4-minmax-neonfma.c
  src/f32-igemm/gen/4x8s4-minmax-neonfma.c
  src/f32-igemm/gen/6x8s4-minmax-neonfma.c
  src/f32-igemm/gen/8x8s4-minmax-neonfma.c
  src/f32-dwconv/gen/up4x4-minmax-neonfma.c
  src/f32-dwconv/gen/up4x4-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-neonfma.c
  src/f32-dwconv/gen/up8x4-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-neonfma.c
  src/f32-dwconv/gen/up4x9-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-neonfma.c
  src/f32-dwconv/gen/up8x9-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-neonfma.c
  src/f32-dwconv/gen/up4x25-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-neonfma.c
  src/f32-dwconv/gen/up8x25-minmax-neonfma-acc2.c
  src/f32-gemm/gen/1x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/4x8-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen/4x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen/1x8s4-minmax-neonfma.c
  src/f32-gemm/gen/4x8s4-minmax-neonfma.c
  src/f32-gemm/gen/6x8s4-minmax-neonfma.c
  src/f32-gemm/gen/8x8s4-minmax-neonfma.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen-inc/1x8s4inc-minmax-neonfma.c
  src/f32-gemm/gen-inc/4x8s4inc-minmax-neonfma.c
  src/f32-gemm/gen-inc/6x8s4inc-minmax-neonfma.c
  src/f32-gemm/gen-inc/8x8s4inc-minmax-neonfma.c
  src/f32-hswish/gen/neonfma-x4.c
  src/f32-hswish/gen/neonfma-x8.c
  src/f32-ppmm/gen/4x8-minmax-neonfma.c
  src/f32-ppmm/gen/8x8-minmax-neonfma.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20-acc5.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x8.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20-acc5.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2fma-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2fma-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2fma-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2fma-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2fma-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2fma-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr1recps1fma-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr1recps1fma-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr1recps1fma-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr1recps1fma-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr1recps1fma-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr1recps1fma-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2recps-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2recps-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2recps-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2recps-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2recps-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-nr2recps-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2fma-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2fma-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2fma-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2fma-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2fma-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2fma-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr1recps1fma-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr1recps1fma-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr1recps1fma-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr1recps1fma-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr1recps1fma-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr1recps1fma-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2recps-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2recps-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2recps-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2recps-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2recps-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-nr2recps-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2fma-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2fma-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2fma-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2fma-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2fma-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2fma-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr1recps1fma-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr1recps1fma-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr1recps1fma-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr1recps1fma-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr1recps1fma-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr1recps1fma-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2recps-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2recps-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2recps-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2recps-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2recps-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-nr2recps-x24.c
  src/f32-vmulcaddc/gen/c4-minmax-neonfma-2x.c
  src/f32-vmulcaddc/gen/c8-minmax-neonfma-2x.c
  src/math/exp-neonfma-lut64-p2.c
  src/math/exp-neonfma-p5.c
  src/math/expminus-neonfma-lut2048-p1.c
  src/math/expminus-neonfma-lut64-p2.c
  src/math/expminus-neonfma-p5.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-nr2fma.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-nr2recps.c
  src/math/sigmoid-neonfma-rr1-lut64-p2-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr1-lut64-p2-nr2fma.c
  src/math/sigmoid-neonfma-rr1-lut64-p2-nr2recps.c
  src/math/sigmoid-neonfma-rr1-p5-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr1-p5-nr2fma.c
  src/math/sigmoid-neonfma-rr1-p5-nr2recps.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-nr2fma.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-nr2recps.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-nr2fma.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-nr2recps.c
  src/math/sigmoid-neonfma-rr2-p5-nr1recps1fma.c
  src/math/sigmoid-neonfma-rr2-p5-nr2fma.c
  src/math/sigmoid-neonfma-rr2-p5-nr2recps.c)

SET(XNNPACK_AARCH64_NEONFMA_MICROKERNEL_SRCS
  src/f32-vbinary/gen/vdiv-minmax-neon-x4.c
  src/f32-vbinary/gen/vdiv-minmax-neon-x8.c
  src/f32-vbinary/gen/vdivc-minmax-neon-x4.c
  src/f32-vbinary/gen/vdivc-minmax-neon-x8.c
  src/f32-vbinary/gen/vrdivc-minmax-neon-x4.c
  src/f32-vbinary/gen/vrdivc-minmax-neon-x8.c
  src/f32-gemm/gen/1x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/4x2-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/4x8-minmax-neonfma-lane-ld128.c
  src/f32-gemm/gen/4x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/5x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld128.c
  src/f32-gemm/gen-inc/1x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-lane-ld128.c
  src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/5x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-lane-ld64.c
  src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-lane-ld128.c
  src/f32-igemm/gen/1x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x2-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x4-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/4x8-minmax-neonfma-lane-ld128.c
  src/f32-igemm/gen/4x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-lane-ld64.c
  src/f32-igemm/gen/6x8-minmax-neonfma-lane-ld128.c
  src/f32-conv-hwc/3x3s2p1c3x4-neonfma-2x2.c
  src/f32-conv-hwc/3x3s2p1c3x8-neonfma-2x2.c
  src/f32-conv-hwc2spchw/3x3s2p1c3x4-neonfma-2x2.c
  src/f32-dwconv-spchw/3x3p1-neonfma.c
  src/f32-dwconv-spchw/5x5p2-neonfma.c
  src/f32-dwconv-spchw/3x3s2p1-neonfma.c
  src/f32-dwconv-spchw/5x5s2p2-neonfma.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-div-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-div-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-div-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-div-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-div-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-p5-div-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-div-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-div-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-div-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-div-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-div-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut64-p2-div-x24.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-div-x4.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-div-x8.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-div-x12.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-div-x16.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-div-x20.c
  src/f32-sigmoid/gen/neonfma-rr1-lut2048-p1-div-x24.c
  src/f32-spmm/gen/12x1-minmax-neonfma.c
  src/f32-spmm/gen/12x2-minmax-neonfma.c
  src/f32-spmm/gen/12x4-minmax-neonfma.c
  src/f32-spmm/gen/16x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/16x1-minmax-neonfma-unroll2.c
  src/f32-spmm/gen/16x1-minmax-neonfma.c
  src/f32-spmm/gen/16x2-minmax-neonfma.c
  src/f32-spmm/gen/16x4-minmax-neonfma.c
  src/f32-spmm/gen/4x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/4x1-minmax-neonfma-unroll2.c
  src/f32-spmm/gen/4x1-minmax-neonfma.c
  src/f32-spmm/gen/4x2-minmax-neonfma.c
  src/f32-spmm/gen/4x4-minmax-neonfma.c
  src/f32-spmm/gen/8x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/8x1-minmax-neonfma-unroll2.c
  src/f32-spmm/gen/8x1-minmax-neonfma.c
  src/f32-spmm/gen/8x2-minmax-neonfma.c
  src/f32-spmm/gen/8x4-minmax-neonfma.c
  src/math/sigmoid-neonfma-rr1-lut2048-p1-div.c
  src/math/sigmoid-neonfma-rr1-lut64-p2-div.c
  src/math/sigmoid-neonfma-rr1-p5-div.c
  src/math/sigmoid-neonfma-rr2-lut2048-p1-div.c
  src/math/sigmoid-neonfma-rr2-lut64-p2-div.c
  src/math/sigmoid-neonfma-rr2-p5-div.c)

SET(XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS
  src/f16-gemm/gen/1x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/4x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/6x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen/8x8-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/1x8inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/4x8inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/6x8inc-minmax-neonfp16arith-ld64.c
  src/f16-gemm/gen-inc/8x8inc-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/1x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/4x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/6x8-minmax-neonfp16arith-ld64.c
  src/f16-igemm/gen/8x8-minmax-neonfp16arith-ld64.c
  src/f16-spmm/gen/8x1-minmax-neonfp16arith.c
  src/f16-spmm/gen/8x1-minmax-neonfp16arith-unroll2.c
  src/f16-spmm/gen/16x1-minmax-neonfp16arith.c
  src/f16-spmm/gen/16x1-minmax-neonfp16arith-unroll2.c
  src/f16-spmm/gen/24x1-minmax-neonfp16arith.c
  src/f16-spmm/gen/24x1-minmax-neonfp16arith-unroll2.c
  src/f16-spmm/gen/32x1-minmax-neonfp16arith.c
  src/f16-spmm/gen/32x1-minmax-neonfp16arith-unroll2.c)

SET(XNNPACK_SSE_MICROKERNEL_SRCS
  src/f32-avgpool/9p8x-minmax-sse-c4.c
  src/f32-avgpool/9x-minmax-sse-c4.c
  src/f32-clamp/gen/sse-x4.c
  src/f32-clamp/gen/sse-x8.c
  src/f32-dwconv-spchw/3x3p1-sse.c
  src/f32-dwconv-spchw/3x3s2p1-sse.c
  src/f32-dwconv/gen/up4x25-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x25-minmax-sse.c
  src/f32-dwconv/gen/up4x4-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x4-minmax-sse.c
  src/f32-dwconv/gen/up4x9-minmax-sse-acc2.c
  src/f32-dwconv/gen/up4x9-minmax-sse.c
  src/f32-dwconv/gen/up8x25-minmax-sse-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-sse.c
  src/f32-dwconv/gen/up8x4-minmax-sse-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-sse.c
  src/f32-dwconv/gen/up8x9-minmax-sse-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-sse.c
  src/f32-gavgpool-spchw/sse-x4.c
  src/f32-gavgpool/7p7x-minmax-sse-c4.c
  src/f32-gavgpool/7x-minmax-sse-c4.c
  src/f32-gemm/gen/1x8-minmax-sse-dup.c
  src/f32-gemm/gen/1x8-minmax-sse-load1.c
  src/f32-gemm/gen/1x8s4-minmax-sse.c
  src/f32-gemm/gen/4x2c4-minmax-sse.c
  src/f32-gemm/gen/4x8-minmax-sse-dup.c
  src/f32-gemm/gen/4x8-minmax-sse-load1.c
  src/f32-gemm/gen/4x8s4-minmax-sse.c
  src/f32-gemm/gen-inc/1x8inc-minmax-sse-dup.c
  src/f32-gemm/gen-inc/1x8inc-minmax-sse-load1.c
  src/f32-gemm/gen-inc/1x8s4inc-minmax-sse.c
  src/f32-gemm/gen-inc/4x8inc-minmax-sse-dup.c
  src/f32-gemm/gen-inc/4x8inc-minmax-sse-load1.c
  src/f32-gemm/gen-inc/4x8s4inc-minmax-sse.c
  src/f32-hswish/gen/sse-x4.c
  src/f32-hswish/gen/sse-x8.c
  src/f32-ibilinear/gen/sse-c4.c
  src/f32-ibilinear/gen/sse-c8.c
  src/f32-igemm/gen/1x8-minmax-sse-dup.c
  src/f32-igemm/gen/1x8-minmax-sse-load1.c
  src/f32-igemm/gen/1x8s4-minmax-sse.c
  src/f32-igemm/gen/4x2c4-minmax-sse.c
  src/f32-igemm/gen/4x8-minmax-sse-dup.c
  src/f32-igemm/gen/4x8-minmax-sse-load1.c
  src/f32-igemm/gen/4x8s4-minmax-sse.c
  src/f32-maxpool/9p8x-minmax-sse-c4.c
  src/f32-pavgpool/9p8x-minmax-sse-c4.c
  src/f32-pavgpool/9x-minmax-sse-c4.c
  src/f32-ppmm/gen/4x8-minmax-sse.c
  src/f32-rmax/sse.c
  src/f32-spmm/gen/4x1-minmax-sse.c
  src/f32-spmm/gen/8x1-minmax-sse.c
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
  src/f32-vbinary/gen/vsub-minmax-sse-x4.c
  src/f32-vbinary/gen/vsub-minmax-sse-x8.c
  src/f32-vbinary/gen/vsubc-minmax-sse-x4.c
  src/f32-vbinary/gen/vsubc-minmax-sse-x8.c
  src/f32-vmulcaddc/gen/c4-minmax-sse-2x.c
  src/f32-vmulcaddc/gen/c8-minmax-sse-2x.c
  src/x32-packx/x4-sse.c)

SET(XNNPACK_SSE2_MICROKERNEL_SRCS
  src/f32-argmaxpool/4x-sse2-c4.c
  src/f32-argmaxpool/9p8x-sse2-c4.c
  src/f32-argmaxpool/9x-sse2-c4.c
  src/f32-prelu/gen/sse2-2x4.c
  src/f32-prelu/gen/sse2-2x8.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x4.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x8.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x8-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x12.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x12-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x12-acc3.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x16.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x16-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x16-acc4.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x20.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x20-acc2.c
  src/f32-raddstoreexpminusmax/gen/sse2-p5-x20-acc5.c
  src/f32-sigmoid/gen/sse2-p5-div-x4.c
  src/f32-sigmoid/gen/sse2-p5-div-x8.c
  src/f32-sigmoid/gen/sse2-p5-div-x12.c
  src/f32-sigmoid/gen/sse2-p5-div-x16.c
  src/f32-sigmoid/gen/sse2-p5-div-x20.c
  src/f32-sigmoid/gen/sse2-p5-div-x24.c
  src/q8-avgpool/9p8x-minmax-sse2-c8.c
  src/q8-avgpool/9x-minmax-sse2-c8.c
  src/q8-igemm/4x4c2-minmax-sse2.c
  src/q8-dwconv/up8x9-minmax-sse2.c
  src/q8-gavgpool/7p7x-minmax-sse2-c8.c
  src/q8-gavgpool/7x-minmax-sse2-c8.c
  src/q8-gemm/2x4c8-minmax-sse2.c
  src/q8-gemm/4x4c2-minmax-sse2.c
  src/q8-vadd/minmax-sse2.c
  src/u8-clamp/sse2-x64.c
  src/u8-maxpool/9p8x-minmax-sse2-c16.c
  src/u8-rmax/sse2.c
  src/x32-pad/x2-sse2.c
  src/x32-unpool/sse2.c
  src/x32-zip/x2-sse2.c
  src/x32-zip/x3-sse2.c
  src/x32-zip/x4-sse2.c
  src/x32-zip/xm-sse2.c
  src/x8-zip/x2-sse2.c
  src/x8-zip/x3-sse2.c
  src/x8-zip/x4-sse2.c
  src/x8-zip/xm-sse2.c
  src/math/exp-sse2-p5.c
  src/math/expminus-sse2-p5.c
  src/math/sigmoid-sse2-p5-div.c
  src/requantization/precise-sse2.c
  src/requantization/fp32-sse2.c
  src/requantization/q31-sse2.c
  src/requantization/gemmlowp-sse2.c)

SET(XNNPACK_SSSE3_MICROKERNEL_SRCS
  src/requantization/precise-ssse3.c
  src/requantization/q31-ssse3.c
  src/requantization/gemmlowp-ssse3.c)

SET(XNNPACK_SSE41_MICROKERNEL_SRCS
  src/f32-prelu/gen/sse41-2x4.c
  src/f32-prelu/gen/sse41-2x8.c
  src/f32-sigmoid/gen/sse41-p5-div-x4.c
  src/f32-sigmoid/gen/sse41-p5-div-x8.c
  src/f32-sigmoid/gen/sse41-p5-div-x12.c
  src/f32-sigmoid/gen/sse41-p5-div-x16.c
  src/f32-sigmoid/gen/sse41-p5-div-x20.c
  src/f32-sigmoid/gen/sse41-p5-div-x24.c
  src/requantization/precise-sse4.c
  src/requantization/q31-sse4.c
  src/requantization/gemmlowp-sse4.c)

SET(XNNPACK_AVX_MICROKERNEL_SRCS
  src/f32-clamp/gen/avx-x8.c
  src/f32-clamp/gen/avx-x16.c
  src/f32-dwconv/gen/up16x4-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-avx.c
  src/f32-dwconv/gen/up8x4-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-avx.c
  src/f32-dwconv/gen/up16x9-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-avx.c
  src/f32-dwconv/gen/up8x9-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-avx.c
  src/f32-dwconv/gen/up16x25-minmax-avx-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-avx.c
  src/f32-dwconv/gen/up8x25-minmax-avx-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-avx.c
  src/f32-gemm/gen/1x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/4x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/5x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/6x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/7x8-minmax-avx-broadcast.c
  src/f32-gemm/gen/1x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/3x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/4x16-minmax-avx-broadcast.c
  src/f32-gemm/gen/5x16-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/1x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/4x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/5x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/6x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/7x8inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/1x16inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/3x16inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/4x16inc-minmax-avx-broadcast.c
  src/f32-gemm/gen-inc/5x16inc-minmax-avx-broadcast.c
  src/f32-hswish/gen/avx-x8.c
  src/f32-hswish/gen/avx-x16.c
  src/f32-igemm/gen/1x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/4x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/5x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/6x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/7x8-minmax-avx-broadcast.c
  src/f32-igemm/gen/1x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/3x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/4x16-minmax-avx-broadcast.c
  src/f32-igemm/gen/5x16-minmax-avx-broadcast.c
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
  src/f32-vbinary/gen/vsub-minmax-avx-x8.c
  src/f32-vbinary/gen/vsub-minmax-avx-x16.c
  src/f32-vbinary/gen/vsubc-minmax-avx-x8.c
  src/f32-vbinary/gen/vsubc-minmax-avx-x16.c
  src/f32-vscale/avx-unroll32.c)

SET(XNNPACK_FMA3_MICROKERNEL_SRCS
  src/f32-dwconv/gen/up16x4-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-fma3.c
  src/f32-dwconv/gen/up8x4-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x4-minmax-fma3.c
  src/f32-dwconv/gen/up16x9-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-fma3.c
  src/f32-dwconv/gen/up8x9-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x9-minmax-fma3.c
  src/f32-dwconv/gen/up16x25-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-fma3.c
  src/f32-dwconv/gen/up8x25-minmax-fma3-acc2.c
  src/f32-dwconv/gen/up8x25-minmax-fma3.c
  src/f32-gemm/gen/1x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/4x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/5x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/6x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/7x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/8x8-minmax-fma3-broadcast.c
  src/f32-gemm/gen/1x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/3x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/4x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/5x16-minmax-fma3-broadcast.c
  src/f32-gemm/gen/1x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/3x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/4x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen/5x16s4-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/1x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/4x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/5x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/6x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/7x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/8x8inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/1x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/3x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/4x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/5x16inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/1x16s4inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/3x16s4inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/4x16s4inc-minmax-fma3-broadcast.c
  src/f32-gemm/gen-inc/5x16s4inc-minmax-fma3-broadcast.c
  src/f32-hswish/gen/fma3-x8.c
  src/f32-hswish/gen/fma3-x16.c
  src/f32-igemm/gen/1x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/4x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/5x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/6x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/7x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/8x8-minmax-fma3-broadcast.c
  src/f32-igemm/gen/1x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/3x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/4x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/5x16-minmax-fma3-broadcast.c
  src/f32-igemm/gen/1x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/3x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/4x16s4-minmax-fma3-broadcast.c
  src/f32-igemm/gen/5x16s4-minmax-fma3-broadcast.c)

SET(XNNPACK_AVX2_MICROKERNEL_SRCS
  src/f32-raddexpminusmax/gen/avx2-p5-x64.c
  src/f32-raddexpminusmax/gen/avx2-p5-x64-acc2.c
  src/f32-raddexpminusmax/gen/avx2-p5-x64-acc4.c
  src/f32-raddexpminusmax/gen/avx2-p5-x72.c
  src/f32-raddexpminusmax/gen/avx2-p5-x72-acc3.c
  src/f32-raddexpminusmax/gen/avx2-p5-x80.c
  src/f32-raddexpminusmax/gen/avx2-p5-x80-acc2.c
  src/f32-raddexpminusmax/gen/avx2-p5-x80-acc5.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96-acc2.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96-acc3.c
  src/f32-raddexpminusmax/gen/avx2-p5-x96-acc6.c
  src/f32-raddextexp/gen/avx2-p5-x64.c
  src/f32-raddextexp/gen/avx2-p5-x64-acc2.c
  src/f32-raddextexp/gen/avx2-p5-x64-acc4.c
  src/f32-raddextexp/gen/avx2-p5-x72.c
  src/f32-raddextexp/gen/avx2-p5-x72-acc3.c
  src/f32-raddextexp/gen/avx2-p5-x80.c
  src/f32-raddextexp/gen/avx2-p5-x80-acc2.c
  src/f32-raddextexp/gen/avx2-p5-x80-acc5.c
  src/f32-raddextexp/gen/avx2-p5-x96.c
  src/f32-raddextexp/gen/avx2-p5-x96-acc2.c
  src/f32-raddextexp/gen/avx2-p5-x96-acc3.c
  src/f32-raddextexp/gen/avx2-p5-x96-acc6.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x64.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x64-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x64-acc4.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x72.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x72-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x80.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x80-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x80-acc5.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc6.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x8.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x16.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x24.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x32.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x40.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x48.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x56.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x64.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x72.c
  src/f32-sigmoid/gen/avx2-rr1-p5-div-x80.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x8.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x16.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x24.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x32.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x40.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x48.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x56.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x64.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x72.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr1fma-x80.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x8.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x16.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x24.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x32.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x40.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x48.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x56.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x64.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x72.c
  src/f32-sigmoid/gen/avx2-rr1-p5-nr2fma-x80.c
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
  src/math/exp-avx2-p5.c
  src/math/exp-avx2-perm-p3.c
  src/math/exp-avx2-perm-p4.c
  src/math/expminus-avx2-p5.c
  src/math/extexp-avx2-p5.c
  src/math/sigmoid-avx2-rr2-p5-div.c
  src/math/sigmoid-avx2-rr1-p5-div.c
  src/math/sigmoid-avx2-rr2-p5-nr2fma.c
  src/math/sigmoid-avx2-rr1-p5-nr2fma.c
  src/math/sigmoid-avx2-rr2-p5-nr1fma.c
  src/math/sigmoid-avx2-rr1-p5-nr1fma.c)

SET(XNNPACK_AVX512F_MICROKERNEL_SRCS
  src/f32-clamp/gen/avx512f-x16.c
  src/f32-clamp/gen/avx512f-x32.c
  src/f32-dwconv/gen/up32x4-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up32x4-minmax-avx512f.c
  src/f32-dwconv/gen/up16x4-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x4-minmax-avx512f.c
  src/f32-dwconv/gen/up32x9-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up32x9-minmax-avx512f.c
  src/f32-dwconv/gen/up16x9-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x9-minmax-avx512f.c
  src/f32-dwconv/gen/up32x25-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up32x25-minmax-avx512f.c
  src/f32-dwconv/gen/up16x25-minmax-avx512f-acc2.c
  src/f32-dwconv/gen/up16x25-minmax-avx512f.c
  src/f32-gemm/gen/1x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/4x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/5x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/6x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/7x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen/8x16-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/1x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/4x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/5x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/6x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/7x16inc-minmax-avx512f-broadcast.c
  src/f32-gemm/gen-inc/8x16inc-minmax-avx512f-broadcast.c
  src/f32-hswish/gen/avx512f-x16.c
  src/f32-hswish/gen/avx512f-x32.c
  src/f32-igemm/gen/1x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/4x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/5x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/6x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/7x16-minmax-avx512f-broadcast.c
  src/f32-igemm/gen/8x16-minmax-avx512f-broadcast.c
  src/f32-prelu/gen/avx512f-2x16.c
  src/f32-prelu/gen/avx512f-2x32.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x128.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x128-acc2.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x128-acc4.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x144.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x144-acc3.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x160.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x160-acc2.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x160-acc5.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192-acc2.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192-acc3.c
  src/f32-raddexpminusmax/gen/avx512f-p5-scalef-x192-acc6.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x128.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x128-acc2.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x128-acc4.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x144.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x144-acc3.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x160.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x160-acc2.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x160-acc5.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc2.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc3.c
  src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc6.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128-acc4.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x144.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x144-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160-acc5.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc2.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc3.c
  src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc6.c
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
  src/f32-vbinary/gen/vsub-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vsub-minmax-avx512f-x32.c
  src/f32-vbinary/gen/vsubc-minmax-avx512f-x16.c
  src/f32-vbinary/gen/vsubc-minmax-avx512f-x32.c
  src/f32-vscale/avx512f-unroll64.c
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
  src/math/exp-avx512f-p5-scalef.c
  src/math/exp-avx512f-p5.c
  src/math/exp-avx512f-perm-p3.c
  src/math/exp-avx512f-perm2-p2.c
  src/math/extexp-avx512f-p5.c)

SET(XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS
  src/q8-dwconv/up8x9-minmax-aarch32-neon.S
  src/f32-gemm/4x8-minmax-aarch32-neon-cortex-a53.S
  src/f32-gemm/4x8-minmax-aarch32-neon-cortex-a55.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a75.S
  src/f32-gemm/gen/4x8-minmax-aarch32-neon-pld-cortex-a75.S
  src/f32-gemm/4x8-minmax-aarch32-neon-ld64.S
  src/f32-igemm/4x8-minmax-aarch32-neon-ld64.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-cortex-a75.S
  src/f32-igemm/gen/4x8-minmax-aarch32-neon-pld-cortex-a75.S
  src/f32-igemm/4x8-minmax-aarch32-neon-cortex-a53.S
  src/f32-igemm/4x8-minmax-aarch32-neon-cortex-a55.S)

SET(XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS
  src/f16-gemm/gen/1x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/4x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/6x16-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen-inc/1x16inc-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen-inc/4x16inc-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen-inc/6x16inc-minmax-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/1x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/4x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/6x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/8x8-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen-inc/1x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen-inc/4x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen-inc/6x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen-inc/8x8inc-minmax-aarch64-neonfp16arith-ld64.S
  src/f32-dwconv/up4x9-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-dwconv/up4x9-minmax-aarch64-neonfma.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/1x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/4x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/5x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen/5x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a73.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ios.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen-inc/1x12inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/4x12inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-ld64.S
  src/f32-gemm/gen-inc/5x8inc-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen-inc/5x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a73.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-ios.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-ld128.S
  src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-ld64.S
  src/f32-igemm/1x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/1x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/4x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/4x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/4x12-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-igemm/gen/4x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/5x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-igemm/gen/5x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/6x8-minmax-aarch64-neonfma-cortex-a73.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-cortex-a57.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-ios.S)

SET(XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SCALAR_MICROKERNEL_SRCS})
IF(NOT EMSCRIPTEN AND NOT MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Intel")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_PSIMD_FASTMATH_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_PSIMD_ACCMATH_MICROKERNEL_SRCS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEON_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR IOS_ARCH MATCHES "^arm64.*")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEON_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64|AMD64)$" OR IOS_ARCH MATCHES "^(i386|x86_64|AMD64)$")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE2_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSSE3_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE41_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_FMA3_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX2_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX512F_MICROKERNEL_SRCS})
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
  SET_PROPERTY(SOURCE ${XNNPACK_PSIMD_FASTMATH_MICROKERNEL_SRCS} ${XNNPACK_PSIMD_ACCMATH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon ")
  SET_PROPERTY(SOURCE ${XNNPACK_NEON_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon ")
  SET_PROPERTY(SOURCE ${XNNPACK_NEONFMA_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon-vfpv4 ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR IOS_ARCH MATCHES "^arm64.*")
  SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86|x86_64|AMD64)$" OR IOS_ARCH MATCHES "^(i386|x86_64|AMD64)$")
  IF(MSVC)
    IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86" OR CMAKE_SIZEOF_VOID_P EQUAL 4)
      SET_PROPERTY(SOURCE ${XNNPACK_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE ")
      SET_PROPERTY(SOURCE ${XNNPACK_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_SSSE3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
      SET_PROPERTY(SOURCE ${XNNPACK_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:SSE2 ")
    ENDIF()
    SET_PROPERTY(SOURCE ${XNNPACK_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${XNNPACK_FMA3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX512 ")
  ELSE()
    SET_PROPERTY(SOURCE ${XNNPACK_PSIMD_ACCMATH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_PSIMD_FASTMATH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse ")
    SET_PROPERTY(SOURCE ${XNNPACK_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_SSSE3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mssse3 ")
    SET_PROPERTY(SOURCE ${XNNPACK_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse4.1 ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx ")
    SET_PROPERTY(SOURCE ${XNNPACK_FMA3_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfma ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfma -mavx2 ")
    SET_PROPERTY(SOURCE ${XNNPACK_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx512f ")
    IF(MINGW OR CMAKE_SYSTEM_NAME MATCHES "^(CYGWIN|MSYS)$")
      # Work-around for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782
      SET_PROPERTY(SOURCE ${XNNPACK_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -fno-asynchronous-unwind-tables ")
    ENDIF()
  ENDIF()
ENDIF()
TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE "XNN_LOG_LEVEL=$<IF:$<CONFIG:Debug>,5,0>")
IF(MSVC)
  # Even though MSVC has __restrict, it can't be used in all the same contexts as the C99 restrict keyword
  TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE "restrict=")
  SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O2 >")
  SET_PROPERTY(SOURCE ${XNNPACK_HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O2 >")
  SET_PROPERTY(SOURCE ${XNNPACK_COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: /O1 >")
ELSE()
  SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O2 >")
  SET_PROPERTY(SOURCE ${XNNPACK_HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O2 >")
  SET_PROPERTY(SOURCE ${XNNPACK_COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -Os >")
ENDIF()
IF(NOT EMSCRIPTEN)
  SET_PROPERTY(SOURCE ${XNNPACK_PSIMD_ACCMATH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O3 >")
  SET_PROPERTY(SOURCE ${XNNPACK_PSIMD_FASTMATH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "$<$<NOT:$<CONFIG:Debug>>: -O3 -ffast-math >")
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

# ---[ Configure psimd
IF(NOT TARGET psimd)
  IF(NOT XNNPACK_USE_SYSTEM_LIBS)
    ADD_SUBDIRECTORY(
      "${PSIMD_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/psimd")
  ELSE()
    FIND_FILE(PSIMD_HDR psimd.h PATH_SUFFIXES include)
    IF(NOT PSIMD_HDR)
      MESSAGE(FATAL_ERROR "Cannot find psimd")
    ENDIF()
    ADD_LIBRARY(psimd STATIC "${PSIMD_HDR}")
    SET_PROPERTY(TARGET psimd PROPERTY LINKER_LANGUAGE C)
  ENDIF()
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE psimd)

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
  ADD_EXECUTABLE(add-nc-test test/add-nc.cc)
  SET_TARGET_PROPERTIES(add-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(add-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(add-nc-test add-nc-test)

  ADD_EXECUTABLE(add-nd-test test/add-nd.cc)
  SET_TARGET_PROPERTIES(add-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(add-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add-nd-test PRIVATE XNNPACK gtest gtest_main)
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

  ADD_EXECUTABLE(channel-pad-nc-test test/channel-pad-nc.cc)
  SET_TARGET_PROPERTIES(channel-pad-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(channel-pad-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(channel-pad-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(channel-pad-nc-test channel-pad-nc-test)

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

  ADD_EXECUTABLE(convolution-nhwc-test test/convolution-nhwc.cc)
  SET_TARGET_PROPERTIES(convolution-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(convolution-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(convolution-nhwc-test convolution-nhwc-test)

  ADD_EXECUTABLE(convolution-nchw-test test/convolution-nchw.cc)
  SET_TARGET_PROPERTIES(convolution-nchw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(convolution-nchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-nchw-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(convolution-nchw-test convolution-nchw-test)

  ADD_EXECUTABLE(deconvolution-nhwc-test test/deconvolution-nhwc.cc)
  SET_TARGET_PROPERTIES(deconvolution-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(deconvolution-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(deconvolution-nhwc-test deconvolution-nhwc-test)

  ADD_EXECUTABLE(divide-nd-test test/divide-nd.cc)
  SET_TARGET_PROPERTIES(divide-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(divide-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(divide-nd-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(divide-nd-test divide-nd-test)

  ADD_EXECUTABLE(fully-connected-nc-test test/fully-connected-nc.cc)
  SET_TARGET_PROPERTIES(fully-connected-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(fully-connected-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(fully-connected-nc-test fully-connected-nc-test)

  ADD_EXECUTABLE(global-average-pooling-nwc-test test/global-average-pooling-nwc.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-nwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-nwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-nwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(global-average-pooling-nwc-test global-average-pooling-nwc-test)

  ADD_EXECUTABLE(global-average-pooling-ncw-test test/global-average-pooling-ncw.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-ncw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-ncw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-ncw-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(global-average-pooling-ncw-test global-average-pooling-ncw-test)

  ADD_EXECUTABLE(hardswish-nc-test test/hardswish-nc.cc)
  SET_TARGET_PROPERTIES(hardswish-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(hardswish-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(hardswish-nc-test PRIVATE XNNPACK gtest gtest_main)
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

  ADD_EXECUTABLE(maximum-nd-test test/multiply-nd.cc)
  SET_TARGET_PROPERTIES(maximum-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(maximum-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(maximum-nd-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(maximum-nd-test maximum-nd-test)

  ADD_EXECUTABLE(minimum-nd-test test/multiply-nd.cc)
  SET_TARGET_PROPERTIES(minimum-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(minimum-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(minimum-nd-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(minimum-nd-test minimum-nd-test)

  ADD_EXECUTABLE(multiply-nd-test test/multiply-nd.cc)
  SET_TARGET_PROPERTIES(multiply-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(multiply-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(multiply-nd-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(multiply-nd-test multiply-nd-test)

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

  ADD_EXECUTABLE(subtract-nd-test test/subtract-nd.cc)
  SET_TARGET_PROPERTIES(subtract-nd-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(subtract-nd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(subtract-nd-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(subtract-nd-test subtract-nd-test)

  ADD_EXECUTABLE(unpooling-nhwc-test test/unpooling-nhwc.cc)
  SET_TARGET_PROPERTIES(unpooling-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(unpooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(unpooling-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(unpooling-nhwc-test unpooling-nhwc-test)

  # ---[ Build microkernel-level unit tests
  ADD_EXECUTABLE(f16-gemm-minmax-test test/f16-gemm-minmax.cc)
  SET_TARGET_PROPERTIES(f16-gemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-gemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-gemm-minmax-test f16-gemm-minmax-test)

  ADD_EXECUTABLE(f16-spmm-minmax-test test/f16-spmm-minmax.cc)
  SET_TARGET_PROPERTIES(f16-spmm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-spmm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-spmm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-spmm-minmax-test f16-spmm-minmax-test)

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

  ADD_EXECUTABLE(f32-clamp-test test/f32-clamp.cc)
  SET_TARGET_PROPERTIES(f32-clamp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-clamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-clamp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-clamp-test f32-clamp-test)

  ADD_EXECUTABLE(f32-conv-hwc-test test/f32-conv-hwc.cc)
  SET_TARGET_PROPERTIES(f32-conv-hwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-conv-hwc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-conv-hwc-test f32-conv-hwc-test)

  ADD_EXECUTABLE(f32-conv-hwc2spchw-test test/f32-conv-hwc2spchw.cc)
  SET_TARGET_PROPERTIES(f32-conv-hwc2spchw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc2spchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-conv-hwc2spchw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-conv-hwc2spchw-test f32-conv-hwc2spchw-test)

  ADD_EXECUTABLE(f32-dwconv-spchw-test test/f32-dwconv-spchw.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-spchw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-spchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-dwconv-spchw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-dwconv-spchw-test f32-dwconv-spchw-test)

  ADD_EXECUTABLE(f32-dwconv-test test/f32-dwconv.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-dwconv-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-dwconv-test f32-dwconv-test)

  ADD_EXECUTABLE(f32-dwconv-minmax-test test/f32-dwconv-minmax.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-dwconv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-dwconv-minmax-test f32-dwconv-minmax-test)

  ADD_EXECUTABLE(f32-gavgpool-spchw-test test/f32-gavgpool-spchw.cc)
  SET_TARGET_PROPERTIES(f32-gavgpool-spchw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gavgpool-spchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gavgpool-spchw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gavgpool-spchw-test f32-gavgpool-spchw-test)

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

  ADD_EXECUTABLE(f32-hswish-test test/f32-hswish.cc)
  SET_TARGET_PROPERTIES(f32-hswish-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-hswish-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-hswish-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-hswish-test f32-hswish-test)

  ADD_EXECUTABLE(f32-ibilinear-test test/f32-ibilinear.cc)
  SET_TARGET_PROPERTIES(f32-ibilinear-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-ibilinear-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-ibilinear-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-ibilinear-test f32-ibilinear-test)

  ADD_EXECUTABLE(f32-igemm-test test/f32-igemm.cc)
  SET_TARGET_PROPERTIES(f32-igemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-igemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-igemm-test f32-igemm-test)

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

  ADD_EXECUTABLE(f16-igemm-minmax-test test/f16-igemm-minmax.cc)
  SET_TARGET_PROPERTIES(f16-igemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-igemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-igemm-minmax-test f16-igemm-minmax-test)

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

  ADD_EXECUTABLE(f32-sigmoid-test test/f32-sigmoid.cc)
  SET_TARGET_PROPERTIES(f32-sigmoid-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-sigmoid-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-sigmoid-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-sigmoid-test f32-sigmoid-test)

  ADD_EXECUTABLE(f32-spmm-minmax-test test/f32-spmm-minmax.cc)
  SET_TARGET_PROPERTIES(f32-spmm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-spmm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-spmm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-spmm-minmax-test f32-spmm-minmax-test)

  ADD_EXECUTABLE(f32-vadd-minmax-test test/f32-vadd-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vadd-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vadd-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vadd-minmax-test f32-vadd-minmax-test)

  ADD_EXECUTABLE(f32-vaddc-minmax-test test/f32-vaddc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vaddc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vaddc-minmax-test f32-vaddc-minmax-test)

  ADD_EXECUTABLE(f32-vdiv-minmax-test test/f32-vdiv-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vdiv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdiv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdiv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdiv-minmax-test f32-vdiv-minmax-test)

  ADD_EXECUTABLE(f32-vdivc-minmax-test test/f32-vdivc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vdivc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vdivc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vdivc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vdivc-minmax-test f32-vdivc-minmax-test)

  ADD_EXECUTABLE(f32-vrdivc-minmax-test test/f32-vrdivc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vrdivc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrdivc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrdivc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrdivc-minmax-test f32-vrdivc-minmax-test)

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

  ADD_EXECUTABLE(f32-vmul-minmax-test test/f32-vmul-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vmul-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmul-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmul-minmax-test f32-vmul-minmax-test)

  ADD_EXECUTABLE(f32-vmulc-minmax-test test/f32-vmulc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vmulc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulc-minmax-test f32-vmulc-minmax-test)

  ADD_EXECUTABLE(f32-vmulcaddc-minmax-test test/f32-vmulcaddc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vmulcaddc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulcaddc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulcaddc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulcaddc-minmax-test f32-vmulcaddc-minmax-test)

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

  ADD_EXECUTABLE(f32-vsub-minmax-test test/f32-vsub-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vsub-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsub-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsub-minmax-test f32-vsub-minmax-test)

  ADD_EXECUTABLE(f32-vsubc-minmax-test test/f32-vsubc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vsubc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsubc-minmax-test f32-vsubc-minmax-test)

  ADD_EXECUTABLE(f32-vrsubc-minmax-test test/f32-vrsubc-minmax.cc)
  SET_TARGET_PROPERTIES(f32-vrsubc-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrsubc-minmax-test f32-vrsubc-minmax-test)

  ADD_EXECUTABLE(q8-avgpool-minmax-test test/q8-avgpool-minmax.cc)
  SET_TARGET_PROPERTIES(q8-avgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-avgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-avgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-avgpool-minmax-test q8-avgpool-minmax-test)

  ADD_EXECUTABLE(q8-dwconv-minmax-test test/q8-dwconv-minmax.cc)
  SET_TARGET_PROPERTIES(q8-dwconv-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-dwconv-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-dwconv-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-dwconv-minmax-test q8-dwconv-minmax-test)

  ADD_EXECUTABLE(q8-gavgpool-minmax-test test/q8-gavgpool-minmax.cc)
  SET_TARGET_PROPERTIES(q8-gavgpool-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-gavgpool-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-gavgpool-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-gavgpool-test q8-gavgpool-minmax-test)

  ADD_EXECUTABLE(q8-gemm-minmax-test test/q8-gemm-minmax.cc)
  SET_TARGET_PROPERTIES(q8-gemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-gemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-gemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-gemm-minmax-test q8-gemm-minmax-test)

  ADD_EXECUTABLE(q8-igemm-minmax-test test/q8-igemm-minmax.cc)
  SET_TARGET_PROPERTIES(q8-igemm-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-igemm-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-igemm-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-igemm-minmax-test q8-igemm-minmax-test)

  ADD_EXECUTABLE(q8-vadd-minmax-test test/q8-vadd-minmax.cc)
  SET_TARGET_PROPERTIES(q8-vadd-minmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-vadd-minmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-vadd-minmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-vadd-minmax-test q8-vadd-minmax-test)

  ADD_EXECUTABLE(u8-clamp-test test/u8-clamp.cc)
  SET_TARGET_PROPERTIES(u8-clamp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-clamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-clamp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-clamp-test u8-clamp-test)

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

  ADD_EXECUTABLE(requantization-test test/requantization.cc)
  SET_TARGET_PROPERTIES(requantization-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(requantization-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(requantization-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(requantization-test requantization-test)
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
  ADD_EXECUTABLE(f32-exp-eval eval/f32-exp.cc)
  SET_TARGET_PROPERTIES(f32-exp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-exp-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-exp-eval PRIVATE XNNPACK benchmark fp16)

  ADD_EXECUTABLE(f32-expminus-eval eval/f32-expminus.cc)
  SET_TARGET_PROPERTIES(f32-expminus-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-expminus-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-expminus-eval PRIVATE XNNPACK benchmark fp16)

  ADD_EXECUTABLE(f32-extexp-eval eval/f32-extexp.cc)
  SET_TARGET_PROPERTIES(f32-extexp-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-extexp-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-extexp-eval PRIVATE XNNPACK benchmark fp16)

  ADD_EXECUTABLE(f32-sigmoid-eval eval/f32-sigmoid.cc)
  SET_TARGET_PROPERTIES(f32-sigmoid-eval PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-sigmoid-eval PRIVATE src)
  TARGET_LINK_LIBRARIES(f32-sigmoid-eval PRIVATE XNNPACK benchmark fp16)

  # ---[ Build end-to-end microbenchmarks
  ADD_LIBRARY(bench-models STATIC
    models/mobilenet-v1.cc
    models/mobilenet-v2.cc
    models/mobilenet-v3-large.cc
    models/mobilenet-v3-small.cc)
  SET_TARGET_PROPERTIES(bench-models PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(bench-models PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(bench-models PRIVATE XNNPACK benchmark bench-utils)

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
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-e2e-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" src)
  TARGET_LINK_LIBRARIES(f32-gemm-e2e-bench PRIVATE XNNPACK benchmark bench-models bench-utils)

  # ---[ Build operator-level microbenchmarks
  ADD_EXECUTABLE(add-bench bench/add.cc)
  SET_TARGET_PROPERTIES(add-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(add-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(add-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(average-pooling-bench bench/average-pooling.cc)
  SET_TARGET_PROPERTIES(average-pooling-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(average-pooling-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(average-pooling-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(channel-shuffle-bench bench/channel-shuffle.cc)
  SET_TARGET_PROPERTIES(channel-shuffle-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(channel-shuffle-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(channel-shuffle-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(convolution-bench bench/convolution.cc)
  SET_TARGET_PROPERTIES(convolution-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(convolution-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(convolution-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(deconvolution-bench bench/deconvolution.cc)
  SET_TARGET_PROPERTIES(deconvolution-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(deconvolution-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(global-average-pooling-bench bench/global-average-pooling.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(global-average-pooling-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(max-pooling-bench bench/max-pooling.cc)
  SET_TARGET_PROPERTIES(max-pooling-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(max-pooling-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(prelu-bench bench/prelu.cc)
  SET_TARGET_PROPERTIES(prelu-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(prelu-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(prelu-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(sigmoid-bench bench/sigmoid.cc)
  SET_TARGET_PROPERTIES(sigmoid-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(sigmoid-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(sigmoid-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  ADD_EXECUTABLE(softmax-bench bench/softmax.cc)
  SET_TARGET_PROPERTIES(softmax-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(softmax-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(softmax-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  # ---[ Build microkernel-level microbenchmarks
  ADD_EXECUTABLE(f16-igemm-bench bench/f16-igemm.cc)
  SET_TARGET_PROPERTIES(f16-igemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f16-igemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f16-igemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f16-gemm-bench bench/f16-gemm.cc)
  SET_TARGET_PROPERTIES(f16-gemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f16-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-conv-hwc-bench bench/f32-conv-hwc.cc)
  SET_TARGET_PROPERTIES(f32-conv-hwc-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-conv-hwc-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-conv-hwc-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-dwconv-spchw-bench bench/f32-dwconv-spchw.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-spchw-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-spchw-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-spchw-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-dwconv-spchw-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(f32-dwconv-bench bench/f32-dwconv.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-dwconv-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

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

  ADD_EXECUTABLE(f32-sigmoid-bench bench/f32-sigmoid.cc)
  SET_TARGET_PROPERTIES(f32-sigmoid-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-sigmoid-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-sigmoid-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-sigmoid-bench PRIVATE XNNPACK benchmark bench-utils)

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

  ADD_EXECUTABLE(q8-gemm-bench bench/q8-gemm.cc)
  SET_TARGET_PROPERTIES(q8-gemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(q8-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(q8-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)

  ADD_EXECUTABLE(requantization-bench bench/requantization.cc)
  SET_TARGET_PROPERTIES(requantization-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(requantization-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(requantization-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(requantization-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)
ENDIF()
