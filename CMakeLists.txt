# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project and semantic versioning.
PROJECT(XNNPACK C CXX ASM)

# ---[ Options.
SET(XNNPACK_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE XNNPACK_LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(XNNPACK_BUILD_TESTS "Build XNNPACK unit tests" ON)
OPTION(XNNPACK_BUILD_BENCHMARKS "Build XNNPACK benchmarks" ON)

# ---[ CMake options
IF(XNNPACK_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Build flags
IF(NOT CMAKE_SYSTEM_PROCESSOR)
  IF(IOS)
    LIST(LENGTH IOS_ARCH IOS_ARCH_COUNT)
    IF(IOS_ARCH_COUNT GREATER 1)
      MESSAGE(FATAL_ERROR "Unsupported XNNPACK build with multiple iOS architectures (${IOS_ARCH}). "
        "Specify a single architecture in IOS_ARCH and re-configure. ")
    ENDIF()
    IF(NOT IOS_ARCH MATCHES "^(i386|x86_64|armv7.*|arm64.*)$")
      MESSAGE(FATAL_ERROR "Unrecognized IOS_ARCH = ${IOS_ARCH}")
    ENDIF()
  ELSE()
    MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR is not defined")
  ENDIF()
ELSEIF(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64|armv[5-8].*|aarch64)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
ENDIF()

IF(NOT CMAKE_SYSTEM_NAME)
  MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_NAME not defined")
ELSEIF(NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux|Android)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
ENDIF()

# ---[ Download deps
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

# ---[ XNNPACK library
SET(XNNPACK_OPERATOR_SRCS
  src/add-nc.c
  src/argmax-pooling-nhwc.c
  src/average-pooling-nhwc.c
  src/channel-pad-nc.c
  src/channel-shuffle-nc.c
  src/clamp-nc.c
  src/convolution-nchw.c
  src/convolution-nhwc.c
  src/deconvolution-nhwc.c
  src/fully-connected-nc.c
  src/global-average-pooling-ncw.c
  src/global-average-pooling-nwc.c
  src/hardswish-nc.c
  src/leaky-relu-nc.c
  src/max-pooling-nhwc.c
  src/multiply-nd.c
  src/prelu-nc.c
  src/resize-bilinear-nhwc.c
  src/sigmoid-nc.c
  src/softargmax-nc.c
  src/unpooling-nhwc.c)

SET(XNNPACK_COLD_SRCS ${XNNPACK_OPERATOR_SRCS})
LIST(APPEND XNNPACK_COLD_SRCS
  src/init.c
  src/operator-delete.c)

SET(XNNPACK_HOT_SRCS
  src/indirection.c
  src/operator-run.c)

SET(XNNPACK_SCALAR_MICROKERNEL_SRCS
  src/f32-argmaxpool/mp9p8q-scalar.c
  src/f32-argmaxpool/up4-scalar.c
  src/f32-argmaxpool/up9-scalar.c
  src/f32-avgpool/mp9p8q-scalar.c
  src/f32-avgpool/up9-scalar.c
  src/f32-bilinear/scalar-c1.c
  src/f32-bilinear/scalar-c2.c
  src/f32-bilinear/scalar-c4.c
  src/f32-binop/vadd-scalar-x1.c
  src/f32-binop/vadd-scalar-x2.c
  src/f32-binop/vadd-scalar-x4.c
  src/f32-binop/vaddc-scalar-x1.c
  src/f32-binop/vaddc-scalar-x2.c
  src/f32-binop/vaddc-scalar-x4.c
  src/f32-binop/vmul-scalar-x1.c
  src/f32-binop/vmul-scalar-x2.c
  src/f32-binop/vmul-scalar-x4.c
  src/f32-binop/vmulc-scalar-x1.c
  src/f32-binop/vmulc-scalar-x2.c
  src/f32-binop/vmulc-scalar-x4.c
  src/f32-binop/vsub-scalar-x1.c
  src/f32-binop/vsub-scalar-x2.c
  src/f32-binop/vsub-scalar-x4.c
  src/f32-binop/vsubc-scalar-x1.c
  src/f32-binop/vsubc-scalar-x2.c
  src/f32-binop/vsubc-scalar-x4.c
  src/f32-binop/vrsubc-scalar-x1.c
  src/f32-binop/vrsubc-scalar-x2.c
  src/f32-binop/vrsubc-scalar-x4.c
  src/f32-clamp/scalar.c
  src/f32-conv-hwc2spchw/3x3s2p1c3x4-scalar-1x1.c
  src/f32-igemm/1x4-scalar.c
  src/f32-igemm/2x4-scalar.c
  src/f32-igemm/4x2-scalar.c
  src/f32-igemm/4x4-scalar.c
  src/f32-dwconv/up1x25-scalar.c
  src/f32-dwconv/up1x25-scalar-acc2.c
  src/f32-dwconv/up2x25-scalar.c
  src/f32-dwconv/up2x25-scalar-acc2.c
  src/f32-dwconv/up1x4-scalar.c
  src/f32-dwconv/up1x4-scalar-acc2.c
  src/f32-dwconv/up2x4-scalar.c
  src/f32-dwconv/up2x4-scalar-acc2.c
  src/f32-dwconv/up1x9-scalar.c
  src/f32-dwconv/up1x9-scalar-acc2.c
  src/f32-dwconv/up2x9-scalar.c
  src/f32-dwconv/up2x9-scalar-acc2.c
  src/f32-dwconv-spchw/3x3p1-scalar.c
  src/f32-dwconv-spchw/5x5p2-scalar.c
  src/f32-dwconv-spchw/3x3s2p1-scalar.c
  src/f32-dwconv-spchw/5x5s2p2-scalar.c
  src/f32-gavgpool-spchw/scalar-x1.c
  src/f32-gavgpool/mp7p7q-scalar.c
  src/f32-gavgpool/up7-scalar.c
  src/f32-gemm/1x4-scalar.c
  src/f32-gemm/2x4-scalar.c
  src/f32-gemm/4x2-scalar.c
  src/f32-gemm/4x4-scalar.c
  src/f32-gemminc/1x4-scalar.c
  src/f32-gemminc/2x4-scalar.c
  src/f32-gemminc/4x4-scalar.c
  src/f32-hswish/scalar.c
  src/f32-maxpool/9p8q-scalar.c
  src/f32-pavgpool/mp9p8q-scalar.c
  src/f32-pavgpool/up9-scalar.c
  src/f32-ppmm/2x4-scalar.c
  src/f32-ppmm/3x3-scalar.c
  src/f32-ppmm/4x2-scalar.c
  src/f32-ppmm/4x4-scalar.c
  src/f32-prelu/scalar-2x1.c
  src/f32-prelu/scalar-2x4.c
  src/f32-rmax/scalar.c
  src/f32-spmm/1x1-scalar-pipelined.c
  src/f32-spmm/1x1-scalar.c
  src/f32-spmm/2x1-scalar-pipelined.c
  src/f32-spmm/2x1-scalar.c
  src/f32-spmm/4x1-scalar-pipelined.c
  src/f32-spmm/4x1-scalar.c
  src/f32-spmm/8x1-scalar-pipelined.c
  src/f32-spmm/8x1-scalar.c
  src/f32-spmm/8x2-scalar.c
  src/f32-spmm/8x4-scalar.c
  src/f32-vmulcaddc/c1-scalar-2x.c
  src/f32-vmulcaddc/c2-scalar-2x.c
  src/f32-vmulcaddc/c4-scalar-2x.c
  src/q8-avgpool/mp9p8q-scalar.c
  src/q8-avgpool/up9-scalar.c
  src/q8-igemm/2x2-scalar.c
  src/q8-dwconv/up1x9-scalar.c
  src/q8-gavgpool/mp7p7q-scalar.c
  src/q8-gavgpool/up7-scalar.c
  src/q8-gemm/2x2-scalar.c
  src/q8-vadd/scalar.c
  src/u8-clamp/scalar.c
  src/u8-lut32norm/scalar.c
  src/u8-maxpool/9p8q-scalar.c
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
  src/x8-zip/xm-scalar.c)

SET(XNNPACK_PSIMD_MICROKERNEL_SRCS
  src/f32-argmaxpool/mp9p8q-psimd.c
  src/f32-argmaxpool/up4-psimd.c
  src/f32-argmaxpool/up9-psimd.c
  src/f32-avgpool/mp9p8q-psimd.c
  src/f32-avgpool/up9-psimd.c
  src/f32-bilinear/psimd-c4.c
  src/f32-bilinear/psimd-c8.c
  src/f32-binop/vadd-psimd-x4.c
  src/f32-binop/vadd-psimd-x8.c
  src/f32-binop/vaddc-psimd-x4.c
  src/f32-binop/vaddc-psimd-x8.c
  src/f32-binop/vmul-psimd-x4.c
  src/f32-binop/vmul-psimd-x8.c
  src/f32-binop/vmulc-psimd-x4.c
  src/f32-binop/vmulc-psimd-x8.c
  src/f32-binop/vrsubc-psimd-x4.c
  src/f32-binop/vrsubc-psimd-x8.c
  src/f32-binop/vsub-psimd-x4.c
  src/f32-binop/vsub-psimd-x8.c
  src/f32-binop/vsubc-psimd-x4.c
  src/f32-binop/vsubc-psimd-x8.c
  src/f32-clamp/psimd.c
  src/f32-igemm/1x8-psimd-loadsplat.c
  src/f32-igemm/1x8-psimd-splat.c
  src/f32-igemm/1x8s4-psimd.c
  src/f32-igemm/4x2c4-psimd.c
  src/f32-igemm/4x8-psimd-loadsplat.c
  src/f32-igemm/4x8-psimd-splat.c
  src/f32-igemm/4x8s4-psimd.c
  src/f32-igemm/6x8-psimd-loadsplat.c
  src/f32-igemm/6x8-psimd-splat.c
  src/f32-igemm/6x8s4-psimd.c
  src/f32-dwconv/up4x25-psimd.c
  src/f32-dwconv/up4x4-psimd.c
  src/f32-dwconv/up4x9-psimd.c
  src/f32-dwconv/up4x25-psimd-acc2.c
  src/f32-dwconv/up4x4-psimd-acc2.c
  src/f32-dwconv/up4x9-psimd-acc2.c
  src/f32-dwconv/up8x25-psimd.c
  src/f32-dwconv/up8x4-psimd.c
  src/f32-dwconv/up8x9-psimd.c
  src/f32-dwconv/up8x25-psimd-acc2.c
  src/f32-dwconv/up8x4-psimd-acc2.c
  src/f32-dwconv/up8x9-psimd-acc2.c
  src/f32-gavgpool/mp7p7q-psimd.c
  src/f32-gavgpool/up7-psimd.c
  src/f32-gemm/1x8-psimd-loadsplat.c
  src/f32-gemm/1x8-psimd-splat.c
  src/f32-gemm/1x8s4-psimd.c
  src/f32-gemm/4x8-psimd-loadsplat.c
  src/f32-gemm/4x8-psimd-splat.c
  src/f32-gemm/4x8s4-psimd.c
  src/f32-gemm/6x8-psimd-loadsplat.c
  src/f32-gemm/6x8-psimd-splat.c
  src/f32-gemm/6x8s4-psimd.c
  src/f32-gemminc/1x8-psimd-loadsplat.c
  src/f32-gemminc/1x8-psimd-splat.c
  src/f32-gemminc/1x8s4-psimd.c
  src/f32-gemminc/4x8-psimd-loadsplat.c
  src/f32-gemminc/4x8-psimd-splat.c
  src/f32-gemminc/4x8s4-psimd.c
  src/f32-gemminc/6x8-psimd-loadsplat.c
  src/f32-gemminc/6x8-psimd-splat.c
  src/f32-gemminc/6x8s4-psimd.c
  src/f32-hswish/psimd.c
  src/f32-maxpool/9p8q-psimd.c
  src/f32-pavgpool/mp9p8q-psimd.c
  src/f32-pavgpool/up9-psimd.c
  src/f32-ppmm/4x8-psimd.c
  src/f32-prelu/psimd-2x4.c
  src/f32-prelu/psimd-2x8.c
  src/f32-vmulcaddc/c4-psimd-2x.c
  src/f32-vmulcaddc/c8-psimd-2x.c
  src/x32-packx/x4-psimd.c
  src/x32-pad/x2-psimd.c
  src/x32-unpool/psimd.c
  src/x32-zip/x2-psimd.c
  src/x32-zip/x3-psimd.c
  src/x32-zip/x4-psimd.c
  src/x32-zip/xm-psimd.c)

SET(XNNPACK_NEON_MICROKERNEL_SRCS
  src/f32-avgpool/mp9p8q-neon.c
  src/f32-avgpool/up9-neon.c
  src/f32-bilinear/neon-c4.c
  src/f32-bilinear/neon-c8.c
  src/f32-binop/vadd-neon-x4.c
  src/f32-binop/vadd-neon-x8.c
  src/f32-binop/vaddc-neon-x4.c
  src/f32-binop/vaddc-neon-x8.c
  src/f32-binop/vmul-neon-x4.c
  src/f32-binop/vmul-neon-x8.c
  src/f32-binop/vmulc-neon-x4.c
  src/f32-binop/vmulc-neon-x8.c
  src/f32-binop/vrsubc-neon-x4.c
  src/f32-binop/vrsubc-neon-x8.c
  src/f32-binop/vsub-neon-x4.c
  src/f32-binop/vsub-neon-x8.c
  src/f32-binop/vsubc-neon-x4.c
  src/f32-binop/vsubc-neon-x8.c
  src/f32-clamp/neon.c
  src/f32-igemm/1x8-neon-ld64.c
  src/f32-igemm/4x2-neon-ld64.c
  src/f32-igemm/4x4-neon-ld64.c
  src/f32-igemm/4x8-neon-ld128.c
  src/f32-igemm/4x8-neon-ld64.c
  src/f32-igemm/6x8-neon-ld64.c
  src/f32-dwconv/up4x9-neon.c
  src/f32-dwconv/up4x9-neon-acc2.c
  src/f32-dwconv/up8x9-neon.c
  src/f32-dwconv/up8x9-neon-acc2.c
  src/f32-gavgpool-spchw/neon-x4.c
  src/f32-gavgpool/mp7p7q-neon.c
  src/f32-gavgpool/up7-neon.c
  src/f32-gemm/1x8-neon-ld64.c
  src/f32-gemm/4x2-neon-ld64.c
  src/f32-gemm/4x8-neon-ld128.c
  src/f32-gemm/4x8-neon-ld64.c
  src/f32-gemm/5x8-neon-ld64.c
  src/f32-gemm/6x8-neon-ld64.c
  src/f32-gemminc/1x8-neon-ld64.c
  src/f32-gemminc/4x8-neon-ld128.c
  src/f32-gemminc/4x8-neon-ld64.c
  src/f32-gemminc/5x8-neon-ld64.c
  src/f32-gemminc/6x8-neon-ld64.c
  src/f32-hswish/neon.c
  src/f32-pavgpool/mp9p8q-neon.c
  src/f32-pavgpool/up9-neon.c
  src/f32-ppmm/4x8-neon.c
  src/f32-ppmm/8x8-neon.c
  src/f32-prelu/neon-2x4.c
  src/f32-prelu/neon-2x8.c
  src/f32-rmax/neon.c
  src/f32-sigmoid/neon-frac-p9-p10-nr1recps-x16.c
  src/f32-vmulcaddc/c4-neon-2x.c
  src/f32-vmulcaddc/c8-neon-2x.c
  src/q8-avgpool/mp9p8q-neon.c
  src/q8-avgpool/up9-neon.c
  src/q8-igemm/4x8-neon.c
  src/q8-igemm/8x8-neon.c
  src/q8-dwconv/up8x9-neon.c
  src/q8-gavgpool/mp7p7q-neon.c
  src/q8-gavgpool/up7-neon.c
  src/q8-gemm/4x8-neon.c
  src/q8-gemm/8x8-neon.c
  src/q8-vadd/neon.c
  src/u8-clamp/neon.c
  src/u8-maxpool/9p8q-neon.c
  src/u8-rmax/neon.c
  src/x32-packx/x4-neon-st4.c
  src/x32-pad/x2-neon.c
  src/x32-zip/x2-neon.c
  src/x32-zip/x3-neon.c
  src/x32-zip/x4-neon.c
  src/x32-zip/xm-neon.c
  src/x8-zip/x2-neon.c
  src/x8-zip/x3-neon.c
  src/x8-zip/x4-neon.c
  src/x8-zip/xm-neon.c)

SET(XNNPACK_NEONFMA_MICROKERNEL_SRCS
  src/f32-bilinear/neonfma-c4.c
  src/f32-bilinear/neonfma-c8.c
  src/f32-igemm/1x8-neonfma-ld64.c
  src/f32-igemm/4x2-neonfma-ld64.c
  src/f32-igemm/4x4-neonfma-ld64.c
  src/f32-igemm/4x8-neonfma-ld128.c
  src/f32-igemm/4x8-neonfma-ld64.c
  src/f32-igemm/6x8-neonfma-ld64.c
  src/f32-dwconv/up4x9-neonfma.c
  src/f32-dwconv/up4x9-neonfma-acc2.c
  src/f32-dwconv/up8x9-neonfma.c
  src/f32-dwconv/up8x9-neonfma-acc2.c
  src/f32-gemm/1x8-neonfma-ld64.c
  src/f32-gemm/4x2-neonfma-ld64.c
  src/f32-gemm/4x8-neonfma-ld128.c
  src/f32-gemm/4x8-neonfma-ld64.c
  src/f32-gemm/5x8-neonfma-ld64.c
  src/f32-gemm/6x8-neonfma-ld64.c
  src/f32-gemminc/1x8-neonfma-ld64.c
  src/f32-gemminc/4x8-neonfma-ld128.c
  src/f32-gemminc/4x8-neonfma-ld64.c
  src/f32-gemminc/5x8-neonfma-ld64.c
  src/f32-gemminc/6x8-neonfma-ld64.c
  src/f32-hswish/neonfma.c
  src/f32-ppmm/4x8-neonfma.c
  src/f32-ppmm/8x8-neonfma.c
  src/f32-sigmoid/neonfma-p5-nr2fma-x16.c
  src/f32-vmulcaddc/c4-neonfma-2x.c
  src/f32-vmulcaddc/c8-neonfma-2x.c
  src/math/exp-neonfma-lut64-p2.c
  src/math/exp-neonfma-p5.c
  src/math/expminus-neonfma-p5.c
  src/math/sigmoid-neonfma-p5-nr2fma.c)

SET(AARCH64_XNNPACK_NEONFMA_MICROKERNEL_SRCS
  src/f32-conv-hwc/3x3s2p1c3x4-neonfma-2x2.c
  src/f32-conv-hwc/3x3s2p1c3x8-neonfma-2x2.c
  src/f32-conv-hwc2spchw/3x3s2p1c3x4-neonfma-2x2.c
  src/f32-dwconv-spchw/3x3p1-neonfma.c
  src/f32-dwconv-spchw/5x5p2-neonfma.c
  src/f32-dwconv-spchw/3x3s2p1-neonfma.c
  src/f32-dwconv-spchw/5x5s2p2-neonfma.c
  src/f32-spmm/12x1-neonfma.c
  src/f32-spmm/12x2-neonfma.c
  src/f32-spmm/12x4-neonfma.c
  src/f32-spmm/16x1-neonfma-pipelined.c
  src/f32-spmm/16x1-neonfma-unroll2.c
  src/f32-spmm/16x1-neonfma.c
  src/f32-spmm/16x2-neonfma.c
  src/f32-spmm/16x4-neonfma.c
  src/f32-spmm/4x1-neonfma-pipelined.c
  src/f32-spmm/4x1-neonfma-unroll2.c
  src/f32-spmm/4x1-neonfma.c
  src/f32-spmm/4x2-neonfma.c
  src/f32-spmm/4x4-neonfma.c
  src/f32-spmm/8x1-neonfma-pipelined.c
  src/f32-spmm/8x1-neonfma-unroll2.c
  src/f32-spmm/8x1-neonfma.c
  src/f32-spmm/8x2-neonfma.c
  src/f32-spmm/8x4-neonfma.c)

SET(XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS
  src/f16-gemm/4x8-neonfp16arith-ld64.c
  src/f16-gemm/6x8-neonfp16arith-ld64.c
  src/f16-gemm/8x8-neonfp16arith-ld64.c)

SET(XNNPACK_SSE_MICROKERNEL_SRCS
  src/f32-avgpool/mp9p8q-sse.c
  src/f32-avgpool/up9-sse.c
  src/f32-bilinear/sse-c4.c
  src/f32-bilinear/sse-c8.c
  src/f32-binop/vadd-sse-x4.c
  src/f32-binop/vadd-sse-x8.c
  src/f32-binop/vaddc-sse-x4.c
  src/f32-binop/vaddc-sse-x8.c
  src/f32-binop/vmul-sse-x4.c
  src/f32-binop/vmul-sse-x8.c
  src/f32-binop/vmulc-sse-x4.c
  src/f32-binop/vmulc-sse-x8.c
  src/f32-binop/vrsubc-sse-x4.c
  src/f32-binop/vrsubc-sse-x8.c
  src/f32-binop/vsub-sse-x4.c
  src/f32-binop/vsub-sse-x8.c
  src/f32-binop/vsubc-sse-x4.c
  src/f32-binop/vsubc-sse-x8.c
  src/f32-clamp/sse.c
  src/f32-igemm/1x8-sse-dup.c
  src/f32-igemm/1x8-sse-load1.c
  src/f32-igemm/1x8s4-sse.c
  src/f32-igemm/4x2c4-sse.c
  src/f32-igemm/4x8-sse-dup.c
  src/f32-igemm/4x8-sse-load1.c
  src/f32-igemm/4x8s4-sse.c
  src/f32-dwconv/up4x25-sse.c
  src/f32-dwconv/up4x4-sse.c
  src/f32-dwconv/up4x9-sse.c
  src/f32-dwconv/up4x25-sse-acc2.c
  src/f32-dwconv/up4x4-sse-acc2.c
  src/f32-dwconv/up4x9-sse-acc2.c
  src/f32-dwconv/up8x25-sse.c
  src/f32-dwconv/up8x4-sse.c
  src/f32-dwconv/up8x9-sse.c
  src/f32-dwconv/up8x25-sse-acc2.c
  src/f32-dwconv/up8x4-sse-acc2.c
  src/f32-dwconv/up8x9-sse-acc2.c
  src/f32-gavgpool-spchw/sse-x4.c
  src/f32-gavgpool/mp7p7q-sse.c
  src/f32-gavgpool/up7-sse.c
  src/f32-gemm/1x8-sse-dup.c
  src/f32-gemm/1x8-sse-load1.c
  src/f32-gemm/1x8s4-sse.c
  src/f32-gemm/4x8-sse-dup.c
  src/f32-gemm/4x8-sse-load1.c
  src/f32-gemm/4x8s4-sse.c
  src/f32-gemminc/1x8-sse-dup.c
  src/f32-gemminc/1x8-sse-load1.c
  src/f32-gemminc/1x8s4-sse.c
  src/f32-gemminc/4x8-sse-dup.c
  src/f32-gemminc/4x8-sse-load1.c
  src/f32-gemminc/4x8s4-sse.c
  src/f32-hswish/sse.c
  src/f32-maxpool/9p8q-sse.c
  src/f32-pavgpool/mp9p8q-sse.c
  src/f32-pavgpool/up9-sse.c
  src/f32-dwconv-spchw/3x3p1-sse.c
  src/f32-dwconv-spchw/3x3s2p1-sse.c
  src/f32-ppmm/4x8-sse.c
  src/f32-rmax/sse.c
  src/f32-spmm/4x1-sse.c
  src/f32-spmm/8x1-sse.c
  src/f32-vmulcaddc/c4-sse-2x.c
  src/f32-vmulcaddc/c8-sse-2x.c
  src/x32-packx/x4-sse.c)

SET(XNNPACK_SSE2_MICROKERNEL_SRCS
  src/f32-argmaxpool/mp9p8q-sse2.c
  src/f32-argmaxpool/up4-sse2.c
  src/f32-argmaxpool/up9-sse2.c
  src/f32-prelu/sse2-2x4.c
  src/f32-prelu/sse2-2x8.c
  src/q8-avgpool/mp9p8q-sse2.c
  src/q8-avgpool/up9-sse2.c
  src/q8-igemm/4x4c2-sse2.c
  src/q8-dwconv/up8x9-sse2.c
  src/q8-gavgpool/mp7p7q-sse2.c
  src/q8-gavgpool/up7-sse2.c
  src/q8-gemm/2x4c8-sse2.c
  src/q8-gemm/4x4c2-sse2.c
  src/q8-vadd/sse2.c
  src/u8-clamp/sse2.c
  src/u8-maxpool/9p8q-sse2.c
  src/u8-rmax/sse2.c
  src/x32-pad/x2-sse2.c
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
  src/math/sigmoid-sse2-p5-div.c)

SET(XNNPACK_SSE41_MICROKERNEL_SRCS
  src/f32-prelu/sse41-2x4.c
  src/f32-prelu/sse41-2x8.c)

SET(XNNPACK_AVX_MICROKERNEL_SRCS
  src/f32-rmax/avx.c
  src/f32-vscale/avx-unroll32.c)

SET(XNNPACK_AVX2_MICROKERNEL_SRCS
  src/f32-raddexpminusmax/avx2-p5-unroll64.c
  src/f32-raddextexp/avx2-p5-unroll64.c
  src/f32-raddstoreexpminusmax/avx2-p5-unroll64.c
  src/f32-vscaleexpminusmax/avx2-p5-unroll64.c
  src/f32-vscaleextexp/avx2-p5-unroll64.c
  src/math/exp-avx2-p5.c
  src/math/exp-avx2-perm-p3.c
  src/math/exp-avx2-perm-p4.c
  src/math/expminus-avx2-p5.c
  src/math/extexp-avx2-p5.c)

SET(XNNPACK_AVX512F_MICROKERNEL_SRCS
  src/f32-raddexpminusmax/avx512f-p5-scalef-unroll128.c
  src/f32-raddextexp/avx512f-p5-scalef-unroll128.c
  src/f32-raddstoreexpminusmax/avx512f-p5-scalef-unroll128.c
  src/f32-rmax/avx512f.c
  src/f32-vscale/avx512f-unroll64.c
  src/f32-vscaleexpminusmax/avx512f-p5-scalef-unroll128.c
  src/f32-vscaleextexp/avx512f-p5-scalef-unroll128.c
  src/math/exp-avx512f-p5-scalef.c
  src/math/exp-avx512f-p5.c
  src/math/exp-avx512f-perm-p3.c
  src/math/exp-avx512f-perm2-p2.c
  src/math/extexp-avx512f-p5.c)

SET(XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS
  src/q8-dwconv/up8x9-aarch32-neon.S)

SET(XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS
  src/f32-dwconv/up4x9-aarch64-neonfma-cortex-a55.S
  src/f32-dwconv/up4x9-aarch64-neonfma.S
  src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/1x8-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/1x8-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/4x12-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/4x8-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/4x8-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/4x8-aarch64-neonfma-ld128.S
  src/f32-gemm/4x8-aarch64-neonfma-ld64.S
  src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/6x8-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/6x8-aarch64-neonfma-cortex-a57.S
  src/f32-gemm/6x8-aarch64-neonfma-cortex-a73.S
  src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/6x8-aarch64-neonfma-ld128.S
  src/f32-gemm/6x8-aarch64-neonfma-ld64.S
  src/f32-gemminc/1x12-aarch64-neonfma-cortex-a53.S
  src/f32-gemminc/1x8-aarch64-neonfma-cortex-a53.S
  src/f32-gemminc/1x8-aarch64-neonfma-cortex-a57.S
  src/f32-gemminc/1x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemminc/4x12-aarch64-neonfma-cortex-a53.S
  src/f32-gemminc/4x8-aarch64-neonfma-cortex-a53.S
  src/f32-gemminc/4x8-aarch64-neonfma-cortex-a57.S
  src/f32-gemminc/4x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemminc/4x8-aarch64-neonfma-ld128.S
  src/f32-gemminc/4x8-aarch64-neonfma-ld64.S
  src/f32-gemminc/5x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemminc/6x8-aarch64-neonfma-cortex-a53.S
  src/f32-gemminc/6x8-aarch64-neonfma-cortex-a57.S
  src/f32-gemminc/6x8-aarch64-neonfma-cortex-a73.S
  src/f32-gemminc/6x8-aarch64-neonfma-cortex-a75.S
  src/f32-gemminc/6x8-aarch64-neonfma-ld128.S
  src/f32-gemminc/6x8-aarch64-neonfma-ld64.S
  src/f32-igemm/1x12-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/1x8-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/1x8-aarch64-neonfma-cortex-a57.S
  src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/4x12-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/5x8-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/6x8-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/6x8-aarch64-neonfma-cortex-a57.S
  src/f32-igemm/6x8-aarch64-neonfma-cortex-a73.S
  src/f32-igemm/6x8-aarch64-neonfma-cortex-a75.S)

SET(XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SCALAR_MICROKERNEL_SRCS})
IF(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_PSIMD_MICROKERNEL_SRCS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEON_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH32_ASM_MICROKERNEL_SRCS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR IOS_ARCH MATCHES "^arm64.*")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEON_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${AARCH64_XNNPACK_NEONFMA_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AARCH64_ASM_MICROKERNEL_SRCS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64)$" OR IOS_ARCH MATCHES "^(i386|x86_64)$")
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE2_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_SSE41_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX2_MICROKERNEL_SRCS})
  LIST(APPEND XNNPACK_MICROKERNEL_SRCS ${XNNPACK_AVX512F_MICROKERNEL_SRCS})
ENDIF()

IF(XNNPACK_LIBRARY_TYPE STREQUAL "default")
  ADD_LIBRARY(XNNPACK ${XNNPACK_COLD_SRCS} ${XNNPACK_HOT_SRCS} ${XNNPACK_MICROKERNEL_SRCS})
ELSEIF(XNNPACK_LIBRARY_TYPE STREQUAL "shared")
  ADD_LIBRARY(XNNPACK SHARED ${XNNPACK_COLD_SRCS} ${XNNPACK_HOT_SRCS} ${XNNPACK_MICROKERNEL_SRCS})
ELSEIF(XNNPACK_LIBRARY_TYPE STREQUAL "static")
  ADD_LIBRARY(XNNPACK STATIC ${XNNPACK_COLD_SRCS} ${XNNPACK_HOT_SRCS} ${XNNPACK_MICROKERNEL_SRCS})
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported XNNPACK library type \"${XNNPACK_LIBRARY_TYPE}\". Must be \"static\", \"shared\", or \"default\"")
ENDIF()
SET_TARGET_PROPERTIES(XNNPACK PROPERTIES
  C_STANDARD 99
  C_EXTENSIONS YES)
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -marm ")
  SET_PROPERTY(SOURCE ${XNNPACK_PSIMD_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon ")
  SET_PROPERTY(SOURCE ${XNNPACK_NEON_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon ")
  SET_PROPERTY(SOURCE ${XNNPACK_NEONFMA_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon-vfpv4 ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${XNNPACK_AARCH32_ASM_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR IOS_ARCH MATCHES "^arm64.*")
  SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${XNNPACK_AARCH64_ASM_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64)$" OR IOS_ARCH MATCHES "^(i386|x86_64)$")
  SET_PROPERTY(SOURCE ${XNNPACK_SSE_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse ")
  SET_PROPERTY(SOURCE ${XNNPACK_SSE2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse2 ")
  SET_PROPERTY(SOURCE ${XNNPACK_SSE41_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -msse4.1 ")
  SET_PROPERTY(SOURCE ${XNNPACK_AVX_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx ")
  SET_PROPERTY(SOURCE ${XNNPACK_AVX2_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfma -mavx2 ")
  SET_PROPERTY(SOURCE ${XNNPACK_AVX512F_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx512f ")
ENDIF()
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE XNN_LOG_LEVEL=5)
ELSE()
  TARGET_COMPILE_DEFINITIONS(XNNPACK PRIVATE XNN_LOG_LEVEL=0)
  SET_PROPERTY(SOURCE ${XNNPACK_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
  SET_PROPERTY(SOURCE ${XNNPACK_HOT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
  SET_PROPERTY(SOURCE ${XNNPACK_COLD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -Os ")
ENDIF()
IF(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  SET_PROPERTY(SOURCE ${XNNPACK_PSIMD_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O3 -ffast-math ")
ENDIF()

TARGET_INCLUDE_DIRECTORIES(XNNPACK PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(XNNPACK PRIVATE src)
SET_TARGET_PROPERTIES(XNNPACK PROPERTIES PUBLIC_HEADER include/XNNPACK.h)

# ---[ Configure clog
IF(NOT TARGET clog)
  SET(CLOG_BUILD_TESTS OFF CACHE BOOL "")
  SET(CLOG_RUNTIME_TYPE "${CPUINFO_RUNTIME_TYPE}" CACHE STRING "")
  ADD_SUBDIRECTORY(
    "${CLOG_SOURCE_DIR}/deps/clog"
    "${CMAKE_BINARY_DIR}/clog")
  # We build static version of clog but a dynamic library may indirectly depend on it
  SET_PROPERTY(TARGET clog PROPERTY POSITION_INDEPENDENT_CODE ON)
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE clog)

# ---[ Configure cpuinfo
IF(NOT TARGET cpuinfo)
  SET(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${CPUINFO_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/cpuinfo")
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE cpuinfo)

# ---[ Configure pthreadpool
IF(NOT TARGET pthreadpool)
  SET(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
  SET(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
  SET(PTHREADPOOL_ALLOW_DEPRECATED_API OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${PTHREADPOOL_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/pthreadpool")
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PUBLIC pthreadpool)

# ---[ Configure FXdiv
IF(NOT TARGET fxdiv)
  SET(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
  SET(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${FXDIV_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/FXdiv")
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE fxdiv)

# ---[ Configure psimd
IF(NOT TARGET psimd)
  ADD_SUBDIRECTORY(
    "${PSIMD_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/psimd")
ENDIF()
TARGET_LINK_LIBRARIES(XNNPACK PRIVATE psimd)

# ---[ Configure FP16
IF(NOT TARGET fp16)
  SET(FP16_BUILD_TESTS OFF CACHE BOOL "")
  SET(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${FP16_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/FP16")
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

  # ---[ Build operator-level unit tests
  ADD_EXECUTABLE(add-nc-test test/add-nc.cc)
  SET_TARGET_PROPERTIES(add-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(add-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(add-nc-test add-nc-test)

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

  ADD_EXECUTABLE(softargmax-nc-test test/softargmax-nc.cc)
  SET_TARGET_PROPERTIES(softargmax-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(softargmax-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(softargmax-nc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(softargmax-nc-test softargmax-nc-test)

  ADD_EXECUTABLE(unpooling-nhwc-test test/unpooling-nhwc.cc)
  SET_TARGET_PROPERTIES(unpooling-nhwc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(unpooling-nhwc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(unpooling-nhwc-test PRIVATE XNNPACK gtest gtest_main)
  ADD_TEST(unpooling-nhwc-test unpooling-nhwc-test)

  # ---[ Build microkernel-level unit tests
  ADD_EXECUTABLE(f16-gemm-test test/f16-gemm.cc)
  SET_TARGET_PROPERTIES(f16-gemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f16-gemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f16-gemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f16-gemm-test f16-gemm-test)

  ADD_EXECUTABLE(f32-argmaxpool-test test/f32-argmaxpool.cc)
  SET_TARGET_PROPERTIES(f32-argmaxpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-argmaxpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-argmaxpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-argmaxpool-test f32-argmaxpool-test)

  ADD_EXECUTABLE(f32-avgpool-test test/f32-avgpool.cc)
  SET_TARGET_PROPERTIES(f32-avgpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-avgpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-avgpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-avgpool-test f32-avgpool-test)

  ADD_EXECUTABLE(f32-bilinear-test test/f32-bilinear.cc)
  SET_TARGET_PROPERTIES(f32-bilinear-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-bilinear-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-bilinear-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-bilinear-test f32-bilinear-test)

  ADD_EXECUTABLE(f32-clamp-nc-test test/f32-clamp.cc)
  SET_TARGET_PROPERTIES(f32-clamp-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-clamp-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-clamp-nc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-clamp-nc-test f32-clamp-nc-test)

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

  ADD_EXECUTABLE(f32-gavgpool-spchw-test test/f32-gavgpool-spchw.cc)
  SET_TARGET_PROPERTIES(f32-gavgpool-spchw-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gavgpool-spchw-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gavgpool-spchw-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gavgpool-spchw-test f32-gavgpool-spchw-test)

  ADD_EXECUTABLE(f32-gavgpool-test test/f32-gavgpool.cc)
  SET_TARGET_PROPERTIES(f32-gavgpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gavgpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gavgpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gavgpool-test f32-gavgpool-test)

  ADD_EXECUTABLE(f32-gemm-test test/f32-gemm.cc)
  SET_TARGET_PROPERTIES(f32-gemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gemm-test f32-gemm-test)

  ADD_EXECUTABLE(f32-gemminc-test test/f32-gemminc.cc)
  SET_TARGET_PROPERTIES(f32-gemminc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-gemminc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-gemminc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-gemminc-test f32-gemminc-test)

  ADD_EXECUTABLE(f32-hswish-test test/f32-hswish.cc)
  SET_TARGET_PROPERTIES(f32-hswish-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-hswish-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-hswish-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-hswish-test f32-hswish-test)

  ADD_EXECUTABLE(f32-igemm-test test/f32-igemm.cc)
  SET_TARGET_PROPERTIES(f32-igemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-igemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-igemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-igemm-test f32-igemm-test)

  ADD_EXECUTABLE(f32-maxpool-test test/f32-maxpool.cc)
  SET_TARGET_PROPERTIES(f32-maxpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-maxpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-maxpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-maxpool-test f32-maxpool-test)

  ADD_EXECUTABLE(f32-pavgpool-test test/f32-pavgpool.cc)
  SET_TARGET_PROPERTIES(f32-pavgpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-pavgpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-pavgpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-pavgpool-test f32-pavgpool-test)

  ADD_EXECUTABLE(f32-ppmm-test test/f32-ppmm.cc)
  SET_TARGET_PROPERTIES(f32-ppmm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-ppmm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-ppmm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-ppmm-test f32-ppmm-test)

  ADD_EXECUTABLE(f32-prelu-nc-test test/f32-prelu.cc)
  SET_TARGET_PROPERTIES(f32-prelu-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-prelu-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-prelu-nc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-prelu-nc-test f32-prelu-nc-test)

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

  ADD_EXECUTABLE(f32-spmm-test test/f32-spmm.cc)
  SET_TARGET_PROPERTIES(f32-spmm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-spmm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-spmm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-spmm-test f32-spmm-test)

  ADD_EXECUTABLE(f32-vadd-nc-test test/f32-vadd.cc)
  SET_TARGET_PROPERTIES(f32-vadd-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vadd-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vadd-nc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vadd-nc-test f32-vadd-nc-test)

  ADD_EXECUTABLE(f32-vaddc-test test/f32-vaddc.cc)
  SET_TARGET_PROPERTIES(f32-vaddc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vaddc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vaddc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vaddc-test f32-vaddc-test)

  ADD_EXECUTABLE(f32-vmul-test test/f32-vmul.cc)
  SET_TARGET_PROPERTIES(f32-vmul-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmul-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmul-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmul-test f32-vmul-test)

  ADD_EXECUTABLE(f32-vmulc-test test/f32-vmulc.cc)
  SET_TARGET_PROPERTIES(f32-vmulc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulc-test f32-vmulc-test)

  ADD_EXECUTABLE(f32-vmulcaddc-test test/f32-vmulcaddc.cc)
  SET_TARGET_PROPERTIES(f32-vmulcaddc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vmulcaddc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vmulcaddc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vmulcaddc-test f32-vmulcaddc-test)

  ADD_EXECUTABLE(f32-vscaleexpminusmax-test test/f32-vscaleexpminusmax.cc)
  SET_TARGET_PROPERTIES(f32-vscaleexpminusmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vscaleexpminusmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vscaleexpminusmax-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vscaleexpminusmax-test f32-vscaleexpminusmax-test)

  ADD_EXECUTABLE(f32-vscaleextexp-test test/f32-vscaleextexp.cc)
  SET_TARGET_PROPERTIES(f32-vscaleexpminusmax-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vscaleextexp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vscaleextexp-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vscaleextexp-test f32-vscaleextexp-test)

  ADD_EXECUTABLE(f32-vsub-test test/f32-vsub.cc)
  SET_TARGET_PROPERTIES(f32-vsub-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsub-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsub-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsub-test f32-vsub-test)

  ADD_EXECUTABLE(f32-vsubc-test test/f32-vsubc.cc)
  SET_TARGET_PROPERTIES(f32-vsubc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vsubc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vsubc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vsubc-test f32-vsubc-test)

  ADD_EXECUTABLE(f32-vrsubc-test test/f32-vrsubc.cc)
  SET_TARGET_PROPERTIES(f32-vrsubc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-vrsubc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(f32-vrsubc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(f32-vrsubc-test f32-vrsubc-test)

  ADD_EXECUTABLE(q8-avgpool-test test/q8-avgpool.cc)
  SET_TARGET_PROPERTIES(q8-avgpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-avgpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-avgpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-avgpool-test q8-avgpool-test)

  ADD_EXECUTABLE(q8-dwconv-test test/q8-dwconv.cc)
  SET_TARGET_PROPERTIES(q8-dwconv-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-dwconv-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-dwconv-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-dwconv-test q8-dwconv-test)

  ADD_EXECUTABLE(q8-gavgpool-test test/q8-gavgpool.cc)
  SET_TARGET_PROPERTIES(q8-gavgpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-gavgpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-gavgpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-gavgpool-test q8-gavgpool-test)

  ADD_EXECUTABLE(q8-gemm-test test/q8-gemm.cc)
  SET_TARGET_PROPERTIES(q8-gemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-gemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-gemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-gemm-test q8-gemm-test)

  ADD_EXECUTABLE(q8-igemm-test test/q8-igemm.cc)
  SET_TARGET_PROPERTIES(q8-igemm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-igemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-igemm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-igemm-test q8-igemm-test)

  ADD_EXECUTABLE(q8-vadd-nc-test test/q8-vadd.cc)
  SET_TARGET_PROPERTIES(q8-vadd-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-vadd-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8-vadd-nc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8-vadd-nc-test q8-vadd-nc-test)

  ADD_EXECUTABLE(u8-clamp-nc-test test/u8-clamp.cc)
  SET_TARGET_PROPERTIES(u8-clamp-nc-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-clamp-nc-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-clamp-nc-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-clamp-nc-test u8-clamp-nc-test)

  ADD_EXECUTABLE(u8-lut32norm-test test/u8-lut32norm.cc)
  SET_TARGET_PROPERTIES(u8-lut32norm-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-lut32norm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-lut32norm-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-lut32norm-test u8-lut32norm-test)

  ADD_EXECUTABLE(u8-maxpool-test test/u8-maxpool.cc)
  SET_TARGET_PROPERTIES(u8-maxpool-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(u8-maxpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8-maxpool-test PRIVATE XNNPACK cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8-maxpool-test u8-maxpool-test)

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
  ADD_EXECUTABLE(end2end-bench bench/end2end.cc models/mobilenet-v1.cc models/mobilenet-v2.cc)
  SET_TARGET_PROPERTIES(end2end-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(end2end-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(end2end-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(f32-dwconv-e2e-bench bench/f32-dwconv-e2e.cc models/mobilenet-v1.cc models/mobilenet-v2.cc)
  SET_TARGET_PROPERTIES(f32-dwconv-e2e-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-dwconv-e2e-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" src)
  TARGET_LINK_LIBRARIES(f32-dwconv-e2e-bench PRIVATE XNNPACK benchmark bench-utils)

  ADD_EXECUTABLE(f32-gemm-e2e-bench bench/f32-gemm-e2e.cc models/mobilenet-v1.cc models/mobilenet-v2.cc)
  SET_TARGET_PROPERTIES(f32-gemm-e2e-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(f32-gemm-e2e-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" src)
  TARGET_LINK_LIBRARIES(f32-gemm-e2e-bench PRIVATE XNNPACK benchmark bench-utils)

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

  ADD_EXECUTABLE(softargmax-bench bench/softargmax.cc)
  SET_TARGET_PROPERTIES(softargmax-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(softargmax-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(softargmax-bench PRIVATE XNNPACK cpuinfo benchmark bench-utils)

  # ---[ Build microkernel-level microbenchmarks
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

  ADD_EXECUTABLE(f32-softargmax-bench bench/f32-softargmax.cc)
  SET_TARGET_PROPERTIES(f32-softargmax-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(f32-softargmax-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(f32-softargmax-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(f32-softargmax-bench PRIVATE XNNPACK fp16 benchmark bench-utils)

  ADD_EXECUTABLE(q8-gemm-bench bench/q8-gemm.cc)
  SET_TARGET_PROPERTIES(q8-gemm-bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
  TARGET_INCLUDE_DIRECTORIES(q8-gemm-bench PRIVATE src)
  TARGET_INCLUDE_DIRECTORIES(q8-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  TARGET_LINK_LIBRARIES(q8-gemm-bench PRIVATE XNNPACK cpuinfo fp16 benchmark bench-utils)
ENDIF()
