# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---[ Macro to shard a test.
MACRO(SHARD_TEST TEST_NAME NUM_SHARDS)
  MATH(EXPR TEST_COST "${NUM_SHARDS} * 1000")
  GET_PROPERTY(TEST_LABELS TEST ${TEST_NAME} PROPERTY LABELS)
  SET_PROPERTY(
      TEST ${TEST_NAME}
        APPEND
        PROPERTY ENVIRONMENT "GTEST_TOTAL_SHARDS=${NUM_SHARDS};"
                "GTEST_SHARD_INDEX=0")
  SET_PROPERTY(
      TEST ${TEST_NAME} PROPERTY COST ${TEST_COST})
  MATH(EXPR NUM_SHARDS_MINUS_ONE "${NUM_SHARDS} - 1")
  FOREACH(SHARD RANGE 1 ${NUM_SHARDS_MINUS_ONE})
    ADD_TEST(NAME ${TEST_NAME}-${SHARD} COMMAND ${TEST_NAME})
    SET_PROPERTY(
        TEST ${TEST_NAME}-${SHARD}
          APPEND
          PROPERTY ENVIRONMENT "GTEST_TOTAL_SHARDS=${NUM_SHARDS};"
                  "GTEST_SHARD_INDEX=${SHARD}")
    SET_PROPERTY(
        TEST ${TEST_NAME}-${SHARD} PROPERTY COST ${TEST_COST})
    SET_PROPERTY(TEST ${TEST_NAME}-${SHARD} PROPERTY LABELS ${TEST_LABELS})
  ENDFOREACH()
ENDMACRO()

# ---[ Build google test
IF(NOT TARGET gtest)
  IF(XNNPACK_USE_SYSTEM_LIBS)
    FIND_PACKAGE(GTest REQUIRED)
  ELSE()
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googletest")
  ENDIF()
ENDIF()

# Create a dummy library that contains shared dependencies and definitions
# for all XNNPACK test targets.
ADD_LIBRARY(xnnpack-test INTERFACE)
IF (MSVC)
  TARGET_COMPILE_OPTIONS(xnnpack-test INTERFACE "$<$<NOT:$<CONFIG:Debug>>:/O2>" "/DEBUG:FASTLINK" "/Zi")
ELSE()
  TARGET_COMPILE_OPTIONS(xnnpack-test INTERFACE "$<$<NOT:$<CONFIG:Debug>>:-O2>" "-g")
ENDIF()
TARGET_LINK_LIBRARIES(xnnpack-test INTERFACE
  GTest::gmock
  GTest::gtest
  GTest::gtest_main
  XNNPACK)

# Test helper libraries
ADD_LIBRARY(next-prime STATIC next_prime.cc)

ADD_LIBRARY(gemm-microkernel-tester STATIC gemm-microkernel-tester.cc)
TARGET_LINK_LIBRARIES(gemm-microkernel-tester PRIVATE xnnpack-test packing)
TARGET_LINK_LIBRARIES(gemm-microkernel-tester PRIVATE pthreadpool)
TARGET_LINK_LIBRARIES(gemm-microkernel-tester PUBLIC next-prime)
IF(XNNPACK_ENABLE_KLEIDIAI)
  TARGET_LINK_LIBRARIES(gemm-microkernel-tester PRIVATE kleidiai)
ENDIF()

ADD_LIBRARY(dwconv-microkernel-tester STATIC dwconv-microkernel-tester.cc)
TARGET_LINK_LIBRARIES(dwconv-microkernel-tester PRIVATE xnnpack-test pthreadpool)
TARGET_LINK_LIBRARIES(dwconv-microkernel-tester PUBLIC next-prime)

ADD_LIBRARY(vbinary-microkernel-tester STATIC vbinary-microkernel-tester.cc)
SET_TARGET_PROPERTIES(vbinary-microkernel-tester PROPERTIES CXX_EXTENSIONS YES)
TARGET_LINK_LIBRARIES(vbinary-microkernel-tester PRIVATE xnnpack-test pthreadpool)

ADD_LIBRARY(unary-ops STATIC unary-ops.cc)
TARGET_LINK_LIBRARIES(unary-ops PRIVATE xnnpack-test )

ADD_LIBRARY(packq-microkernel-tester STATIC packq-microkernel-tester.cc)
TARGET_LINK_LIBRARIES(packq-microkernel-tester PRIVATE xnnpack-test pthreadpool)
IF(XNNPACK_ENABLE_KLEIDIAI)
  TARGET_LINK_LIBRARIES(packq-microkernel-tester PRIVATE kleidiai)
ENDIF()

ADD_SUBDIRECTORY(operators)
ADD_SUBDIRECTORY(subgraph)

# ---[ Build microkernel-level unit tests
SET(MICROKERNEL_UNIT_TESTS
    avgpool-minmax
    maxpool-minmax
    f16-conv-hwc2chw
    f16-f32acc-rdsum
    f16-f32acc-rsum
    f16-ibilinear
    f16-ibilinear-chw
    f16-raddstoreexpminusmax
    f16-rmax
    f16-rmin
    f16-rminmax
    f16-rsum
    f16-spmm-minmax
    f16-vcmul
    f16-vmulcaddc-minmax
    f32-argmaxpool
    f32-conv-hwc
    f32-conv-hwc2chw
    f32-ibilinear
    f32-ibilinear-chw
    f32-raddexpminusmax
    f32-raddextexp
    f32-raddstoreexpminusmax
    f32-rdsum
    f32-rmax
    f32-rmin
    f32-rminmax
    f32-rsum
    f32-spmm-minmax
    f32-vcmul
    f32-vmulcaddc-minmax
    f32-vscaleexpminusmax
    f32-vscaleextexp
    indirection
    packing
    qs8-packw
    qs8-qc4w-packw
    qs8-rdsum-minmax-fp32
    qs8-rsum
    qu8-rdsum
    qu8-rsum
    qu8-vlrelu
    s8-ibilinear
    u8-ibilinear
    u8-lut32norm
    u8-rmax
    x16-packw
    x32-packw
    x32-packx
    x32-unpool
    x8-lut
    x8-packw
    xN-transpose
    xx-fill
    xx-pad)
FOREACH(TEST ${MICROKERNEL_UNIT_TESTS})
  ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
  TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      hardware-config
      indirection
      logging
      microkernels-all
      microkernel-utils
      microparams-init
      next-prime
      packing
      pthreadpool
      xnnpack-test)
  ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
  SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
ENDFOREACH()
SHARD_TEST(avgpool-minmax-test 10)
SHARD_TEST(maxpool-minmax-test 10)
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^riscv")
  SHARD_TEST(f32-argmaxpool-test 20)
  SHARD_TEST(f32-rdsum-test 10)
ENDIF()

SET(MICROKERNEL_DWCONV_UNIT_TESTS
    f16-dwconv-minmax
    f16-dwconv2d-chw
    f32-dwconv-minmax
    f32-dwconv
    f32-dwconv2d-chw
    qs8-dwconv-minmax-fp32
    qs8-dwconv-minmax-rndnu
    qs8-qc8w-dwconv-minmax-fp32
    qu8-dwconv-minmax-fp32
    qu8-dwconv-minmax-rndnu)
FOREACH(TEST ${MICROKERNEL_DWCONV_UNIT_TESTS})
  ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
  TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      dwconv-microkernel-tester
      hardware-config
      logging
      microkernels-all
      microparams-init
      pthreadpool
      xnnpack-test)
  ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
  SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
ENDFOREACH()
SHARD_TEST(f32-dwconv-minmax-test 10)
SHARD_TEST(f32-dwconv-test 10)
SHARD_TEST(qs8-dwconv-minmax-fp32-test 10)
SHARD_TEST(qs8-qc8w-dwconv-minmax-fp32-test 10)

SET(MICROKERNEL_GEMM_UNIT_TESTS
    bf16-gemm-minmax
    f16-f32acc-gemm-minmax
    f16-f32acc-igemm-minmax
    f16-gemm-minmax
    f16-igemm-minmax
    f32-gemm
    f32-gemm-goi-minmax
    f32-gemm-minmax
    f32-gemm-relu
    f32-gemminc-minmax
    f32-igemm
    f32-igemm-minmax
    f32-igemm-relu
    f32-ppmm-minmax
    f32-qc4w-gemm-minmax
    f32-qc8w-gemm
    f32-qc8w-gemm-minmax
    f32-qc8w-gemm-relu
    pf32-gemm-minmax
    pqs8-qc8w-gemm-minmax
    qd8-f16-qb4w-gemm-minmax
    qd8-f16-qc4w-gemm-minmax
    qd8-f16-qc8w-gemm-minmax
    qd8-f32-qb4w-gemm-minmax
    qd8-f32-qc4w-gemm-minmax
    qd8-f32-qc8w-gemm-minmax
    qd8-f32-qc8w-igemm-minmax
    qp8-f32-qb4w-gemm-minmax
    qp8-f32-qc4w-gemm-minmax
    qp8-f32-qc8w-gemm-minmax
    qs8-qc8w-gemm-minmax-fp32
    qs8-qc8w-igemm-minmax-fp32
    qu8-gemm-minmax-fp32
    qu8-gemm-minmax-rndnu
    qu8-igemm-minmax-fp32
    qu8-igemm-minmax-rndnu)
FOREACH(TEST ${MICROKERNEL_GEMM_UNIT_TESTS})
  FILE(GLOB TEST_SOURCES "${TEST}*.cc")
  IF(TEST_SOURCES)
      ADD_EXECUTABLE(${TEST}-test ${TEST_SOURCES})
      TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
          gemm-microkernel-tester
          hardware-config
          logging
          microkernels-all
          microparams-init
          pthreadpool
          xnnpack-test)
      ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
      SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
  ENDIF()
ENDFOREACH()
SHARD_TEST(qd8-f16-qc8w-gemm-minmax-test 10)
SHARD_TEST(qd8-f32-qc4w-gemm-minmax-test 10)
SHARD_TEST(qd8-f32-qc8w-gemm-minmax-test 10)
SHARD_TEST(qd8-f32-qc8w-igemm-minmax-test 10)
SHARD_TEST(qs8-qc8w-gemm-minmax-fp32-test 10)
SHARD_TEST(qs8-qc8w-igemm-minmax-fp32-test 10)

SET(MICROKERNEL_PACKQ_UNIT_TESTS
    x8-packq)
FOREACH(TEST ${MICROKERNEL_PACKQ_UNIT_TESTS})
  ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
  TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      packq-microkernel-tester
      hardware-config
      logging
      microkernels-all
      microparams-init
      pthreadpool
      xnnpack-test)
  ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
  SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
ENDFOREACH()

SET(MICROKERNEL_VBINARY_UNIT_TESTS
    f16-vadd
    f16-vaddc
    f16-vdiv
    f16-vdivc
    f16-vmax
    f16-vmaxc
    f16-vmin
    f16-vminc
    f16-vmul
    f16-vmulc
    f16-vprelu
    f16-vpreluc
    f16-vrpreluc
    f16-vrdivc
    f16-vrsubc
    f16-vsqrdiff
    f16-vsqrdiffc
    f16-vsub
    f16-vsubc
    f32-vadd
    f32-vaddc
    f32-vcopysign
    f32-vcopysignc
    f32-vdiv
    f32-vdivc
    f32-vmax
    f32-vmaxc
    f32-vmin
    f32-vminc
    f32-vmul
    f32-vmulc
    f32-vprelu
    f32-vpreluc
    f32-vrpreluc
    f32-vrcopysignc
    f32-vrdivc
    f32-vrsubc
    f32-vsqrdiff
    f32-vsqrdiffc
    f32-vsub
    f32-vsubc
    qs8-vadd-minmax
    qs8-vaddc-minmax
    qs8-vmul-minmax-fp32
    qs8-vmulc-minmax-fp32
    qs8-vprelu
    qs8-vpreluc
    qs8-vrpreluc
    qu8-vadd-minmax
    qu8-vaddc-minmax
    qu8-vmul-minmax-fp32
    qu8-vmul-minmax-rndnu
    qu8-vmulc-minmax-fp32
    qu8-vmulc-minmax-rndnu
    qu8-vprelu
    qu8-vpreluc
    qu8-vrpreluc)
FOREACH(TEST ${MICROKERNEL_VBINARY_UNIT_TESTS})
  ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
  TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      vbinary-microkernel-tester
      hardware-config
      logging
      microkernels-all
      microparams-init
      pthreadpool
      xnnpack-test)
  ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
  SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
ENDFOREACH()

SET(MICROKERNEL_VCVT_TESTS
    f16-f32-vcvt
    f16-qs8-vcvt
    f16-qu8-vcvt
    f32-f16-vcvt
    f32-qs8-vcvt
    f32-qu8-vcvt
    qs8-f16-vcvt
    qs8-f32-vcvt
    qs8-vcvt
    qu8-f32-vcvt
    qu8-vcvt)
LIST(REMOVE_ITEM MICROKERNEL_VCVT_TESTS ${SHARDED_TESTS})
FOREACH(TEST ${MICROKERNEL_VCVT_TESTS})
  ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
  TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      datatype
      hardware-config
      logging
      microkernels-all
      microparams-init
      pthreadpool
      xnnpack-test)
  ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
  SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
ENDFOREACH()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^riscv")
  SHARD_TEST(f16-qs8-vcvt-test 10)
  SHARD_TEST(f16-qu8-vcvt-test 10)
  SHARD_TEST(f32-qs8-vcvt-test 10)
  SHARD_TEST(f32-qu8-vcvt-test 10)
  SHARD_TEST(qs8-f32-vcvt-test 10)
  SHARD_TEST(qu8-f32-vcvt-test 10)
  SHARD_TEST(qs8-vcvt-test 10)
  SHARD_TEST(qu8-vcvt-test 10)
ENDIF()

SET(MICROKERNEL_VUNARY_TESTS
    f16-vabs
    f16-vapproxgelu
    f16-vclamp
    f16-velu
    f16-vgelu
    f16-vhswish
    f16-vlrelu
    f16-vneg
    f16-vrndd
    f16-vrndne
    f16-vrndu
    f16-vrndz
    f16-vrsqrt
    f16-vsigmoid
    f16-vsqr
    f16-vsqrt
    f16-vtanh
    f32-vabs
    f32-vclamp
    f32-vcos
    f32-velu
    f32-vexp
    f32-vgelu
    f32-vhswish
    f32-vlog
    f32-vlrelu
    f32-vneg
    f32-vrelu
    f32-vrndd
    f32-vrndne
    f32-vrndu
    f32-vrndz
    f32-vrsqrt
    f32-vsigmoid
    f32-vsin
    f32-vsqr
    f32-vsqrt
    f32-vtanh
    qs8-vlrelu
    s8-vclamp
    u8-vclamp)
LIST(REMOVE_ITEM MICROKERNEL_VUNARY_TESTS ${SHARDED_TESTS})
FOREACH(TEST ${MICROKERNEL_VUNARY_TESTS})
  ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
  TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      hardware-config
      logging
      microkernels-all
      microparams-init
      pthreadpool
      next-prime
      xnnpack-test)
  ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
  SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
ENDFOREACH()
IF(XNNPACK_TARGET_PROCESSOR MATCHES "^riscv")
  SHARD_TEST(f32-vclamp-test 10)
  SHARD_TEST(f32-velu-test 10)
  SHARD_TEST(f32-vlrelu-test 10)
  SHARD_TEST(qs8-vlrelu-test 10)
  SHARD_TEST(qu8-vlrelu-test 10)
  SHARD_TEST(s8-vclamp-test 10)
  SHARD_TEST(u8-vclamp-test 10)
ENDIF()

# Special-case
IF(XNNPACK_ENABLE_KLEIDIAI)
  TARGET_LINK_LIBRARIES(x8-packq-test PRIVATE kleidiai)
ENDIF()

ADD_EXECUTABLE(buffer-test buffer.cc)
TARGET_LINK_LIBRARIES(buffer-test PRIVATE
  xnnpack-test
  datatype)
ADD_TEST(NAME buffer-test COMMAND buffer-test)
SET_TARGET_PROPERTIES(buffer-test PROPERTIES CXX_EXTENSIONS YES)

IF(XNNPACK_BUILD_LIBRARY)
  ADD_EXECUTABLE(weights-cache-test weights-cache.cc)
  TARGET_LINK_LIBRARIES(weights-cache-test PRIVATE XNNPACK pthreadpool GTest::gtest GTest::gtest_main)

  ADD_EXECUTABLE(mutex-test mutex.cc)
  TARGET_LINK_LIBRARIES(mutex-test PRIVATE GTest::gtest GTest::gtest_main pthreadpool)
  TARGET_LINK_LIBRARIES(mutex-test PRIVATE logging mutex)

  ADD_EXECUTABLE(microkernel-utils-test microkernel-utils.cc)
  TARGET_LINK_LIBRARIES(microkernel-utils-test PRIVATE microkernel-utils hardware-config logging GTest::gtest GTest::gtest_main pthreadpool)
ENDIF()
