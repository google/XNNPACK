# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test helper libraries
ADD_LIBRARY(runtime-flags STATIC runtime-flags.cc)
TARGET_LINK_LIBRARIES(runtime-flags PRIVATE xnnpack-test)

ADD_LIBRARY(subgraph-tester STATIC subgraph-tester.cc)
TARGET_LINK_LIBRARIES(subgraph-tester PRIVATE xnnpack-test pthreadpool runtime-flags)

IF(XNNPACK_BUILD_LIBRARY)
  # ---[ Build size tests
  ADD_EXECUTABLE(subgraph-size-test subgraph-size.c)
  TARGET_LINK_LIBRARIES(subgraph-size-test PRIVATE xnnpack-test)

  # ---[ Build subgraph optimizations unit tests
  SET(LIBRARY_SUBGRAPH_OPTIMIZATION_TESTS
      fusion
      memory-planner
      subgraph-fp16
      subgraph
      runtime
      subgraph-nchw
      persistent
      workspace)
  FOREACH(TEST ${LIBRARY_SUBGRAPH_OPTIMIZATION_TESTS})
    ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
    TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      runtime-flags
      subgraph-tester
      xnnpack-test)
    ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
  ENDFOREACH()

  # ---[ Build subgraph-level unit tests
  SET(LIBRARY_SUBGRAPH_UNIT_TESTS
      argmax-pooling-2d
      average-pooling-2d
      batch-matrix-multiply
      binary
      concatenate
      copy
      depth-to-space-2d
      depthwise-convolution-2d
      even-split
      fully-connected
      max-pooling-2d
      softmax
      space-to-depth-2d
      split-fuse
      static-constant-pad
      static-reduce
      static-reshape
      static-resize-bilinear-2d
      static-slice
      static-transpose
      unary
      unpooling-2d)
  FOREACH(TEST ${LIBRARY_SUBGRAPH_UNIT_TESTS})
    ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
    TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      datatype
      logging
      runtime-flags
      subgraph-tester
      unary-ops
      xnnpack-test)
    ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
    SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
  ENDFOREACH()
  IF(XNNPACK_TARGET_PROCESSOR MATCHES "^riscv")
    SHARD_TEST(static-reduce-test 20)
  ENDIF()

  SET(LIBRARY_SUBGRAPH_CONVOLUTION_UNIT_TESTS
      convolution-2d
      deconvolution-2d)
  FOREACH(TEST ${LIBRARY_SUBGRAPH_CONVOLUTION_UNIT_TESTS})
    ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
    TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      convolution-test-helpers
      xnnpack-test
      runtime-flags
      subgraph-tester
      xnnpack-test)
    ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
    SET_TARGET_PROPERTIES(${TEST}-test PROPERTIES CXX_EXTENSIONS YES)
  ENDFOREACH()
  SHARD_TEST(deconvolution-2d-test 10)
ENDIF()
