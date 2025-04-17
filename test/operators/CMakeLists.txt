# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(convolution-test-helpers STATIC convolution-test-helpers.cc)
TARGET_LINK_LIBRARIES(convolution-test-helpers PRIVATE xnnpack-test xnnpack-base)

IF(XNNPACK_BUILD_LIBRARY)
  # ---[ Build operator-level unit tests
  SET(LIBRARY_OPERATOR_TESTS
    argmax-pooling-nhwc
    average-pooling-nhwc
    batch-matrix-multiply-nc
    binary-elementwise-nd
    constant-pad-nd-eager
    constant-pad-nd
    convolution-nchw
    convolution-nhwc
    copy-nc-eager
    copy-nc
    deconvolution-nhwc-qd8-f32-qc8w
    deconvolution-nhwc
    depth-to-space-nchw2nhwc
    depth-to-space-nhwc
    dynamic-fully-connected-nc
    fully-connected-nc
    max-pooling-nhwc
    reduce-nd
    resize-bilinear-nchw
    resize-bilinear-nhwc
    rope-nthc
    slice-nd-eager
    slice-nd
    softmax-nc
    space-to-depth-nhwc
    transpose-nd-eager
    transpose-nd
    unary-elementwise-nc
    unpooling-nhwc
  )
  FOREACH(TEST ${LIBRARY_OPERATOR_TESTS})
    ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
    TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
      convolution-test-helpers
      hardware-config
      logging
      microkernels-all
      microparams-init
      next-prime
      pthreadpool
      unary-ops
      xnnpack-test)
    ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
    SET_PROPERTY(TEST ${TEST}-test PROPERTY LABELS "operator")
  ENDFOREACH()
  SHARD_TEST(average-pooling-nhwc-test 5)
  SHARD_TEST(binary-elementwise-nd-test 10)
  SHARD_TEST(batch-matrix-multiply-nc-test 5)
  SHARD_TEST(constant-pad-nd-test 5)
  SHARD_TEST(constant-pad-nd-eager-test 5)
  SHARD_TEST(convolution-nchw-test 10)
  SHARD_TEST(convolution-nhwc-test 10)
  SHARD_TEST(deconvolution-nhwc-test 10)
  SHARD_TEST(fully-connected-nc-test 10)
  SHARD_TEST(max-pooling-nhwc-test 10)
  SHARD_TEST(slice-nd-test 5)
  SHARD_TEST(transpose-nd-test 10)
  SHARD_TEST(transpose-nd-eager-test 5)
ENDIF()

# ---[ Normalization unit tests
SET(NORMALIZATION_UNIT_TESTS
    slice-normalization
    reduce-normalization
    transpose-normalization)
FOREACH(TEST ${NORMALIZATION_UNIT_TESTS})
  ADD_EXECUTABLE(${TEST}-test ${TEST}.cc)
  TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE xnnpack-test pthreadpool normalization)
  ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
ENDFOREACH()

IF(XNNPACK_BUILD_LIBRARY)
  ADD_EXECUTABLE(operator-utils-test operator-utils.cc)
  TARGET_LINK_LIBRARIES(operator-utils-test PRIVATE XNNPACK GTest::gtest GTest::gtest_main pthreadpool)
ENDIF()
