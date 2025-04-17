# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

IF(XNNPACK_BUILD_LIBRARY)
  # ---[ Build operator-level microbenchmarks
  SET(LIBRARY_OPERATOR_BENCHMARKS
      average-pooling
      batch-matrix-multiply
      binary
      convolution
      deconvolution
      fully-connected
      max-pooling
      prelu
      resize-bilinear-nhwc
      softmax
      unary)
  FOREACH(BENCH ${LIBRARY_OPERATOR_BENCHMARKS})
    ADD_EXECUTABLE(${BENCH}-bench ${BENCH}.cc)
    TARGET_LINK_LIBRARIES(${BENCH}-bench PRIVATE
      bench-utils
      benchmark::benchmark
      datatype
      XNNPACK
    )
  ENDFOREACH()
ENDIF()
