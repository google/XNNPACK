# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

IF(XNNPACK_BUILD_LIBRARY)
  # ---[ Build end-to-end microbenchmarks
  ADD_LIBRARY(models STATIC
    fp32-attention.cc
    fp32-depthwise-separable.cc
    fp32-elementwise.cc
    fp32-layer-norm.cc
    fp32-mobilenet-v1.cc
    fp32-mobilenet-v2.cc
    fp32-mobilenet-v3-large.cc
    fp32-mobilenet-v3-small.cc
    fp32-softmax.cc
    qd8-attention.cc
    qs8-mobilenet-v2.cc)
  SET_TARGET_PROPERTIES(models PROPERTIES CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(models PRIVATE XNNPACK)

  ADD_EXECUTABLE(bench-models benchmark.cc)
  TARGET_LINK_LIBRARIES(bench-models PRIVATE
    bench-utils
    benchmark::benchmark
    models
    XNNPACK)
ENDIF()
