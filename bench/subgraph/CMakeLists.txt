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
    fp32-l2-norm.cc
    fp32-layer-norm.cc
    fp32-mobilenet-v1.cc
    fp32-mobilenet-v2.cc
    fp32-mobilenet-v3-large.cc
    fp32-mobilenet-v3-small.cc
    fp32-softmax.cc
    fp32-transformer.cc
    qd8-attention.cc
    qd8-transformer.cc
    qs8-mobilenet-v2.cc)
  SET_TARGET_PROPERTIES(models PROPERTIES CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(models PRIVATE XNNPACK)

  ADD_LIBRARY(model_runtime STATIC
    model_runtime.cc)
  SET_TARGET_PROPERTIES(model_runtime PROPERTIES CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(model_runtime PRIVATE
    bench-utils
    benchmark::benchmark
    XNNPACK)

  ADD_EXECUTABLE(bench-models benchmark.cc)
  TARGET_LINK_LIBRARIES(bench-models PRIVATE
    bench-utils
    benchmark::benchmark
    models
    model_runtime
    XNNPACK)

  ADD_EXECUTABLE(bench-fully-connected fully_connected.cc)
  TARGET_LINK_LIBRARIES(bench-fully-connected PRIVATE
    bench-utils
    benchmark::benchmark
    model_runtime
    next-prime
    XNNPACK)
ENDIF()
