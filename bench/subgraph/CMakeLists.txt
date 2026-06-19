# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019-2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

IF(XNNPACK_BUILD_LIBRARY)
  # ---[ Build end-to-end microbenchmarks
  ADD_LIBRARY(models STATIC
    fp32-mobilenet-v1.cc
    fp32-mobilenet-v2.cc
    fp32-mobilenet-v3-large.cc
    fp32-mobilenet-v3-small.cc
    fp32-transformer.cc
    qd8-transformer.cc
    qs8-mobilenet-v2.cc)
  SET_TARGET_PROPERTIES(models PROPERTIES CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(models PRIVATE XNNPACK xnnpack-test)

  ADD_LIBRARY(subgraph-benchmark STATIC
    benchmark.cc)
  SET_TARGET_PROPERTIES(subgraph-benchmark PROPERTIES CXX_EXTENSIONS YES)
  TARGET_LINK_LIBRARIES(subgraph-benchmark PRIVATE
    benchmark::benchmark
    xnnpack-bench-utils
    xnnpack-test
    XNNPACK)

  SET(SUBGRAPH_BENCHMARKS
      attention
      depthwise-separable
      elementwise
      fully-connected
      layer-norm
      mobilenet
      softmax
      transformer)
  FOREACH(BENCH ${SUBGRAPH_BENCHMARKS})
    ADD_EXECUTABLE(subgraph-${BENCH}-bench ${BENCH}.cc)
    TARGET_LINK_LIBRARIES(subgraph-${BENCH}-bench PRIVATE
      xnnpack-test
      xnnpack-bench-utils
      benchmark::benchmark
      subgraph-benchmark
      models
      XNNPACK)
    SET_TARGET_PROPERTIES(subgraph-${BENCH}-bench PROPERTIES CXX_EXTENSIONS YES)
  ENDFOREACH()
ENDIF()
