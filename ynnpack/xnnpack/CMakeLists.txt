# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_xnnpack_utils STATIC utils.cc)
TARGET_LINK_LIBRARIES(ynnpack_xnnpack_utils PUBLIC ynnpack ynnpack_base ynnpack_subgraph pthreadpool)

ADD_LIBRARY(ynnpack_dynamic_quantization STATIC dynamic_quantization.cc)
TARGET_LINK_LIBRARIES(ynnpack_dynamic_quantization PUBLIC
  ynnpack_xnnpack_utils
  ynnpack
  ynnpack_base
  ynnpack_kernels_dot
  ynnpack_subgraph
  pthreadpool
  slinky_builder
  slinky_runtime
)

ADD_LIBRARY(ynnpack_as_xnnpack STATIC
  deprecated.cc
  experimental.cc
  operator.cc
  runtime.cc
  subgraph.cc
  tensor.cc
  weights_cache.cc
  workspace.cc
)

TARGET_COMPILE_DEFINITIONS(ynnpack_as_xnnpack PUBLIC XNNPACK_USING_YNNPACK=1)
TARGET_LINK_LIBRARIES(ynnpack_as_xnnpack PUBLIC
  ynnpack_dynamic_quantization
  ynnpack_xnnpack_utils
  ynnpack
  ynnpack_base
  ynnpack_subgraph
  pthreadpool
  slinky_thread_pool_impl
)

IF(YNNPACK_BUILD_TESTS)
  ADD_EXECUTABLE(ynnpack_dynamic_quantization_test dynamic_quantization_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_dynamic_quantization_test ynnpack_dynamic_quantization ynnpack ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random ynnpack_base_test_tensor ynnpack_base_test_util ynnpack_subgraph_test_builder gtest gtest_main)
  ADD_TEST(NAME ynnpack_dynamic_quantization_test COMMAND ynnpack_dynamic_quantization_test)
ENDIF()
