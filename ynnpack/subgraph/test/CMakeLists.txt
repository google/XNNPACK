# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_subgraph_test_scheduler STATIC scheduler.cc)
TARGET_LINK_LIBRARIES(ynnpack_subgraph_test_scheduler PUBLIC ynnpack ynnpack_base slinky_thread_pool_impl gtest)

ADD_LIBRARY(ynnpack_subgraph_test_builder STATIC subgraph_builder.cc)
TARGET_LINK_LIBRARIES(ynnpack_subgraph_test_builder PUBLIC
  ynnpack_subgraph_test_scheduler
  ynnpack
  ynnpack_base
  ynnpack_subgraph
  gtest
)

FUNCTION(YNN_ADD_SUBGRAPH_TEST NAME)
  ADD_EXECUTABLE(ynnpack_subgraph_${NAME}_test ${NAME}.cc)
  TARGET_LINK_LIBRARIES(ynnpack_subgraph_${NAME}_test
    ynnpack
    ynnpack_base
    ynnpack_subgraph
    ynnpack_subgraph_test_builder
    ynnpack_base_test_tensor
    gmock
    gtest
    gtest_main
  )
  ADD_TEST(NAME ynnpack_subgraph_${NAME}_test COMMAND ynnpack_subgraph_${NAME}_test)
ENDFUNCTION()

YNN_ADD_SUBGRAPH_TEST(broadcast_like)
YNN_ADD_SUBGRAPH_TEST(concatenate)
YNN_ADD_SUBGRAPH_TEST(copy)
YNN_ADD_SUBGRAPH_TEST(errors)
YNN_ADD_SUBGRAPH_TEST(even_split)
YNN_ADD_SUBGRAPH_TEST(fuse_dim)
YNN_ADD_SUBGRAPH_TEST(fuse_dims)
YNN_ADD_SUBGRAPH_TEST(get_tensor_shape)
YNN_ADD_SUBGRAPH_TEST(lut)
YNN_ADD_SUBGRAPH_TEST(reduce)
