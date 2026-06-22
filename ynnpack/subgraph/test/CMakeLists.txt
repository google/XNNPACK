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
YNN_ADD_SUBGRAPH_TEST(dynamic_quantization)
YNN_ADD_SUBGRAPH_TEST(errors)
YNN_ADD_SUBGRAPH_TEST(even_split)
YNN_ADD_SUBGRAPH_TEST(fuse_dim)
YNN_ADD_SUBGRAPH_TEST(fuse_dims)
YNN_ADD_SUBGRAPH_TEST(gather)
YNN_ADD_SUBGRAPH_TEST(get_tensor_shape)
YNN_ADD_SUBGRAPH_TEST(iota)
YNN_ADD_SUBGRAPH_TEST(reduce)
YNN_ADD_SUBGRAPH_TEST(reduce_dot)
YNN_ADD_SUBGRAPH_TEST(reduce_window)
YNN_ADD_SUBGRAPH_TEST(runtime)
YNN_ADD_SUBGRAPH_TEST(slice_like)
YNN_ADD_SUBGRAPH_TEST(split_dim)
YNN_ADD_SUBGRAPH_TEST(stack)
YNN_ADD_SUBGRAPH_TEST(static_broadcast)
YNN_ADD_SUBGRAPH_TEST(static_expand_dims)
YNN_ADD_SUBGRAPH_TEST(static_pad)
YNN_ADD_SUBGRAPH_TEST(static_reshape)
YNN_ADD_SUBGRAPH_TEST(static_slice)
YNN_ADD_SUBGRAPH_TEST(static_transpose)
YNN_ADD_SUBGRAPH_TEST(stencil_copy)
YNN_ADD_SUBGRAPH_TEST(utils)
YNN_ADD_SUBGRAPH_TEST(polynomial)
YNN_ADD_SUBGRAPH_TEST(partial_reduce_dot)
YNN_ADD_SUBGRAPH_TEST(sub_byte_types)

YNN_ADD_SUBGRAPH_TEST(binary)
TARGET_LINK_LIBRARIES(ynnpack_subgraph_binary_test ynnpack_kernels_binary_reference)

YNN_ADD_SUBGRAPH_TEST(unary)
TARGET_LINK_LIBRARIES(ynnpack_subgraph_unary_test ynnpack_kernels_unary_reference)

YNN_ADD_SUBGRAPH_TEST(dot)
YNN_ADD_SUBGRAPH_TEST(gelu)
YNN_ADD_SUBGRAPH_TEST(subgraphs)
YNN_ADD_SUBGRAPH_TEST(eliminate_common_subgraphs)
YNN_ADD_SUBGRAPH_TEST(fold_constants)
YNN_ADD_SUBGRAPH_TEST(invalidate_dead_values)
YNN_ADD_SUBGRAPH_TEST(slinky_thread_pool)

ADD_EXECUTABLE(ynnpack_subgraph_fusion_test
  fusion_arithmetic.cc
  fusion_copy.cc
  fusion_lut.cc
  fusion_reduce.cc
)
TARGET_LINK_LIBRARIES(ynnpack_subgraph_fusion_test
  ynnpack
  ynnpack_base
  ynnpack_subgraph
  ynnpack_subgraph_test_builder
  ynnpack_base_test_tensor
  ynnpack_kernels_ternary
  gmock
  gtest
  gtest_main
)
ADD_TEST(NAME ynnpack_subgraph_fusion_test COMMAND ynnpack_subgraph_fusion_test)

IF(YNNPACK_BUILD_BENCHMARKS)
  ADD_EXECUTABLE(ynnpack_subgraph_dot_bench dot_bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_subgraph_dot_bench ynnpack ynnpack_base ynnpack_subgraph_test_scheduler benchmark::benchmark)

  ADD_EXECUTABLE(ynnpack_subgraph_pi_bench pi_bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_subgraph_pi_bench ynnpack ynnpack_base ynnpack_subgraph ynnpack_subgraph_test_scheduler benchmark::benchmark_main)

  ADD_EXECUTABLE(ynnpack_subgraph_slinky_bench slinky_bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_subgraph_slinky_bench ynnpack_subgraph benchmark::benchmark_main slinky_runtime)
ENDIF()
