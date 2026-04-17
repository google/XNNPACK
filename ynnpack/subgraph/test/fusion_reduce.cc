// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::AllOf;

TEST(fusion, reduce_reduce_keep_dims) {
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 6, a_id)
      .AddOutput(ynn_type_fp32, 6, x_id)
      .AddTensor(ynn_type_fp32, 6, v1_id);

  // reduce axes {1} then {2}, both keep_dims=true
  builder
      .AddReduce(ynn_reduce_sum, {1, 3}, a_id, YNN_INVALID_VALUE_ID, v1_id,
                 YNN_NODE_FLAG_KEEP_DIMS)
      .AddReduce(ynn_reduce_sum, {2, 4}, v1_id, YNN_INVALID_VALUE_ID, x_id,
                 YNN_NODE_FLAG_KEEP_DIMS);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, AllOf(IsReduce(ynn_reduce_sum), InputsInclude(a_id)));
  const auto& reduce = std::get<ynn_node::reduce>(node.op);
  EXPECT_EQ(reduce.k_dims, 0b011110);
  EXPECT_TRUE(reduce.keep_dims);
}

TEST(fusion, reduce_reduce_no_keep_dims) {
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 6, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, 4, v1_id);

  // reduce axis {1} then {1} (which was axis 2), both keep_dims=false
  builder
      .AddReduce(ynn_reduce_sum, {1, 2}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddReduce(ynn_reduce_sum, {1, 2}, v1_id, YNN_INVALID_VALUE_ID, x_id, 0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, AllOf(IsReduce(ynn_reduce_sum), InputsInclude(a_id)));
  const auto& reduce = std::get<ynn_node::reduce>(node.op);
  EXPECT_EQ(reduce.k_dims, 0b011110);
  EXPECT_FALSE(reduce.keep_dims);
}

TEST(fusion, reduce_sum_squared_sum) {
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 4, a_id)
      .AddOutput(ynn_type_fp32, 4, x_id)
      .AddTensor(ynn_type_fp32, 4, v1_id);

  builder
      .AddReduce(ynn_reduce_sum_squared, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id,
                 YNN_NODE_FLAG_KEEP_DIMS)
      .AddReduce(ynn_reduce_sum, {2}, v1_id, YNN_INVALID_VALUE_ID, x_id,
                 YNN_NODE_FLAG_KEEP_DIMS);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node,
              AllOf(IsReduce(ynn_reduce_sum_squared), InputsInclude(a_id)));
}

TEST(fusion, reduce_reduce_keep_dims_mismatch) {
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 4, a_id)
      .AddOutput(ynn_type_fp32, 3, x_id)
      .AddTensor(ynn_type_fp32, 4, v1_id);

  // reduce axis {1} keep_dims=true, then {2} keep_dims=false
  builder
      .AddReduce(ynn_reduce_sum, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id,
                 YNN_NODE_FLAG_KEEP_DIMS)
      .AddReduce(ynn_reduce_sum, {2}, v1_id, YNN_INVALID_VALUE_ID, x_id, 0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // Should NOT fuse.
  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

TEST(fusion, reduce_reduce_non_identity_init) {
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  uint32_t init2_id = builder.DefineScalar(1.0f);
  builder.AddInput(ynn_type_fp32, 4, a_id)
      .AddOutput(ynn_type_fp32, 4, x_id)
      .AddTensor(ynn_type_fp32, 4, v1_id);

  builder
      .AddReduce(ynn_reduce_sum, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id,
                 YNN_NODE_FLAG_KEEP_DIMS)
      .AddReduce(ynn_reduce_sum, {2}, v1_id, init2_id, x_id,
                 YNN_NODE_FLAG_KEEP_DIMS);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // Should NOT fuse because init2 is 1.0f, not identity 0.0f.
  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

}  // namespace ynn
