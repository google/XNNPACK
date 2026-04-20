// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::ElementsAre;
using ::testing::Not;

TEST(fold_constants, simple) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t constant_id = 2;
  uint32_t temp_id = YNN_INVALID_VALUE_ID;

  constexpr float constant_values[] = {0.0f, 1.0f, 2.0f, 3.0f};
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output_id)
      .AddTensor(type_of<float>(), {4}, constant_id, &constant_values)
      .AddTensor(type_of<float>(), 1, temp_id);

  const uint32_t one_id = builder.DefineScalar(1.0f);
  builder.AddBinary(ynn_binary_add, constant_id, one_id, temp_id)
      .AddBinary(ynn_binary_add, input_id, temp_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  const ynn_value& constant = subgraph->value(temp_id);
  ASSERT_THAT(subgraph, HasValidValueId(constant_id));
  EXPECT_TRUE(constant.is_static());
  EXPECT_THAT(ValuesIn<float>(constant), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  ASSERT_THAT(subgraph, HasValidNodeCount(1));

  subgraph->invalidate_dead_values();
  EXPECT_THAT(subgraph, Not(HasValidValueId(constant_id)));
}

TEST(fold_constants, transitive) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t constant_id = 2;
  uint32_t temp_id = YNN_INVALID_VALUE_ID;
  uint32_t temp2_id = YNN_INVALID_VALUE_ID;

  constexpr float constant_values[] = {0.0f, 1.0f, 2.0f, 3.0f};
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output_id)
      .AddTensor(type_of<float>(), {4}, constant_id, &constant_values)
      .AddTensor(type_of<float>(), 1, temp_id)
      .AddTensor(type_of<float>(), 1, temp2_id);

  const uint32_t one_id = builder.DefineScalar(1.0f);
  const uint32_t two_id = builder.DefineScalar(2.0f);
  builder.AddBinary(ynn_binary_multiply, constant_id, two_id, temp2_id)
      .AddBinary(ynn_binary_add, temp2_id, one_id, temp_id)
      .AddBinary(ynn_binary_add, input_id, temp_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  const ynn_value& constant = subgraph->value(temp_id);
  ASSERT_THAT(subgraph, HasValidValueId(constant_id));
  EXPECT_TRUE(constant.is_static());
  EXPECT_THAT(ValuesIn<float>(constant), ElementsAre(1.0f, 3.0f, 5.0f, 7.0f));
  ASSERT_THAT(subgraph, HasValidNodeCount(1));

  subgraph->invalidate_dead_values();
  EXPECT_THAT(subgraph, Not(HasValidValueId(constant_id)));
  EXPECT_THAT(subgraph, Not(HasValidValueId(temp2_id)));
}

TEST(fold_constants, cheap_node_not_folded) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t expanded_id = 2;

  SubgraphBuilder builder(3);
  uint32_t scalar_id = builder.DefineScalar(1.0f);
  builder.AddInput(type_of<float>(), {4}, input_id)
      .AddOutput(type_of<float>(), {4}, output_id)
      .AddTensor(type_of<float>(), 2, expanded_id);

  // constant -> reshape (cheap) -> add (non-foldable)
  // temp_cheap_id should not be folded because it is cheap and feeds into a
  // non-foldable node.
  builder.AddExpandDims({0, 1}, scalar_id, expanded_id)
      .AddBinary(ynn_binary_add, input_id, expanded_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_FALSE(subgraph->value(expanded_id).is_static());
  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

TEST(fold_constants, cheap_node_folded_in_chain) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t constant_id = 2;
  uint32_t temp_cheap_id = YNN_INVALID_VALUE_ID;
  uint32_t temp_expensive_id = YNN_INVALID_VALUE_ID;

  constexpr float constant_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), {2, 2}, input_id)
      .AddOutput(type_of<float>(), {2, 2}, output_id)
      .AddTensor(type_of<float>(), {4}, constant_id, &constant_values)
      .AddTensor(type_of<float>(), {2, 2}, temp_cheap_id)
      .AddTensor(type_of<float>(), {2, 2}, temp_expensive_id);

  const uint32_t one_id = builder.DefineScalar(1.0f);

  // constant -> reshape (cheap) -> add (expensive) -> add (non-foldable)
  // Both reshape and the first add should be folded.
  builder.AddReshape({2, 2}, constant_id, temp_cheap_id)
      .AddBinary(ynn_binary_add, temp_cheap_id, one_id, temp_expensive_id)
      .AddBinary(ynn_binary_add, input_id, temp_expensive_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_TRUE(subgraph->value(temp_expensive_id).is_static());
  EXPECT_TRUE(subgraph->value(temp_cheap_id).is_static());
  EXPECT_THAT(subgraph, HasValidNodeCount(1));
}

TEST(fold_constants, expensive_node_folded_before_cheap_node) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t constant_id = 2;
  uint32_t temp_expensive_id = YNN_INVALID_VALUE_ID;
  uint32_t temp_cheap_id = YNN_INVALID_VALUE_ID;

  constexpr float constant_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), {4}, input_id)
      .AddOutput(type_of<float>(), {4}, output_id)
      .AddTensor(type_of<float>(), {4}, constant_id, &constant_values)
      .AddTensor(type_of<float>(), {4}, temp_expensive_id)
      .AddTensor(type_of<float>(), {4}, temp_cheap_id);

  const uint32_t one_id = builder.DefineScalar(1.0f);

  // constant -> add (expensive) -> reshape (cheap) -> add (non-foldable)
  // add (expensive) should be folded, but reshape (cheap) should not.
  builder.AddBinary(ynn_binary_add, constant_id, one_id, temp_expensive_id)
      .AddReshape({4}, temp_expensive_id, temp_cheap_id)
      .AddBinary(ynn_binary_add, input_id, temp_cheap_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_TRUE(subgraph->value(temp_expensive_id).is_static());
  EXPECT_FALSE(subgraph->value(temp_cheap_id).is_static());
  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

TEST(fold_constants, expensive_node_folded_before_multiple_cheap_nodes) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t constant_id = 2;
  uint32_t temp_expensive_id = YNN_INVALID_VALUE_ID;
  uint32_t temp_cheap1_id = YNN_INVALID_VALUE_ID;
  uint32_t temp_cheap2_id = YNN_INVALID_VALUE_ID;

  constexpr float constant_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), {1, 1, 4}, input_id)
      .AddOutput(type_of<float>(), {1, 1, 4}, output_id)
      .AddTensor(type_of<float>(), {4}, constant_id, &constant_values)
      .AddTensor(type_of<float>(), {4}, temp_expensive_id)
      .AddTensor(type_of<float>(), {1, 1, 4}, temp_cheap1_id)
      .AddTensor(type_of<float>(), {1, 1, 4}, temp_cheap2_id);

  const uint32_t one_id = builder.DefineScalar(1.0f);

  // constant -> add (expensive) -> expand_dims (cheap) -> broadcast (cheap) ->
  // add (non-foldable) add (expensive) should be folded, but expand_dims and
  // broadcast should not.
  builder.AddBinary(ynn_binary_add, constant_id, one_id, temp_expensive_id)
      .AddExpandDims({0, 1}, temp_expensive_id, temp_cheap1_id)
      .AddBroadcast(std::vector<size_t>{1, 1, 4}, temp_cheap1_id,
                    temp_cheap2_id)
      .AddBinary(ynn_binary_add, input_id, temp_cheap2_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_TRUE(subgraph->value(temp_expensive_id).is_static());
  EXPECT_FALSE(subgraph->value(temp_cheap1_id).is_static());
  EXPECT_FALSE(subgraph->value(temp_cheap2_id).is_static());
  EXPECT_THAT(subgraph, HasValidNodeCount(3));
}

TEST(fold_constants, scalar_fp16_to_fp32_folded) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t fp16_id = YNN_INVALID_VALUE_ID;
  uint32_t fp32_id = YNN_INVALID_VALUE_ID;

  half constant_value = 1.0f;
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), {}, input_id)
      .AddOutput(type_of<float>(), {}, output_id)
      .AddTensor(type_of<half>(), {}, fp16_id, &constant_value)
      .AddTensor(type_of<float>(), {}, fp32_id);

  builder.AddConvert(fp16_id, type_of<float>(), fp32_id)
      .AddBinary(ynn_binary_add, input_id, fp32_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_TRUE(subgraph->value(fp32_id).is_static());
  EXPECT_THAT(ValuesIn<float>(subgraph->value(fp32_id)), ElementsAre(1.0f));
  EXPECT_THAT(subgraph, HasValidNodeCount(1));
}

TEST(fold_constants, vector_fp16_to_fp32_not_folded) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t fp16_id = YNN_INVALID_VALUE_ID;
  uint32_t fp32_id = YNN_INVALID_VALUE_ID;

  half constant_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), {4}, input_id)
      .AddOutput(type_of<float>(), {4}, output_id)
      .AddTensor(type_of<half>(), {4}, fp16_id, &constant_values)
      .AddTensor(type_of<float>(), {4}, fp32_id);

  builder.AddConvert(fp16_id, type_of<float>(), fp32_id)
      .AddBinary(ynn_binary_add, input_id, fp32_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_FALSE(subgraph->value(fp32_id).is_static());
  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

}  // namespace ynn
