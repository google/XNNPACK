// Copyright 2026 Google LLC
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

using ::testing::AllOf;
using ::testing::Not;

TEST(eliminate_common_subgraphs, simple_binary) {
  // Graph:
  // input_id + input_id -> temp1_id -> output1_id
  // input_id + input_id -> temp2_id -> output2_id
  //
  // Graph after eliminate_common_subgraphs:
  // input_id + input_id -> temp1_id -> output1_id
  //                                  \ -> output2_id
  const uint32_t input_id = 0;
  const uint32_t output1_id = 1;
  const uint32_t output2_id = 2;
  uint32_t temp1_id = YNN_INVALID_VALUE_ID;
  uint32_t temp2_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output1_id)
      .AddOutput(type_of<float>(), 1, output2_id)
      .AddTensor(type_of<float>(), 1, temp1_id)
      .AddTensor(type_of<float>(), 1, temp2_id);

  // temp1 = input + input.
  builder.AddBinary(ynn_binary_add, input_id, input_id, temp1_id);
  // temp2 = input + input (should be eliminate_common_subgraphs'd).
  builder.AddBinary(ynn_binary_add, input_id, input_id, temp2_id);
  builder.AddCopy(temp1_id, output1_id);
  builder.AddCopy(temp2_id, output2_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  ASSERT_THAT(subgraph, HasValidNodeCount(4));  // 2 add, 2 copy.

  subgraph.eliminate_common_subgraphs();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(
      subgraph,
      AllOf(HasValidNodeCount(3),  // 1 add, 2 copy.
            HasValidValueIds(input_id, temp1_id, output1_id, output2_id),
            Not(HasValidValueId(temp2_id))));
  EXPECT_THAT(ProducerOf(output2_id, subgraph), InputsAre(temp1_id));
  EXPECT_THAT(ProducerOf(output1_id, subgraph), InputsAre(temp1_id));
}

TEST(eliminate_common_subgraphs, chained) {
  // Graph:
  // input_id -> (negate) -> b1_id -> (abs) -> c1_id -> (copy) -> output1_id
  //           \ -> (negate) -> b2_id -> (abs) -> c2_id -> (copy) -> output2_id
  //
  // Graph after eliminate_common_subgraphs:
  // input_id -> (negate) -> b1_id -> (abs) c1_id -> (copy) -> output1_id
  //                                               \ -> (copy) -> output2_id
  const uint32_t input_id = 0;
  const uint32_t output1_id = 1;
  const uint32_t output2_id = 2;
  uint32_t b1_id = YNN_INVALID_VALUE_ID;
  uint32_t b2_id = YNN_INVALID_VALUE_ID;
  uint32_t c1_id = YNN_INVALID_VALUE_ID;
  uint32_t c2_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output1_id)
      .AddOutput(type_of<float>(), 1, output2_id)
      .AddTensor(type_of<float>(), 1, b1_id)
      .AddTensor(type_of<float>(), 1, b2_id)
      .AddTensor(type_of<float>(), 1, c1_id)
      .AddTensor(type_of<float>(), 1, c2_id);

  builder.AddUnary(ynn_unary_negate, input_id, b1_id);
  builder.AddUnary(ynn_unary_negate, input_id, b2_id);
  builder.AddUnary(ynn_unary_abs, b1_id, c1_id);
  builder.AddUnary(ynn_unary_abs, b2_id, c2_id);
  builder.AddCopy(c1_id, output1_id);
  builder.AddCopy(c2_id, output2_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  ASSERT_THAT(subgraph, HasValidNodeCount(6));  // 2 negate, 2 abs, 2 copy.

  subgraph.eliminate_common_subgraphs();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(4),  // 1 negate, 1 abs, 2 copy.
                              HasValidValueIds(input_id, b1_id, c1_id,
                                               output1_id, output2_id),
                              Not(HasValidValueIds(b2_id, c2_id))));
  EXPECT_THAT(ProducerOf(output1_id, subgraph), InputsAre(c1_id));
  EXPECT_THAT(ProducerOf(output2_id, subgraph), InputsAre(c1_id));
}

TEST(eliminate_common_subgraphs, diamond_reconvergence) {
  // Graph:
  // input_id -> (abs) -> a_id -> (negate) -> b_id
  //                            \ -> (negate) -> c_id
  // b_id + b_id -> d_id -> (copy) -> output1_id
  // c_id + c_id -> e_id -> (copy) -> output2_id
  //
  // Graph after eliminate_common_subgraphs:
  // input_id -> (abs) -> a_id -> (negate) -> b_id
  // b_id + b_id -> d_id -> (copy) -> output1_id
  //                          \ -> (copy) -> output2_id
  const uint32_t input_id = 0;
  const uint32_t output1_id = 1;
  const uint32_t output2_id = 2;
  uint32_t a_id = YNN_INVALID_VALUE_ID;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t c_id = YNN_INVALID_VALUE_ID;
  uint32_t d_id = YNN_INVALID_VALUE_ID;
  uint32_t e_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output1_id)
      .AddOutput(type_of<float>(), 1, output2_id)
      .AddTensor(type_of<float>(), 1, a_id)
      .AddTensor(type_of<float>(), 1, b_id)
      .AddTensor(type_of<float>(), 1, c_id)
      .AddTensor(type_of<float>(), 1, d_id)
      .AddTensor(type_of<float>(), 1, e_id);

  builder.AddUnary(ynn_unary_abs, input_id, a_id);
  builder.AddUnary(ynn_unary_negate, a_id, b_id);
  builder.AddUnary(ynn_unary_negate, a_id, c_id);
  builder.AddBinary(ynn_binary_add, b_id, b_id, d_id);
  builder.AddBinary(ynn_binary_add, c_id, c_id, e_id);
  builder.AddCopy(d_id, output1_id);
  builder.AddCopy(e_id, output2_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  // Nodes: Abs, Negate, Negate, Add, Add, Copy, Copy = 7 nodes.
  ASSERT_THAT(subgraph, HasValidNodeCount(7));

  subgraph.eliminate_common_subgraphs();
  subgraph.invalidate_dead_values();

  // Expected: Abs, Negate, Add, Copy, Copy = 5 nodes.
  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(5),
                              HasValidValueIds(input_id, a_id, b_id, d_id,
                                               output1_id, output2_id),
                              Not(HasValidValueIds(c_id, e_id))));
  EXPECT_THAT(ProducerOf(output1_id, subgraph), InputsAre(d_id));
  EXPECT_THAT(ProducerOf(output2_id, subgraph), InputsAre(d_id));
}

TEST(eliminate_common_subgraphs, no_merge_different_ops) {
  // Graph:
  // input_id + input_id -> t1_id -> output1_id
  // input_id * input_id -> t2_id -> output2_id
  //
  // Graph after eliminate_common_subgraphs:
  // input_id + input_id -> t1_id -> output1_id
  // input_id * input_id -> t2_id -> output2_id
  const uint32_t input_id = 0;
  const uint32_t output1_id = 1;
  const uint32_t output2_id = 2;
  uint32_t t1_id = YNN_INVALID_VALUE_ID;
  uint32_t t2_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output1_id)
      .AddOutput(type_of<float>(), 1, output2_id)
      .AddTensor(type_of<float>(), 1, t1_id)
      .AddTensor(type_of<float>(), 1, t2_id);

  builder.AddBinary(ynn_binary_add, input_id, input_id, t1_id);
  builder.AddBinary(ynn_binary_multiply, input_id, input_id, t2_id);
  builder.AddCopy(t1_id, output1_id);
  builder.AddCopy(t2_id, output2_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  ASSERT_THAT(subgraph, HasValidNodeCount(4));  // Add, Mul, Copy, Copy.

  subgraph.eliminate_common_subgraphs();
  subgraph.invalidate_dead_values();

  // No change.
  EXPECT_THAT(subgraph, AllOf(HasValidNodeCount(4),
                              HasValidValueIds(input_id, t1_id, t2_id,
                                               output1_id, output2_id)));
}

TEST(eliminate_common_subgraphs, no_merge_different_inputs) {
  // Graph:
  // input1_id + input1_id -> t1_id -> output1_id
  // input1_id + input2_id -> t2_id -> output2_id
  //
  // Graph after eliminate_common_subgraphs:
  // input1_id + input1_id -> t1_id -> output1_id
  // input1_id + input2_id -> t2_id -> output2_id
  const uint32_t input1_id = 0;
  const uint32_t input2_id = 1;
  const uint32_t output1_id = 2;
  const uint32_t output2_id = 3;
  uint32_t t1_id = YNN_INVALID_VALUE_ID;
  uint32_t t2_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(4);
  builder.AddInput(type_of<float>(), 1, input1_id)
      .AddInput(type_of<float>(), 1, input2_id)
      .AddOutput(type_of<float>(), 1, output1_id)
      .AddOutput(type_of<float>(), 1, output2_id)
      .AddTensor(type_of<float>(), 1, t1_id)
      .AddTensor(type_of<float>(), 1, t2_id);

  builder.AddBinary(ynn_binary_add, input1_id, input1_id, t1_id);
  builder.AddBinary(ynn_binary_add, input1_id, input2_id, t2_id);
  builder.AddCopy(t1_id, output1_id);
  builder.AddCopy(t2_id, output2_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  ASSERT_THAT(subgraph, HasValidNodeCount(4));  // Add, Add, Copy, Copy.

  subgraph.eliminate_common_subgraphs();
  subgraph.invalidate_dead_values();

  // No change.
  EXPECT_THAT(subgraph, AllOf(HasValidNodeCount(4),
                              HasValidValueIds(input1_id, input2_id, t1_id,
                                               t2_id, output1_id, output2_id)));
}

}  // namespace ynn
