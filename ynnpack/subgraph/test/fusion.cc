// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstdint>
#include <variant>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

int valid_value_count(ynn_subgraph_t subgraph) {
  return std::count_if(subgraph->values.begin(), subgraph->values.end(),
                       [](const ynn_value& value) { return value.is_valid(); });
}

int valid_node_count(ynn_subgraph_t subgraph) {
  return std::count_if(subgraph->nodes.begin(), subgraph->nodes.end(),
                       [](const ynn_node& node) { return node.is_valid(); });
}

TEST(fusion, multiply_add) {
  // rewrite add(multiply(a, b), c) -> multiply_add(a, b, c)
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t c_id = 2;
  const uint32_t x_id = 3;
  SubgraphBuilder builder(4);
  uint32_t ab_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 2, b_id)
      .AddInput(ynn_type_fp32, 2, c_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, 2, ab_id);
  builder.AddBinary(ynn_binary_multiply, a_id, b_id, ab_id)
      .AddBinary(ynn_binary_add, ab_id, c_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 4);
  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 3);
  ASSERT_TRUE(std::get_if<ynn_node::opaque>(&output->op));
}

TEST(fusion, negate_multiply) {
  // negate(mul(a, b)) -> subtract_multiply(0, a, b)
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 3;
  SubgraphBuilder builder(4);
  uint32_t ab_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_int32, 2, a_id)
      .AddInput(ynn_type_int32, 2, b_id)
      .AddOutput(ynn_type_int32, 2, x_id)
      .AddTensor(ynn_type_int32, 2, ab_id);
  builder.AddBinary(ynn_binary_multiply, a_id, b_id, ab_id)
      .AddUnary(ynn_unary_negate, ab_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 4);
  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 3);
  ASSERT_TRUE(std::get_if<ynn_node::opaque>(&output->op));
}

TEST(fusion, subtract_multiply) {
  // subtract(a, mul(b, c) -> subtract_multiply(a, b, c)
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t c_id = 2;
  const uint32_t x_id = 3;
  SubgraphBuilder builder(4);
  uint32_t bc_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_int32, 2, a_id)
      .AddInput(ynn_type_int32, 2, b_id)
      .AddInput(ynn_type_int32, 2, c_id)
      .AddOutput(ynn_type_int32, 2, x_id)
      .AddTensor(ynn_type_int32, 2, bc_id);
  builder.AddBinary(ynn_binary_multiply, b_id, c_id, bc_id)
      .AddBinary(ynn_binary_subtract, a_id, bc_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 4);
  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 3);
  ASSERT_TRUE(std::get_if<ynn_node::opaque>(&output->op));
}

TEST(fusion, convert_int32_to_fp32_binary) {
  // convert(a) -> a*scale if a has no zero point.
  const uint32_t a_id = 0;
  const uint32_t scale_id = 1;
  const uint32_t x_id = 2;
  const uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_fp32, 2, scale_id)
      .AddInput(ynn_type_int32, 2, a_id, zero_point_id, scale_id)
      .AddOutput(ynn_type_fp32, 2, x_id);
  builder.AddUnary(ynn_unary_convert, a_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 2);
  ASSERT_TRUE(std::get_if<ynn_node::binary_elementwise>(&output->op));
}

TEST(fusion, convert_int32_to_fp32_ternary) {
  // convert(a) -> a*scale1*scale2 if a has no zero point.
  const uint32_t a_id = 0;
  const uint32_t scale1_id = 1;
  const uint32_t scale2_id = 2;
  const uint32_t x_id = 3;
  const uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder builder(4);
  builder.AddTensor(ynn_type_fp32, 2, scale_id);
  builder.AddInput(ynn_type_fp32, 2, scale1_id)
      .AddInput(ynn_type_fp32, 2, scale2_id)
      .AddInput(ynn_type_int32, 2, a_id, zero_point_id, scale_id)
      .AddOutput(ynn_type_fp32, 2, x_id);
  builder.AddBinary(ynn_binary_multiply, scale1_id, scale2_id, scale_id)
      .AddUnary(ynn_unary_convert, a_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 4);
  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 3);
  ASSERT_TRUE(std::get_if<ynn_node::opaque>(&output->op));
}

TEST(fusion, convert_fp32_to_int8_ternary) {
  // convert(a) -> quantize(a, scale, zero_point)
  const uint32_t a_id = 0;
  const uint32_t scale_id = 1;
  const uint32_t x_id = 2;
  const uint32_t zero_point_id = 3;
  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_fp32, 2, scale_id)
      .AddInput(ynn_type_int32, 2, zero_point_id)
      .AddInput(ynn_type_fp32, 2, a_id)
      .AddOutput(ynn_type_int8, 2, x_id, zero_point_id, scale_id);
  builder.AddUnary(ynn_unary_convert, a_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 4);
  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 3);
  ASSERT_TRUE(std::get_if<ynn_node::opaque>(&output->op));
}

TEST(fusion, convert_fp32_to_uint8_ternary) {
  // convert(a) -> quantize(a, scale, zero_point)
  const uint32_t a_id = 0;
  const uint32_t scale_id = 1;
  const uint32_t x_id = 2;
  const uint32_t zero_point_id = 3;
  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_fp32, 2, scale_id)
      .AddInput(ynn_type_int32, 2, zero_point_id)
      .AddInput(ynn_type_fp32, 2, a_id)
      .AddOutput(ynn_type_uint8, 2, x_id, zero_point_id, scale_id);
  builder.AddUnary(ynn_unary_convert, a_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 4);
  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 3);
  ASSERT_TRUE(std::get_if<ynn_node::opaque>(&output->op));
}

TEST(fusion, clamp_min_max) {
  // rewrite min(max(a, b), c)
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t c_id = 2;
  const uint32_t x_id = 3;
  SubgraphBuilder builder(4);
  uint32_t max_ab_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 2, b_id)
      .AddInput(ynn_type_fp32, 2, c_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, 2, max_ab_id);
  builder.AddBinary(ynn_binary_max, a_id, b_id, max_ab_id)
      .AddBinary(ynn_binary_min, max_ab_id, c_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 4);
  const ynn_node* output = subgraph.get_producer(x_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 3);
  ASSERT_EQ(output->inputs[0], a_id);
  ASSERT_EQ(output->inputs[1], b_id);
  ASSERT_EQ(output->inputs[2], c_id);
  ASSERT_TRUE(std::get_if<ynn_node::opaque>(&output->op));
}

TEST(fusion, broadcast_of_static) {
  // rewrite a + broadcast(b) -> a + b if b is static.
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t broadcast_b_id = YNN_INVALID_VALUE_ID;
  static const float value[] = {1.0f};
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, {10, 1}, b_id, value)
      .AddTensor(ynn_type_fp32, 2, broadcast_b_id);
  builder.AddBroadcastLike({0, 1}, b_id, a_id, broadcast_b_id)
      .AddBinary(ynn_binary_min, a_id, broadcast_b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_EQ(valid_value_count(&subgraph), 3);
}

}  // namespace ynn
