// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ::testing::ElementsAre;
using ::testing::IsSupersetOf;

bool operator==(const ynn_node::stencil_copy::stencil& a,
                const ynn_node::stencil_copy::stencil& b) {
  return a.axis == b.axis && a.new_axis == b.new_axis && a.extent == b.extent &&
         a.stride == b.stride && a.dilation == b.dilation;
}

namespace ynn {

int valid_value_count(ynn_subgraph_t subgraph) {
  return std::count_if(subgraph->values.begin(), subgraph->values.end(),
                       [](const ynn_value& value) { return value.is_valid(); });
}

int valid_node_count(ynn_subgraph_t subgraph) {
  return std::count_if(subgraph->nodes.begin(), subgraph->nodes.end(),
                       [](const ynn_node& node) { return node.is_valid(); });
}

bool is_binary(const ynn_node& node, ynn_binary_operator op) {
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&node.op);
  return binary && binary->op == op;
}

bool is_ternary(const ynn_node& node, ternary_op op) {
  const ynn_node::ternary_elementwise* ternary =
      std::get_if<ynn_node::ternary_elementwise>(&node.op);
  return ternary && ternary->op == op;
}

bool is_reduce(const ynn_node& node, ynn_reduce_operator op) {
  const ynn_node::reduce* reduce = std::get_if<ynn_node::reduce>(&node.op);
  return reduce && reduce->op == op;
}

bool is_stencil_copy(
    const ynn_node& node,
    const std::vector<ynn_node::stencil_copy::stencil>& stencils) {
  const ynn_node::stencil_copy* stencil_copy =
      std::get_if<ynn_node::stencil_copy>(&node.op);
  return stencil_copy && stencil_copy->stencils == stencils;
}

bool is_transpose_a(const ynn_node& node, int tile_k, int m_dim) {
  const ynn_node::transpose_a* transpose_a =
      std::get_if<ynn_node::transpose_a>(&node.op);
  return transpose_a && transpose_a->tile_k == tile_k &&
         transpose_a->m_dim == m_dim;
}

bool is_lut(const ynn_node& node) {
  return std::holds_alternative<ynn_node::lut>(node.op);
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
  ASSERT_TRUE(is_ternary(*output, ternary_op::multiply_add));
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
  ASSERT_THAT(output->inputs, IsSupersetOf({a_id, b_id}));
  ASSERT_TRUE(is_ternary(*output, ternary_op::subtract_multiply));
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
  ASSERT_THAT(output->inputs, ElementsAre(a_id, b_id, c_id));
  ASSERT_TRUE(is_ternary(*output, ternary_op::subtract_multiply));
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
  ASSERT_THAT(output->inputs, ElementsAre(a_id, scale_id));
  ASSERT_TRUE(is_binary(*output, ynn_binary_multiply));
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
  ASSERT_THAT(output->inputs, ElementsAre(a_id, scale1_id, scale2_id));
  ASSERT_TRUE(is_ternary(*output, ternary_op::multiply));
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
  ASSERT_THAT(output->inputs, ElementsAre(a_id, scale_id, zero_point_id));
  ASSERT_TRUE(is_ternary(*output, ternary_op::quantize_int8));
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
  ASSERT_THAT(output->inputs, ElementsAre(a_id, scale_id, zero_point_id));
  ASSERT_TRUE(is_ternary(*output, ternary_op::quantize_uint8));
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
  ASSERT_THAT(output->inputs, ElementsAre(a_id, b_id, c_id));
  ASSERT_TRUE(is_ternary(*output, ternary_op::clamp));
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

TEST(fusion, transpose_stencil_copy) {
  // rewrite transpose_a(stencil_copy(x)) -> stencil_copy(transpose_a(x))
  const uint32_t x_id = 0;
  uint32_t y_id = 1;
  const uint32_t z_id = 2;
  SubgraphBuilder builder(3);

  builder.AddInput(ynn_type_fp32, {10, 20}, x_id)
      .AddTensor(ynn_type_fp32, 3, y_id)
      .AddOutput(ynn_type_fp32, 3, z_id);

  // stencil_copy: x -> y. Insert dim at axis 0.
  // x: [10, 20]. y: [3, 8, 20].
  builder.AddStencilCopy(
      /*stencil_axes=*/{0},
      /*new_axes=*/{0},
      /*stencil_dims=*/{3},
      /*stencil_strides=*/{1},
      /*stencil_dilations=*/{1}, x_id,
      /*padding_id=*/YNN_INVALID_VALUE_ID, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  // transpose_a: y -> z.
  // m_dim = 1. (Dimension of size 8).
  ynn_node transpose_node;
  transpose_node.op = ynn_node::transpose_a{
      .tile_k = 4,
      .m_dim = 1,
  };
  transpose_node.inputs = {y_id};
  transpose_node.outputs = {z_id};
  subgraph.add_node(transpose_node);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  const ynn_node* z_producer = subgraph.get_producer(z_id);
  ASSERT_NE(z_producer, nullptr);
  ASSERT_THAT(z_producer->inputs, ElementsAre(y_id, YNN_INVALID_VALUE_ID));
  ASSERT_TRUE(
      is_stencil_copy(*z_producer, {{/*axis=*/1, /*new_axis=*/2, /*extent=*/3,
                                     /*stride=*/1, /*dilation=*/1}}));

  // Check transposed m_dim. Original was 1. Inserted dim at 0.
  // Note: YNNPACK uses Slinky dimension ordering (innermost first).
  // Input [10, 20]. Slinky dims: 0->20, 1->10.
  // Output [1, 10, 20]. Slinky dims: 0->20, 1->10, 2->1.
  // Insertion at API 0 corresponds to Slinky dim 2.
  // m_dim was 1 (size 10).
  // new_axis (2) > m_dim (1), so m_dim is not decremented.
  const ynn_node* y_producer = subgraph.get_producer(y_id);
  ASSERT_NE(y_producer, nullptr);
  ASSERT_THAT(y_producer->inputs, ElementsAre(x_id));
  ASSERT_TRUE(is_transpose_a(*y_producer, /*tile_k=*/4, /*m_dim=*/1));
}

TEST(fusion, transpose_stencil_copy_grouped) {
  // rewrite transpose_a(stencil_copy(x)) -> stencil_copy(transpose_a(x))
  const uint32_t x_id = 0;
  uint32_t y_id = 1;
  const uint32_t z_id = 2;
  SubgraphBuilder builder(3);

  builder.AddInput(ynn_type_fp32, {10, 20}, x_id)
      .AddTensor(ynn_type_fp32, 4, y_id)
      .AddOutput(ynn_type_fp32, 4, z_id);

  // stencil_copy: x -> y. Insert stencil at axis 0, and a dummy dimension at
  // axis 2.
  // x: [10, 20]. y: [3, 8, 1, 20].
  builder.AddStencilCopy(
      /*stencil_axes=*/{0, 1},
      /*new_axes=*/{0, 2},
      /*stencil_dims=*/{3, 1},
      /*stencil_strides=*/{1, 1},
      /*stencil_dilations=*/{1, 1}, x_id,
      /*padding_id=*/YNN_INVALID_VALUE_ID, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  // transpose_a: y -> z.
  // m_dim = 2. (Dimension of size 1).
  ynn_node transpose_node;
  transpose_node.op = ynn_node::transpose_a{
      .tile_k = 4,
      .m_dim = 1,
  };
  transpose_node.inputs = {y_id};
  transpose_node.outputs = {z_id};
  subgraph.add_node(transpose_node);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The rewrite should not be applied in this case, because the transpose is of
  // a dimension that was created by the stencil copy.
  const ynn_node* z_producer = subgraph.get_producer(z_id);
  ASSERT_NE(z_producer, nullptr);
  ASSERT_THAT(z_producer->inputs, ElementsAre(y_id));
  ASSERT_TRUE(is_transpose_a(*z_producer, /*tile_k=*/4, /*m_dim=*/1));

  const ynn_node* y_producer = subgraph.get_producer(y_id);
  ASSERT_NE(y_producer, nullptr);
  ASSERT_THAT(y_producer->inputs, ElementsAre(x_id, YNN_INVALID_VALUE_ID));
  ASSERT_TRUE(
      is_stencil_copy(*y_producer, {{/*axis=*/0, /*new_axis=*/1, /*extent=*/1,
                                     /*stride=*/1, /*dilation=*/1},
                                    {/*axis=*/1, /*new_axis=*/3, /*extent=*/3,
                                     /*stride=*/1, /*dilation=*/1}}));
}

namespace {

void TestReduceSumOfConvert(ynn_type input_type, ynn_type intermediate_type,
                            ynn_reduce_operator op) {
  const uint32_t x_id = 0;
  uint32_t converted_x_id = 1;
  const uint32_t y_id = 2;
  SubgraphBuilder builder(3);
  builder.AddInput(input_type, 2, x_id)
      .AddTensor(intermediate_type, 2, converted_x_id)
      .AddOutput(intermediate_type, 1, y_id);
  builder.AddUnary(ynn_unary_convert, x_id, converted_x_id)
      .AddReduce(op, {1}, converted_x_id, YNN_INVALID_VALUE_ID, y_id, 0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  ASSERT_TRUE(subgraph.value(x_id).is_valid());
  ASSERT_TRUE(subgraph.value(y_id).is_valid());
  ASSERT_FALSE(subgraph.value(converted_x_id).is_valid());

  const ynn_node* output = subgraph.get_producer(y_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 2);
  ASSERT_EQ(output->inputs[0], x_id);
  ASSERT_TRUE(is_reduce(*output, op));
}

void TestReduceSumOfConvertQuantized(ynn_reduce_operator reduce_op) {
  const uint32_t x_id = 0;
  uint32_t converted_x_id = 1;
  const uint32_t y_id = 2;
  uint32_t scale_id = 3;
  uint32_t zero_point_id = 4;
  SubgraphBuilder builder(5);

  // Define scale and zero point.
  builder.AddTensor(ynn_type_fp32, {1}, scale_id, /*data=*/nullptr)
      .AddTensor(ynn_type_int32, {1}, zero_point_id, /*data=*/nullptr);

  // Input with quantization params.
  builder.AddInput(ynn_type_int8, 2, x_id, zero_point_id, scale_id);

  builder.AddTensor(ynn_type_int32, 2, converted_x_id)
      .AddOutput(ynn_type_int32, 1, y_id);
  builder.AddUnary(ynn_unary_convert, x_id, converted_x_id);

  ynn_node reduce_node;
  reduce_node.op = ynn_node::reduce{
      .k_dims = ynn::axes_set(2),  // Reduce axis 1.
      .op = reduce_op,
      .keep_dims = false,
  };
  reduce_node.inputs = {converted_x_id, YNN_INVALID_VALUE_ID};
  reduce_node.outputs = {y_id};

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.add_node(reduce_node);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // Should NOT fuse.
  // We expect the reduce node to still consume converted_x_id, not x_id.
  const ynn_node* output = subgraph.get_producer(y_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs[0], converted_x_id);

  const ynn_node* convert_node = subgraph.get_producer(converted_x_id);
  ASSERT_NE(convert_node, nullptr);
  ASSERT_EQ(convert_node->inputs[0], x_id);

  ASSERT_TRUE(subgraph.value(x_id).is_valid());
  ASSERT_TRUE(subgraph.value(y_id).is_valid());
  ASSERT_TRUE(subgraph.value(converted_x_id).is_valid());
}

}  // namespace

// reduce_sum tests.

TEST(fusion, reduce_sum_of_convert_fp16) {
  // reduce_sum(convert_fp32(x_fp16)) -> reduce_sum(x_fp16)
  TestReduceSumOfConvert(ynn_type_fp16, ynn_type_fp32, ynn_reduce_sum);
}

TEST(fusion, reduce_sum_of_convert_bf16) {
  // reduce_sum(convert_fp32(x_bf16)) -> reduce_sum(x_bf16)
  TestReduceSumOfConvert(ynn_type_bf16, ynn_type_fp32, ynn_reduce_sum);
}

TEST(fusion, reduce_sum_of_convert_int8) {
  // reduce_sum(convert_int32(x_int8)) -> reduce_sum(x_int8)
  TestReduceSumOfConvert(ynn_type_int8, ynn_type_int32, ynn_reduce_sum);
}

TEST(fusion, reduce_sum_of_convert_int8_quantized) {
  // reduce_sum(convert_int32(x_int8)) -> NO CHANGE if x_int8 has
  // quantization params.
  TestReduceSumOfConvertQuantized(ynn_reduce_sum);
}

// reduce_sum_squared tests.

TEST(fusion, reduce_sum_squared_of_convert_fp16) {
  // reduce_sum_squared(convert_fp32(x_fp16)) -> reduce_sum_squared(x_fp16)
  TestReduceSumOfConvert(ynn_type_fp16, ynn_type_fp32, ynn_reduce_sum_squared);
}

TEST(fusion, reduce_sum_squared_of_convert_bf16) {
  // reduce_sum_squared(convert_fp32(x_bf16)) -> reduce_sum_squared(x_bf16)
  TestReduceSumOfConvert(ynn_type_bf16, ynn_type_fp32, ynn_reduce_sum_squared);
}

TEST(fusion, reduce_sum_squared_of_convert_int8) {
  // reduce_sum_squared(convert_int32(x_int8)) -> reduce_sum_squared(x_int8)
  TestReduceSumOfConvert(ynn_type_int8, ynn_type_int32, ynn_reduce_sum_squared);
}

TEST(fusion, reduce_sum_squared_of_convert_int8_quantized) {
  // reduce_sum_squared(convert_int32(x_int8)) -> NO CHANGE if x_int8 has
  // quantization params.
  TestReduceSumOfConvertQuantized(ynn_reduce_sum_squared);
}

// reduce_sum -> reduce_sum_squared tests.

namespace {

void TestReduceSumOfSquared(ynn_type type) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  SubgraphBuilder builder(2);
  uint32_t sq_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(type, 2, x_id)
      .AddOutput(type, 1, y_id)
      .AddTensor(type, 2, sq_id);
  builder.AddBinary(ynn_binary_multiply, x_id, x_id, sq_id)
      .AddReduce(ynn_reduce_sum, {1}, sq_id, YNN_INVALID_VALUE_ID, y_id, 0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  // x and y should be valid, sq should be invalid/removed.
  ASSERT_TRUE(subgraph.value(x_id).is_valid());
  ASSERT_TRUE(subgraph.value(y_id).is_valid());
  ASSERT_FALSE(subgraph.value(sq_id).is_valid());

  const ynn_node* output = subgraph.get_producer(y_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 2);
  ASSERT_EQ(output->inputs[0], x_id);
  ASSERT_TRUE(is_reduce(*output, ynn_reduce_sum_squared));
}

}  // namespace

TEST(fusion, reduce_sum_of_squared_f32) {
  // reduce_sum(x_f32 * x_f32) -> reduce_sum_squared(x_f32)
  TestReduceSumOfSquared(ynn_type_fp32);
}

TEST(fusion, reduce_sum_of_squared_int32) {
  // reduce_sum(x_int32 * x_int32) -> reduce_sum_squared(x_int32)
  TestReduceSumOfSquared(ynn_type_int32);
}

namespace {

void TestReduceSumOfSquaredWithConvert(ynn_type input_type,
                                       ynn_type intermediate_type) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  uint32_t intermediate1_id = 2;
  uint32_t intermediate2_id = 3;

  SubgraphBuilder builder(4);
  builder.AddInput(input_type, 2, x_id).AddOutput(intermediate_type, 1, y_id);

  // x_id -> (convert) -> intermediate1_id -> (multiply) -> intermediate2_id
  // -> (reduce) -> y_id.
  builder.AddTensor(intermediate_type, 2, intermediate1_id)
      .AddTensor(intermediate_type, 2, intermediate2_id);
  builder.AddUnary(ynn_unary_convert, x_id, intermediate1_id)
      .AddBinary(ynn_binary_multiply, intermediate1_id, intermediate1_id,
                 intermediate2_id)
      .AddReduce(ynn_reduce_sum, {1}, intermediate2_id, YNN_INVALID_VALUE_ID,
                 y_id, 0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  // x and y should be valid. Intermediate values should be invalid/removed.
  ASSERT_TRUE(subgraph.value(x_id).is_valid());
  ASSERT_TRUE(subgraph.value(y_id).is_valid());
  ASSERT_FALSE(subgraph.value(intermediate1_id).is_valid());
  ASSERT_FALSE(subgraph.value(intermediate2_id).is_valid());

  const ynn_node* output = subgraph.get_producer(y_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->inputs.size(), 2);
  ASSERT_EQ(output->inputs[0], x_id);
  ASSERT_TRUE(is_reduce(*output, ynn_reduce_sum_squared));
}

}  // namespace

TEST(fusion, reduce_sum_of_squared_with_convert_fp16) {
  // reduce_sum(fp32(x_fp16) * fp32(x_fp16)) -> reduce_sum_squared(x_fp16).
  TestReduceSumOfSquaredWithConvert(ynn_type_fp16, ynn_type_fp32);
}

TEST(fusion, reduce_sum_of_squared_with_convert_bf16) {
  // reduce_sum(fp32(x_bf16) * fp32(x_bf16)) -> reduce_sum_squared(x_bf16).
  TestReduceSumOfSquaredWithConvert(ynn_type_bf16, ynn_type_fp32);
}

TEST(fusion, reduce_sum_of_squared_with_convert_int8) {
  // reduce_sum(int32(x_int8) * int32(x_int8)) -> reduce_sum_squared(x_int8).
  TestReduceSumOfSquaredWithConvert(ynn_type_int8, ynn_type_int32);
}

namespace {

template <typename A, typename X>
void RunSubgraph(ynn_subgraph_t subgraph,
                 const std::vector<uint32_t>& input_ids,
                 const std::vector<std::vector<A>>& input_datas,
                 const std::vector<ynn::TensorShape>& input_shapes,
                 uint32_t output_id, std::vector<X>& output_data,
                 bool optimize) {
  ynn::Runtime runtime(subgraph, nullptr, 0, optimize);
  ASSERT_EQ(input_ids.size(), input_datas.size());
  ASSERT_EQ(input_ids.size(), input_shapes.size());
  for (size_t i = 0; i < input_ids.size(); ++i) {
    runtime.ReshapeExternalTensor(
        input_shapes[i], const_cast<A*>(input_datas[i].data()), input_ids[i]);
  }
  runtime.SetupExternalTensor(output_data.data(), output_id);
  ASSERT_EQ(runtime.ReshapeRuntime().Status(), ynn_status_success);
  ASSERT_EQ(runtime.InvokeRuntime().Status(), ynn_status_success);
}

}  // namespace

TEST(fusion, unary_lut_single_unsupported) {
  // y = negate(x). Negate is not supported for single-node LUT replacement.
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  uint32_t scale_id = 2;
  uint32_t zero_point_id = 3;
  static const float scale_val[] = {1.0f};
  static const int32_t zero_point_val[] = {0};

  SubgraphBuilder builder(/*external_value_count=*/4);

  builder.AddTensor(ynn_type_fp32, {1}, scale_id, scale_val)
      .AddTensor(ynn_type_int32, {1}, zero_point_id, zero_point_val);

  builder.AddInput(ynn_type_int8, {256}, x_id, zero_point_id, scale_id)
      .AddOutput(ynn_type_int8, {256}, y_id, zero_point_id, scale_id);

  builder.AddUnary(ynn_unary_negate, x_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // Expect 2 nodes: one for `make_unary_params` (opaque) and one for `negate`
  // (unary).
  EXPECT_EQ(valid_node_count(&subgraph), 2);
  const ynn_node* output = subgraph.get_producer(y_id);
  ASSERT_NE(output, nullptr);
  ASSERT_FALSE(is_lut(*output));

  const ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&output->op);
  ASSERT_NE(unary, nullptr);
  ASSERT_EQ(unary->op, ynn_unary_negate);
}

TEST(fusion, unary_lut_single_simple) {
  // x -> sigmoid -> y.
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  uint32_t x_scale_id = 2;
  uint32_t x_zero_point_id = 3;
  uint32_t y_scale_id = 4;
  uint32_t y_zero_point_id = 5;
  static const float scale_val[] = {1.0f / 255.0f};
  static const int32_t zero_point_val[] = {-128};

  SubgraphBuilder builder(/*external_value_count=*/6);

  builder.AddTensor(ynn_type_fp32, {1}, x_scale_id, scale_val)
      .AddTensor(ynn_type_int32, {1}, x_zero_point_id, zero_point_val)
      .AddTensor(ynn_type_fp32, {1}, y_scale_id, scale_val)
      .AddTensor(ynn_type_int32, {1}, y_zero_point_id, zero_point_val);

  builder.AddInput(ynn_type_int8, 1, x_id, x_zero_point_id, x_scale_id)
      .AddOutput(ynn_type_int8, 1, y_id, y_zero_point_id, y_scale_id);

  builder.AddUnary(ynn_unary_sigmoid, x_id, y_id);

  std::vector<int8_t> input_data(256);
  std::iota(input_data.begin(), input_data.end(), -128);
  std::vector<int8_t> output_before(256);
  RunSubgraph<int8_t, int8_t>(builder.GetSubgraph(), {x_id}, {input_data},
                              {ynn::TensorShape({input_data.size()})}, y_id,
                              output_before,
                              /*optimize=*/false);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  const ynn_node* output = subgraph.get_producer(y_id);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(is_lut(*output));
  ASSERT_THAT(output->inputs,
              ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
  EXPECT_TRUE(subgraph.value(output->inputs[1]).is_valid());

  // Check that the rewritten subgraph includes quantization parameters.
  EXPECT_EQ(subgraph.value(x_id).scale_id, x_scale_id);
  EXPECT_EQ(subgraph.value(x_id).zero_point_id, x_zero_point_id);
  EXPECT_EQ(subgraph.value(y_id).scale_id, y_scale_id);
  EXPECT_EQ(subgraph.value(y_id).zero_point_id, y_zero_point_id);

  std::vector<int8_t> output_after(256);
  RunSubgraph<int8_t, int8_t>(builder.GetSubgraph(), {x_id}, {input_data},
                              {ynn::TensorShape({input_data.size()})}, y_id,
                              output_after,
                              /*optimize=*/true);
  EXPECT_EQ(output_after, output_before);
}

TEST(fusion, unary_lut_single) {
  // Similar to `unary_lut_single_simple`, but checks that the subgraphs before
  // and after the rewrite are the same.
  //
  // a * b -> x
  // x -> sigmoid -> y
  // y + c -> d
  uint32_t a_id = 0;
  uint32_t b_id = 1;
  uint32_t c_id = 2;
  uint32_t x_id = 3;
  uint32_t y_id = 4;
  uint32_t d_id = 5;
  uint32_t scale_id = 6;
  uint32_t zero_point_id = 7;

  static const float scale_val[] = {1.0f / 255.0f};
  static const int32_t zero_point_val[] = {-128};

  SubgraphBuilder builder(/*external_value_count=*/8);

  builder.AddTensor(ynn_type_fp32, {1}, scale_id, scale_val)
      .AddTensor(ynn_type_int32, {1}, zero_point_id, zero_point_val);

  builder.AddInput(ynn_type_int8, {256}, a_id)
      .AddInput(ynn_type_int8, {256}, b_id)
      .AddInput(ynn_type_int8, {256}, c_id)
      .AddOutput(ynn_type_int8, {256}, d_id, zero_point_id, scale_id);

  builder.AddTensor(ynn_type_int8, {256}, x_id)
      .AddTensor(ynn_type_int8, {256}, y_id);

  builder.AddBinary(ynn_binary_multiply, a_id, b_id, x_id)
      .AddUnary(ynn_unary_sigmoid, x_id, y_id)
      .AddBinary(ynn_binary_add, y_id, c_id, d_id);

  std::vector<int8_t> a_data(256);
  std::iota(a_data.begin(), a_data.end(), -128);
  std::vector<int8_t> b_data(256, 2);
  std::vector<int8_t> c_data(256, 3);
  std::vector<int8_t> output_before(256);

  RunSubgraph<int8_t, int8_t>(
      builder.GetSubgraph(), {a_id, b_id, c_id}, {a_data, b_data, c_data},
      {ynn::TensorShape({a_data.size()}), ynn::TensorShape({b_data.size()}),
       ynn::TensorShape({c_data.size()})},
      d_id, output_before,
      /*optimize=*/false);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 3);

  const ynn_node* x_producer = subgraph.get_producer(x_id);
  ASSERT_NE(x_producer, nullptr);
  ASSERT_TRUE(is_binary(*x_producer, ynn_binary_multiply));

  const ynn_node* y_producer = subgraph.get_producer(y_id);
  ASSERT_NE(y_producer, nullptr);
  ASSERT_TRUE(is_lut(*y_producer));
  ASSERT_THAT(y_producer->inputs,
              ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
  EXPECT_TRUE(subgraph.value(y_producer->inputs[1]).is_valid());

  const ynn_node* d_producer = subgraph.get_producer(d_id);
  ASSERT_NE(d_producer, nullptr);
  ASSERT_TRUE(is_binary(*d_producer, ynn_binary_add));

  std::vector<int8_t> output_after(256);
  RunSubgraph<int8_t, int8_t>(
      builder.GetSubgraph(), {a_id, b_id, c_id}, {a_data, b_data, c_data},
      {ynn::TensorShape({a_data.size()}), ynn::TensorShape({b_data.size()}),
       ynn::TensorShape({c_data.size()})},
      d_id, output_after,
      /*optimize=*/true);

  EXPECT_EQ(output_after, output_before);
}

TEST(fusion, unary_lut_elu_chain) {
  // Implements ELU operation:
  //
  // if x > 0, y = x.
  // if x <= 0, y = alpha * (exp(x) - 1).
  //
  // Note that this graph involves nodes with multiple consumers that eventually
  // feed into a single output node.
  //
  //
  // x -> (convert) -> a
  //
  //  a ->
  //      \ -> (min) -> b -> (expm1) -> c
  // 0.0f ->
  //
  //    c ->
  //        \ -> (multiply) -> e
  // d_const ->
  //
  // a ->
  //     \ -> (max) -> f -> (convert) -> y
  //   e ->
  //
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  uint32_t a_id = 2;
  uint32_t b_id = 3;
  uint32_t c_id = 4;
  uint32_t e_id = 5;
  uint32_t f_id = 6;
  uint32_t zero_id = 7;
  uint32_t alpha_const_id = 8;

  SubgraphBuilder builder(/*external_value_count=*/9);
  builder.AddInput(ynn_type_uint8, 3, x_id)
      .AddOutput(ynn_type_uint8, 3, y_id)
      // Intermediate tensors (fp32)
      .AddTensor(ynn_type_fp32, 3, a_id)
      .AddTensor(ynn_type_fp32, 3, b_id)
      .AddTensor(ynn_type_fp32, 3, c_id)
      .AddTensor(ynn_type_fp32, 3, e_id)
      .AddTensor(ynn_type_fp32, 3, f_id);

  static const float zero_val[] = {0.0f};
  static const float d_val[] = {
      1.0f};  // Value doesn't matter for fusion structure

  builder.AddTensor(ynn_type_fp32, {1}, zero_id, zero_val)
      .AddTensor(ynn_type_fp32, {1}, alpha_const_id, d_val);

  // Build the chain
  builder.AddUnary(ynn_unary_convert, x_id, a_id)
      .AddBinary(ynn_binary_min, a_id, zero_id, b_id)
      .AddUnary(ynn_unary_exp, b_id, c_id)
      .AddBinary(ynn_binary_multiply, c_id, alpha_const_id, e_id)
      .AddBinary(ynn_binary_max, a_id, e_id, f_id)
      .AddUnary(ynn_unary_convert, f_id, y_id);

  std::vector<uint8_t> input_data(256);
  std::iota(input_data.begin(), input_data.end(), 0);
  std::vector<uint8_t> output_before(256);
  RunSubgraph<uint8_t, uint8_t>(builder.GetSubgraph(), {x_id}, {input_data},
                                {ynn::TensorShape({4, 8, 8})}, y_id,
                                output_before,
                                /*optimize=*/false);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_EQ(valid_node_count(&subgraph), 1);
  const ynn_node* output = subgraph.get_producer(y_id);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(is_lut(*output));
  // LUT inputs: First is index (x), second is table (generated const).
  ASSERT_THAT(output->inputs,
              ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
  EXPECT_TRUE(subgraph.value(output->inputs[1]).is_valid());

  std::vector<uint8_t> output_after(256);
  RunSubgraph<uint8_t, uint8_t>(builder.GetSubgraph(), {x_id}, {input_data},
                                {ynn::TensorShape({4, 8, 8})}, y_id,
                                output_after,
                                /*optimize=*/true);
  EXPECT_EQ(output_after, output_before);
}

}  // namespace ynn
