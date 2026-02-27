// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::AllOf;
using ::testing::Not;

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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::multiply_add), HasInputCount(3)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::subtract_multiply),
                    InputsInclude(a_id, b_id)));
}

TEST(fusion, multiply_multiply) {
  // multiply(mul(a, b), c) -> multiply(a, b, c)
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
      .AddBinary(ynn_binary_multiply, ab_id, c_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::multiply),
                    InputsInclude(a_id, b_id, c_id)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::subtract_multiply),
                    InputsAre(a_id, b_id, c_id)));
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

  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsBinary(ynn_binary_multiply), InputsAre(a_id, scale_id)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::multiply),
                    InputsAre(a_id, scale1_id, scale2_id)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::quantize_int8),
                    InputsAre(a_id, scale_id, zero_point_id)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::quantize_uint8),
                    InputsAre(a_id, scale_id, zero_point_id)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(4)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsTernary(ternary_op::clamp), InputsAre(a_id, b_id, c_id)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
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

  EXPECT_THAT(ProducerOf(z_id, subgraph),
              AllOf(IsStencilCopy(std::vector<ynn_node::stencil_copy::stencil>{
                        {/*axis=*/1, /*new_axis=*/2, /*extent=*/3,
                         /*stride=*/1, /*dilation=*/1}}),
                    InputsAre(y_id, YNN_INVALID_VALUE_ID)));

  // Check transposed m_dim. Original was 1. Inserted dim at 0.
  // Note: YNNPACK uses Slinky dimension ordering (innermost first).
  // Input [10, 20]. Slinky dims: 0->20, 1->10.
  // Output [1, 10, 20]. Slinky dims: 0->20, 1->10, 2->1.
  // Insertion at API 0 corresponds to Slinky dim 2.
  // m_dim was 1 (size 10).
  // new_axis (2) > m_dim (1), so m_dim is not decremented.
  EXPECT_THAT(ProducerOf(y_id, subgraph),
              AllOf(IsTransposeA(/*tile_k=*/4, /*m_dim=*/1), InputsAre(x_id)));
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
  EXPECT_THAT(ProducerOf(z_id, subgraph),
              AllOf(IsTransposeA(/*tile_k=*/4, /*m_dim=*/1), InputsAre(y_id)));

  EXPECT_THAT(ProducerOf(y_id, subgraph),
              AllOf(IsStencilCopy(std::vector<ynn_node::stencil_copy::stencil>{
                        {/*axis=*/0, /*new_axis=*/1, /*extent=*/1,
                         /*stride=*/1, /*dilation=*/1},
                        {/*axis=*/1, /*new_axis=*/3, /*extent=*/3,
                         /*stride=*/1, /*dilation=*/1}}),
                    InputsAre(x_id, YNN_INVALID_VALUE_ID)));
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

  ASSERT_THAT(subgraph,
              AllOf(HasValidNodeCount(1), HasValidValueIds(x_id, y_id),
                    Not(HasValidValueId(converted_x_id))));
  EXPECT_THAT(ProducerOf(y_id, subgraph),
              AllOf(IsReduce(op), HasInputCount(2),
                    InputsAre(x_id, IsValidValueIn(subgraph))));
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
  EXPECT_THAT(ProducerOf(y_id, subgraph), InputsInclude(converted_x_id));
  EXPECT_THAT(ProducerOf(converted_x_id, subgraph), InputsInclude(x_id));
  EXPECT_THAT(subgraph, HasValidValueIds(x_id, y_id, converted_x_id));
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

  // x and y should be valid, sq should be invalid/removed.
  ASSERT_THAT(subgraph,
              AllOf(HasValidNodeCount(1), HasValidValueIds(x_id, y_id),
                    Not(HasValidValueIds(sq_id))));
  EXPECT_THAT(ProducerOf(y_id, subgraph),
              AllOf(IsReduce(ynn_reduce_sum_squared), HasInputCount(2),
                    InputsAre(x_id, IsValidValueIn(subgraph))));
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

  // x and y should be valid. Intermediate values should be invalid/removed.
  ASSERT_THAT(subgraph,
              AllOf(HasValidNodeCount(1), HasValidValueIds(x_id, y_id),
                    Not(HasValidValueIds(intermediate1_id, intermediate2_id))));
  EXPECT_THAT(ProducerOf(y_id, subgraph),
              AllOf(IsReduce(ynn_reduce_sum_squared), HasInputCount(2),
                    InputsAre(x_id, IsValidValueIn(subgraph))));
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

TEST(fusion, reduce_sum_of_squared_blocked_by_non_copy) {
  // reduce_sum(stencil_copy(add(x*x, bias))) should NOT be rewritten because
  // add is not a copy node.
  const uint32_t x_id = 0;
  const uint32_t bias_id = 1;
  const uint32_t y_id = 2;
  SubgraphBuilder builder(3);

  uint32_t sq_id = YNN_INVALID_VALUE_ID;
  uint32_t added_id = YNN_INVALID_VALUE_ID;
  uint32_t stencil_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, 1, x_id)
      .AddInput(ynn_type_fp32, 1, bias_id)
      .AddOutput(ynn_type_fp32, 1, y_id)
      .AddTensor(ynn_type_fp32, 1, sq_id)
      .AddTensor(ynn_type_fp32, 1, added_id)
      .AddTensor(ynn_type_fp32, 2, stencil_id);

  builder.AddBinary(ynn_binary_multiply, x_id, x_id, sq_id)
      .AddBinary(ynn_binary_add, sq_id, bias_id, added_id)
      .AddStencilCopy({0}, {1}, {3}, {3}, {1}, added_id, YNN_INVALID_VALUE_ID,
                      stencil_id)
      .AddReduce(ynn_reduce_sum, {1}, stencil_id, YNN_INVALID_VALUE_ID, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The reduce should still be reduce_sum (not rewritten).
  EXPECT_THAT(ProducerOf(y_id, subgraph), IsReduce(ynn_reduce_sum));
}

TEST(fusion, reduce_sum_of_squared_blocked_by_nontrivial_padding) {
  // reduce_sum(stencil_copy(static_pad(x*x, 1.0))) should NOT be rewritten
  // because the padding value is not 0 or 1.
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  SubgraphBuilder builder(2);

  uint32_t sq_id = YNN_INVALID_VALUE_ID;
  uint32_t padded_id = YNN_INVALID_VALUE_ID;
  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  const uint32_t padding_val_id = builder.DefineScalar(2.0f);

  builder.AddInput(ynn_type_fp32, 1, x_id)
      .AddOutput(ynn_type_fp32, 1, y_id)
      .AddTensor(ynn_type_fp32, 1, sq_id)
      .AddTensor(ynn_type_fp32, 1, padded_id)
      .AddTensor(ynn_type_fp32, 2, stencil_id);

  builder.AddBinary(ynn_binary_multiply, x_id, x_id, sq_id)
      .AddPad({0}, {0}, {2}, sq_id, padding_val_id, padded_id)
      .AddStencilCopy({0}, {1}, {3}, {3}, {1}, padded_id, YNN_INVALID_VALUE_ID,
                      stencil_id)
      .AddReduce(ynn_reduce_sum, {1}, stencil_id, YNN_INVALID_VALUE_ID, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The reduce should still be reduce_sum (not rewritten).
  EXPECT_THAT(ProducerOf(y_id, subgraph), IsReduce(ynn_reduce_sum));
}

TEST(fusion, reduce_sum_of_squared_blocked_by_stencil_copy_nontrivial_padding) {
  // reduce_sum(stencil_copy(x*x, 1.0)) should NOT be rewritten
  // because the padding value is not 0 or 1.
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  SubgraphBuilder builder(2);

  uint32_t sq_id = YNN_INVALID_VALUE_ID;
  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  const uint32_t padding_val_id = builder.DefineScalar(2.0f);

  builder.AddInput(ynn_type_fp32, 1, x_id)
      .AddOutput(ynn_type_fp32, 1, y_id)
      .AddTensor(ynn_type_fp32, 1, sq_id)
      .AddTensor(ynn_type_fp32, 2, stencil_id);

  builder.AddBinary(ynn_binary_multiply, x_id, x_id, sq_id)
      .AddStencilCopy({0}, {1}, {3}, {3}, {1}, sq_id, padding_val_id,
                      stencil_id)
      .AddReduce(ynn_reduce_sum, {1}, stencil_id, YNN_INVALID_VALUE_ID, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The reduce should still be reduce_sum (not rewritten).
  EXPECT_THAT(ProducerOf(y_id, subgraph), IsReduce(ynn_reduce_sum));
}

TEST(fusion, reduce_sum_of_squared_windowed) {
  // reduce_sum(stencil_copy(static_pad(x*x))) ->
  //   reduce_sum_squared(stencil_copy(static_pad(x)))
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  SubgraphBuilder builder(2);

  uint32_t sq_id = YNN_INVALID_VALUE_ID;
  uint32_t padded_id = YNN_INVALID_VALUE_ID;
  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  const uint32_t padding_val_id = builder.DefineScalar(0.0f);

  builder.AddInput(ynn_type_fp32, 1, x_id)
      .AddOutput(ynn_type_fp32, 1, y_id)
      .AddTensor(ynn_type_fp32, 1, sq_id)
      .AddTensor(ynn_type_fp32, 1, padded_id)
      .AddTensor(ynn_type_fp32, 2, stencil_id);

  builder.AddBinary(ynn_binary_multiply, x_id, x_id, sq_id)
      .AddPad({0}, {0}, {2}, sq_id, padding_val_id, padded_id)
      .AddStencilCopy({0}, {1}, {3}, {3}, {1}, padded_id, YNN_INVALID_VALUE_ID,
                      stencil_id)
      .AddReduce(ynn_reduce_sum, {1}, stencil_id, YNN_INVALID_VALUE_ID, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The multiply node and sq_id should be removed.
  ASSERT_THAT(subgraph,
              AllOf(Not(HasValidValueId(sq_id)),
                    HasValidValueIds(x_id, y_id, padded_id, stencil_id)));
  // The reduce should now be reduce_sum_squared.
  EXPECT_THAT(ProducerOf(y_id, subgraph), IsReduce(ynn_reduce_sum_squared));
  // The pad should now take x_id as input (not sq_id).
  EXPECT_THAT(ProducerOf(padded_id, subgraph), InputsInclude(x_id));
}

}  // namespace ynn
