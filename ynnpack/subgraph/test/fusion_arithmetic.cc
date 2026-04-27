// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/subgraph/elementwise.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::AllOf;

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

TEST(fusion, divide_sqrt) {
  // rewrite x/sqrt(y) -> x*rsqrt(y)
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);
  uint32_t sqrt_y_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, x_id)
      .AddInput(ynn_type_fp32, 2, y_id)
      .AddOutput(ynn_type_fp32, 2, out_id)
      .AddTensor(ynn_type_fp32, 2, sqrt_y_id);
  builder.AddUnary(ynn_unary_square_root, y_id, sqrt_y_id)
      .AddBinary(ynn_binary_divide, x_id, sqrt_y_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(2), HasValidValueCount(4)));
  EXPECT_THAT(
      ProducerOf(out_id, subgraph),
      AllOf(IsBinary(ynn_binary_multiply), InputsInclude(x_id, sqrt_y_id)));
  EXPECT_THAT(
      ProducerOf(sqrt_y_id, subgraph),
      AllOf(IsUnary(ynn_unary_reciprocal_square_root), InputsAre(y_id)));
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
  EXPECT_THAT(
      ProducerOf(x_id, subgraph),
      AllOf(IsTernary(ternary_op::multiply), InputsInclude(a_id, b_id, c_id)));
}

TEST(fusion, exp_multiplier) {
  // exp(multiply(a, C)) -> exp(a, multiplier = log2(e) * C)
  const uint32_t a_id = 0;
  const uint32_t c_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t ac_id = YNN_INVALID_VALUE_ID;
  float c = 0.5f;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddScalar(c, c_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, 2, ac_id);
  builder.AddBinary(ynn_binary_multiply, a_id, c_id, ac_id)
      .AddUnary(ynn_unary_exp, ac_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, IsUnary(ynn_unary_exp));
  const auto& unary = std::get<ynn_node::unary_elementwise>(node.op);
  EXPECT_NEAR(unary.params.exp.input_multiplier, std::log2(std::exp(1.0f)) * c,
              1e-6f);
}

TEST(fusion, exp_negate) {
  // exp(negate(a)) -> exp(a, multiplier = -log2(e))
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t a_negated_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, 2, a_negated_id);
  builder.AddUnary(ynn_unary_negate, a_id, a_negated_id)
      .AddUnary(ynn_unary_exp, a_negated_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, IsUnary(ynn_unary_exp));
  const auto& unary = std::get<ynn_node::unary_elementwise>(node.op);
  EXPECT_NEAR(unary.params.exp.input_multiplier, -std::log2(std::exp(1.0f)),
              1e-6f);
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
      .AddInput(ynn_type_int32, 2, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id);
  builder.AddDequantize(a_id, zero_point_id, scale_id, ynn_type_fp32, x_id);

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
      .AddInput(ynn_type_int32, 2, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id);
  builder.AddBinary(ynn_binary_multiply, scale1_id, scale2_id, scale_id)
      .AddDequantize(a_id, zero_point_id, scale_id, ynn_type_fp32, x_id);

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
      .AddOutput(ynn_type_int8, 2, x_id);
  builder.AddQuantize(a_id, ynn_type_int8, zero_point_id, scale_id, x_id);

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
      .AddOutput(ynn_type_uint8, 2, x_id);
  builder.AddQuantize(a_id, ynn_type_uint8, zero_point_id, scale_id, x_id);

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

TEST(fusion, binary_convert) {
  // x_bf16 * y_fp32 -> bf16 should be implemented with a single op.
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);

  builder.AddInput(ynn_type_bf16, {10}, x_id)
      .AddInput(ynn_type_fp32, {10}, y_id)
      .AddOutput(ynn_type_bf16, {10}, out_id);

  builder.AddBinary(ynn_binary_multiply, x_id, y_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph, HasValidNodeCount(1));
  const ynn_node& node = ProducerOf(out_id, subgraph);
  EXPECT_THAT(node,
              AllOf(IsBinary(ynn_binary_multiply), InputsAre(x_id, y_id)));
}

TEST(fusion, dequantize_dot) {
  // rewrite multiply(dot(A, B, subtract_multiply(0, a, b)), c, d) ->
  // dequantize_dot(dot(A, B, YNN_INVALID_VALUE_ID), a, b, c, d, 0)
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t a_offset_id = 2;
  const uint32_t b_offset_id = 3;
  const uint32_t c_id = 4;
  const uint32_t d_id = 5;
  const uint32_t x_id = 6;
  SubgraphBuilder builder(7);

  uint32_t sm_id = YNN_INVALID_VALUE_ID;
  uint32_t dot_output_id = YNN_INVALID_VALUE_ID;
  uint32_t zero_id = builder.DefineScalar(0.0f);

  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 2, b_id)
      .AddInput(ynn_type_fp32, 1, a_offset_id)
      .AddInput(ynn_type_fp32, 1, b_offset_id)
      .AddInput(ynn_type_fp32, 1, c_id)
      .AddInput(ynn_type_fp32, 1, d_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, 1, sm_id)
      .AddTensor(ynn_type_fp32, 2, dot_output_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  ynn_node sm_node;
  ynn::define_ternary(
      subgraph, sm_node, zero_id, a_offset_id, b_offset_id, sm_id,
      ternary_op::subtract_multiply,
      get_ternary_kernel(ternary_op::subtract_multiply, ynn_type_fp32,
                         ynn_type_fp32, ynn_type_fp32, ynn_type_fp32));
  subgraph.add_node(std::move(sm_node));

  builder.AddDot(1, a_id, b_id, sm_id, dot_output_id);

  ynn_node mul_node;
  ynn::define_ternary(
      subgraph, mul_node, dot_output_id, c_id, d_id, x_id, ternary_op::multiply,
      get_ternary_kernel(ternary_op::multiply, ynn_type_fp32, ynn_type_fp32,
                         ynn_type_fp32, ynn_type_fp32));
  subgraph.add_node(std::move(mul_node));

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(3), HasValidValueCount(10)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsRescaleDot(), HasInputCount(6)));
}

TEST(fusion, dequantize_dot_add) {
  // rewrite add(dequantize_dot(..., 0), x) -> dequantize_dot(..., x)
  const uint32_t dot_id = 0;
  const uint32_t a_offset_id = 1;
  const uint32_t b_offset_id = 2;
  const uint32_t a_scale_id = 3;
  const uint32_t b_scale_id = 4;
  const uint32_t x_offset_id = 5;
  const uint32_t out_id = 6;
  SubgraphBuilder builder(7);

  uint32_t dequantize_dot_out_id = YNN_INVALID_VALUE_ID;
  uint32_t zero_id = builder.DefineScalar(0.0f);

  builder.AddInput(ynn_type_fp32, 2, dot_id)
      .AddInput(ynn_type_fp32, 1, a_offset_id)
      .AddInput(ynn_type_fp32, 1, b_offset_id)
      .AddInput(ynn_type_fp32, 1, a_scale_id)
      .AddInput(ynn_type_fp32, 1, b_scale_id)
      .AddInput(ynn_type_fp32, 1, x_offset_id)
      .AddOutput(ynn_type_fp32, 2, out_id)
      .AddTensor(ynn_type_fp32, 2, dequantize_dot_out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  ynn_node rescale_node;
  ynn::define_dequantize_dot(subgraph, rescale_node, ynn_type_fp32, dot_id,
                             a_offset_id, b_offset_id, a_scale_id, b_scale_id,
                             zero_id, dequantize_dot_out_id,
                             ynn::dequantize_dot_params{});
  subgraph.add_node(std::move(rescale_node));

  builder.AddBinary(ynn_binary_add, dequantize_dot_out_id, x_offset_id, out_id);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(7)));
  const ynn_node& final_node = ProducerOf(out_id, subgraph);
  EXPECT_THAT(final_node, IsRescaleDot());
  EXPECT_EQ(final_node.inputs[5], x_offset_id);
}

TEST(fusion, output_convert_convert) {
  // rewrite convert(float, convert(half, a)).
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t fp16_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp16, 2, fp16_id);
  builder.AddUnary(ynn_unary_convert, a_id, fp16_id)
      .AddUnary(ynn_unary_convert, fp16_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // We want to rewrite this to be simply `a`, but because a is an input and the
  // result is an output of the graph, we need to copy the input to the output.
  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
  EXPECT_THAT(ProducerOf(x_id, subgraph), AllOf(IsCopy(), InputsAre(a_id)));
}

TEST(fusion, convert_convert) {
  // rewrite convert<fp32>(convert<bf16>(-a_fp32)) to just -a_fp32.
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t bf16_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_fp32, 2, b_id)
      .AddTensor(ynn_type_bf16, 2, bf16_id);
  builder.AddUnary(ynn_unary_negate, a_id, b_id)
      .AddUnary(ynn_unary_convert, b_id, bf16_id)
      .AddUnary(ynn_unary_convert, bf16_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
  EXPECT_THAT(ProducerOf(x_id, subgraph),
              AllOf(IsUnary(ynn_unary_negate), InputsAre(a_id)));
}

TEST(fusion, dequantize_quantize) {
  // rewrite dequantize(quantize(a_fp32)) -> a_fp32.
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t zero_point_id = builder.DefineScalar(5);
  uint32_t scale_id = builder.DefineScalar(0.5f);
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddTensor(ynn_type_int8, 2, b_id);
  builder.AddQuantize(a_id, ynn_type_int8, zero_point_id, scale_id, b_id)
      .AddDequantize(b_id, zero_point_id, scale_id, ynn_type_fp32, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
  EXPECT_THAT(ProducerOf(x_id, subgraph), AllOf(IsCopy(), InputsAre(a_id)));
}

TEST(fusion, bf16_elementwise) {
  for (bool consistent_arithmetic : {false, true}) {
    // We don't have bf16 binary elementwise ops, we will insert converts to
    // make this work, and then adjacent elementwise ops can be simplified.
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t x_id = 2;
    SubgraphBuilder builder(
        3, consistent_arithmetic ? YNN_FLAG_CONSISTENT_ARITHMETIC : 0);
    uint32_t c_id = YNN_INVALID_VALUE_ID;
    builder.AddInput(ynn_type_bf16, 2, a_id)
        .AddInput(ynn_type_bf16, 2, b_id)
        .AddOutput(ynn_type_bf16, 2, x_id)
        .AddTensor(ynn_type_bf16, 2, c_id);
    builder.AddBinary(ynn_binary_squared_difference, a_id, b_id, c_id)
        .AddBinary(ynn_binary_add, a_id, c_id, x_id);

    ynn_subgraph& subgraph = *builder.GetSubgraph();

    // The graph should be:
    // c_bf16 = convert<bf16>(convert<fp32>(a_bf16) * convert<fp32>(b_bf16)))
    // x_bf16 = convert<bf16>(convert<fp32>(a_bf16) + convert<fp32>(c_bf16)))
    ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(8)));

    subgraph.fusion();
    subgraph.eliminate_common_subgraphs();
    subgraph.invalidate_dead_values();

    if (consistent_arithmetic) {
      // The graph should now be:
      // a_fp32 = convert<fp32>(a_bf16)
      // c_bf16 = convert<bf16>(a_fp32 * convert<fp32>(b_bf16))
      // x_bf16 = convert<bf16>(a_fp32 + convert<fp32>(c_bf16))
      ASSERT_THAT(subgraph, HasValidNodeCount(7));
      EXPECT_THAT(ProducerOf(c_id, subgraph), IsUnary(ynn_unary_convert));
    } else {
      // The graph should now be:
      // a_fp32 = convert<fp32>(a_bf16)
      // c_fp32 = a_fp32 * convert<fp32>(b_bf16)
      // x_bf16 = convert<bf16>(a_fp32 + c_fp32)
      ASSERT_THAT(subgraph, HasValidNodeCount(5));
    }
    EXPECT_THAT(ProducerOf(x_id, subgraph), IsUnary(ynn_unary_convert));
  }
}

TEST(fusion, iota_multiply_add) {
  // rewrite add(multiply(iota, A), B) -> iota(folded)
  const uint32_t begin_id = 0;
  const uint32_t stride_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);

  uint32_t iota_out_id = YNN_INVALID_VALUE_ID;
  uint32_t mul_out_id = YNN_INVALID_VALUE_ID;

  float A = 2.0f;
  float B = 3.0f;
  uint32_t A_id = builder.DefineScalar(A);
  uint32_t B_id = builder.DefineScalar(B);

  builder.AddInput(ynn_type_fp32, 1, begin_id)
      .AddInput(ynn_type_fp32, 1, stride_id)
      .AddOutput(ynn_type_fp32, 1, out_id)
      .AddTensor(ynn_type_fp32, 1, iota_out_id)
      .AddTensor(ynn_type_fp32, 1, mul_out_id);

  builder.AddIota(ynn_type_fp32, {10}, begin_id, stride_id, iota_out_id)
      .AddBinary(ynn_binary_multiply, iota_out_id, A_id, mul_out_id)
      .AddBinary(ynn_binary_add, mul_out_id, B_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, HasValidNodeCount(1));
  const ynn_node& final_node = ProducerOf(out_id, subgraph);
  EXPECT_THAT(final_node, IsIota());

  const ynn_node::iota* iota_op = std::get_if<ynn_node::iota>(&final_node.op);
  ASSERT_NE(iota_op, nullptr);
  EXPECT_EQ(iota_op->params.scale, A);
  EXPECT_EQ(iota_op->params.offset, B);
}

TEST(fusion, iota_multi_consumer) {
  // iota -> v1
  // v1 -> other_consumer (v1 is an output)
  // v1 -> multiply(A) -> v2

  const uint32_t begin_id = 0;
  const uint32_t stride_id = 1;
  const uint32_t v1_id = 2;
  const uint32_t v2_id = 3;
  SubgraphBuilder builder(4);

  float A = 2.0f;
  uint32_t A_id = builder.DefineScalar(A);

  builder.AddInput(ynn_type_fp32, 1, begin_id)
      .AddInput(ynn_type_fp32, 1, stride_id)
      .AddOutput(ynn_type_fp32, 1, v1_id)
      .AddOutput(ynn_type_fp32, 1, v2_id);

  builder.AddIota(ynn_type_fp32, {10}, begin_id, stride_id, v1_id)
      .AddBinary(ynn_binary_multiply, v1_id, A_id, v2_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, HasValidNodeCount(2));

  const ynn_node& node1 = ProducerOf(v1_id, subgraph);
  EXPECT_THAT(node1, IsIota());
  const ynn_node::iota* iota1 = std::get_if<ynn_node::iota>(&node1.op);
  EXPECT_EQ(iota1->params.scale, 1.0f);
  EXPECT_EQ(iota1->params.offset, 0.0f);

  const ynn_node& node2 = ProducerOf(v2_id, subgraph);
  EXPECT_THAT(node2, IsIota());
  const ynn_node::iota* iota2 = std::get_if<ynn_node::iota>(&node2.op);
  EXPECT_EQ(iota2->params.scale, A);
  EXPECT_EQ(iota2->params.offset, 0.0f);
}

}  // namespace ynn
