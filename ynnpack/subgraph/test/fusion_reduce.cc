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
using ::testing::Not;

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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
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

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
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
                    InputsAre(x_id, YNN_INVALID_VALUE_ID)));
}

void TestReduceSumOfConvertQuantized(ynn_reduce_operator reduce_op) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t scale_id = 2;
  const uint32_t zero_point_id = 3;
  SubgraphBuilder builder(4);

  // Define scale and zero point.
  builder.AddInput(ynn_type_fp32, 0, scale_id)
      .AddInput(ynn_type_int32, 0, zero_point_id)
      .AddInput(ynn_type_int8, 2, input_id)
      .AddOutput(ynn_type_fp32, 1, output_id);

  uint32_t dequantized_x_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(ynn_type_fp32, 2, dequantized_x_id);
  builder
      .AddDequantize(input_id, zero_point_id, scale_id, ynn_type_fp32,
                     dequantized_x_id)
      .AddReduce(reduce_op, {1}, dequantized_x_id, YNN_INVALID_VALUE_ID,
                 output_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // Should NOT fuse.
  // We expect the reduce node to still consume dequantized_x_id, not x_id.
  EXPECT_THAT(ProducerOf(output_id, subgraph), InputsInclude(dequantized_x_id));
  EXPECT_THAT(ProducerOf(dequantized_x_id, subgraph),
              AllOf(IsDequantize(), InputsInclude(input_id)));
  EXPECT_THAT(subgraph,
              HasValidValueIds(input_id, output_id, dequantized_x_id));
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

void TestReduceSumOfSquared(ynn_type type, bool use_square) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  SubgraphBuilder builder(2);
  uint32_t sq_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(type, 2, x_id)
      .AddOutput(type, 1, y_id)
      .AddTensor(type, 2, sq_id);
  if (use_square) {
    builder.AddBinary(ynn_binary_multiply, x_id, x_id, sq_id);
  } else {
    builder.AddUnary(ynn_unary_square, x_id, sq_id);
  }
  builder.AddReduce(ynn_reduce_sum, {1}, sq_id, YNN_INVALID_VALUE_ID, y_id, 0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // x and y should be valid, sq should be invalid/removed.
  ASSERT_THAT(subgraph,
              AllOf(HasValidNodeCount(1), HasValidValueIds(x_id, y_id),
                    Not(HasValidValueIds(sq_id))));
  EXPECT_THAT(ProducerOf(y_id, subgraph),
              AllOf(IsReduce(ynn_reduce_sum_squared), HasInputCount(2),
                    InputsAre(x_id, YNN_INVALID_VALUE_ID)));
}

}  // namespace

TEST(fusion, reduce_sum_of_squared_f32) {
  // reduce_sum(x_f32 * x_f32) -> reduce_sum_squared(x_f32)
  for (bool use_square : {false, true}) {
    TestReduceSumOfSquared(ynn_type_fp32, use_square);
  }
}

TEST(fusion, reduce_sum_of_squared_int32) {
  // reduce_sum(x_int32 * x_int32) -> reduce_sum_squared(x_int32)
  for (bool use_square : {false, true}) {
    TestReduceSumOfSquared(ynn_type_int32, use_square);
  }
}

namespace {

void TestReduceSumOfSquaredWithConvert(ynn_type input_type,
                                       ynn_type intermediate_type,
                                       bool use_square) {
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
  builder.AddUnary(ynn_unary_convert, x_id, intermediate1_id);
  if (use_square) {
    builder.AddBinary(ynn_binary_multiply, intermediate1_id, intermediate1_id,
                      intermediate2_id);
  } else {
    builder.AddUnary(ynn_unary_square, intermediate1_id, intermediate2_id);
  }
  builder.AddReduce(ynn_reduce_sum, {1}, intermediate2_id, YNN_INVALID_VALUE_ID,
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
                    InputsAre(x_id, YNN_INVALID_VALUE_ID)));
}

}  // namespace

TEST(fusion, reduce_sum_of_squared_with_convert_fp16) {
  // reduce_sum(fp32(x_fp16) * fp32(x_fp16)) -> reduce_sum_squared(x_fp16).
  for (bool use_square : {false, true}) {
    TestReduceSumOfSquaredWithConvert(ynn_type_fp16, ynn_type_fp32, use_square);
  }
}

TEST(fusion, reduce_sum_of_squared_with_convert_bf16) {
  // reduce_sum(fp32(x_bf16) * fp32(x_bf16)) -> reduce_sum_squared(x_bf16).
  for (bool use_square : {false, true}) {
    TestReduceSumOfSquaredWithConvert(ynn_type_bf16, ynn_type_fp32, use_square);
  }
}

TEST(fusion, reduce_sum_of_squared_with_convert_int8) {
  // reduce_sum(int32(x_int8) * int32(x_int8)) -> reduce_sum_squared(x_int8).
  for (bool use_square : {false, true}) {
    TestReduceSumOfSquaredWithConvert(ynn_type_int8, ynn_type_int32,
                                      use_square);
  }
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

TEST(fusion, reduce_expand_dims) {
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 4, a_id)
      .AddOutput(ynn_type_fp32, 4, x_id)
      .AddTensor(ynn_type_fp32, 2, v1_id);

  builder
      .AddReduce(ynn_reduce_sum, {1, 2}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddExpandDims({1, 2}, v1_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(2)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, AllOf(IsReduce(ynn_reduce_sum), InputsInclude(a_id)));
  const auto& reduce = std::get<ynn_node::reduce>(node.op);
  EXPECT_EQ(reduce.k_dims, 0b0110);
  EXPECT_TRUE(reduce.keep_dims);
}

TEST(fusion, reduce_expand_dims_initializer) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 4, a_id)
      .AddInput(ynn_type_fp32, 2, b_id)
      .AddOutput(ynn_type_fp32, 4, x_id)
      .AddTensor(ynn_type_fp32, 2, v1_id);

  builder.AddReduce(ynn_reduce_sum, {1, 2}, a_id, b_id, v1_id, 0)
      .AddExpandDims({1, 2}, v1_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(2), HasValidValueCount(4)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node,
              AllOf(IsReduce(ynn_reduce_sum), InputsInclude(a_id, v1_id)));
  EXPECT_THAT(ProducerOf(v1_id, subgraph),
              AllOf(IsExpandDims(), InputsAre(b_id)));
  const auto& reduce = std::get<ynn_node::reduce>(node.op);
  EXPECT_EQ(reduce.k_dims, 0b0110);
  EXPECT_TRUE(reduce.keep_dims);
}

TEST(fusion, reduce_sum_add) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // v1 = sum(a, identity=0)
  // x = add(v1, b)
  // Should fuse to x = sum(a, b)
  builder.AddReduce(ynn_reduce_sum, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddBinary(ynn_binary_add, v1_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, AllOf(IsReduce(ynn_reduce_sum), InputsAre(a_id, b_id)));
}

TEST(fusion, reduce_sum_squared_add) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // v1 = sum_squared(a, identity=0)
  // x = add(v1, b)
  // Should fuse to x = sum_squared(a, b)
  builder
      .AddReduce(ynn_reduce_sum_squared, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id,
                 0)
      .AddBinary(ynn_binary_add, v1_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node,
              AllOf(IsReduce(ynn_reduce_sum_squared), InputsAre(a_id, b_id)));
}

TEST(fusion, reduce_min_min) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // v1 = min_reduce(a, identity=inf)
  // x = min(v1, b)
  // Should fuse to x = min_reduce(a, b)
  builder.AddReduce(ynn_reduce_min, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddBinary(ynn_binary_min, v1_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, AllOf(IsReduce(ynn_reduce_min), InputsAre(a_id, b_id)));
}

TEST(fusion, reduce_max_max) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // v1 = max_reduce(a, identity=-inf)
  // x = max(v1, b)
  // Should fuse to x = max_reduce(a, b)
  builder.AddReduce(ynn_reduce_max, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddBinary(ynn_binary_max, v1_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, AllOf(IsReduce(ynn_reduce_max), InputsAre(a_id, b_id)));
}

TEST(fusion, reduce_sum_add_commutative) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // v1 = sum(a, identity=0)
  // x = add(b, v1)
  // Should fuse to x = sum(a, b)
  builder.AddReduce(ynn_reduce_sum, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddBinary(ynn_binary_add, b_id, v1_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(1), HasValidValueCount(3)));
  const ynn_node& node = ProducerOf(x_id, subgraph);
  EXPECT_THAT(node, AllOf(IsReduce(ynn_reduce_sum), InputsAre(a_id, b_id)));
}

TEST(fusion, reduce_sum_add_non_identity) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  uint32_t init_id = builder.DefineScalar(1.0f);
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // v1 = sum(a, identity=1.0)
  // x = add(v1, b)
  // Should NOT fuse
  builder.AddReduce(ynn_reduce_sum, {1}, a_id, init_id, v1_id, 0)
      .AddBinary(ynn_binary_add, v1_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

TEST(fusion, reduce_sum_multiply) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // v1 = sum(a, identity=0)
  // x = multiply(v1, b)
  // Should NOT fuse
  builder.AddReduce(ynn_reduce_sum, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddBinary(ynn_binary_multiply, v1_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

TEST(fusion, reduce_sum_add_of_reduce) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t reduce_id = YNN_INVALID_VALUE_ID;
  uint32_t abs_reduce_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 1, b_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, reduce_id)
      .AddTensor(ynn_type_fp32, 1, abs_reduce_id);

  // reduce = sum(a, identity=0)
  // x = add(reduce, abs(reduce))
  // We can't rewrite this to sum(a, abs(reduce)) because it would create a
  // cycle in the graph.
  // This rewrite shouldn't happen because the reduce op has more than one user.
  builder
      .AddReduce(ynn_reduce_sum, {1}, a_id, YNN_INVALID_VALUE_ID, reduce_id, 0)
      .AddUnary(ynn_unary_abs, reduce_id, abs_reduce_id)
      .AddBinary(ynn_binary_add, abs_reduce_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph, HasValidNodeCount(3));
}

}  // namespace ynn
