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
using ::testing::ElementsAre;
using ::testing::Not;

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

TEST(fusion, reduce_sum_broadcast_add) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  uint32_t v1_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 3, b_id)
      .AddOutput(ynn_type_fp32, 3, x_id)
      .AddTensor(ynn_type_fp32, 1, v1_id);

  // Similar to the case above, but the add is broadcasting, which reduce cannot
  // implement.
  builder.AddReduce(ynn_reduce_sum, {1}, a_id, YNN_INVALID_VALUE_ID, v1_id, 0)
      .AddBinary(ynn_binary_add, v1_id, b_id, x_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(2), HasValidValueCount(4)));
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

TEST(fusion, reduce_static_transpose) {
  // reduce(static_transpose(x)) -> static_transpose(reduce(x))
  const uint32_t x_id = 0;
  uint32_t transposed_id = 1;
  const uint32_t y_id = 2;
  SubgraphBuilder builder(3);

  builder.AddInput(ynn_type_fp32, 3, x_id)
      .AddOutput(ynn_type_fp32, 2, y_id)
      .AddTensor(ynn_type_fp32, 3, transposed_id);

  // Transpose (2, 0, 1). So dimension 0 -> 2, 1 -> 0, 2 -> 1.
  std::vector<int32_t> perm = {2, 0, 1};
  builder.AddTranspose(perm, x_id, transposed_id)
      .AddReduce(ynn_reduce_sum, {1}, transposed_id, YNN_INVALID_VALUE_ID, y_id,
                 /*flags=*/0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, HasValidNodeCount(2));

  const ynn_node& trans_node = ProducerOf(y_id, subgraph);
  EXPECT_TRUE(
      std::holds_alternative<ynn_node::static_transpose>(trans_node.op));
  const uint32_t r_id = trans_node.inputs[0];

  const ynn_node& reduce_node = ProducerOf(r_id, subgraph);
  EXPECT_THAT(reduce_node, AllOf(IsReduce(ynn_reduce_sum),
                                 InputsAre(x_id, YNN_INVALID_VALUE_ID)));

  const auto& reduce_op = std::get<ynn_node::reduce>(reduce_node.op);
  // Original user reduce axis is 1 -> Slinky axis 1. (k_dims[1] = true).
  // Transpose permutation is {2, 0, 1}, which translates to slinky permutation
  // {1, 2, 0}. Mapped axis = perm[1] = 2.
  EXPECT_EQ(reduce_op.k_dims, 0b100);
}

TEST(fusion, reduce_static_transpose_identity) {
  // reduce(static_transpose(x)) -> reduce(x) where the resulting transpose is
  // an identity.
  const uint32_t x_id = 0;
  uint32_t transposed_id = 1;
  const uint32_t y_id = 2;
  SubgraphBuilder builder(3);

  builder.AddInput(ynn_type_fp32, 3, x_id)
      .AddOutput(ynn_type_fp32, 2, y_id)
      .AddTensor(ynn_type_fp32, 3, transposed_id);

  // Transpose (0, 2, 1). So dimension 0 -> 0, 1 -> 2, 2 -> 1.
  std::vector<int32_t> perm = {0, 2, 1};
  builder.AddTranspose(perm, x_id, transposed_id)
      .AddReduce(ynn_reduce_sum, {2}, transposed_id, YNN_INVALID_VALUE_ID, y_id,
                 /*flags=*/0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph,
              AllOf(HasValidNodeCount(1), HasValidValueIds(x_id, y_id),
                    Not(HasValidValueId(transposed_id))));

  const ynn_node& reduce_node = ProducerOf(y_id, subgraph);
  EXPECT_THAT(reduce_node, AllOf(IsReduce(ynn_reduce_sum),
                                 InputsAre(x_id, YNN_INVALID_VALUE_ID)));

  const auto& reduce_op = std::get<ynn_node::reduce>(reduce_node.op);
  // Original user reduce axis is 2 -> Slinky axis 0. (k_dims[0] = true).
  // Transpose permutation is {0, 2, 1}, which translates to slinky permutation
  // {1, 0, 2}. Mapped axis = perm[0] = 1.
  EXPECT_EQ(reduce_op.k_dims, 0b010);
}

TEST(fusion, reduce_new_dimension_static_transpose) {
  const uint32_t x_id = 0;
  uint32_t transposed_id = 1;
  const uint32_t y_id = 2;
  SubgraphBuilder builder(3);

  builder.AddInput(ynn_type_fp32, 2, x_id)
      .AddOutput(ynn_type_fp32, 2, y_id)
      .AddTensor(ynn_type_fp32, 3, transposed_id);

  // Transpose (2, 0, 1). So dimension 0 -> 2, 1 -> 0, 2 -> 1.
  std::vector<int32_t> perm = {2, YNN_MAX_TENSOR_RANK, 1};
  builder.AddTranspose(perm, x_id, transposed_id)
      .AddReduce(ynn_reduce_sum, {1}, transposed_id, YNN_INVALID_VALUE_ID, y_id,
                 /*flags=*/0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  ASSERT_THAT(subgraph, HasValidNodeCount(2));

  // This should not be rewritten because the reduction is of a new dimension
  // inserted by the transpose.
  EXPECT_THAT(ProducerOf(y_id, subgraph), IsReduce(ynn_reduce_sum));
}

TEST(fusion, reduce_sum_to_dot_f32) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t y_id = 2;
  SubgraphBuilder builder(3);
  uint32_t mul_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, {4, 8, 1}, a_id)
      .AddInput(ynn_type_fp32, {1, 8, 3}, b_id)
      .AddOutput(ynn_type_fp32, {4, 3}, y_id)
      .AddTensor(ynn_type_fp32, {4, 8, 3}, mul_id);

  // y = sum(a * b, axis=1)
  // This should rewrite to a dot.
  builder.AddBinary(ynn_binary_multiply, a_id, b_id, mul_id)
      .AddReduce(ynn_reduce_sum, {1}, mul_id, YNN_INVALID_VALUE_ID, y_id, 0);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph, Not(HasValidValueId(mul_id)));
  const ynn_node& dot_node = ProducerOf(y_id, subgraph);
  ASSERT_THAT(dot_node, IsDot());
  EXPECT_THAT(
      ProducerOf(dot_node.inputs[0], subgraph),
      AllOf(IsStaticTransposeWithPerm(ElementsAre(1, 2)), InputsAre(a_id)));
}

}  // namespace ynn
