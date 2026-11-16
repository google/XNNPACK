// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/dot.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

class DynamicallyQuantizedDot : public testing::TestWithParam<ynn_type> {};

TEST_P(DynamicallyQuantizedDot, Transpose) {
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;

  SubgraphBuilder builder(4);

  builder.AddInput(ynn_type_fp32, {16, 8}, a_id)
      .AddInput(ynn_type_fp32, {2, 1, 8}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b_id)
      .AddOutput(ynn_type_int32, {8, 32}, output_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;

  builder.AddTensor(ynn_type_fp32, {1, 8}, scale_id)
      .AddTensor(ynn_type_int32, {1, 8}, zp_id)
      .AddTensor(ynn_type_int8, {16, 8}, quantized_id)
      .AddTensor(ynn_type_int8, {8, 16}, transposed_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddTranspose({1, 0}, quantized_id, transposed_id);
  builder.AddDot(1, transposed_id, b_id, YNN_INVALID_VALUE_ID, output_id);

  // `ynn_define_dot` should not rewrite upstream types before `fusion()`.
  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(subgraph.value(transposed_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  const bool expect_rewrite = prefer_uint8_dot(rhs_type);
  if (expect_rewrite) {
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_uint8));
    EXPECT_THAT(subgraph.value(transposed_id), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(128));
    EXPECT_THAT(ProducerOf(quantized_id, subgraph), IsQuantize());
    EXPECT_THAT(ProducerOf(output_id, subgraph), IsDot());
  } else {
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
    EXPECT_THAT(subgraph.value(transposed_id), HasType(ynn_type_int8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  }
}

TEST_P(DynamicallyQuantizedDot, SplitDim) {
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;

  SubgraphBuilder builder(4);

  builder.AddInput(ynn_type_fp32, {8, 16}, a_id)
      .AddInput(ynn_type_fp32, {2, 8, 1}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b_id)
      .AddOutput(ynn_type_int32, {2, 4, 32}, output_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  uint32_t split_id = YNN_INVALID_VALUE_ID;

  builder.AddTensor(ynn_type_fp32, {8, 1}, scale_id)
      .AddTensor(ynn_type_int32, {8, 1}, zp_id)
      .AddTensor(ynn_type_int8, {8, 16}, quantized_id)
      .AddTensor(ynn_type_int8, {2, 4, 16}, split_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddSplitDim(0, {2, 4}, quantized_id, split_id);
  builder.AddDot(1, split_id, b_id, YNN_INVALID_VALUE_ID, output_id);

  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  const bool expect_rewrite = prefer_uint8_dot(rhs_type);
  if (expect_rewrite) {
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_uint8));
    EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(128));
    EXPECT_THAT(ProducerOf(quantized_id, subgraph), IsQuantize());
    EXPECT_THAT(ProducerOf(output_id, subgraph), IsDot());
  } else {
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
    EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_int8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  }
}

TEST_P(DynamicallyQuantizedDot, SplitDimAndTransposeChain) {
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;

  SubgraphBuilder builder(4);

  builder.AddInput(ynn_type_fp32, {8, 16}, a_id)
      .AddInput(ynn_type_fp32, {2, 8, 1}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b_id)
      .AddOutput(ynn_type_int32, {4, 2, 32}, output_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  uint32_t split_id = YNN_INVALID_VALUE_ID;
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;

  builder.AddTensor(ynn_type_fp32, {8, 1}, scale_id)
      .AddTensor(ynn_type_int32, {8, 1}, zp_id)
      .AddTensor(ynn_type_int8, {8, 16}, quantized_id)
      .AddTensor(ynn_type_int8, {2, 4, 16}, split_id)
      .AddTensor(ynn_type_int8, {4, 2, 16}, transposed_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddSplitDim(0, {2, 4}, quantized_id, split_id);
  builder.AddTranspose({1, 0, 2}, split_id, transposed_id);
  builder.AddDot(1, transposed_id, b_id, YNN_INVALID_VALUE_ID, output_id);

  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_int8));
  EXPECT_THAT(subgraph.value(transposed_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  const bool expect_rewrite = prefer_uint8_dot(rhs_type);
  if (expect_rewrite) {
    // All values along the layout transform path must be rewritten to uint8.
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_uint8));
    EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_uint8));
    EXPECT_THAT(subgraph.value(transposed_id), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(128));
    EXPECT_THAT(ProducerOf(quantized_id, subgraph), IsQuantize());
    EXPECT_THAT(ProducerOf(output_id, subgraph), IsDot());
  } else {
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
    EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_int8));
    EXPECT_THAT(subgraph.value(transposed_id), HasType(ynn_type_int8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  }
}

TEST_P(DynamicallyQuantizedDot, MultipleDotConsumers) {
  // When a dynamically quantized tensor fans out to multiple dots (e.g. Q/K/V
  // or gate/up projections), all branches can be rewritten in place to uint8
  // without inserting requantize_to_uint8 nodes.
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b1_id = 2;
  const uint32_t b2_id = 3;
  const uint32_t out1_id = 4;
  const uint32_t out2_id = 5;

  SubgraphBuilder builder(6);

  builder.AddInput(ynn_type_fp32, {8, 16}, a_id)
      .AddInput(ynn_type_fp32, {2, 8, 1}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b1_id)
      .AddInput(rhs_type, {16, 32}, b2_id)
      .AddOutput(ynn_type_int32, {8, 32}, out1_id)
      .AddOutput(ynn_type_int32, {2, 4, 32}, out2_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  uint32_t split_id = YNN_INVALID_VALUE_ID;

  builder.AddTensor(ynn_type_fp32, {8, 1}, scale_id)
      .AddTensor(ynn_type_int32, {8, 1}, zp_id)
      .AddTensor(ynn_type_int8, {8, 16}, quantized_id)
      .AddTensor(ynn_type_int8, {2, 4, 16}, split_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddDot(1, quantized_id, b1_id, YNN_INVALID_VALUE_ID, out1_id);
  builder.AddSplitDim(0, {2, 4}, quantized_id, split_id);
  builder.AddDot(1, split_id, b2_id, YNN_INVALID_VALUE_ID, out2_id);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  const bool expect_rewrite = prefer_uint8_dot(rhs_type);
  if (expect_rewrite) {
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_uint8));
    EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(128));
    EXPECT_EQ(ProducerOf(out1_id, subgraph).inputs[0], quantized_id);
    EXPECT_EQ(ProducerOf(out2_id, subgraph).inputs[0], split_id);
  } else {
    EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
    EXPECT_THAT(subgraph.value(split_id), HasType(ynn_type_int8));
    EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  }
}

TEST_P(DynamicallyQuantizedDot, SharedWithNonDotConsumer) {
  // If `quantized_id` also feeds a non-dot consumer, rewriting it in place
  // would corrupt that consumer. It must stay int8 and the dot must fall back
  // to `requantize_to_uint8`.
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;
  const uint32_t other_output_id = 4;

  SubgraphBuilder builder(5);

  builder.AddInput(ynn_type_fp32, {8, 16}, a_id)
      .AddInput(ynn_type_fp32, {2, 8, 1}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b_id)
      .AddOutput(ynn_type_int32, {8, 32}, output_id)
      .AddOutput(ynn_type_fp32, {8, 16}, other_output_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;

  builder.AddTensor(ynn_type_fp32, {8, 1}, scale_id)
      .AddTensor(ynn_type_int32, {8, 1}, zp_id)
      .AddTensor(ynn_type_int8, {8, 16}, quantized_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddDot(1, quantized_id, b_id, YNN_INVALID_VALUE_ID, output_id);
  builder.AddUnary(ynn_unary_convert, quantized_id, other_output_id);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  const ynn_node& dot_node = ProducerOf(output_id, subgraph);
  if (prefer_uint8_dot(rhs_type)) {
    EXPECT_THAT(subgraph.value(dot_node.inputs[0]), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(dot_node.inputs[0], subgraph),
                IsUnary(ynn_unary_requantize_to_uint8));
  } else {
    EXPECT_EQ(dot_node.inputs[0], quantized_id);
  }
}

TEST_P(DynamicallyQuantizedDot, SharedLayoutTransformWithNonDotConsumer) {
  // If a layout-transformed descendant (`transposed_id`) also feeds a non-dot
  // consumer, keep the int8 chain and fall back to `requantize_to_uint8`.
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;
  const uint32_t other_output_id = 4;

  SubgraphBuilder builder(5);

  builder.AddInput(ynn_type_fp32, {16, 8}, a_id)
      .AddInput(ynn_type_fp32, {2, 1, 8}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b_id)
      .AddOutput(ynn_type_int32, {8, 32}, output_id)
      .AddOutput(ynn_type_fp32, {8, 16}, other_output_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;

  builder.AddTensor(ynn_type_fp32, {1, 8}, scale_id)
      .AddTensor(ynn_type_int32, {1, 8}, zp_id)
      .AddTensor(ynn_type_int8, {16, 8}, quantized_id)
      .AddTensor(ynn_type_int8, {8, 16}, transposed_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddTranspose({1, 0}, quantized_id, transposed_id);
  builder.AddDot(1, transposed_id, b_id, YNN_INVALID_VALUE_ID, output_id);
  builder.AddUnary(ynn_unary_convert, transposed_id, other_output_id);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(subgraph.value(transposed_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  const ynn_node& dot_node = ProducerOf(output_id, subgraph);
  if (prefer_uint8_dot(rhs_type)) {
    EXPECT_THAT(subgraph.value(dot_node.inputs[0]), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(dot_node.inputs[0], subgraph),
                IsUnary(ynn_unary_requantize_to_uint8));
  } else {
    EXPECT_EQ(dot_node.inputs[0], transposed_id);
  }
}

TEST_P(DynamicallyQuantizedDot, ExternalOutputQuantizedTensor) {
  // Do not rewrite `quantized_id` in place when it is an external output; fall
  // back to `requantize_to_uint8` so the external tensor stays int8.
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;
  const uint32_t quantized_id = 4;

  SubgraphBuilder builder(5);

  builder.AddInput(ynn_type_fp32, {8, 16}, a_id)
      .AddInput(ynn_type_fp32, {2, 8, 1}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b_id)
      .AddOutput(ynn_type_int32, {8, 32}, output_id)
      .AddOutput(ynn_type_int8, {8, 16}, quantized_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(ynn_type_fp32, {8, 1}, scale_id)
      .AddTensor(ynn_type_int32, {8, 1}, zp_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddDot(1, quantized_id, b_id, YNN_INVALID_VALUE_ID, output_id);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  const ynn_node& dot_node = ProducerOf(output_id, subgraph);
  if (prefer_uint8_dot(rhs_type)) {
    EXPECT_THAT(subgraph.value(dot_node.inputs[0]), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(dot_node.inputs[0], subgraph),
                IsUnary(ynn_unary_requantize_to_uint8));
  } else {
    EXPECT_EQ(dot_node.inputs[0], quantized_id);
  }
}

TEST_P(DynamicallyQuantizedDot, ExternalOutputZeroPoint) {
  // Shifting `output_zero_point` to 128 would alter `zp_id`; if `zp_id` is an
  // external output, keep it unchanged and fall back to `requantize_to_uint8`.
  const ynn_type rhs_type = GetParam();
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;
  uint32_t zp_id = 4;

  SubgraphBuilder builder(5);

  builder.AddInput(ynn_type_fp32, {8, 16}, a_id)
      .AddInput(ynn_type_fp32, {2, 8, 1}, min_max_id)
      .AddInput(rhs_type, {16, 32}, b_id)
      .AddOutput(ynn_type_int32, {8, 32}, output_id)
      .AddOutput(ynn_type_int32, {8, 1}, zp_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(ynn_type_fp32, {8, 1}, scale_id)
      .AddTensor(ynn_type_int8, {8, 16}, quantized_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddDot(1, quantized_id, b_id, YNN_INVALID_VALUE_ID, output_id);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  const ynn_node& dot_node = ProducerOf(output_id, subgraph);
  if (prefer_uint8_dot(rhs_type)) {
    EXPECT_THAT(subgraph.value(dot_node.inputs[0]), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(dot_node.inputs[0], subgraph),
                IsUnary(ynn_unary_requantize_to_uint8));
  } else {
    EXPECT_EQ(dot_node.inputs[0], quantized_id);
  }
}

TEST(DynamicallyQuantizedDotSelf, DotConsumesBothInputs) {
  // When `quantized_id` is both `A` and `B` of the same dot, rewriting it in
  // place would also change `B` to uint8; fall back to `requantize_to_uint8`.
  const uint32_t a_id = 0;
  const uint32_t min_max_id = 1;
  const uint32_t output_id = 2;

  SubgraphBuilder builder(3);

  builder.AddInput(ynn_type_fp32, {16, 16}, a_id)
      .AddInput(ynn_type_fp32, {2, 16, 1}, min_max_id)
      .AddOutput(ynn_type_int32, {16, 16}, output_id);

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(ynn_type_fp32, {16, 1}, scale_id)
      .AddTensor(ynn_type_int32, {16, 1}, zp_id)
      .AddTensor(ynn_type_int8, {16, 16}, quantized_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  ynn_status status = ynn_define_dynamic_quantization(
      &subgraph, min_max_id, ynn_type_int8, &zp_id, &scale_id, 0);
  ASSERT_EQ(status, ynn_status_success);

  builder.AddQuantize(a_id, ynn_type_int8, zp_id, scale_id, quantized_id);
  builder.AddDot(1, quantized_id, quantized_id, YNN_INVALID_VALUE_ID,
                 output_id);

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(subgraph.value(quantized_id), HasType(ynn_type_int8));
  EXPECT_THAT(ProducerOf(zp_id, subgraph), IsDynamicQuantization(0));
  const ynn_node& dot_node = ProducerOf(output_id, subgraph);
  EXPECT_EQ(dot_node.inputs[1], quantized_id);
  if (prefer_uint8_dot(ynn_type_int8)) {
    EXPECT_THAT(subgraph.value(dot_node.inputs[0]), HasType(ynn_type_uint8));
    EXPECT_THAT(ProducerOf(dot_node.inputs[0], subgraph),
                IsUnary(ynn_unary_requantize_to_uint8));
  } else {
    EXPECT_EQ(dot_node.inputs[0], quantized_id);
  }
}

INSTANTIATE_TEST_SUITE_P(RhsType, DynamicallyQuantizedDot,
                         testing::Values(ynn_type_int8, ynn_type_int4,
                                         ynn_type_int2),
                         [](const testing::TestParamInfo<ynn_type>& info) {
                           return to_string(info.param);
                         });

}  // namespace
}  // namespace ynn
