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
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::AllOf;

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

TEST(fusion, convert_broadcast_unary) {
  // rewrite abs(static_broadcast(x)) -> static_broadcast(abs(x))
  const uint32_t x_id = 0;
  const uint32_t out_id = 1;
  SubgraphBuilder builder(2);
  uint32_t broadcast_x_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, 2, x_id)
      .AddOutput(ynn_type_fp32, 2, out_id)
      .AddTensor(ynn_type_fp32, 2, broadcast_x_id);

  builder.AddStaticBroadcast({10, 0}, x_id, broadcast_x_id)
      .AddUnary(ynn_unary_abs, broadcast_x_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_THAT(ProducerOf(out_id, subgraph),
              AllOf(IsStaticBroadcast(), InputsAre(broadcast_x_id)));
}

TEST(fusion, convert_broadcast_binary) {
  // rewrite add(convert(broadcast_like(x, y)), y) ->
  // add(broadcast_like(convert(x), y), y)
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);
  uint32_t broadcast_x_id = YNN_INVALID_VALUE_ID;
  uint32_t converted_broadcast_x_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp16, {1, 10}, x_id)
      .AddInput(ynn_type_fp32, {5, 10}, y_id)
      .AddOutput(ynn_type_fp32, {5, 10}, out_id)
      .AddTensor(ynn_type_fp16, {5, 10}, broadcast_x_id)
      .AddTensor(ynn_type_fp32, {5, 10}, converted_broadcast_x_id);

  builder.AddBroadcastLike({0}, x_id, y_id, broadcast_x_id)
      .AddUnary(ynn_unary_convert, broadcast_x_id, converted_broadcast_x_id)
      .AddBinary(ynn_binary_add, converted_broadcast_x_id, y_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The graph should now have:
  // x_id -> convert -> converted_x_id
  // converted_x_id -> add (with y_id) -> out_id
  // (broadcast_like was moved past convert and then removed as it is
  // redundant for add).

  const ynn_node& add_node = ProducerOf(out_id, subgraph);
  EXPECT_THAT(add_node, IsBinary(ynn_binary_add));
  uint32_t convert_out_id =
      add_node.inputs[0] == y_id ? add_node.inputs[1] : add_node.inputs[0];

  const ynn_node& convert_node = ProducerOf(convert_out_id, subgraph);
  EXPECT_THAT(convert_node, IsUnary(ynn_unary_convert));
  EXPECT_EQ(convert_node.inputs[0], x_id);
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
                        {/*axis=*/2, /*new_axis=*/3, /*extent=*/3,
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

TEST(fusion, remove_static_broadcast_from_elementwise_binary) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);
  uint32_t broadcast_x_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, {1, 10}, x_id)
      .AddInput(ynn_type_fp32, {5, 10}, y_id)
      .AddOutput(ynn_type_fp32, {5, 10}, out_id)
      .AddTensor(ynn_type_fp32, {5, 10}, broadcast_x_id);

  builder.AddStaticBroadcast({5, 0}, x_id, broadcast_x_id)
      .AddBinary(ynn_binary_add, broadcast_x_id, y_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The static_broadcast should be removed because it is implied by the other
  // operand.
  EXPECT_THAT(ProducerOf(out_id, subgraph),
              AllOf(IsBinary(ynn_binary_add), InputsInclude(x_id, y_id)));
}

TEST(fusion, keep_static_broadcast_from_dynamic_shape_elementwise_binary) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);
  uint32_t broadcast_x_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, 2, x_id)
      .AddInput(ynn_type_fp32, 2, y_id)
      .AddOutput(ynn_type_fp32, 2, out_id)
      .AddTensor(ynn_type_fp32, 2, broadcast_x_id);

  builder.AddStaticBroadcast({5, 0}, x_id, broadcast_x_id)
      .AddBinary(ynn_binary_add, broadcast_x_id, y_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The static_broadcast should not be removed because we don't know if it is
  // needed or not.
  EXPECT_THAT(
      ProducerOf(out_id, subgraph),
      AllOf(IsBinary(ynn_binary_add), InputsInclude(broadcast_x_id, y_id)));
}

TEST(fusion, keep_static_broadcast_from_elementwise_binary) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);
  uint32_t broadcast_x_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, {1, 10}, x_id)
      .AddInput(ynn_type_fp32, {1, 10}, y_id)
      .AddOutput(ynn_type_fp32, {5, 10}, out_id)
      .AddTensor(ynn_type_fp32, {5, 10}, broadcast_x_id);

  builder.AddStaticBroadcast({5, 0}, x_id, broadcast_x_id)
      .AddBinary(ynn_binary_add, broadcast_x_id, y_id, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The static_broadcast should be moved to the output instead of removed.
  EXPECT_THAT(ProducerOf(out_id, subgraph), IsStaticBroadcast());
  uint32_t add_out_id = ProducerOf(out_id, subgraph).inputs[0];
  EXPECT_THAT(ProducerOf(add_out_id, subgraph),
              AllOf(IsBinary(ynn_binary_add), InputsInclude(x_id, y_id)));
}

TEST(fusion, keep_static_broadcast_multiple_consumers) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t out_id = 2;
  const uint32_t out2_id = 3;
  SubgraphBuilder builder(4);
  uint32_t broadcast_x_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, {1, 10}, x_id)
      .AddInput(ynn_type_fp32, {1, 10}, y_id)
      .AddOutput(ynn_type_fp32, {5, 10}, out_id)
      .AddOutput(ynn_type_fp32, {5, 10}, out2_id)
      .AddTensor(ynn_type_fp32, {5, 10}, broadcast_x_id);

  builder.AddStaticBroadcast({5, 0}, x_id, broadcast_x_id)
      .AddBinary(ynn_binary_add, broadcast_x_id, y_id, out_id)
      .AddUnary(ynn_unary_abs, broadcast_x_id, out2_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // We can't rewrite this because the broadcast has two consumers. We could
  // improve the rewrite to handle this case.
  EXPECT_THAT(
      ProducerOf(out_id, subgraph),
      AllOf(IsBinary(ynn_binary_add), InputsInclude(broadcast_x_id, y_id)));
}

}  // namespace ynn
