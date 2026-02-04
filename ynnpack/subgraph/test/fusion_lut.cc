#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

namespace ynn {

namespace {

int valid_node_count(const ynn_subgraph* subgraph) {
  return std::count_if(subgraph->nodes.begin(), subgraph->nodes.end(),
                       [](const ynn_node& node) { return node.is_valid(); });
}

bool is_lut(const ynn_node& node) {
  return std::holds_alternative<ynn_node::lut>(node.op);
}

bool is_binary(const ynn_node& node, ynn_binary_operator op) {
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&node.op);
  return binary && binary->op == op;
}

template <typename A, typename X>
void RunSubgraph(ynn_subgraph_t subgraph,
                 const std::vector<uint32_t>& input_ids,
                 const std::vector<std::vector<A>>& input_datas,
                 const std::vector<TensorShape>& input_shapes,
                 uint32_t output_id, std::vector<X>& output_data,
                 bool optimize) {
  Runtime runtime(subgraph, nullptr, 0, optimize);
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

// Runs fusion on the subgraph and checks the structure if `subgraph_check` is
// provided.
void FuseAndCheck(SubgraphBuilder& builder,
                  std::function<void(const ynn_subgraph&)> subgraph_check) {
  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph.fusion();
  subgraph.invalidate_dead_values();
  if (subgraph_check) {
    subgraph_check(subgraph);
  }
}

// Runs the subgraph with fusion disabled and enabled and checks that the
// outputs are the same.
template <typename A, typename X>
void RunFuseCompare(
    SubgraphBuilder& builder, const std::vector<uint32_t>& input_ids,
    const std::vector<std::vector<A>>& input_datas,
    const std::vector<TensorShape>& input_shapes, uint32_t output_id,
    size_t output_size,
    std::function<void(const ynn_subgraph&)> subgraph_check = nullptr) {
  std::vector<X> output_before(output_size);
  RunSubgraph<A, X>(builder.GetSubgraph(), input_ids, input_datas, input_shapes,
                    output_id, output_before, /*optimize=*/false);

  FuseAndCheck(builder, subgraph_check);

  std::vector<X> output_after(output_size);
  RunSubgraph<A, X>(builder.GetSubgraph(), input_ids, input_datas, input_shapes,
                    output_id, output_after, /*optimize=*/true);
  EXPECT_THAT(output_after, ElementsAreArray(output_before));
}

}  // namespace

TEST(fusion_lut, single_node_unsupported) {
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

  FuseAndCheck(builder, [&](const ynn_subgraph& subgraph) {
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
  });
}

TEST(fusion_lut, single_node_simple) {
  // y_int8 = sigmoid(x_int8).
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

  RunFuseCompare<int8_t, int8_t>(
      builder, {x_id}, {input_data}, {TensorShape({input_data.size()})}, y_id,
      256, [&](const ynn_subgraph& subgraph) {
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
      });
}

TEST(fusion_lut, single_node) {
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

  builder.AddInput(ynn_type_int8, {256}, a_id, zero_point_id, scale_id)
      .AddInput(ynn_type_int8, {256}, b_id, zero_point_id, scale_id)
      .AddInput(ynn_type_int8, {256}, c_id, zero_point_id, scale_id)
      .AddOutput(ynn_type_int8, {256}, d_id, zero_point_id, scale_id);

  builder
      .AddTensor(ynn_type_int8, {256}, x_id, /*data=*/nullptr, zero_point_id,
                 scale_id)
      .AddTensor(ynn_type_int8, {256}, y_id, /*data=*/nullptr, zero_point_id,
                 scale_id);

  builder.AddBinary(ynn_binary_multiply, a_id, b_id, x_id)
      .AddUnary(ynn_unary_sigmoid, x_id, y_id)
      .AddBinary(ynn_binary_add, y_id, c_id, d_id);

  std::vector<int8_t> a_data(256);
  std::iota(a_data.begin(), a_data.end(), -128);
  std::vector<int8_t> b_data(256, 2);
  std::vector<int8_t> c_data(256, 3);

  RunFuseCompare<int8_t, int8_t>(
      builder, {a_id, b_id, c_id}, {a_data, b_data, c_data},
      {TensorShape({a_data.size()}), TensorShape({b_data.size()}),
       TensorShape({c_data.size()})},
      d_id, 256, [&](const ynn_subgraph& subgraph) {
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
      });
}

TEST(fusion_lut, multiple_unary_chain) {
  // x -> (convert) -> a -> (square) -> b -> (exp) -> c -> (convert) -> y.
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  uint32_t a_id = 2;
  uint32_t b_id = 3;
  uint32_t c_id = 4;

  SubgraphBuilder builder(/*external_value_count=*/8);

  builder.AddInput(ynn_type_uint8, {256}, x_id)
      .AddOutput(ynn_type_uint8, {256}, y_id);

  builder.AddTensor(ynn_type_fp32, {256}, a_id)
      .AddTensor(ynn_type_fp32, {256}, b_id)
      .AddTensor(ynn_type_fp32, {256}, c_id);

  builder.AddUnary(ynn_unary_convert, x_id, a_id)
      .AddUnary(ynn_unary_square, a_id, b_id)
      .AddUnary(ynn_unary_exp, b_id, c_id)
      .AddUnary(ynn_unary_convert, c_id, y_id);

  std::vector<uint8_t> input_data(256);
  std::iota(input_data.begin(), input_data.end(), -128);

  RunFuseCompare<uint8_t, uint8_t>(
      builder, {x_id}, {input_data}, {TensorShape({input_data.size()})}, y_id,
      256, [&](const ynn_subgraph& subgraph) {
        ASSERT_EQ(valid_node_count(&subgraph), 1);
        const ynn_node* output = subgraph.get_producer(y_id);
        ASSERT_NE(output, nullptr);
        ASSERT_TRUE(is_lut(*output));
        ASSERT_THAT(output->inputs,
                    ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
        EXPECT_TRUE(subgraph.value(output->inputs[1]).is_valid());
      });
}

TEST(fusion_lut, elu_chain) {
  // Implements ELU operation:
  //
  // if x > 0, y = x.
  // if x <= 0, y = alpha * (exp(x) - 1).
  //
  // This graph involves nodes with multiple consumers (a has 2 consumers) that
  // eventually feed into a single output node. It also involves binary nodes
  // with constants.
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

  RunFuseCompare<uint8_t, uint8_t>(
      builder, {x_id}, {input_data}, {TensorShape({4, 8, 8})}, y_id, 256,
      [&](const ynn_subgraph& subgraph) {
        ASSERT_EQ(valid_node_count(&subgraph), 1);
        const ynn_node* output = subgraph.get_producer(y_id);
        ASSERT_NE(output, nullptr);
        ASSERT_TRUE(is_lut(*output));
        // LUT inputs: First is index (x), second is table (generated const).
        ASSERT_THAT(output->inputs,
                    ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
        EXPECT_TRUE(subgraph.value(output->inputs[1]).is_valid());
      });
}

TEST(fusion_lut, branching_2_luts) {
  // x -> (abs) -> t -> (ceil) -> y
  //                 \ -> (floor) -> z.
  //
  // Results in 2 luts:
  // x -> (lut_xy) -> y
  //     \
  //      -> (lut_xz) -> z
  uint32_t x_id = 0;
  uint32_t t_id = 1;
  uint32_t y_id = 2;
  uint32_t z_id = 3;
  uint32_t scale_id = 4;
  uint32_t zero_point_id = 5;

  static const float scale_val[] = {1.0f};
  static const int32_t zero_point_val[] = {0};

  SubgraphBuilder builder(/*external_value_count=*/6);

  builder.AddTensor(ynn_type_fp32, {1}, scale_id, scale_val)
      .AddTensor(ynn_type_int32, {1}, zero_point_id, zero_point_val);

  builder.AddInput(ynn_type_int8, {256}, x_id, zero_point_id, scale_id)
      .AddTensor(ynn_type_int8, {256}, t_id, /*data=*/nullptr, zero_point_id,
                 scale_id)
      .AddOutput(ynn_type_int8, {256}, y_id, zero_point_id, scale_id)
      .AddOutput(ynn_type_int8, {256}, z_id, zero_point_id, scale_id);

  builder.AddUnary(ynn_unary_abs, x_id, t_id)
      .AddUnary(ynn_unary_ceil, t_id, y_id)
      .AddUnary(ynn_unary_floor, t_id, z_id);

  FuseAndCheck(builder, [&](const ynn_subgraph& subgraph) {
    EXPECT_EQ(valid_node_count(&subgraph), 2);

    const ynn_node* y_producer = subgraph.get_producer(y_id);
    ASSERT_NE(y_producer, nullptr);
    ASSERT_TRUE(is_lut(*y_producer));
    ASSERT_THAT(y_producer->inputs,
                ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
    EXPECT_TRUE(subgraph.value(y_producer->inputs[1]).is_valid());

    const ynn_node* z_producer = subgraph.get_producer(z_id);
    ASSERT_NE(z_producer, nullptr);
    ASSERT_TRUE(is_lut(*z_producer));
    ASSERT_THAT(z_producer->inputs,
                ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
    EXPECT_TRUE(subgraph.value(z_producer->inputs[1]).is_valid());
  });
}

TEST(fusion_lut, input_type_unsupported) {
  // x_fp32 -> exp -> a_fp32 -> convert -> y_uint8
  uint32_t x_id = 0;
  uint32_t a_id = 1;
  uint32_t y_id = 2;

  SubgraphBuilder builder(/*external_value_count=*/3);

  builder.AddInput(ynn_type_fp32, {256}, x_id)
      .AddTensor(ynn_type_fp32, {256}, a_id)
      .AddOutput(ynn_type_uint8, {256}, y_id);

  builder.AddUnary(ynn_unary_exp, x_id, a_id)
      .AddUnary(ynn_unary_convert, a_id, y_id);

  FuseAndCheck(builder, [&](const ynn_subgraph& subgraph) {
    // We expect 3 nodes: `exp`, `convert` and `make_unary_params` node attached
    // to `convert`.
    EXPECT_EQ(valid_node_count(&subgraph), 3);

    const ynn_node* a_producer = subgraph.get_producer(a_id);
    ASSERT_NE(a_producer, nullptr);
    const ynn_node::unary_elementwise* exp_op =
        std::get_if<ynn_node::unary_elementwise>(&a_producer->op);
    ASSERT_NE(exp_op, nullptr);
    EXPECT_EQ(exp_op->op, ynn_unary_exp);

    const ynn_node* y_producer = subgraph.get_producer(y_id);
    ASSERT_NE(y_producer, nullptr);
    ASSERT_FALSE(is_lut(*y_producer));
    const ynn_node::unary_elementwise* convert_op =
        std::get_if<ynn_node::unary_elementwise>(&y_producer->op);
    ASSERT_NE(convert_op, nullptr);
    EXPECT_EQ(convert_op->op, ynn_unary_convert);
  });
}

TEST(fusion_lut, binary_scalar_constant) {
  // x -> convert -> a
  // a + C -> b
  // b -> convert -> y
  // If C is scalar, it should fuse to LUT.
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  uint32_t a_id = 2;
  uint32_t b_id = 3;
  uint32_t c_id = 4;

  SubgraphBuilder builder(/*external_value_count=*/5);

  builder.AddInput(ynn_type_uint8, {256}, x_id)
      .AddOutput(ynn_type_uint8, {256}, y_id);

  builder.AddTensor(ynn_type_fp32, {256}, a_id)
      .AddTensor(ynn_type_fp32, {256}, b_id);

  static const float c_val[] = {1.0f};
  builder.AddTensor(ynn_type_fp32, {1}, c_id, c_val);

  builder.AddUnary(ynn_unary_convert, x_id, a_id)
      .AddBinary(ynn_binary_add, a_id, c_id, b_id)
      .AddUnary(ynn_unary_convert, b_id, y_id);

  std::vector<uint8_t> input_data(256);
  std::iota(input_data.begin(), input_data.end(), 0);

  RunFuseCompare<uint8_t, uint8_t>(
      builder, {x_id}, {input_data}, {TensorShape({input_data.size()})}, y_id,
      256, [&](const ynn_subgraph& subgraph) {
        ASSERT_EQ(valid_node_count(&subgraph), 1);
        const ynn_node* output = subgraph.get_producer(y_id);
        ASSERT_NE(output, nullptr);
        ASSERT_TRUE(is_lut(*output));
        ASSERT_THAT(output->inputs,
                    ElementsAre(x_id, testing::Ne(YNN_INVALID_VALUE_ID)));
        EXPECT_TRUE(subgraph.value(output->inputs[1]).is_valid());
      });
}

TEST(fusion_lut, binary_nonscalar_constant_unsupported) {
  // x -> convert -> a
  // a + C -> b
  // b -> convert -> y
  // If C is not scalar, it should not fuse to LUT.
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  uint32_t a_id = 2;
  uint32_t b_id = 3;
  uint32_t c_id = 4;

  SubgraphBuilder builder(/*external_value_count=*/5);

  builder.AddInput(ynn_type_uint8, {256}, x_id)
      .AddOutput(ynn_type_uint8, {256}, y_id);

  builder.AddTensor(ynn_type_fp32, {256}, a_id)
      .AddTensor(ynn_type_fp32, {256}, b_id);

  std::vector<float> c_val(256);
  std::iota(c_val.begin(), c_val.end(), 0.0f);
  builder.AddTensor(ynn_type_fp32, {256}, c_id, c_val.data());

  builder.AddUnary(ynn_unary_convert, x_id, a_id)
      .AddBinary(ynn_binary_add, a_id, c_id, b_id)
      .AddUnary(ynn_unary_convert, b_id, y_id);

  std::vector<uint8_t> input_data(256);
  std::iota(input_data.begin(), input_data.end(), 0);

  RunFuseCompare<uint8_t, uint8_t>(
      builder, {x_id}, {input_data}, {TensorShape({input_data.size()})}, y_id,
      256, [&](const ynn_subgraph& subgraph) {
        const ynn_node* y_producer = subgraph.get_producer(y_id);
        ASSERT_NE(y_producer, nullptr);
        ASSERT_FALSE(is_lut(*y_producer));
        const ynn_node::unary_elementwise* convert_op =
            std::get_if<ynn_node::unary_elementwise>(&y_producer->op);
        ASSERT_NE(convert_op, nullptr);
        EXPECT_EQ(convert_op->op, ynn_unary_convert);

        const ynn_node* b_producer = subgraph.get_producer(b_id);
        ASSERT_NE(b_producer, nullptr);
        ASSERT_TRUE(is_binary(*b_producer, ynn_binary_add));
      });
}

}  // namespace ynn
