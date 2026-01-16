// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/fusion.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

using ::testing::UnorderedElementsAre;

TEST(FindUnaryLut, SingleNodeInt8ToInt8) {
  // x -> (negate) -> y.
  SubgraphBuilder builder(/*external_value_count=*/2);
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  builder.AddInput(ynn_type_int8, 2, x_id).AddOutput(ynn_type_int8, 2, y_id);
  builder.AddUnary(ynn_unary_negate, x_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph_analysis analysis(subgraph);

  const ynn_node* node = subgraph.get_producer(y_id);
  ASSERT_NE(node, nullptr);

  subgraph_candidate candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*node), analysis);

  EXPECT_EQ(candidate.size, 1);
  EXPECT_EQ(candidate.input_id, x_id);
  EXPECT_EQ(candidate.output_id, y_id);
  EXPECT_THAT(candidate.nodes, UnorderedElementsAre(node));
}

TEST(FindUnaryLut, ChainInt8ToFloatToInt8) {
  // x -> (convert) -> t -> (convert) -> y.
  SubgraphBuilder builder(/*external_value_count=*/3);
  uint32_t x_id = 0;
  uint32_t t_id = 1;
  uint32_t y_id = 2;
  builder.AddInput(ynn_type_int8, 2, x_id)
      .AddTensor(ynn_type_fp32, 2, t_id)
      .AddOutput(ynn_type_int8, 2, y_id);
  builder.AddUnary(ynn_unary_convert, x_id, t_id)
      .AddUnary(ynn_unary_convert, t_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph_analysis analysis(subgraph);

  const ynn_node* convert_1 = subgraph.get_producer(t_id);
  subgraph_candidate candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*convert_1), analysis);

  // `t_id` is fp32 so should be invalid.
  EXPECT_EQ(candidate.size, 0);

  const ynn_node* convert_2 = subgraph.get_producer(y_id);
  candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*convert_2), analysis);

  EXPECT_EQ(candidate.size, 2);
  EXPECT_EQ(candidate.input_id, x_id);
  EXPECT_EQ(candidate.output_id, y_id);
  EXPECT_THAT(candidate.nodes, UnorderedElementsAre(convert_1, convert_2));
  EXPECT_THAT(candidate.values, UnorderedElementsAre(t_id));
}

TEST(FindUnaryLut, BinaryWithConstant) {
  //     x ->
  //         \ -> (add) -> y
  // c_const ->
  SubgraphBuilder builder(3);
  uint32_t x_id = 0;
  uint32_t c_id = 1;
  uint32_t y_id = 2;
  int8_t const_val = 1;
  builder.AddInput(ynn_type_int8, 2, x_id)
      .AddTensor(ynn_type_int8, {1}, c_id, &const_val)
      .AddOutput(ynn_type_int8, 2, y_id);
  builder.AddBinary(ynn_binary_add, x_id, c_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph_analysis analysis(subgraph);

  const ynn_node* node = subgraph.get_producer(y_id);
  subgraph_candidate candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*node), analysis);

  EXPECT_EQ(candidate.size, 1);
  EXPECT_EQ(candidate.input_id, x_id);
  EXPECT_THAT(candidate.nodes, UnorderedElementsAre(node));
}

// Long chain with unary and binary nodes:
//
//    x ->
//        \ -> (multiply) -> b
// a_const ->
//
//    b ->
//        \ -> (add) -> d -> (convert) -> e -> (sigmoid) -> f -> (convert) -> y
// c_const ->
//
TEST(FindUnaryLut, UnaryAndBinarywithConstants) {
  SubgraphBuilder builder(8);
  uint32_t x_id = 0;
  uint32_t a_id = 1;
  uint32_t b_id = 2;
  uint32_t c_id = 3;
  uint32_t d_id = 4;
  uint32_t e_id = 5;
  uint32_t f_id = 6;
  uint32_t y_id = 7;

  int8_t a_val = 2;
  int8_t c_val = 3;

  builder.AddInput(ynn_type_int8, 2, x_id)
      .AddTensor(ynn_type_int8, {1}, a_id, &a_val)
      .AddTensor(ynn_type_int8, 2, b_id)
      .AddTensor(ynn_type_int8, {1}, c_id, &c_val)
      .AddTensor(ynn_type_int8, 2, d_id)
      .AddTensor(ynn_type_fp32, 2, e_id)
      .AddTensor(ynn_type_fp32, 2, f_id)
      .AddOutput(ynn_type_int8, 2, y_id);

  builder.AddBinary(ynn_binary_multiply, x_id, a_id, b_id)
      .AddBinary(ynn_binary_add, b_id, c_id, d_id)
      .AddUnary(ynn_unary_convert, d_id, e_id)
      .AddUnary(ynn_unary_sigmoid, e_id, f_id)
      .AddUnary(ynn_unary_convert, f_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph_analysis analysis(subgraph);

  const ynn_node* mul_node = subgraph.get_producer(b_id);
  const ynn_node* add_node = subgraph.get_producer(d_id);
  const ynn_node* convert_1 = subgraph.get_producer(e_id);
  const ynn_node* sigmoid_node = subgraph.get_producer(f_id);
  const ynn_node* convert_2 = subgraph.get_producer(y_id);

  // We should expect multiple candidates.
  subgraph_candidate candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*add_node), analysis);
  EXPECT_EQ(candidate.size, 2);
  EXPECT_EQ(candidate.input_id, x_id);
  EXPECT_EQ(candidate.output_id, d_id);
  EXPECT_THAT(candidate.nodes, UnorderedElementsAre(mul_node, add_node));

  candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*mul_node), analysis);
  EXPECT_EQ(candidate.size, 1);
  EXPECT_EQ(candidate.input_id, x_id);
  EXPECT_EQ(candidate.output_id, b_id);
  EXPECT_THAT(candidate.nodes, UnorderedElementsAre(mul_node));

  candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*convert_2), analysis);
  EXPECT_EQ(candidate.size, 5);
  EXPECT_EQ(candidate.input_id, x_id);
  EXPECT_EQ(candidate.output_id, y_id);
  EXPECT_THAT(candidate.nodes,
              UnorderedElementsAre(mul_node, add_node, convert_1, sigmoid_node,
                                   convert_2));
  EXPECT_THAT(candidate.values, UnorderedElementsAre(b_id, d_id, e_id, f_id));
}

TEST(FindUnaryLut, EluChain) {
  // x -> (convert) -> a
  //
  //  a ->
  //      \ -> (min) -> b -> (exp) -> c
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
  SubgraphBuilder builder(9);
  uint32_t x_id = 0;
  uint32_t a_id = 1;
  uint32_t zero_id = 2;
  uint32_t b_id = 3;
  uint32_t c_id = 4;
  uint32_t d_id = 5;
  uint32_t e_id = 6;
  uint32_t f_id = 7;
  uint32_t y_id = 8;

  float zero_val = 0.0f;
  float d_val = 1.0f;

  builder.AddInput(ynn_type_int8, 2, x_id)
      .AddTensor(ynn_type_fp32, 2, a_id)
      .AddTensor(ynn_type_fp32, {1}, zero_id, &zero_val)
      .AddTensor(ynn_type_fp32, 2, b_id)
      .AddTensor(ynn_type_fp32, 2, c_id)
      .AddTensor(ynn_type_fp32, {1}, d_id, &d_val)
      .AddTensor(ynn_type_fp32, 2, e_id)
      .AddTensor(ynn_type_fp32, 2, f_id)
      .AddOutput(ynn_type_int8, 2, y_id);

  builder.AddUnary(ynn_unary_convert, x_id, a_id)
      .AddBinary(ynn_binary_min, a_id, zero_id, b_id)
      .AddUnary(ynn_unary_exp, b_id, c_id)
      .AddBinary(ynn_binary_multiply, c_id, d_id, e_id)
      .AddBinary(ynn_binary_max, a_id, e_id, f_id)
      .AddUnary(ynn_unary_convert, f_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph_analysis analysis(subgraph);

  const ynn_node* convert_1 = subgraph.get_producer(a_id);
  const ynn_node* min_node = subgraph.get_producer(b_id);
  const ynn_node* exp_node = subgraph.get_producer(c_id);
  const ynn_node* mul_node = subgraph.get_producer(e_id);
  const ynn_node* max_node = subgraph.get_producer(f_id);
  const ynn_node* convert_2 = subgraph.get_producer(y_id);

  subgraph_candidate candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*convert_2), analysis);

  EXPECT_EQ(candidate.size, 6);
  EXPECT_EQ(candidate.input_id, x_id);
  EXPECT_EQ(candidate.output_id, y_id);
  EXPECT_THAT(candidate.nodes,
              UnorderedElementsAre(convert_1, min_node, exp_node, mul_node,
                                   max_node, convert_2));
  EXPECT_THAT(candidate.values,
              UnorderedElementsAre(a_id, b_id, c_id, e_id, f_id));
}

TEST(FindUnaryLut, BranchingInvalidates) {
  // x -> (negate) -> t -> (negate) -> y
  //                   \ -> (negate) -> z.
  SubgraphBuilder builder(5);
  uint32_t x_id = 0;
  uint32_t t_id = 1;
  uint32_t y_id = 2;
  uint32_t z_id = 3;

  builder.AddInput(ynn_type_int8, 2, x_id)
      .AddTensor(ynn_type_int8, 2, t_id)
      .AddOutput(ynn_type_int8, 2, y_id)
      .AddOutput(ynn_type_int8, 2, z_id);

  builder.AddUnary(ynn_unary_negate, x_id, t_id)
      .AddUnary(ynn_unary_negate, t_id, y_id)
      .AddUnary(ynn_unary_negate, t_id, z_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph_analysis analysis(subgraph);

  const ynn_node* negate_y = subgraph.get_producer(y_id);
  subgraph_candidate candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*negate_y), analysis);
  EXPECT_EQ(candidate.size, 1);
  EXPECT_EQ(candidate.input_id, t_id);
  EXPECT_EQ(candidate.output_id, y_id);
  EXPECT_THAT(candidate.nodes, UnorderedElementsAre(negate_y));

  const ynn_node* negate_z = subgraph.get_producer(z_id);
  candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*negate_z), analysis);
  EXPECT_EQ(candidate.size, 1);
}

TEST(FindUnaryLut, InvalidWrongInputType) {
  // x(float) -> convert -> y(int8)
  SubgraphBuilder builder(3);
  uint32_t x_id = 0;
  uint32_t y_id = 1;
  builder.AddInput(ynn_type_fp32, 2, x_id).AddOutput(ynn_type_int8, 2, y_id);
  builder.AddUnary(ynn_unary_convert, x_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  subgraph_analysis analysis(subgraph);

  const ynn_node* node = subgraph.get_producer(y_id);
  subgraph_candidate candidate = find_subgraph_for_unary_lut(
      subgraph, const_cast<ynn_node&>(*node), analysis);
  EXPECT_EQ(candidate.size, 0);
  EXPECT_EQ(candidate.input_id, YNN_INVALID_VALUE_ID);
}

}  // namespace
}  // namespace ynn
