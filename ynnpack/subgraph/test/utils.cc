// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/utils.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

TEST(CloneSubgraphSubset, SimpleChain) {
  // Original subgraph:
  //   a -> (abs) -> b -> (negate) -> c.
  //
  // Cloned subgraph (input a, output b):
  //   a -> (abs) -> b.
  const int kShapeSize = 10;
  uint32_t a_id = 0;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t c_id = YNN_INVALID_VALUE_ID;

  // Create original subgraph.
  SubgraphBuilder builder(/*external_value_count=*/1);
  builder.AddInput(ynn_type_fp32, {kShapeSize}, a_id);
  // Make `b_id` an external output in order to check for numerical correctness.
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, b_id, /*data=*/nullptr,
                    /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                    /*scale_id=*/YNN_INVALID_VALUE_ID,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, c_id);
  builder.AddUnary(ynn_unary_abs, a_id, b_id)
      .AddUnary(ynn_unary_negate, b_id, c_id);
  const ynn_subgraph& original = *builder.GetSubgraph();

  std::cout << "Original subgraph: " << "\n";
  original.dump(std::cout);

  // Clone subgraph.
  uint32_t cloned_a_id = YNN_INVALID_VALUE_ID;
  uint32_t cloned_b_id = YNN_INVALID_VALUE_ID;
  auto cloned_subgraph =
      clone_subgraph_subset(original, a_id, b_id, cloned_a_id, cloned_b_id);

  std::cout << "Cloned subgraph: " << "\n";
  cloned_subgraph->dump(std::cout);

  ASSERT_TRUE(cloned_subgraph.has_value());
  EXPECT_EQ(cloned_subgraph->values.size(), 2);
  EXPECT_EQ(cloned_subgraph->nodes.size(), 1);
  EXPECT_NE(cloned_a_id, YNN_INVALID_VALUE_ID);
  EXPECT_NE(cloned_b_id, YNN_INVALID_VALUE_ID);

  const auto& node = cloned_subgraph->nodes[0];
  const auto* op = std::get_if<ynn_node::unary_elementwise>(&node.op);
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->op, ynn_unary_abs);

  // Check numerical correctness of the cloned subgraph.
  TestScheduler scheduler(/*thread_count=*/4);
  Runtime orig_runtime(builder.GetSubgraph(), &scheduler);
  ASSERT_EQ(orig_runtime.Status(), ynn_status_success);
  Runtime cloned_runtime(&*cloned_subgraph, &scheduler);
  ASSERT_EQ(cloned_runtime.Status(), ynn_status_success);

  std::vector<float> a_data(kShapeSize);
  std::iota(a_data.begin(), a_data.end(), -5.0f);
  std::vector<float> b_data(kShapeSize);
  std::vector<float> cloned_b_data(kShapeSize);

  orig_runtime.ReshapeExternalTensor({kShapeSize}, a_data.data(), a_id)
      .ReshapeRuntime()
      .SetupExternalTensor(b_data.data(), b_id)
      .InvokeRuntime();

  cloned_runtime.ReshapeExternalTensor({kShapeSize}, a_data.data(), cloned_a_id)
      .ReshapeRuntime()
      .SetupExternalTensor(cloned_b_data.data(), cloned_b_id)
      .InvokeRuntime();

  for (size_t i = 0; i < b_data.size(); ++i) {
    EXPECT_FLOAT_EQ(cloned_b_data[i], b_data[i]);
  }
}

TEST(CloneSubgraphSubset, BranchingFails) {
  // Original subgraph:
  // a -> (abs) -> b ->
  //                    \ -> (add) -> c
  // d -> (negate) -> e ->
  const int kShapeSize = 10;
  uint32_t a_id = 0, d_id = 1;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t c_id = YNN_INVALID_VALUE_ID;
  uint32_t e_id = YNN_INVALID_VALUE_ID;

  // Create original subgraph.
  SubgraphBuilder builder(/*external_value_count=*/2);
  builder.AddInput(ynn_type_fp32, {kShapeSize}, a_id)
      .AddInput(ynn_type_fp32, {kShapeSize}, d_id);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, b_id)
      .AddTensor(ynn_type_fp32, {kShapeSize}, e_id);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, c_id, /*data=*/nullptr,
                    /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                    /*scale_id=*/YNN_INVALID_VALUE_ID,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  builder.AddUnary(ynn_unary_abs, a_id, b_id)
      .AddUnary(ynn_unary_negate, d_id, e_id)
      .AddBinary(ynn_binary_add, b_id, e_id, c_id);
  const ynn_subgraph& original = *builder.GetSubgraph();

  // Clone subgraph.
  uint32_t cloned_a_id = YNN_INVALID_VALUE_ID;
  uint32_t cloned_c_id = YNN_INVALID_VALUE_ID;
  auto cloned_subgraph =
      clone_subgraph_subset(original, a_id, c_id, cloned_a_id, cloned_c_id);
  EXPECT_FALSE(cloned_subgraph.has_value());
}

TEST(CloneSubgraphSubset, MiddleCut) {
  // Original subgraph:
  //   a -> (abs) -> b -> (negate) -> c -> (square) -> d.
  //
  // Cloned subgraph (input b, output c):
  //   b -> (negate) -> c.
  const int kShapeSize = 10;
  uint32_t a_id = 0;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t c_id = YNN_INVALID_VALUE_ID;
  uint32_t d_id = YNN_INVALID_VALUE_ID;

  // Create original subgraph.
  SubgraphBuilder builder(/*external_value_count=*/1);
  builder.AddInput(ynn_type_fp32, {kShapeSize}, a_id);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, b_id, /*data=*/nullptr,
                    /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                    /*scale_id=*/YNN_INVALID_VALUE_ID,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, c_id, /*data=*/nullptr,
                    /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                    /*scale_id=*/YNN_INVALID_VALUE_ID,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, d_id);
  builder.AddUnary(ynn_unary_abs, a_id, b_id)
      .AddUnary(ynn_unary_negate, b_id, c_id)
      .AddUnary(ynn_unary_square, c_id, d_id);
  const ynn_subgraph& original = *builder.GetSubgraph();

  // Clone subgraph.
  uint32_t cloned_b_id = YNN_INVALID_VALUE_ID;
  uint32_t cloned_c_id = YNN_INVALID_VALUE_ID;
  auto cloned_subgraph =
      clone_subgraph_subset(original, b_id, c_id, cloned_b_id, cloned_c_id);
  EXPECT_EQ(cloned_subgraph->nodes.size(), 1);
  EXPECT_NE(cloned_b_id, YNN_INVALID_VALUE_ID);
  EXPECT_NE(cloned_c_id, YNN_INVALID_VALUE_ID);

  const auto& node = cloned_subgraph->nodes[0];
  const auto* op = std::get_if<ynn_node::unary_elementwise>(&node.op);
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->op, ynn_unary_negate);

  // Check numerical correctness of the cloned subgraph.
  TestScheduler scheduler(/*thread_count=*/4);
  Runtime orig_runtime(builder.GetSubgraph(), &scheduler);
  ASSERT_EQ(orig_runtime.Status(), ynn_status_success);
  Runtime subset_runtime(&*cloned_subgraph, &scheduler);
  ASSERT_EQ(subset_runtime.Status(), ynn_status_success);

  std::vector<float> a_data(kShapeSize);
  std::iota(a_data.begin(), a_data.end(), -5.0f);
  std::vector<float> b_data(kShapeSize), c_data(kShapeSize),
      cloned_c_data(kShapeSize);

  orig_runtime.ReshapeExternalTensor({kShapeSize}, a_data.data(), a_id)
      .ReshapeRuntime()
      .SetupExternalTensor(b_data.data(), b_id)
      .SetupExternalTensor(c_data.data(), c_id)
      .InvokeRuntime();

  subset_runtime.ReshapeExternalTensor({kShapeSize}, b_data.data(), cloned_b_id)
      .ReshapeRuntime()
      .SetupExternalTensor(cloned_c_data.data(), cloned_c_id)
      .InvokeRuntime();

  for (size_t i = 0; i < c_data.size(); ++i) {
    EXPECT_FLOAT_EQ(cloned_c_data[i], c_data[i]);
  }
}

TEST(CloneSubgraphSubset, QuantizationParams) {
  // Original subgraph:
  //   a -> (abs) -> b.
  //
  // Cloned subgraph (input a, output b):
  //   a -> (abs) -> b.
  const int kShapeSize = 10;
  uint32_t a_id = 0;
  uint32_t b_id = YNN_INVALID_VALUE_ID;

  // Create original subgraph.
  SubgraphBuilder builder(/*external_value_count=*/1);
  uint32_t a_zero_point_id = builder.DefineScalar<int32_t>(-128);
  uint32_t a_scale_id = builder.DefineScalar(0.5f);
  uint32_t b_zero_point_id = builder.DefineScalar<int32_t>(128);
  uint32_t b_scale_id = builder.DefineScalar(0.25f);
  builder.AddInput(ynn_type_int8, {kShapeSize}, a_id, a_zero_point_id,
                   a_scale_id);
  builder.AddTensor(ynn_type_int8, {kShapeSize}, b_id, nullptr, b_zero_point_id,
                    b_scale_id,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  builder.AddUnary(ynn_unary_abs, a_id, b_id);
  const ynn_subgraph& subgraph = *builder.GetSubgraph();

  // Clone subgraph.
  uint32_t cloned_a_id = YNN_INVALID_VALUE_ID;
  uint32_t cloned_b_id = YNN_INVALID_VALUE_ID;
  auto cloned_subgraph =
      clone_subgraph_subset(subgraph, a_id, b_id, cloned_a_id, cloned_b_id);
  // Should have a, b and their quantization params.
  EXPECT_GE(cloned_subgraph->values.size(), 6);
  EXPECT_NE(cloned_a_id, YNN_INVALID_VALUE_ID);
  EXPECT_NE(cloned_b_id, YNN_INVALID_VALUE_ID);

  // Make sure that quantization params are present in the cloned subgraph.
  bool found_a_scale = false;
  bool found_b_scale = false;
  bool found_a_zero_point = false;
  bool found_b_zero_point = false;
  for (const auto& val : cloned_subgraph->values) {
    if (val.is_static()) {
      if (val.type == ynn_type_fp32) {
        auto opt_val = val.as_scalar_float();
        if (opt_val.has_value() && *opt_val == 0.5f) {
          found_a_scale = true;
        } else if (opt_val.has_value() && *opt_val == 0.25f) {
          found_b_scale = true;
        }
      } else if (val.type == ynn_type_int32) {
        auto val_int = val.static_scalar_value<int32_t>();
        if (val_int == -128) {
          found_a_zero_point = true;
        } else if (val_int == 128) {
          found_b_zero_point = true;
        }
      }
    }
  }
  EXPECT_TRUE(found_a_scale);
  EXPECT_TRUE(found_b_scale);
  EXPECT_TRUE(found_a_zero_point);
  EXPECT_TRUE(found_b_zero_point);

  // Check numerical correctness of the cloned subgraph.
  TestScheduler scheduler(/*thread_count=*/4);
  Runtime orig_runtime(builder.GetSubgraph(), &scheduler);
  ASSERT_EQ(orig_runtime.Status(), ynn_status_success);
  Runtime subset_runtime(&*cloned_subgraph, &scheduler);
  ASSERT_EQ(subset_runtime.Status(), ynn_status_success);

  std::vector<int8_t> a_data(kShapeSize);
  for (int i = 0; i < kShapeSize; ++i) a_data[i] = i - 5;
  std::vector<int8_t> b_data(kShapeSize), cloned_b_data(kShapeSize);

  orig_runtime.ReshapeExternalTensor({kShapeSize}, a_data.data(), a_id)
      .ReshapeRuntime()
      .SetupExternalTensor(b_data.data(), b_id)
      .InvokeRuntime();

  subset_runtime.ReshapeExternalTensor({kShapeSize}, a_data.data(), cloned_a_id)
      .ReshapeRuntime()
      .SetupExternalTensor(cloned_b_data.data(), cloned_b_id)
      .InvokeRuntime();

  for (size_t i = 0; i < b_data.size(); ++i) {
    EXPECT_EQ(cloned_b_data[i], b_data[i]);
  }
}

TEST(CloneSubgraphSubset, DisconnectedInput) {
  // Original subgraph:
  //   a -> (abs) -> b
  //   c -> (negate) -> d
  //
  // Cloned subgraph (input c, output b):
  //   Should be nullptr.
  const int kShapeSize = 10;
  uint32_t a_id = 0;
  uint32_t c_id = 1;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t d_id = YNN_INVALID_VALUE_ID;

  // Create original subgraph.
  SubgraphBuilder builder(/*external_value_count=*/2);
  builder.AddInput(ynn_type_fp32, {kShapeSize}, a_id);
  builder.AddInput(ynn_type_fp32, {kShapeSize}, c_id);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, b_id, /*data=*/nullptr,
                    /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                    /*scale_id=*/YNN_INVALID_VALUE_ID,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  builder.AddTensor(ynn_type_fp32, {kShapeSize}, d_id);
  builder.AddUnary(ynn_unary_abs, a_id, b_id)
      .AddUnary(ynn_unary_negate, c_id, d_id);
  const ynn_subgraph& original = *builder.GetSubgraph();

  // Clone subgraph.
  uint32_t cloned_b_id = YNN_INVALID_VALUE_ID;
  uint32_t cloned_c_id = YNN_INVALID_VALUE_ID;
  auto cloned_subgraph =
      clone_subgraph_subset(original, c_id, b_id, cloned_c_id, cloned_b_id);
  EXPECT_FALSE(cloned_subgraph.has_value());
}

}  // namespace
}  // namespace ynn
