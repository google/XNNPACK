// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include "runtime-tester.h"
#include "subgraph-tester.h"
#include <gtest/gtest.h>

namespace xnnpack {

TEST(SUBGRAPH, hanging_nodes) {
  auto tester = SubgraphTester(6);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, 1)
    .AddStaticTensorF32({32}, TensorType::kDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 32}, 3)
    .AddOutputTensorF32({32}, 4)
    .AddDynamicTensorF32({32}, 5)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{1, 1, 1, 1},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 32,
        }, 0, 1, 2, 3)
    .AddGlobalAveragePooling(3, 4)
    // Add hanging node
    .AddGlobalAveragePooling(3, 5)
    .Optimize();

  // The hanging node is still there.
  ASSERT_EQ(tester.NumNodes(), 3);
  // But empty.
  ASSERT_EQ(tester.Node(2)->compute_type, xnn_compute_type_invalid);
}

TEST(SUBGRAPH, multiple_outputs_with_hanging_nodes) {
  auto tester = SubgraphTester(4);
  tester
    .AddDynamicTensorF32({96}, 0)
    .AddDynamicTensorF32({32}, 1)
    .AddDynamicTensorF32({32}, 2)
    .AddOutputTensorF32({32}, 3)
    // Add split3 with 1 consumed output and two unconsumed outputs.
    .AddEvenSplit3(0, 1, 2, 3)
    .Optimize();

  // The node is still there.
  ASSERT_EQ(tester.NumNodes(), 1);
  // And all four values also.
  ASSERT_EQ(tester.NumValues(), 4);
  // The first two outputs are optimized away.
  ASSERT_EQ(tester.Value(1)->type, xnn_value_type_invalid);
  ASSERT_EQ(tester.Value(2)->type, xnn_value_type_invalid);
  // The last output is consumed.
  ASSERT_EQ(tester.Value(3)->type, xnn_value_type_dense_tensor);
}

TEST(SUBGRAPH, even_split3_first_two_outputs_optimized_away) {
  RuntimeTester tester(5);
  // auto tester = RuntimeTester(4);
  constexpr size_t size = 9;
  float inputs[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  tester
    .AddStaticTensorF32({size}, TensorType::kDense, 0, 0, inputs)
    .AddDynamicTensorF32({3}, 1)
    .AddDynamicTensorF32({3}, 2)
    .AddOutputTensorF32({3}, 3)
    // Add split3 with 1 consumed output and two unconsumed outputs.
    .AddEvenSplit3(0, 1, 2, 3);
  // Regression test for a crash where we could not deal with a split where the
  // 0th output is not used (and optimized away).
  auto output = tester.RunWithFusion<float>();
  std::vector<float> expected = {6, 7, 8};
  ASSERT_EQ(expected, output);
}

}  // namespace xnnpack
