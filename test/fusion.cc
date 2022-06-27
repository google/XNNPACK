// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include "runtime-tester.h"
#include <gtest/gtest.h>

TEST(ADD_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester<float>(4);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input1_id = 0;
  uint32_t input2_id = 1;
  uint32_t intermediate_id = 2;
  uint32_t output_id = 3;
  tester
    .AddInputTensor({1, 2, 2, 3}, input1_id)
    .AddInputTensor({1, 2, 2, 3}, input2_id)
    .AddTensor({1, 2, 2, 3}, kDynamic, intermediate_id)
    .AddOutputTensor({1, 2, 2, 3}, output_id)
    .AddAddition(input1_id, input2_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}
