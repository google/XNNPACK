// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

using ::testing::AllOf;
using ::testing::Each;
using ::testing::Eq;

TEST(SumToDotRankTest, HighRankSumOfMultiplyReduction) {
  // Test high rank (rank 8) sum of a multiply reduction:
  // obs shape: {10, 4, 6}
  // obs_expanded shape: {10, 4, 6, 1, 1, 1, 1, 1}
  // A shape:   {10, 4, 6, 9, 9, 9, 9, 9}
  // mul = obs_expanded * A  -> {10, 4, 6, 9, 9, 9, 9, 9}
  // lik = sum(mul, axis=2)  -> {10, 4, 9, 9, 9, 9, 9}
  const std::vector<size_t> obs_shape = {10, 4, 6};
  const std::vector<size_t> a_shape = {10, 4, 6, 9, 9, 9, 9, 9};

  SubgraphBuilder subgraph(3, 0);

  const uint32_t obs_id = 0;
  const uint32_t a_id = 1;
  const uint32_t output_id = 2;

  subgraph.AddInput(ynn_type_fp32, obs_shape, obs_id)
      .AddInput(ynn_type_fp32, a_shape, a_id)
      .AddOutput(ynn_type_fp32, 7, output_id);

  uint32_t obs_expanded_id = YNN_INVALID_VALUE_ID;
  uint32_t mul_id = YNN_INVALID_VALUE_ID;
  uint32_t init_id = YNN_INVALID_VALUE_ID;

  subgraph.AddTensor(ynn_type_fp32, 8, obs_expanded_id)
      .AddTensor(ynn_type_fp32, a_shape, mul_id)
      .AddScalar<float>(0.0f, init_id);

  subgraph.AddExpandDims({3, 4, 5, 6, 7}, obs_id, obs_expanded_id)
      .AddBinary(ynn_binary_multiply, obs_expanded_id, a_id, mul_id)
      .AddReduce(ynn_reduce_sum, {2}, mul_id, init_id, output_id,
                 /*flags=*/0);

  TestScheduler scheduler(3);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> obs(obs_shape);
  obs.fill(2.0f);

  Tensor<float> a(a_shape);
  a.fill(3.0f);

  runtime.ReshapeExternalTensor(obs_shape, obs.data(), obs_id);
  runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);

  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const std::vector<size_t> output_shape = {10, 4, 9, 9, 9, 9, 9};
  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), output_shape);

  Tensor<float> output(output_shape);
  runtime.SetupExternalTensor(output.data(), output_id).InvokeRuntime();
  EXPECT_EQ(runtime.Status(), ynn_status_success);

  EXPECT_THAT(output, AllOf(Each(Eq(36.0f))));
}

}  // namespace
}  // namespace ynn
