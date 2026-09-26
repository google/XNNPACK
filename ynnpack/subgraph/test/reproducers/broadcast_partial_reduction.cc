// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

// Summing a broadcast over every axis with multiple threads splits the
// reduction into partial reductions computed in parallel. Slinky drops the
// trailing broadcast dimension from each tile's input, but the partial results
// still have an entry per tile in that dimension. The reduce kernel only looked
// at the input's dimensions when choosing which partial result to write, so
// each tile clobbered the partial results of the other tiles and the sum came
// out wrong, depending on how the tiles were scheduled.
TEST(ReduceTest, BroadcastPartialReduction) {
  constexpr size_t kInputSize = 1024;
  const std::vector<size_t> input_shape = {kInputSize};
  const std::vector<size_t> expanded_shape = {1, kInputSize, 1};
  const std::vector<size_t> broadcast_shape = {2, kInputSize, kInputSize};

  SubgraphBuilder subgraph(2);

  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  subgraph.AddInput(ynn_type_fp32, input_shape, input_id)
      .AddOutput(ynn_type_fp32, TensorShape(), output_id);

  uint32_t expanded_id = YNN_INVALID_VALUE_ID;
  uint32_t broadcast_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(ynn_type_fp32, expanded_shape, expanded_id)
      .AddTensor(ynn_type_fp32, broadcast_shape, broadcast_id);

  subgraph.AddExpandDims({0, 2}, input_id, expanded_id)
      .AddStaticBroadcast(broadcast_shape, expanded_id, broadcast_id)
      .AddReduce(ynn_reduce_sum, {0, 1, 2}, broadcast_id,
                 YNN_INVALID_VALUE_ID, output_id);

  TestScheduler scheduler(4);
  if (TestScheduler::num_threads_impl(&scheduler) < 2) {
    GTEST_SKIP() << "Parallel reduction requires multiple worker threads.";
  }
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> input(input_shape);
  input.fill(1.0f);

  runtime.ReshapeExternalTensor(input_shape, input.data(), input_id);
  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), std::vector<size_t>{});

  float output = 0.0f;
  runtime.SetupExternalTensor(&output, output_id);
  for (int i = 0; i < 100; ++i) {
    output = 0.0f;
    runtime.InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_EQ(output, 2097152.0f);
  }
}

}  // namespace
}  // namespace ynn
