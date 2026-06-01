// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

TEST(PartialReduceDot, Test) {
  TestScheduler scheduler(3);

  const size_t M = 128;
  const size_t K = 16;
  const size_t N = 40000;  // Large enough to trigger partial reduction

  Tensor<float> a({M, K});
  Tensor<float> b({K, N});
  a.fill(1.0f);
  b.fill(1.0f);

  SubgraphBuilder subgraph(5);
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  uint32_t dot_out_id = 2;
  const uint32_t reduce_out_id = 3;
  const uint32_t c_id = 4;  // init for reduce

  subgraph.AddInput(type_of<float>(), {M, K}, a_id)
      .AddTensor(b, b_id)
      .AddTensor(type_of<float>(), {M, N}, dot_out_id)
      .AddOutput(type_of<float>(), {M}, reduce_out_id);

  // Dot: a * b
  subgraph.AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, dot_out_id);

  // Reduce: sum(dot_out, axis=1)
  subgraph.AddScalar<float>(0.0f, c_id);
  subgraph.AddReduce(ynn_reduce_sum, {1}, dot_out_id, c_id, reduce_out_id, 0);

  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor({M, K}, a.data(), a_id);

  Tensor<float> c({M});
  runtime.SetupExternalTensor(c.data(), reduce_out_id);

  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Each element of dot_out is K (16.0f).
  // Summing across N (40000) gives 16.0f * 40000 = 640000.0f.
  for (size_t i = 0; i < M; ++i) {
    ASSERT_EQ(c(i), 640000.0f);
  }
}

}  // namespace ynn
