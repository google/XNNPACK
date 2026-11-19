// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include "slinky/builder/simplify.h"

namespace ynn {
namespace {

bool contains(const std::string& str, const std::string& substr) {
  return std::search(str.begin(), str.end(), substr.begin(), substr.end()) !=
         str.end();
}

template <typename AT, typename BT>
void VerifyDotLoopOrder(const std::vector<size_t>& a_shape,
                        const std::vector<size_t>& b_shape,
                        bool expect_split_k) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<AT>(), a_shape, a_id)
      .AddInput(type_of<BT>(), b_shape, b_id)
      .AddOutput(type_of<float>(), TensorShape({a_shape[0], b_shape[1]}),
                 out_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, out_id);

  TestScheduler scheduler(3);
  Runtime runtime(builder.GetSubgraph(), &scheduler,
                  YNN_FLAG_ENABLE_SLINKY_TRACE);

  std::vector<std::string> trace_events;
  std::mutex trace_mutex;  // NOLINT(build/c++11)
  runtime.get()->eval_config.trace_begin =
      [&](const char* name) -> slinky::index_t {
    std::lock_guard<std::mutex> lock(trace_mutex);  // NOLINT(build/c++11)
    trace_events.push_back(name);
    return 0;
  };

  Tensor<AT> a(a_shape);
  Tensor<BT> b(b_shape);
  Tensor<float> out({a_shape[0], b_shape[1]});
  runtime.ReshapeExternalTensor(a.extents(), a.data(), a_id)
      .ReshapeExternalTensor(b.extents(), b.data(), b_id)
      .ReshapeRuntime()
      .SetupExternalTensor(out.data(), out_id)
      .InvokeRuntime();
  EXPECT_EQ(runtime.Status(), ynn_status_success);

  EXPECT_FALSE(trace_events.empty());
  EXPECT_EQ(trace_events.front(), "pipeline");
  bool found_dot = false;
  std::string first_loop;
  for (const std::string& event : trace_events) {
    if (contains(event, "dot")) found_dot = true;
    if (first_loop.empty() &&
        (contains(event, "loop k") || contains(event, "loop d")) &&
        !contains(event, "iteration")) {
      first_loop = event;
    }
  }
  EXPECT_TRUE(found_dot);
  if (expect_split_k) {
    EXPECT_THAT(first_loop, testing::HasSubstr("loop k"));
  } else {
    EXPECT_THAT(first_loop, testing::Not(testing::HasSubstr("loop k")));
  }
}

TEST(DotSchedulingTest, NoSplitK) {
  VerifyDotLoopOrder<float, float>({300, 100}, {100, 400}, false);
}

TEST(DotSchedulingTest, SplitKTrue) {
  VerifyDotLoopOrder<float, float>({300, 8192}, {8192, 400}, true);
}

TEST(DotSchedulingTest, NarrowTypeNoSplitK) {
  VerifyDotLoopOrder<bfloat16, bfloat16>({300, 8192}, {8192, 400}, false);
}

TEST(DotSchedulingTest, NarrowTypeLargeSplitK) {
  VerifyDotLoopOrder<bfloat16, bfloat16>({300, 16384}, {16384, 400}, true);
}

TEST(DotSchedulingTest, SmallMNoSplitK) {
  VerifyDotLoopOrder<float, float>({1, 8192}, {8192, 400}, false);
  VerifyDotLoopOrder<float, float>({16, 8192}, {8192, 400}, false);
  VerifyDotLoopOrder<float, float>({64, 8192}, {8192, 400}, true);
  VerifyDotLoopOrder<bfloat16, bfloat16>({1, 16384}, {16384, 400}, false);
}

TEST(DotSchedulingTest, RaggedDotDecodeParallelizesAcrossActiveGroups) {
  constexpr size_t G = 64;
  constexpr size_t M = 8;
  constexpr size_t K = 256;
  constexpr size_t N = 256;

  Tensor<uint8_t> a({M, K});
  Tensor<int8_t> b({G, K, N});
  Tensor<int32_t> rg({M, 1});
  for (size_t i = 0; i < a.size(); ++i) {
    a.data()[i] = static_cast<uint8_t>((i % 7) + 1);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b.data()[i] = static_cast<int8_t>((i % 5) - 2);
  }
  const int32_t groups[M] = {3, 11, 19, 27, 35, 43, 51, 59};
  for (size_t i = 0; i < M; ++i) {
    rg(i, 0) = groups[i];
  }

  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t rg_id = 2;
  uint32_t out_id = 3;
  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_uint8, {0, K}, a_id)
      .AddTensor(b, b_id)
      .AddInput(ynn_type_int32, {0, 1}, rg_id)
      .AddOutput(ynn_type_int32, TensorShape(2), out_id);
  ASSERT_EQ(
      ynn_define_ragged_dot(builder.GetSubgraph(), /*num_k_dims=*/1, a_id, b_id,
                            YNN_INVALID_VALUE_ID, rg_id, &out_id, /*flags=*/0),
      ynn_status_success);

  TestScheduler scheduler(3);
  Runtime runtime(builder.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Empty batch (M = 0) reshapes and invokes cleanly with a thread pool.
  uint8_t dummy_a = 0;
  int32_t dummy_rg = 0;
  int32_t dummy_out = 0;
  runtime.ReshapeExternalTensor({0, K}, &dummy_a, a_id)
      .ReshapeExternalTensor({0, 1}, &dummy_rg, rg_id)
      .ReshapeRuntime()
      .SetupExternalTensor(&dummy_out, out_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Decode batch (M = 8, G = 64): a single [8, 256] x [256, 256] slice would
  // collapse to 1 serial task, but streaming 8 active expert slices must split
  // N across worker threads.
  Tensor<int32_t> out({M, N});
  runtime.ReshapeExternalTensor(a.extents(), a.data(), a_id)
      .ReshapeExternalTensor(rg.extents(), rg.data(), rg_id)
      .ReshapeRuntime()
      .SetupExternalTensor(out.data(), out_id);
  const size_t tasks_before = scheduler.task_count();
  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  EXPECT_EQ(scheduler.task_count() - tasks_before, 3u);

  for (size_t row = 0; row < M; ++row) {
    const size_t g = static_cast<size_t>(rg(row, 0));
    for (size_t j = 0; j < N; ++j) {
      int32_t expected = 0;
      for (size_t k = 0; k < K; ++k) {
        expected +=
            static_cast<int32_t>(a(row, k)) * static_cast<int32_t>(b(g, k, j));
      }
      ASSERT_EQ(out(row, j), expected);
    }
  }
}

TEST(DotSchedulingTest, RaggedDotPrefillSplitsOuterM) {
  constexpr size_t G = 8;
  constexpr size_t M = 1024;
  constexpr size_t K = 256;
  constexpr size_t N = 16;

  Tensor<uint8_t> a({M, K});
  Tensor<int8_t> b({G, K, N});
  Tensor<int32_t> rg({M, 1});
  for (size_t i = 0; i < a.size(); ++i) {
    a.data()[i] = static_cast<uint8_t>((i % 7) + 1);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b.data()[i] = static_cast<int8_t>((i % 5) - 2);
  }
  for (size_t i = 0; i < M; ++i) {
    rg(i, 0) = static_cast<int32_t>((i * G) / M);
  }

  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t rg_id = 2;
  uint32_t out_id = 3;
  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_uint8, a.extents(), a_id)
      .AddTensor(b, b_id)
      .AddInput(ynn_type_int32, rg.extents(), rg_id)
      .AddOutput(ynn_type_int32, std::vector<size_t>{M, N}, out_id);
  ASSERT_EQ(
      ynn_define_ragged_dot(builder.GetSubgraph(), /*num_k_dims=*/1, a_id, b_id,
                            YNN_INVALID_VALUE_ID, rg_id, &out_id, /*flags=*/0),
      ynn_status_success);

  // N = 16 fits in a single N-block, so parallel tasks can only come from
  // splitting outer M across sorted `row_group` slices.
  TestScheduler scheduler(3);
  Runtime runtime(builder.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<int32_t> out({M, N});
  runtime.SetupExternalTensor(a.data(), a_id)
      .SetupExternalTensor(rg.data(), rg_id)
      .SetupExternalTensor(out.data(), out_id)
      .ReshapeRuntime();
  const size_t tasks_before = scheduler.task_count();
  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  EXPECT_GT(scheduler.task_count() - tasks_before, 0u);

  for (size_t row = 0; row < M; ++row) {
    const size_t g = static_cast<size_t>(rg(row, 0));
    for (size_t j = 0; j < N; ++j) {
      int32_t expected = 0;
      for (size_t k = 0; k < K; ++k) {
        expected +=
            static_cast<int32_t>(a(row, k)) * static_cast<int32_t>(b(g, k, j));
      }
      ASSERT_EQ(out(row, j), expected);
    }
  }
}

}  // namespace
}  // namespace ynn
