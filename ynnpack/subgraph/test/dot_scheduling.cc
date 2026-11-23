// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
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

  ASSERT_EQ(ynn_optimize_subgraph(builder.GetSubgraph(), nullptr, 0),
            ynn_status_success);
  TestScheduler scheduler(3);
  Runtime runtime(builder.GetSubgraph(), &scheduler, /*flags=*/0,
                  /*optimize=*/false);
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
  ASSERT_EQ(ynn_optimize_subgraph(builder.GetSubgraph(), nullptr, 0),
            ynn_status_success);
  TestScheduler scheduler(3);
  Runtime runtime(builder.GetSubgraph(), &scheduler, /*flags=*/0,
                  /*optimize=*/false);
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

template <typename AT, typename BT>
bool DotPacksB(const TensorShape& a_shape, const TensorShape& b_shape,
               bool static_b = false) {
  const uint32_t a_id = 0;
  uint32_t b_id = static_b ? YNN_INVALID_VALUE_ID : 1;
  const uint32_t out_id = static_b ? 1 : 2;
  SubgraphBuilder builder(static_b ? 2 : 3);
  builder.AddInput(type_of<AT>(), a_shape, a_id);
  Tensor<BT> static_b_data;
  if (static_b) {
    static_b_data = Tensor<BT>(
        std::vector<size_t>(b_shape.Dims(), b_shape.Dims() + b_shape.Rank()));
    builder.AddTensor(type_of<BT>(), b_shape, b_id, static_b_data.data());
  } else {
    builder.AddInput(type_of<BT>(), b_shape, b_id);
  }
  builder
      .AddOutput(type_of<float>(),
                 TensorShape(std::max(a_shape.Rank(), b_shape.Rank())), out_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, out_id);

  const ynn_subgraph& subgraph = *builder.GetSubgraph();
  const ynn_node& dot_node = ProducerOf(out_id, subgraph);
  const auto* dot = std::get_if<ynn_node::dot>(&dot_node.op);
  EXPECT_NE(dot, nullptr);
  return dot && dot->pack_b;
}

bool HasUnalignedFp32Kernel() {
#if defined(YNN_ARCH_X86)
  return is_arch_supported(arch_flag::sse2);
#elif defined(YNN_ARCH_ARM64)
  return is_arch_supported(arch_flag::neon);
#else
  return false;
#endif
}

bool HasUnalignedBf16Kernel() {
#if defined(YNN_ARCH_X86)
  return is_arch_supported(arch_flag::avx2_fma3);
#elif defined(YNN_ARCH_ARM64)
  return is_arch_supported(arch_flag::neon);
#else
  return false;
#endif
}

TEST(DotSchedulingTest, PackBFp32) {
  if (!HasUnalignedFp32Kernel()) GTEST_SKIP();
  EXPECT_FALSE((DotPacksB<float, float>({1, 2048}, {2048, 2048})));
  EXPECT_TRUE((DotPacksB<float, float>({512, 2048}, {2048, 2048})));
  EXPECT_TRUE((DotPacksB<float, float>({1, 64}, {64, 64}, /*static_b=*/true)));
  EXPECT_TRUE((DotPacksB<float, float>(TensorShape(2), {2048, 2048})));
  // L2-resident B skips packing for small/moderate M, but packs for very large
  // M.
  EXPECT_FALSE((DotPacksB<float, float>({64, 128}, {128, 128})));
  EXPECT_FALSE((DotPacksB<float, float>({32, 256}, {256, 64})));
  EXPECT_TRUE((DotPacksB<float, float>({4096, 128}, {128, 128})));
}

TEST(DotSchedulingTest, PackBBatchBroadcast) {
  if (!HasUnalignedFp32Kernel()) GTEST_SKIP();
  // A broadcast over B's batch does not reduce per-slice M reuse of B.
  EXPECT_TRUE((DotPacksB<float, float>({1, 256, 2048}, {8, 2048, 2048})));
  // B broadcast over A's batch increases reuse of B across slices.
  EXPECT_TRUE((DotPacksB<float, float>({128, 1, 2048}, {1, 2048, 2048})));
  // Matching batch dims with M=1 per slice only read each B slice once.
  EXPECT_FALSE((DotPacksB<float, float>({8, 1, 2048}, {8, 2048, 2048})));
}

TEST(DotSchedulingTest, PackBBf16) {
  if (!HasUnalignedBf16Kernel()) GTEST_SKIP();
  EXPECT_FALSE((DotPacksB<bfloat16, bfloat16>({1, 2048}, {2048, 2048})));
  EXPECT_TRUE((DotPacksB<bfloat16, bfloat16>({64, 2048}, {2048, 2048})));
  EXPECT_TRUE(
      (DotPacksB<bfloat16, bfloat16>({1, 64}, {64, 64}, /*static_b=*/true)));
}

}  // namespace
}  // namespace ynn
