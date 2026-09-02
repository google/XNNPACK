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
  bool found_pack = false;
  bool found_dot = false;
  std::string first_loop;
  for (const std::string& event : trace_events) {
    if (contains(event, "pack_b")) found_pack = true;
    if (contains(event, "dot")) found_dot = true;
    if (first_loop.empty() &&
        (contains(event, "loop k") || contains(event, "loop d")) &&
        !contains(event, "iteration")) {
      first_loop = event;
    }
  }
  EXPECT_TRUE(found_pack);
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

// A dynamic-shape dot computes its split factors with an opaque runtime call
// behind global variables, so the bare prover can't bound them. Verify that
// choose_split_factors registers bounds facts for those globals, and that
// the facts make the splits provable.
TEST(DotSchedulingTest, OpaqueSplitFactorBoundsAreProvable) {
  ynn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(ynn_create_subgraph(3, 0, &subgraph), ynn_status_success);
  uint32_t a_id = 0, b_id = 1, out_id = 2;
  const size_t b_dims[] = {512, 256};
  // Dynamic (unknown) shape for the input and output makes the splits
  // symbolic.
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_fp32, 2, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_fp32, 2, b_dims, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_fp32, 2, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_dot(subgraph, /*num_k_dims=*/1, a_id, b_id,
                           YNN_INVALID_VALUE_ID, &out_id, 0),
            ynn_status_success);

  TestScheduler scheduler(3);
  Runtime runtime(subgraph, &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ynn_runtime_t rt = runtime.get();

  slinky::expr split_m, split_n, split_k;
  for (const auto& let : rt->globals.lets) {
    const std::string name = rt->globals.symbols.name(let.first);
    if (name.rfind("split_m", 0) == 0) split_m = slinky::expr(let.first);
    if (name.rfind("split_n", 0) == 0) split_n = slinky::expr(let.first);
    if (name.rfind("split_k", 0) == 0) split_k = slinky::expr(let.first);
  }
  ASSERT_TRUE(split_m.defined());
  ASSERT_TRUE(split_n.defined());
  ASSERT_TRUE(split_k.defined());

  // Without facts, the splits are opaque global variables.
  EXPECT_FALSE(slinky::prove_true(split_m <= 2047 * 16));
  EXPECT_FALSE(slinky::prove_true(split_n <= 256));
  EXPECT_FALSE(slinky::prove_true(split_n >= 1));
  EXPECT_FALSE(slinky::prove_true(split_k == 512));
  // With the registered facts, their structure is provable. Each split is
  // min(extent, mult * step) with mult in [1, max_multiplier]:
  // - split_m <= min(m, 2047 * 16), so it never exceeds the packing cap
  //   even with a symbolic m;
  // - split_n is bounded by the static n = 256, and >= min(n, block_n) > 0;
  // - k = 512 is static, so the split_k facts [min(k, 1024),
  //   min(k, 511 * 1024)] collapse to the point 512, making the exact
  //   value provable.
  EXPECT_TRUE(slinky::prove_true(split_m <= 2047 * 16, rt->globals.fact_bounds,
                                 rt->globals.fact_alignment));
  EXPECT_TRUE(slinky::prove_true(split_n <= 256, rt->globals.fact_bounds,
                                 rt->globals.fact_alignment));
  EXPECT_TRUE(slinky::prove_true(split_n >= 1, rt->globals.fact_bounds,
                                 rt->globals.fact_alignment));
  EXPECT_TRUE(slinky::prove_true(split_k == 512, rt->globals.fact_bounds,
                                 rt->globals.fact_alignment));

  ynn_delete_subgraph(subgraph);
}

}  // namespace
}  // namespace ynn
