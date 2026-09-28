// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
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
#include "ynnpack/subgraph/dot.h"
#include "ynnpack/subgraph/runtime.h"
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

TEST(DotSchedulingTest, DefinePackA) {
  const size_t M = 35, K = 50;
  const size_t tile_m = 32, tile_k = 32;
  const size_t blocks_m = 2, tiles_k = 2;
  const uint32_t a_id = 0;
  const uint32_t out_id = 1;
  SubgraphBuilder builder(2);
  builder.AddInput(ynn_type_bf16, {M, K}, a_id)
      .AddOutput(ynn_type_bf16, {blocks_m, tiles_k, tile_m, tile_k}, out_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();
  const uint32_t packed_a_id =
      define_pack_a(subgraph, /*tile_m=*/tile_m, /*tile_k=*/tile_k,
                    /*m_dim=*/1, a_id);
  EXPECT_THAT(ProducerOf(packed_a_id, subgraph),
              AllOf(IsPackA(tile_m, tile_k, 1), InputsAre(a_id)));
  builder.AddCopy(packed_a_id, out_id);

  TestScheduler scheduler(3);
  Runtime runtime(builder.GetSubgraph(), &scheduler);
  EXPECT_EQ(runtime.Status(), ynn_status_success);

  Tensor<bfloat16> a({M, K});
  Tensor<bfloat16> out({blocks_m, tiles_k, tile_m, tile_k});
  for (size_t i = 0; i < M * K; ++i) {
    a.data()[i] = bfloat16(static_cast<float>((i % 13) - 6));
  }

  runtime.ReshapeExternalTensor(a.extents(), a.data(), a_id)
      .ReshapeRuntime()
      .SetupExternalTensor(out.data(), out_id)
      .InvokeRuntime();
  EXPECT_EQ(runtime.Status(), ynn_status_success);

  for (size_t mo = 0; mo < blocks_m; ++mo) {
    for (size_t ko = 0; ko < tiles_k; ++ko) {
      for (size_t mi = 0; mi < tile_m; ++mi) {
        for (size_t ki = 0; ki < tile_k; ++ki) {
          const size_t m = mo * tile_m + mi;
          const size_t k = ko * tile_k + ki;
          const float expected =
              (m < M && k < K) ? static_cast<float>(a({m, k})) : 0.0f;
          EXPECT_FLOAT_EQ(static_cast<float>(out({mo, ko, mi, ki})), expected);
        }
      }
    }
  }
}

}  // namespace
}  // namespace ynn
