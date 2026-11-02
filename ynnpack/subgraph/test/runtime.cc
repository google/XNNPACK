// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {
// By default WASM doesn't have any thread support.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
TEST(runtime, dot_concurrency) {}
#else
TEST(runtime, dot_concurrency) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t c_id = 2;
  constexpr uint32_t init_zero = YNN_INVALID_VALUE_ID;

  TestScheduler scheduler(3);

  auto get_concurrency = [&](SubgraphBuilder& builder) -> int32_t {
    Runtime runtime(builder.GetSubgraph(), &scheduler);
    EXPECT_EQ(runtime.Status(), ynn_status_success);
    int32_t concurrency;
    EXPECT_EQ(runtime.Query(ynn_runtime_property_concurrency, &concurrency),
              ynn_status_success);
    return concurrency;
  };

  // We should be able to statically know this graph will not run a parallel
  // loop.
  SubgraphBuilder small(3);
  small.AddInput(type_of<float>(), {8, 8}, a_id);
  small.AddInput(type_of<float>(), {8, 8}, b_id);
  small.AddOutput(type_of<float>(), {8, 8}, c_id);
  small.AddDot(1, a_id, b_id, init_zero, c_id);
  // TODO(b/458542243): This doesn't actually work because we don't simplify
  // away these loops yet.
  // ASSERT_EQ(get_concurrency(small), 1);

  // We should be able to statically know this graph will run parallel loops.
  SubgraphBuilder big(3);
  big.AddInput(type_of<float>(), {800, 800}, a_id);
  big.AddInput(type_of<float>(), {800, 800}, b_id);
  big.AddOutput(type_of<float>(), {800, 800}, c_id);
  big.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(big), 1);

  // We don't know in this case, we might run a parallel loop if the input is
  // big enough.
  SubgraphBuilder dynamic(3);
  dynamic.AddInput(type_of<float>(), 2, a_id);
  dynamic.AddInput(type_of<float>(), 2, b_id);
  dynamic.AddOutput(type_of<float>(), 2, c_id);
  dynamic.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(dynamic), 1);
}
#endif

TEST(runtime, cross_runtime_isolation) {
  constexpr uint32_t in_id = 0;
  constexpr uint32_t out_id = 1;
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), 1, in_id)
      .AddOutput(type_of<float>(), 1, out_id)
      .AddUnary(ynn_unary_negate, in_id, out_id);

  Runtime A(builder.GetSubgraph());
  Runtime B(builder.GetSubgraph());
  ASSERT_EQ(A.Status(), ynn_status_success);
  ASSERT_EQ(B.Status(), ynn_status_success);

  std::vector<float> in_a(100, 1.0f);
  std::vector<float> in_b(10, 2.0f);
  std::vector<float> out_a(100, 0.0f);
  std::vector<float> out_b(10, 0.0f);

  A.ReshapeExternalTensor(TensorShape({100}), in_a.data(), in_id);
  A.ReshapeRuntime();

  // Runtime B's output shape must not be affected by Runtime A's reshape.
  EXPECT_NE(B.GetExternalTensorShape(out_id), std::vector<size_t>{100});

  B.ReshapeExternalTensor(TensorShape({10}), in_b.data(), in_id);
  B.ReshapeRuntime();

  EXPECT_EQ(A.GetExternalTensorShape(out_id), std::vector<size_t>{100});
  EXPECT_EQ(B.GetExternalTensorShape(out_id), std::vector<size_t>{10});

  A.SetupExternalTensor(out_a.data(), out_id).InvokeRuntime();
  B.SetupExternalTensor(out_b.data(), out_id).InvokeRuntime();

  ASSERT_EQ(A.Status(), ynn_status_success);
  ASSERT_EQ(B.Status(), ynn_status_success);

  for (float v : out_a) {
    EXPECT_FLOAT_EQ(v, -1.0f);
  }
  for (float v : out_b) {
    EXPECT_FLOAT_EQ(v, -2.0f);
  }
}

TEST(runtime, may_block) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t c_id = 2;
  constexpr uint32_t init_zero = YNN_INVALID_VALUE_ID;

  TestScheduler scheduler(3);

  auto query_may_block = [](Runtime& runtime) -> int32_t {
    int32_t may_block = -1;
    EXPECT_EQ(runtime.Query(ynn_runtime_property_may_block, &may_block),
              ynn_status_success);
    return may_block;
  };

  SubgraphBuilder tiny_dot(3);
  tiny_dot.AddInput(type_of<float>(), {8, 8}, a_id);
  tiny_dot.AddInput(type_of<float>(), {8, 8}, b_id);
  tiny_dot.AddOutput(type_of<float>(), {8, 8}, c_id);
  tiny_dot.AddDot(1, a_id, b_id, init_zero, c_id);
  Runtime tiny_runtime(tiny_dot.GetSubgraph(), &scheduler);
  ASSERT_EQ(tiny_runtime.Status(), ynn_status_success);
  EXPECT_EQ(query_may_block(tiny_runtime), 0);

  SubgraphBuilder medium_dot(3);
  medium_dot.AddInput(type_of<float>(), {256, 128}, a_id);
  medium_dot.AddInput(type_of<float>(), {128, 128}, b_id);
  medium_dot.AddOutput(type_of<float>(), {256, 128}, c_id);
  medium_dot.AddDot(1, a_id, b_id, init_zero, c_id);
  Runtime medium_single_thread(medium_dot.GetSubgraph());
  ASSERT_EQ(medium_single_thread.Status(), ynn_status_success);
  EXPECT_EQ(query_may_block(medium_single_thread), 1);

  SubgraphBuilder dynamic_dot(3);
  dynamic_dot.AddInput(type_of<float>(), 2, a_id);
  dynamic_dot.AddInput(type_of<float>(), 2, b_id);
  dynamic_dot.AddOutput(type_of<float>(), 2, c_id);
  dynamic_dot.AddDot(1, a_id, b_id, init_zero, c_id);
  Runtime dynamic_runtime(dynamic_dot.GetSubgraph(), &scheduler);
  ASSERT_EQ(dynamic_runtime.Status(), ynn_status_success);
  EXPECT_EQ(query_may_block(dynamic_runtime), 1);

  std::vector<float> a_small(16, 1.0f);
  std::vector<float> b_small(16, 1.0f);
  dynamic_runtime.ReshapeExternalTensor(TensorShape({4, 4}), a_small.data(),
                                        a_id);
  dynamic_runtime.ReshapeExternalTensor(TensorShape({4, 4}), b_small.data(),
                                        b_id);
  dynamic_runtime.ReshapeRuntime();
  ASSERT_EQ(dynamic_runtime.Status(), ynn_status_success);
  EXPECT_EQ(query_may_block(dynamic_runtime), 0);

  std::vector<float> a_large(256 * 256, 1.0f);
  std::vector<float> b_large(256 * 256, 1.0f);
  dynamic_runtime.ReshapeExternalTensor(TensorShape({256, 256}), a_large.data(),
                                        a_id);
  dynamic_runtime.ReshapeExternalTensor(TensorShape({256, 256}), b_large.data(),
                                        b_id);
  dynamic_runtime.ReshapeRuntime();
  ASSERT_EQ(dynamic_runtime.Status(), ynn_status_success);
  EXPECT_EQ(query_may_block(dynamic_runtime), 1);
}

}  // namespace ynn
