// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
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

constexpr std::align_val_t kAlignment{256};
std::atomic<int> allocations{0};
std::atomic<int> frees{0};

void* Allocate(void* context, size_t size, size_t alignment) {
  if (alignment > static_cast<size_t>(kAlignment)) return nullptr;
  ++allocations;
  return ::operator new(size, kAlignment);
}

void Free(void* context, void* ptr, size_t size) {
  ++frees;
  ::operator delete(ptr, kAlignment);
}

TEST(runtime, allocator) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t c_id = 2;
  uint32_t exp_id = YNN_INVALID_VALUE_ID;

  // dot(a, exp(b)) needs an internal buffer for exp(b).
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 2, a_id)
      .AddInput(type_of<float>(), 2, b_id)
      .AddOutput(type_of<float>(), 2, c_id)
      .AddTensor(type_of<float>(), 2, exp_id)
      .AddUnary(ynn_unary_exp, b_id, exp_id)
      .AddDot(1, a_id, exp_id, YNN_INVALID_VALUE_ID, c_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(ynn_set_runtime_allocator(runtime.get(), Allocate, Free, nullptr),
            ynn_status_success);

  Tensor<float> a({16, 512});
  Tensor<float> b({512, 1024});
  Tensor<float> c({16, 1024});
  runtime.ReshapeExternalTensor({16, 512}, a.data(), a_id)
      .ReshapeExternalTensor({512, 1024}, b.data(), b_id)
      .ReshapeRuntime()
      .SetupExternalTensor(c.data(), c_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  EXPECT_GT(allocations, 0);
  EXPECT_EQ(frees, allocations);
}
}  // namespace ynn
