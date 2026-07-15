// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Tests that the automatic scheduler successfully fuses loops between operators
// in small subgraphs. We verify this by using a custom allocator to track
// max_allocation_size during execution; when loops are fused, intermediate
// buffers (like packed weights or elementwise outputs) are processed and
// allocated per-block inside the loop nest rather than as full buffers up
// front.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

class LoopFusionTest : public testing::Test {
 protected:
  void MakeRuntime(ynn_subgraph_t subgraph) {
    ynn_threadpool_t threadpool = nullptr;
    ASSERT_EQ(ynn_create_threadpool(TestScheduler::scheduler(), &scheduler_,
                                    /*flags=*/0, &threadpool),
              ynn_status_success);
    threadpool_.reset(threadpool);
    ASSERT_EQ(ynn_optimize_subgraph(subgraph, threadpool, /*flags=*/0),
              ynn_status_success);
    ynn_runtime_t runtime = nullptr;
    ASSERT_EQ(ynn_create_runtime(subgraph, threadpool, /*flags=*/0, &runtime),
              ynn_status_success);
    runtime_.reset(runtime);

    runtime_->eval_config.allocate = [this](slinky::var sym,
                                            slinky::raw_buffer* buffer) {
      void* ptr = buffer->allocate(YNN_ALLOCATION_ALIGNMENT);
      if (ptr) {
        max_allocation_size_ =
            std::max(max_allocation_size_, buffer->size_bytes());
      }
      return ptr;
    };
  }

  void ReshapeExternalTensor(uint32_t id, const std::vector<size_t>& shape,
                             void* data) {
    ASSERT_EQ(ynn_set_external_value_shape(runtime_.get(), id, shape.size(),
                                           shape.data()),
              ynn_status_success);
    ASSERT_EQ(ynn_set_external_value_data(runtime_.get(), id, data),
              ynn_status_success);
  }

  void SetupExternalTensor(uint32_t id, void* data) {
    ASSERT_EQ(ynn_set_external_value_data(runtime_.get(), id, data),
              ynn_status_success);
  }

  void RunPipeline() {
    max_allocation_size_ = 0;
    ASSERT_EQ(ynn_reshape_runtime(runtime_.get()), ynn_status_success);
    ASSERT_EQ(ynn_invoke_runtime(runtime_.get()), ynn_status_success);
  }

  TestScheduler scheduler_{3};
  std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)> threadpool_{
      nullptr, ynn_delete_threadpool};
  std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)> runtime_{
      nullptr, ynn_delete_runtime};
  std::size_t max_allocation_size_ = 0;
};

// pack_b should be computed inside the dot's loop nest, so packing happens
// per-block instead of materializing the whole packed buffer up front.
TEST_F(LoopFusionTest, PackFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  Tensor<float> a({16, 512});
  Tensor<float> b({512, 1024});
  Tensor<float> out({16, 1024});
  ReshapeExternalTensor(a_id, {16, 512}, a.data());
  ReshapeExternalTensor(b_id, {512, 1024}, b.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  EXPECT_LT(max_allocation_size_, b.size_bytes());
}

// The pipeline is dot(A, pack(exp(B))). exp's natural loop order does not
// match the dot's loop nest positionally (its n dimension is innermost, while
// the dot's nest iterates n outermost), so fusing it requires the scheduler
// to match loop splits by source region rather than by position.
TEST_F(LoopFusionTest, ProducerOfPackedInputFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  uint32_t exp_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), exp_id)
      .AddUnary(ynn_unary_exp, b_id, exp_id)
      .AddDot(1, a_id, exp_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  Tensor<float> a({16, 512});
  Tensor<float> b({512, 1024});
  Tensor<float> out({16, 1024});
  ReshapeExternalTensor(a_id, {16, 512}, a.data());
  ReshapeExternalTensor(b_id, {512, 1024}, b.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  EXPECT_LT(max_allocation_size_, b.size_bytes());
}

// The pipeline is dot(A, transpose(exp(Bt))). The transpose is folded into the
// packing (always_alias_transpose), so the func chain is exp -> transpose
// (aliased copy) -> pack_b -> dot. In this layout exp's loop order matches the
// dot's loop nest positionally, so fusion of exp is blocked *only* by the
// source region inference breaking at pack's non-identity input bounds.
TEST_F(LoopFusionTest, ProducerOfTransposedPackedInputFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  uint32_t exp_id = YNN_INVALID_VALUE_ID;
  uint32_t transpose_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), exp_id)
      .AddTensor(type_of<float>(), TensorShape(2), transpose_id)
      .AddUnary(ynn_unary_exp, b_id, exp_id)
      .AddTranspose({1, 0}, exp_id, transpose_id)
      .AddDot(1, a_id, transpose_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  Tensor<float> a({16, 512});
  Tensor<float> b({1024, 512});
  Tensor<float> out({16, 1024});
  ReshapeExternalTensor(a_id, {16, 512}, a.data());
  ReshapeExternalTensor(b_id, {1024, 512}, b.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  EXPECT_LT(max_allocation_size_, b.size_bytes());
}

// Two dots accumulated into one output: dot(A, B1, c=dot(A, B0)), like the
// dots of the dot_sum composite. The second dot fuses into the loops of the
// first one; both require specific tile steps for the shared loops, and the
// scheduler reconciles them with a least common multiple (lcm_sat).
TEST_F(LoopFusionTest, TwoDotsShareLoops) {
  const uint32_t a_id = 0;
  const uint32_t b0_id = 1;
  const uint32_t b1_id = 2;
  const uint32_t out_id = 3;
  uint32_t dot0_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(4);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b0_id)
      .AddInput(type_of<float>(), TensorShape(2), b1_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), dot0_id)
      .AddDot(1, a_id, b0_id, YNN_INVALID_VALUE_ID, dot0_id)
      .AddDot(1, a_id, b1_id, dot0_id, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  const size_t M = 128, K = 256, N = 1024;
  Tensor<float> a({M, K});
  Tensor<float> b0({K, N});
  Tensor<float> b1({K, N});
  Tensor<float> out({M, N});
  a.fill(1.0f);
  b0.fill(1.0f);
  b1.fill(2.0f);
  ReshapeExternalTensor(a_id, {M, K}, a.data());
  ReshapeExternalTensor(b0_id, {K, N}, b0.data());
  ReshapeExternalTensor(b1_id, {K, N}, b1.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  // Both dot intermediates should be computed per-block inside the shared
  // loop nest rather than materialized in full.
  EXPECT_LT(max_allocation_size_, M * N * sizeof(float));
  // The reconciled loop steps must still produce correct results.
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      ASSERT_EQ(out({i, j}), 3.0f * K) << i << " " << j;
    }
  }
}

}  // namespace
}  // namespace ynn
