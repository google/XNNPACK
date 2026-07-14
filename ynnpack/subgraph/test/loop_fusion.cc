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

}  // namespace
}  // namespace ynn
