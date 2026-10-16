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
  void MakeRuntime(ynn_subgraph_t subgraph,
                   TestScheduler* scheduler = nullptr) {
    if (!scheduler) scheduler = &scheduler_;
    ynn_threadpool_t threadpool = nullptr;
    ASSERT_EQ(ynn_create_threadpool(TestScheduler::scheduler(), scheduler,
                                    /*flags=*/0, &threadpool),
              ynn_status_success);
    threadpool_.reset(threadpool);
    ASSERT_EQ(ynn_optimize_subgraph(subgraph, threadpool, /*flags=*/0),
              ynn_status_success);
    ynn_runtime_t runtime = nullptr;
    ASSERT_EQ(ynn_create_runtime(subgraph, threadpool, /*flags=*/0, &runtime),
              ynn_status_success);
    runtime_.reset(runtime);

    // Disable block reuse so every allocation reaches this hook.
    runtime_->eval_config.use_memory_pool = false;
    runtime_->eval_config.allocate = [this](std::size_t size,
                                            std::size_t alignment) {
      void* ptr = slinky::allocate_bytes(size, alignment);
      if (ptr) {
        max_allocation_size_ = std::max(max_allocation_size_, size);
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
// first one; the shared steps must respect both dots' blocking requirements.
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

// The two dots prefer different column tiles. A shared tile may shrink, but
// must still cover whole packed blocks for both inputs, including the tail.
TEST_F(LoopFusionTest, TwoDotsShareColumnTiles) {
  constexpr size_t M = 8, K0 = 64, K1 = 512, N = 1025;
  Tensor<float> a0({M, K0}), b0({K0, N});
  Tensor<float> a1({M, K1}), b1({K1, N}), out({M, N});
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0; k < K0; ++k) a0({m, k}) = m + 1;
    for (size_t k = 0; k < K1; ++k) a1({m, k}) = m + 2;
  }
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K0; ++k) b0({k, n}) = 1 + n % 7;
    for (size_t k = 0; k < K1; ++k) b1({k, n}) = 1 + n % 5;
  }

  uint32_t dot_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(5);
  subgraph.AddInput(type_of<float>(), TensorShape(2), 0)
      .AddInput(type_of<float>(), TensorShape(2), 1)
      .AddInput(type_of<float>(), TensorShape(2), 2)
      .AddInput(type_of<float>(), TensorShape(2), 3)
      .AddOutput(type_of<float>(), TensorShape(2), 4)
      .AddTensor(type_of<float>(), TensorShape(2), dot_id)
      .AddDot(1, 0, 1, YNN_INVALID_VALUE_ID, dot_id)
      .AddDot(1, 2, 3, dot_id, 4);

  TestScheduler scheduler(0);
  MakeRuntime(subgraph.GetSubgraph(), &scheduler);
  ReshapeExternalTensor(0, a0.extents(), a0.data());
  ReshapeExternalTensor(1, b0.extents(), b0.data());
  ReshapeExternalTensor(2, a1.extents(), a1.data());
  ReshapeExternalTensor(3, b1.extents(), b1.data());
  SetupExternalTensor(4, out.data());
  RunPipeline();
  EXPECT_LE(max_allocation_size_, K1 * 512 * sizeof(float));
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      EXPECT_EQ(out({m, n}),
                K0 * (m + 1) * (1 + n % 7) + K1 * (m + 2) * (1 + n % 5));
    }
  }
  runtime_.reset();
  threadpool_.reset();
}

TEST_F(LoopFusionTest, PartialReductionTracksSharedStep) {
  TestScheduler scheduler(0);
  // Different K sizes give the dots different column blocking requirements.
  // Both must agree with the partial reduction's accumulation chunk.
  constexpr size_t M = 64, K0 = 64, K1 = 512, N = 40001;
  Tensor<float> a0({M, K0}), b0({K0, N});
  Tensor<float> a1({M, K1}), b1({K1, N}), out({M});
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0; k < K0; ++k) a0({m, k}) = 1 + m % 4;
    for (size_t k = 0; k < K1; ++k) a1({m, k}) = 2 + m % 4;
  }
  b0.fill(1.0f);
  b1.fill(1.0f);

  uint32_t dot0 = YNN_INVALID_VALUE_ID, dot1 = YNN_INVALID_VALUE_ID;
  uint32_t zero = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(5);
  subgraph.AddInput(type_of<float>(), {M, K0}, 0)
      .AddTensor(b0, 1)
      .AddInput(type_of<float>(), {M, K1}, 2)
      .AddTensor(b1, 3)
      .AddOutput(type_of<float>(), {M}, 4)
      .AddTensor(type_of<float>(), TensorShape(2), dot0)
      .AddTensor(type_of<float>(), TensorShape(2), dot1)
      .AddScalar<float>(0.0f, zero)
      .AddDot(1, 0, 1, YNN_INVALID_VALUE_ID, dot0)
      .AddDot(1, 2, 3, dot0, dot1)
      .AddReduce(ynn_reduce_sum, {1}, dot1, zero, 4, 0);

  MakeRuntime(subgraph.GetSubgraph(), &scheduler);
  ReshapeExternalTensor(0, a0.extents(), a0.data());
  ReshapeExternalTensor(2, a1.extents(), a1.data());
  SetupExternalTensor(4, out.data());
  RunPipeline();
  // The partial sums must use the final 512-column step, not the first
  // dot's 64-column step, which would allocate an oversized partial buffer.
  EXPECT_LE(max_allocation_size_, M * 512 * sizeof(float));
  for (size_t m = 0; m < M; ++m) {
    EXPECT_EQ(out(m), N * (K0 * (1 + m % 4) + K1 * (2 + m % 4)));
  }
  runtime_.reset();
  threadpool_.reset();
}

// A short second reduction prefers tall row tiles. Sharing those rows must
// not enlarge the full-width intermediate used by the first dot and norm.
class DotRowTileTest
    : public LoopFusionTest,
      public testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(DotRowTileTest, SmallProducerTileSurvivesFusion) {
  constexpr size_t M = 1025, K = 64, S = 1024, active = 11, N = 64;
  Tensor<float> b({K, S}), v({active, N});
  b.fill(0.0f);
  for (size_t s = 0; s < S; ++s) b({0, s}) = s % 2;
  for (size_t s = 0; s < active; ++s) {
    for (size_t n = 0; n < N; ++n) v({s, n}) = 1 + n % 7;
  }
  const auto [dynamic, transpose_a] = GetParam();
  const std::vector<size_t> a_shape =
      transpose_a ? std::vector<size_t>{K, M} : std::vector<size_t>{M, K};
  uint32_t a_id = 0;
  uint32_t b_id = YNN_INVALID_VALUE_ID, v_id = YNN_INVALID_VALUE_ID;
  uint32_t dot_id = YNN_INVALID_VALUE_ID, exp_id = YNN_INVALID_VALUE_ID;
  uint32_t sum_id = YNN_INVALID_VALUE_ID, norm_id = YNN_INVALID_VALUE_ID;
  uint32_t slice_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(2);
  subgraph
      .AddInput(type_of<float>(),
                dynamic ? TensorShape(2) : TensorShape(a_shape), 0)
      .AddOutput(type_of<float>(), TensorShape(2), 1)
      .AddTensor(type_of<float>(), b.extents(), b_id, b.data())
      .AddTensor(type_of<float>(), v.extents(), v_id, v.data())
      .AddTensor(type_of<float>(), TensorShape(2), dot_id)
      .AddTensor(type_of<float>(), TensorShape(2), exp_id)
      .AddTensor(type_of<float>(), TensorShape(2), sum_id)
      .AddTensor(type_of<float>(), TensorShape(2), norm_id)
      .AddTensor(type_of<float>(), TensorShape(2), slice_id);
  if (transpose_a) {
    a_id = YNN_INVALID_VALUE_ID;
    subgraph.AddTensor(type_of<float>(), TensorShape(2), a_id)
        .AddTranspose({1, 0}, 0, a_id);
  }
  subgraph.AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, dot_id)
      .AddUnary(ynn_unary_exp, dot_id, exp_id)
      .AddReduce(ynn_reduce_sum, {1}, exp_id, YNN_INVALID_VALUE_ID, sum_id,
                 YNN_NODE_FLAG_KEEP_DIMS)
      .AddBinary(ynn_binary_divide, exp_id, sum_id, norm_id)
      .AddSlice({1}, {0}, {active}, {1}, norm_id, slice_id)
      .AddDot(1, slice_id, v_id, YNN_INVALID_VALUE_ID, 1);

  TestScheduler scheduler(0);
  MakeRuntime(subgraph.GetSubgraph(), &scheduler);
  // Reuse a dynamic pipeline at a second shape, including a partial row tile.
  for (size_t rows :
       dynamic ? std::vector<size_t>{M, 259} : std::vector<size_t>{M}) {
    Tensor<float> a(transpose_a ? std::vector<size_t>{K, rows}
                                : std::vector<size_t>{rows, K});
    Tensor<float> out({rows, N});
    a.fill(0.0f);
    for (size_t row = 0; row < rows; ++row) {
      const float value = (static_cast<int>(row % 7) - 3) / 8.0f;
      if (transpose_a)
        a({0, row}) = value;
      else
        a({row, 0}) = value;
    }
    ReshapeExternalTensor(0, a.extents(), a.data());
    SetupExternalTensor(1, out.data());
    RunPipeline();
    EXPECT_LE(max_allocation_size_, 128 * S * sizeof(float));
    for (size_t row = 0; row < rows; ++row) {
      const float odd = std::exp((static_cast<int>(row % 7) - 3) / 8.0f);
      const float norm =
          ((active + 1) / 2 + (active / 2) * odd) / ((S / 2) * (1 + odd));
      for (size_t n = 0; n < N; ++n) {
        EXPECT_NEAR(out({row, n}), norm * (1 + n % 7), 1.0e-6f);
      }
    }
  }
  runtime_.reset();
  threadpool_.reset();
}

INSTANTIATE_TEST_SUITE_P(Shapes, DotRowTileTest,
                         testing::Combine(testing::Bool(), testing::Bool()));

// When B is static, the packing is constant folded during
// ynn_optimize_subgraph: it has no dot loop nest to fuse into, so it runs
// with its own loops, which should still be parallelized.
TEST_F(LoopFusionTest, ConstantFoldedPackIsParallel) {
  const size_t K = 512, N = 4096;
  Tensor<float> b({K, N});
  b.fill(1.0f);

  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddTensor(b, b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, out_id);

  // Constant folding runs the packing during MakeRuntime (in
  // ynn_optimize_subgraph); nothing else is invoked, so any tasks the
  // scheduler saw are the packing's parallel loops.
  MakeRuntime(subgraph.GetSubgraph());
// NOTE(vksnk): We skip WASM_SIMD128, because the default config doesn't enable
// threads so all of the loops become serial.
#if !defined(YNN_ARCH_WASM_SIMD128)
  EXPECT_GT(scheduler_.task_count(), 0);
#endif  // !YNN_ARCH_WASM_SIMD128
}

}  // namespace
}  // namespace ynn
