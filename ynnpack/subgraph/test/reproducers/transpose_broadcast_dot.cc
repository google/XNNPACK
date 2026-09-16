// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

// Grouped-query attention broadcasts one key head over several query heads and
// then feeds the result, transposed, to a dot as the right-hand operand:
//
//   k_tiled = static_broadcast(k, [1, num_heads, kv_len, head_dim])
//   k_t     = static_transpose(k_tiled, [0, 1, 3, 2])
//   scores  = dot(q, k_t)
//
// `rewrite_transpose_broadcast` commutes those two into
// static_broadcast(static_transpose(k)), and the resulting subgraph is
// inconsistent.
//
// Logging the graph will show something like:
//
//   value 4:  fp32 extents={4,3,1,1}
//   value 5:  fp32 extents={4,3,1,1}
//   value 7:  fp32 extents={3,4,2,1}
//   {4} -> {5} static_transpose permutation={1,0,2,3} alias
//   {5} -> {7} static_broadcast new_dims={0,0,2,1}
//   {7} -> {8} pack_b
//
// Transposing {4,3,..} by {1,0,2,3} is {3,4,..}, and the broadcast consumer did
// derive its own shape from {3,4,..}. But value 5, the transpose's output,
// still records the un-permuted shape {4,3,..}.
//
// The static_broadcast only grows dim 2 and cannot turn {4,3,1,1} into
// {3,4,2,1}. The packed operand is read with the wrong strides and every key
// position ends up seeing the same data, so the scores come out constant along
// the key axis.
//
// Notes:
//
//  1. The broadcast operand must be computed, not constant, or it is folded
//     away before the rewrite runs.
//  2. `ynn_define_dot` clones the operand transpose as an aliasing transpose it
//     can fold into `pack_b`, which leaves the original transpose dead but
//     still registered as a second consumer of the broadcast and the rewrite
//     bails when the broadcast has more than one consumer.
//
//     `ynn_subgraph::fusion` only calls `invalidate_dead_values()` when some
//     rewrite fires, so in a graph that is otherwise already optimal the clone
//     survives forever.
//
//     The multiply below exists solely to give the first iteration something to
//     rewrite (it becomes `square`); that lets dead-value elimination run and
//     the transpose/broadcast rewrite fire on the second iteration.
TEST(TransposeBroadcastDotTest, BroadcastOperandTransposedIntoDot) {
  constexpr size_t kNumHeads = 2;
  constexpr size_t kQueryLen = 3;
  constexpr size_t kKeyLen = 3;
  constexpr size_t kHeadDim = 4;

  const std::vector<size_t> q_shape = {1, kNumHeads, kQueryLen, kHeadDim};
  const std::vector<size_t> k_shape = {1, 1, kKeyLen, kHeadDim};
  const std::vector<size_t> k_tiled_shape = {1, kNumHeads, kKeyLen, kHeadDim};
  const std::vector<size_t> k_t_shape = {1, kNumHeads, kHeadDim, kKeyLen};
  const std::vector<size_t> out_shape = {1, kNumHeads, kQueryLen, kKeyLen};

  SubgraphBuilder subgraph(3);

  const uint32_t q_id = 0;
  const uint32_t k_in_id = 1;
  const uint32_t out_id = 2;

  subgraph.AddInput(ynn_type_fp32, q_shape, q_id)
      .AddInput(ynn_type_fp32, k_shape, k_in_id)
      .AddOutput(ynn_type_fp32, out_shape, out_id);

  uint32_t k_squared_id = YNN_INVALID_VALUE_ID;
  uint32_t k_id = YNN_INVALID_VALUE_ID;
  uint32_t k_tiled_id = YNN_INVALID_VALUE_ID;
  uint32_t k_t_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(ynn_type_fp32, k_shape, k_squared_id)
      .AddTensor(ynn_type_fp32, k_shape, k_id)
      .AddTensor(ynn_type_fp32, k_tiled_shape, k_tiled_id)
      .AddTensor(ynn_type_fp32, k_t_shape, k_t_id);

  subgraph.AddBinary(ynn_binary_multiply, k_in_id, k_in_id, k_squared_id)
      .AddUnary(ynn_unary_negate, k_squared_id, k_id)
      .AddStaticBroadcast(k_tiled_shape, k_id, k_tiled_id)
      .AddTranspose({0, 1, 3, 2}, k_tiled_id, k_t_id)
      .AddDot(/*num_k_dims=*/1, q_id, k_t_id, YNN_INVALID_VALUE_ID, out_id);

  TestScheduler scheduler(1);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // q is 1 in the first component of every head/query and 0 elsewhere, so
  // scores[0, h, i, j] reduces to k[0, 0, j, 0] = -k_in[0, 0, j, 0]^2.
  Tensor<float> q(q_shape);
  q.fill(0.0f);
  for (size_t h = 0; h < kNumHeads; ++h) {
    for (size_t i = 0; i < kQueryLen; ++i) {
      q(0, h, i, 0) = 1.0f;
    }
  }

  // Each key position holds a distinct value, so a result that is constant
  // along the key axis is unambiguously wrong.
  Tensor<float> k_in(k_shape);
  k_in.fill(0.0f);
  for (size_t j = 0; j < kKeyLen; ++j) {
    k_in(0, 0, j, 0) = static_cast<float>(j + 1);
  }

  Tensor<float> out(out_shape);

  runtime.ReshapeExternalTensor(q_shape, q.data(), q_id);
  runtime.ReshapeExternalTensor(k_shape, k_in.data(), k_in_id);
  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(out_id), out_shape);

  runtime.SetupExternalTensor(out.data(), out_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Both heads see the same broadcast key, so both expect {-1, -4, -9} for
  // every query position.
  const std::vector<float> expected = {
      -1.0f, -4.0f, -9.0f, -1.0f, -4.0f, -9.0f, -1.0f, -4.0f, -9.0f,
      -1.0f, -4.0f, -9.0f, -1.0f, -4.0f, -9.0f, -1.0f, -4.0f, -9.0f,
  };
  EXPECT_THAT(out, Pointwise(FloatNear(1e-5f), expected));
}

}  // namespace
}  // namespace ynn
