// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution broadcast_dist(0.25);
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // The broadcast shape is a random shape, with 0s randomly added which pass
    // through the input shape.
    std::vector<size_t> broadcast_shape = random_shape(rng, rank);
    for (size_t& dim : broadcast_shape) {
      if (broadcast_dist(rng)) {
        dim = 0;
      }
    }

    // static_broadcast supports adding new dimensions, but only if the static
    // broadcast shape is not trying to pass the input shape through (the static
    // broadcast shape is not 0 in that dimension).
    size_t input_rank = rank;
    while (input_rank >= 1 && broadcast_shape[rank - input_rank] != 0 &&
           broadcast_dist(rng)) {
      input_rank--;
    }

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(input_rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddBroadcast(broadcast_shape, 0, 1);
    ASSERT_EQ(subgraph.CreateRuntime(), xnn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = broadcast_shape;
      std::vector<size_t> output_shape = broadcast_shape;
      for (size_t i = 0; i < rank; ++i) {
        if (input_shape[i] == 0) {
          input_shape[i] = dim_dist(rng);
          output_shape[i] = input_shape[i];
        } else {
          input_shape[i] = 1;
        }
      }
      input_shape.erase(input_shape.begin(),
                        input_shape.begin() + rank - input_rank);

      Tensor<T> input(input_shape, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Check reshaped shape is correct
      subgraph.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.Status(), xnn_status_success);
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      Tensor<T> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Add the new dimensions back to the input and make it broadcastable.
      std::vector<size_t> new_axes(rank - input_rank);
      std::iota(new_axes.begin(), new_axes.end(), 0);
      input = input.expand_dims(new_axes);
      broadcast_extent_1(input);

      Tensor<T> expected(output_shape);
      expected.assign(input);

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

// Runs `input -> static_broadcast -> consumer -> output` with the default
// runtime flags, so the static_broadcast is either elided into the consumer or
// rewritten into a broadcasting add of zero, and checks the output shape and
// values. The consumer is `add(., other)` when `other_shape` is given and
// `abs(.)` otherwise.
static void CheckStaticBroadcast(const std::vector<size_t>& input_shape,
                                 const std::vector<size_t>& broadcast_shape,
                                 const std::vector<size_t>& other_shape,
                                 const std::vector<size_t>& expected_shape,
                                 const std::vector<float>& expected_values) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t other_id = 2;
  const bool binary = !other_shape.empty();
  SubgraphTester subgraph(binary ? 3 : 2);
  subgraph.AddInputTensor(input_shape, xnn_datatype_fp32, input_id)
      .AddOutputTensor(expected_shape, xnn_datatype_fp32, output_id);
  if (binary) {
    subgraph.AddInputTensor(other_shape, xnn_datatype_fp32, other_id);
  }
  uint32_t broadcast_id = XNN_INVALID_VALUE_ID;
  subgraph.AddInternalDynamicTensor(expected_shape, xnn_datatype_fp32,
                                    &broadcast_id, /*flags=*/0);
  subgraph.AddBroadcast(broadcast_shape, input_id, broadcast_id);
  if (binary) {
    subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, broadcast_id,
                       other_id, output_id);
  } else {
    subgraph.AddUnary(xnn_unary_abs, /*params=*/nullptr, broadcast_id,
                      output_id);
  }
  ASSERT_EQ(subgraph.CreateRuntime(), xnn_status_success);

  // input = -1, -2, -3, ...; other = 10, 11, 12, ...
  Tensor<float> input(input_shape, xnnpack::XnnExtraBytes);
  float next_input = 0.0f;
  input.generate([&]() { return next_input -= 1.0f; });
  Tensor<float> other(binary ? other_shape : std::vector<size_t>{1},
                      xnnpack::XnnExtraBytes);
  std::iota(other.begin(), other.end(), 10.0f);
  subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id);
  if (binary) {
    subgraph.ReshapeExternalTensor(other_shape, other.base(), other_id);
  }
  subgraph.ReshapeRuntime();
  ASSERT_EQ(subgraph.Status(), xnn_status_success);
  ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), expected_shape);

  Tensor<float> output(expected_shape);
  subgraph.SetupExternalTensor(output.base(), output_id)
      .SetupRuntime()
      .InvokeRuntime();
  ASSERT_EQ(subgraph.Status(), xnn_status_success);
  ASSERT_THAT(output, testing::ElementsAreArray(expected_values));
}

TEST(StaticBroadcastRewrite, ElidedIntoBinaryOnlyIfShapeUnchanged) {
  // add(broadcast(s[1] -> [4, 5]), other[1, 5]): the add broadcasts only to
  // [1, 5] on its own, so the broadcast may not be elided into it.
  CheckStaticBroadcast(/*input_shape=*/{1}, /*broadcast_shape=*/{4, 5},
                       /*other_shape=*/{1, 5}, /*expected_shape=*/{4, 5},
                       {9, 10, 11, 12, 13, 9, 10, 11, 12, 13, 9, 10, 11, 12,
                        13, 9, 10, 11, 12, 13});
  // add(broadcast(s[1] -> [4, 5]), other[4, 5]): the add broadcasts to [4, 5]
  // on its own, so the broadcast can be elided.
  CheckStaticBroadcast(/*input_shape=*/{1}, /*broadcast_shape=*/{4, 5},
                       /*other_shape=*/{4, 5}, /*expected_shape=*/{4, 5},
                       {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28});
}

TEST(StaticBroadcastRewrite, MaterializesRankIncreasingBroadcast) {
  // abs(broadcast(x[4] -> [4, 4])): the new leading dimension coincides with
  // the input's only dimension, which must not be mistaken for a kept one.
  CheckStaticBroadcast(/*input_shape=*/{4}, /*broadcast_shape=*/{4, 4},
                       /*other_shape=*/{}, /*expected_shape=*/{4, 4},
                       {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
  // The same, with the passed-through dimension given as 0.
  CheckStaticBroadcast(/*input_shape=*/{3}, /*broadcast_shape=*/{4, 0},
                       /*other_shape=*/{}, /*expected_shape=*/{4, 3},
                       {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});
}

template <typename T>
class Broadcast : public ::testing::TestWithParam<int> {};

using BroadcastQS8 = Broadcast<quantized<int8_t>>;
using BroadcastQU8 = Broadcast<quantized<uint8_t>>;
using BroadcastBF16 = Broadcast<xnn_bfloat16>;
using BroadcastF16 = Broadcast<xnn_float16>;
using BroadcastF32 = Broadcast<float>;

TEST_P(BroadcastQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(BroadcastQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(BroadcastBF16, test) { TestImpl<xnn_bfloat16>(GetParam()); }
TEST_P(BroadcastF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(BroadcastF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastQS8, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastQU8, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastBF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastF32, rank_params);

}  // namespace xnnpack
