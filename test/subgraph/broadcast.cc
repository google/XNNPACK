// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

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
        .AddBroadcast(broadcast_shape, 0, 1)
        .CreateRuntime();

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
      if (subgraph.Status() == xnn_status_unsupported_parameter) {
        GTEST_SKIP() << "Broadcast operator unsupported by XNNPACK runtime";
        return;
      }
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
