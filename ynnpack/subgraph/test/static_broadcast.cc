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
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

template <typename T>
void TestImpl(T, size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution broadcast_dist(0.25);
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    quantization_params quantization = random_quantization(type_of<T>(), rng);

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
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), input_rank, 0, quantization)
        .AddOutput(type_of<T>(), rank, 1, quantization)
        .AddBroadcast(broadcast_shape, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

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

      Tensor<T> input(input_shape);
      TypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Check reshaped shape is correct
      runtime.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      Tensor<T> output(output_shape);
      runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

      // Add the new dimensions back to the input and make it broadcastable.
      std::vector<int32_t> new_axes(rank - input_rank);
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

class Broadcast : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(Broadcast, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank); });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_test_rank = 4;

INSTANTIATE_TEST_SUITE_P(
    Broadcast, Broadcast,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, max_test_rank)),
    test_param_to_string<Broadcast::ParamType>);

}  // namespace ynn
