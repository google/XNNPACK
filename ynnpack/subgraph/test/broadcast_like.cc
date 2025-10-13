// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

template <typename Rng>
std::pair<std::vector<size_t>, std::vector<size_t>> random_broadcasted_inputs(
    Rng& rng, std::vector<size_t> output_shape, size_t a_rank, size_t b_rank) {
  // The logic here is simpler if the innermost dimension is first.
  std::reverse(output_shape.begin(), output_shape.end());
  std::vector<size_t> a_shape = output_shape;
  std::vector<size_t> b_shape = output_shape;
  std::bernoulli_distribution broadcast_dist(0.25);
  for (size_t i = 0; i < output_shape.size(); i++) {
    // We only want to broadcast one of the two inputs in this dimension,
    // including broadcasting due to smaller rank.
    if (broadcast_dist(rng) && i < b_rank) {
      a_shape[i] = 1;
    } else if (broadcast_dist(rng) && i < a_rank) {
      b_shape[i] = 1;
    }
  }
  a_shape.resize(a_rank);
  b_shape.resize(b_rank);
  std::reverse(a_shape.begin(), a_shape.end());
  std::reverse(b_shape.begin(), b_shape.end());
  return {a_shape, b_shape};
}

template <typename T>
void TestImpl(T, size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution broadcast_dist(0.25);
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  for (size_t input_rank = 0; input_rank <= rank; input_rank++) {
    for (std::pair<size_t, size_t> input_ranks :
         {std::make_pair(input_rank, rank), std::make_pair(rank, input_rank)}) {
      quantization_params quantization = random_quantization(type_of<T>(), rng);

      std::vector<int32_t> axes(input_ranks.first);
      std::iota(axes.begin(), axes.end(), 0);

      // Define subgraph
      SubgraphBuilder subgraph(3);
      subgraph.AddInput(type_of<T>(), input_ranks.first, 0, quantization)
          .AddInput(type_of<half>(), input_ranks.second, 1)
          .AddOutput(type_of<T>(), rank, 2, quantization)
          .AddBroadcastLike(axes, 0, 1, 2);

      Runtime runtime(subgraph.GetSubgraph());
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      for (int reshape = 0; reshape < 2; ++reshape) {
        std::vector<size_t> output_shape = random_shape(rng, rank);
        std::vector<size_t> input_shape, template_shape;
        std::tie(input_shape, template_shape) = random_broadcasted_inputs(
            rng, output_shape, input_ranks.first, input_ranks.second);

        Tensor<T> input(input_shape);
        TypeGenerator<T> generator(quantization);
        input.generate([&]() { return generator(rng); });

        // Check reshaped shape is correct
        runtime.ReshapeExternalTensor(input_shape, input.base(), 0)
            .ReshapeExternalTensor(template_shape, nullptr, 1)
            .ReshapeRuntime();
        ASSERT_EQ(runtime.GetExternalTensorShape(2), output_shape);

        // Run subgraph
        Tensor<T> output(output_shape);
        runtime.SetupExternalTensor(output.base(), 2).InvokeRuntime();

        // Implement the broadcast so we can check the result.
        std::vector<int32_t> new_dims(output_shape.size() - input.rank());
        std::iota(new_dims.begin(), new_dims.end(), 0);
        input = input.expand_dims(new_dims);
        broadcast_extent_1(input);
        Tensor<T> expected(output_shape);
        expected.assign(input);

        // Verify results.
        ASSERT_THAT(output, testing::ElementsAreArray(expected));
      }
    }
  }
}

class BroadcastLike
    : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(BroadcastLike, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank); });
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastLike, BroadcastLike,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, YNN_MAX_TENSOR_RANK)),
    test_param_to_string<BroadcastLike::ParamType>);

}  // namespace ynn
