// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
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
void TestImpl(T, size_t rank, size_t num_inputs) {
  ReplicableRandomDevice rng;

  std::vector<uint32_t> input_ids(num_inputs);
  std::iota(input_ids.begin(), input_ids.end(), 1);

  for (size_t axis = 0; axis <= rank; ++axis) {
    quantization_params quantization = random_quantization(type_of<T>(), rng);

    // Define subgraph
    SubgraphBuilder subgraph(num_inputs + 1);
    subgraph.AddOutput(type_of<T>(), std::max(rank, axis) + 1, 0, quantization);
    for (size_t i = 0; i < num_inputs; ++i) {
      subgraph.AddInput(type_of<T>(), rank, i + 1, quantization);
    }
    subgraph.AddStack(axis, input_ids, 0);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank, 3, 3);
      std::vector<Tensor<T>> inputs;
      for (size_t i = 0; i < num_inputs; ++i) {
        Tensor<T> input_i(shape);
        TypeGenerator<T> generator(quantization);
        input_i.generate([&]() { return generator(rng); });
        inputs.push_back(std::move(input_i));

        runtime.ReshapeExternalTensor(shape, inputs[i].base(), i + 1);
      }

      std::vector<size_t> expected_shape = shape;
      while (expected_shape.size() < axis) {
        expected_shape.push_back(1);
      }
      expected_shape.insert(expected_shape.begin() + axis, num_inputs);
      Tensor<T> output(expected_shape);
      // Check reshaped shape is correct
      runtime.ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(0), output.extents());

      // Run subgraph
      runtime.SetupExternalTensor(output.base(), 0).InvokeRuntime();

      // Verify results.
      for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor<T> input_i = inputs[i];
        Tensor<T> output_i = output.slice(axis, i).deep_copy();
        ASSERT_THAT(output_i, testing::ElementsAreArray(input_i));
      }
    }
  }
}

class Stack : public ::testing::TestWithParam<std::tuple<ynn_type, int, int>> {
};

TEST_P(Stack, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  int num_inputs = std::get<2>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank, num_inputs); });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_test_rank = 3;

INSTANTIATE_TEST_SUITE_P(
    Stack, Stack,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, max_test_rank), testing::Range(1, 5)),
    test_param_to_string<Stack::ParamType>);

}  // namespace ynn
