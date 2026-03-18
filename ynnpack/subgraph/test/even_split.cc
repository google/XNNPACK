// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
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

// This needs to be in the global namespace for argument dependent lookup to
// work.
using ::ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

struct Param {
  using TupleT = std::tuple<int, int>;
  explicit Param(TupleT p)
      : rank(std::get<0>(p)), num_outputs(std::get<1>(p)) {}

  size_t rank;
  size_t num_outputs;

  std::string Name() const {
    return std::to_string(rank) + "_" + std::to_string(num_outputs);
  }
};

template <typename T>
void TestImpl(T, size_t rank, size_t num_outputs) {
  ReplicableRandomDevice rng;

  std::vector<uint32_t> output_ids(num_outputs);
  std::iota(output_ids.begin(), output_ids.end(), 1);

  for (size_t axis = 0; axis < rank; ++axis) {
    quantization_params quantization = random_quantization(type_of<T>(), rng);

    // Define subgraph
    SubgraphBuilder subgraph(num_outputs + 1);
    subgraph.AddInput(type_of<T>(), rank, 0, quantization);
    for (size_t i = 0; i < num_outputs; ++i) {
      subgraph.AddOutput(type_of<T>(), rank, i + 1, quantization);
    }
    subgraph.AddEvenSplit(axis, 0, output_ids);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> expected_shape = random_shape(rng, rank);
      std::vector<size_t> input_shape = expected_shape;
      input_shape[axis] *= num_outputs;

      Tensor<T> input(input_shape);
      fill_random(input.data(), input.size(), rng, quantization);

      // Check reshaped shape is correct
      runtime.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();
      for (uint32_t i = 0; i < num_outputs; ++i) {
        ASSERT_EQ(runtime.GetExternalTensorShape(i + 1), expected_shape);
      }

      std::vector<Tensor<T>> outputs;
      for (size_t i = 0; i < num_outputs; ++i) {
        Tensor<T> output_i(expected_shape);
        outputs.push_back(std::move(output_i));
        runtime.SetupExternalTensor(outputs[i].base(), i + 1);
      }
      // Run subgraph
      runtime.InvokeRuntime();

      // Verify results.
      size_t offset = 0;
      for (const Tensor<T>& i : outputs) {
        Tensor<T> output_i =
            input.slice(axis, offset, offset + i.extents()[axis]).deep_copy();
        ASSERT_THAT(output_i, testing::ElementsAreArray(i));
        offset += expected_shape[axis];
      }
    }
  }
}

class EvenSplit
    : public ::testing::TestWithParam<std::tuple<ynn_type, int, int>> {};

TEST_P(EvenSplit, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  int num_outputs = std::get<2>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank, num_outputs); });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_test_rank = 3;

INSTANTIATE_TEST_SUITE_P(
    EvenSplit, EvenSplit,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, max_test_rank), testing::Range(2, 5)),
    test_param_to_string<EvenSplit::ParamType>);

}  // namespace ynn
