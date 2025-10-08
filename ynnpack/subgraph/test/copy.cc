// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cstddef>
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

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    quantization_params quantization = random_quantization(type_of<T>(), rng);

    // Define subgraph
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), rank, 0, quantization)
        .AddOutput(type_of<T>(), rank, 1, quantization)
        .AddCopy(0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape);
      TypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Check reshaped shape is correct
      runtime.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(1), input.extents());

      // Run subgraph
      Tensor<T> output(input.extents());
      runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(input));
    }
  }
}

class Copy : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(Copy, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank); });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_test_rank = 3;

INSTANTIATE_TEST_SUITE_P(
    Copy, Copy,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(0, max_test_rank)),
    test_param_to_string<Copy::ParamType>);

}  // namespace ynn
