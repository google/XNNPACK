// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
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
void TestImpl(T, size_t rank) {
  ReplicableRandomDevice rng;

  for (size_t axes_count = 1; axes_count * 2 <= rank; ++axes_count) {
    quantization_params quantization = random_quantization(type_of<T>(), rng);

    std::vector<int32_t> axes(rank / 2);
    std::iota(axes.begin(), axes.end(), 0);
    std::shuffle(axes.begin(), axes.end(), rng);
    axes.resize(axes_count);
    for (int32_t& i : axes) {
      i *= 2;
    }

    // Define subgraph
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), rank, 0, quantization)
        .AddOutput(type_of<T>(), rank + 1 - axes_count, 1, quantization)
        .AddFuseDims(axes, 0, 1);

    std::sort(axes.begin(), axes.end(), std::greater<int32_t>());

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, rank);

      Tensor<T> input(input_shape);
      TypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Check reshaped shape is correct
      runtime.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();

      std::vector<size_t> output_shape(input_shape);
      for (int32_t axis : axes) {
        output_shape[axis] *= input_shape[axis + 1];
        output_shape.erase(output_shape.begin() + axis + 1);
      }
      ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      Tensor<T> output(output_shape);
      runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(input));
    }
  }
}

class FuseDims : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(FuseDims, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank); });
}

INSTANTIATE_TEST_SUITE_P(
    FuseDims, FuseDims,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(2, YNN_MAX_TENSOR_RANK)),
    test_param_to_string<FuseDims::ParamType>);

}  // namespace ynn
