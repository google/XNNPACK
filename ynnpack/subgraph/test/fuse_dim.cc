// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
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

  for (size_t axis = 0; axis < rank; ++axis) {
    for (size_t axes_count = 1; axis + axes_count <= rank; ++axes_count) {
      quantization_params quantization = random_quantization(type_of<T>(), rng);

      // Define subgraph
      SubgraphBuilder subgraph(2);
      subgraph.AddInput(type_of<T>(), rank, 0, quantization)
          .AddOutput(type_of<T>(), rank + 1 - axes_count, 1, quantization)
          .AddFuseDim(axis, axes_count, 0, 1);

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
        output_shape.erase(output_shape.begin() + axis,
                           output_shape.begin() + axis + axes_count);
        size_t fused_size = std::accumulate(
            input_shape.begin() + axis, input_shape.begin() + axis + axes_count,
            static_cast<size_t>(1), std::multiplies<>());
        output_shape.insert(output_shape.begin() + axis, fused_size);
        ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

        // Run subgraph
        Tensor<T> output(output_shape);
        runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

        // Verify results.
        ASSERT_THAT(output, testing::ElementsAreArray(input));
      }
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
                     testing::Range(1, YNN_MAX_TENSOR_RANK)),
    test_param_to_string<FuseDims::ParamType>);

}  // namespace ynn
