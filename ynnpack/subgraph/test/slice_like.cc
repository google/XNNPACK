// Copyright 2025 Google LLC
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

namespace ynn {

template <typename T>
void TestSliceLike(T, size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution slice_dist(0.5);

  for (size_t input_rank = rank; input_rank <= rank; input_rank++) {
    for (size_t template_rank = rank; template_rank <= rank; template_rank++) {
      std::vector<int32_t> axes;
      for (int i = 0; i < rank; ++i) {
        if (slice_dist(rng)) {
          axes.push_back(i);
        }
      }

      // Define subgraph
      SubgraphBuilder subgraph(3);
      subgraph.AddInput(type_of<T>(), input_rank, 0)
          .AddInput(type_of<half>(), template_rank, 1)
          .AddOutput(type_of<T>(), rank, 2)
          .AddSliceLike(axes, 0, 1, 2);

      Runtime runtime(subgraph.GetSubgraph());
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      for (int reshape = 0; reshape < 2; ++reshape) {
        std::vector<size_t> input_shape = random_shape(rng, input_rank, 1, 10);
        std::vector<size_t> template_shape =
            random_shape(rng, template_rank, 1, 10);

        std::vector<size_t> output_shape = input_shape;
        for (int32_t axis : axes) {
          if (axis < template_rank) {
            output_shape[axis] =
                std::min(template_shape[axis], input_shape[axis]);
          } else {
            output_shape[axis] = 1;
          }
        }

        Tensor<T> input(input_shape);
        fill_random(input.data(), input.size(), rng);

        // Check reshaped shape is correct
        runtime.ReshapeExternalTensor(input_shape, input.base(), 0)
            .ReshapeExternalTensor(template_shape, nullptr, 1)
            .ReshapeRuntime();
        ASSERT_EQ(runtime.GetExternalTensorShape(2), output_shape);

        // Run subgraph
        Tensor<T> output(output_shape);
        runtime.SetupExternalTensor(output.base(), 2).InvokeRuntime();

        // Implement the slice so we can check the result.
        std::vector<size_t> begins(input_rank, 0);
        Tensor<T> expected = input.crop(begins, output_shape).deep_copy();

        // Verify results.
        ASSERT_THAT(output, testing::ElementsAreArray(expected));
      }
    }
  }
}

class SliceLike : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(SliceLike, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestSliceLike(type, rank); });
}

INSTANTIATE_TEST_SUITE_P(
    SliceLike, SliceLike,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, YNN_MAX_TENSOR_RANK)),
    test_param_to_string<SliceLike::ParamType>);

}  // namespace ynn
