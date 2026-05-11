// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

template <typename Rng>
size_t random_factor(Rng& rng, size_t n) {
  std::uniform_int_distribution<size_t> dist(
      1, static_cast<size_t>(std::ceil(std::sqrt(n))));

  for (size_t i = 0; i < 100; ++i) {
    size_t factor = dist(rng);
    if (n % factor == 0) {
      return factor;
    }
  }
  return n;
}

template <typename Rng>
std::vector<size_t> random_reshape(Rng& rng, size_t n) {
  std::vector<size_t> new_shape;
  while (new_shape.size() + 1 < YNN_MAX_TENSOR_RANK && n > 1) {
    size_t factor = random_factor(rng, n);
    new_shape.push_back(factor);
    n /= factor;
  }
  if (n != 1) {
    new_shape.push_back(n);
  }
  return new_shape;
}

template <typename T>
void TestImpl(T) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    Tensor<T> output(random_shape(rng));
    Tensor<T> input(random_reshape(rng, output.size()));
    fill_random(input.data(), input.size(), rng);

    std::vector<size_t> dims = output.shape();

    // Randomly set one dimension to 0, indicating reshape should deduce the
    // size of that dimension.
    if (!dims.empty() && bool_dist(rng)) {
      std::uniform_int_distribution<> dim_dist(0, dims.size() - 1);
      dims[dim_dist(rng)] = 0;
    }

    // Define subgraph
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), input.extents(), 0)
        .AddOutput(type_of<T>(), output.extents(), 1)
        .AddReshape(dims, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    runtime.ReshapeExternalTensor(input.extents(), input.base(), 0)
        .ReshapeRuntime();

    // Check reshaped shape is correct
    ASSERT_EQ(runtime.GetExternalTensorShape(1), output.extents());

    // Run subgraph
    runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

    // Verify results.
    ASSERT_THAT(output, testing::ElementsAreArray(input));
  }
}

class Reshape : public ::testing::TestWithParam<ynn_type> {};

TEST_P(Reshape, test) {
  SwitchRealType(GetParam(), [&](auto type) { TestImpl(type); });
}

INSTANTIATE_TEST_SUITE_P(Reshape, Reshape,
                         testing::Values(ynn_type_int8, ynn_type_uint8,
                                         ynn_type_fp16, ynn_type_bf16,
                                         ynn_type_fp32),
                         [](auto p) { return to_string(p.param); });

// A reshape followed by a slice could cause the internal allocation to be too
// small, because the reshape was writing its full output into an allocation
// sized only for the sliced output.
TEST(Reshape, ReshapeSliceRegression) {
  std::vector<size_t> input_shape = {112};
  std::vector<size_t> reshape_shape = {7, 16};
  std::vector<size_t> output_shape = {100};

  Tensor<float> input(input_shape);
  Tensor<float> other(reshape_shape);
  Tensor<float> output(output_shape);

  std::fill(input.data(), input.data() + input.size(), 1.0f);
  std::fill(other.data(), other.data() + other.size(), 2.0f);

  uint32_t input_id = 0;
  uint32_t other_id = 1;
  uint32_t output_id = 2;

  // Intermediate values.
  uint32_t reshape1_id = YNN_INVALID_VALUE_ID;
  uint32_t add_id = YNN_INVALID_VALUE_ID;
  uint32_t reshape2_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, input_shape, input_id)
      .AddInput(ynn_type_fp32, reshape_shape, other_id)
      .AddOutput(ynn_type_fp32, output_shape, output_id)
      .AddTensor(ynn_type_fp32, 2, reshape1_id)
      .AddTensor(ynn_type_fp32, 2, add_id)
      .AddTensor(ynn_type_fp32, 1, reshape2_id)
      .AddReshape(reshape_shape, input_id, reshape1_id)
      .AddBinary(ynn_binary_add, reshape1_id, other_id, add_id)
      .AddReshape(input_shape, add_id, reshape2_id)
      .AddSlice({0}, {0}, {100}, {1}, reshape2_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(input_shape, input.base(), input_id)
      .ReshapeExternalTensor(reshape_shape, other.base(), other_id)
      .ReshapeRuntime();

  runtime.SetupExternalTensor(output.base(), output_id).InvokeRuntime();

  for (size_t i = 0; i < output.size(); ++i) {
    ASSERT_EQ(output.data()[i], 3.0f);
  }
}

}  // namespace ynn
