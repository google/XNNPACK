// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
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
    quantization_params quantization = random_quantization(type_of<T>(), rng);

    Tensor<T> output(random_shape(rng));
    Tensor<T> input(random_reshape(rng, output.size()));
    TypeGenerator<T> generator(quantization);
    input.generate([&]() { return generator(rng); });

    std::vector<size_t> dims = output.shape();

    // Randomly set one dimension to 0, indicating reshape should deduce the
    // size of that dimension.
    if (!dims.empty() && bool_dist(rng)) {
      std::uniform_int_distribution<> dim_dist(0, dims.size() - 1);
      dims[dim_dist(rng)] = 0;
    }

    // Define subgraph
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), input.extents(), 0, quantization)
        .AddOutput(type_of<T>(), output.extents(), 1, quantization)
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

}  // namespace ynn
