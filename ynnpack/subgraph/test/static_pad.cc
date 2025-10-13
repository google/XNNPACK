// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
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

template <typename Rng>
std::vector<int64_t> random_padding(Rng& rng, size_t rank, int64_t min,
                                    int64_t max) {
  std::uniform_int_distribution<int64_t> dim_dist(min, max);
  std::vector<int64_t> padding(rank);
  for (size_t i = 0; i < rank; ++i) {
    padding[i] = dim_dist(rng);
  }
  return padding;
}

std::vector<size_t> max0(const std::vector<int64_t>& v) {
  std::vector<size_t> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = std::max<int64_t>(0, v[i]);
  }
  return result;
}

std::vector<int64_t> negate(std::vector<int64_t> v) {
  for (int64_t& i : v) {
    i = -i;
  }
  return v;
}

std::vector<int32_t> iota(size_t rank) {
  std::vector<int32_t> result(rank);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

std::vector<int64_t> gather(const std::vector<int32_t>& axes,
                            const std::vector<int64_t>& values) {
  std::vector<int64_t> result(axes.size(), 0);
  for (size_t i = 0; i < axes.size(); ++i) {
    result[i] = values[axes[i]];
  }
  return result;
}

template <typename T>
void TestImpl(T, size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    std::vector<int32_t> axes = iota(rank);
    std::vector<int64_t> pre_padding = random_padding(rng, rank, -3, 3);
    std::vector<int64_t> post_padding = random_padding(rng, rank, -3, 3);

    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
      if (bool_dist(rng)) {
        // Randomly remove dimensions from the padding op. To implement the
        // reference result, just set the padding to 0.
        pre_padding[i] = 0;
        post_padding[i] = 0;
        axes.erase(axes.begin() + i);
      }
    }

    float pad_value = 1.0f;

    quantization_params quantization = random_quantization(type_of<T>(), rng);

    // Define subgraph
    SubgraphBuilder subgraph(2);
    uint32_t padding_id = subgraph.DefineScalar<T>(pad_value, quantization);
    subgraph.AddInput(type_of<T>(), rank, 0, quantization)
        .AddOutput(type_of<T>(), rank, 1, quantization)
        .AddPad(axes, gather(axes, pre_padding), gather(axes, post_padding), 0,
                padding_id, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape);
      TypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      std::vector<size_t> output_shape(shape);
      for (size_t i = 0; i < rank; ++i) {
        output_shape[i] = std::max<int64_t>(
            0, output_shape[i] + pre_padding[i] + post_padding[i]);
      }

      // Check reshape is correct
      runtime.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      Tensor<T> output(output_shape);
      runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

      // Make the expected output: fill a buffer with padding, and then copy
      // the unpadded area from the input with the negative padding cropped off.
      Tensor<T> expected(output_shape);
      expected.fill(quantize<T>(pad_value, quantization));
      expected.crop_padding(max0(pre_padding), max0(post_padding))
          .assign(input.crop_padding(max0(negate(pre_padding)),
                                     max0(negate(post_padding))));

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

class Pad : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(Pad, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank); });
}

INSTANTIATE_TEST_SUITE_P(
    Pad, Pad,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, 4)),
    test_param_to_string<Pad::ParamType>);

}  // namespace ynn
