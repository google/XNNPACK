// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
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

template <typename T>
void TestKeepDims(T, size_t rank) {
  ReplicableRandomDevice rng;

  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    std::vector<size_t> dims = random_shape(rng, rank);

    std::vector<int32_t> axes(dims.size());
    std::iota(axes.begin(), axes.end(), 0);
    std::vector<int64_t> begins(dims.size());
    std::vector<int64_t> ends(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      // Test out of bounds slices too.
      const int64_t range = dims[i] * 2;
      auto begin_dist =
          std::uniform_int_distribution<int64_t>(-range, range - 1);
      begins[i] = begin_dist(rng);
      std::uniform_int_distribution<int64_t> end_dist;
      if (begins[i] < 0) {
        // Negative begin, negative end
        end_dist = std::uniform_int_distribution<int64_t>(begins[i], 0);
      } else if (rng() % 2 == 0) {
        // Positive begin, negative end
        end_dist = std::uniform_int_distribution<int64_t>(begins[i] - range, 0);
      } else {
        // Positive begin, positive end
        end_dist = std::uniform_int_distribution<int64_t>(begins[i], range);
      }
      ends[i] = end_dist(rng);
    }

    quantization_params quantization = random_quantization(type_of<T>(), rng);

    std::vector<int64_t> strides(dims.size(), 1);
    // Define subgraph
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), rank, 0, quantization)
        .AddOutput(type_of<T>(), rank, 1, quantization)
        .AddSlice(axes, begins, ends, strides, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);
      for (size_t i = 0; i < rank; ++i) {
        shape[i] += dims[i];
      }

      Tensor<T> input(shape);
      TypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Make a deep copy so the expected result is contiguous.
      Tensor<T> expected = input.slice(begins, ends).deep_copy();

      // Check reshape is correct
      runtime.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(1), expected.extents());

      // Run subgraph
      Tensor<T> output(expected.extents());
      runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

std::vector<int32_t> mask_to_axes(uint32_t mask) {
  std::vector<int32_t> axes;
  for (uint32_t i = 0; i < YNN_MAX_TENSOR_RANK; ++i) {
    if (mask & (1 << i)) {
      axes.push_back(i);
    }
  }
  return axes;
}

template <typename T>
void TestSliceDims(T, size_t rank) {
  ReplicableRandomDevice rng;

  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    std::vector<size_t> dims = random_shape(rng, rank);

    for (uint32_t mask = 1; mask < (1 << dims.size()); ++mask) {
      std::vector<int32_t> axes = mask_to_axes(mask);
      std::reverse(axes.begin(), axes.end());
      std::vector<int64_t> at(axes.size());
      for (size_t i = 0; i < axes.size(); ++i) {
        const int64_t range = dims[axes[i]];
        auto begin_dist =
            std::uniform_int_distribution<int64_t>(-range, range - 1);
        at[i] = begin_dist(rng);
      }

      quantization_params quantization = random_quantization(type_of<T>(), rng);

      // Define subgraph
      SubgraphBuilder subgraph(2);
      subgraph.AddInput(type_of<T>(), rank, 0, quantization)
          .AddOutput(type_of<T>(), rank - axes.size(), 1, quantization)
          .AddSlice(axes, at, {}, {}, 0, 1, YNN_NODE_FLAG_SLICE_DIMS);

      Runtime runtime(subgraph.GetSubgraph());
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      for (int reshape = 0; reshape < 2; ++reshape) {
        std::vector<size_t> shape = random_shape(rng, rank);
        for (size_t i = 0; i < rank; ++i) {
          shape[i] += dims[i];
        }

        Tensor<T> input(shape);
        TypeGenerator<T> generator(quantization);
        input.generate([&]() { return generator(rng); });

        // Make a deep copy so the expected result is contiguous.
        Tensor<T> expected = input;
        for (size_t i = 0; i < axes.size(); ++i) {
          expected = expected.slice(axes[i], at[i]);
          expected = expected.remove_dim(axes[i]);
        }
        expected = expected.deep_copy();

        // Check reshape is correct
        runtime.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
        ASSERT_EQ(runtime.GetExternalTensorShape(1), expected.extents());

        // Run subgraph
        Tensor<T> output(expected.extents());
        runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

        // Verify results.
        ASSERT_THAT(output, testing::ElementsAreArray(expected));
      }
    }
  }
}

class Slice : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(Slice, keep_dims) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestKeepDims(type, rank); });
}

TEST_P(Slice, slice_dims) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestSliceDims(type, rank); });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_rank_for_testing = 4;

INSTANTIATE_TEST_SUITE_P(
    Slice, Slice,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, max_rank_for_testing)),
    test_param_to_string<Slice::ParamType>);

}  // namespace ynn
