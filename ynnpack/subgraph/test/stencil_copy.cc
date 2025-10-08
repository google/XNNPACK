// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

struct StencilParams {
  size_t dim;
  int size;
  int dilation;
  int stride;

  int dilated_kernel_extent() const { return (size - 1) * dilation + 1; }

  int output_extent(int input_extent) const {
    return std::max(0, input_extent - dilated_kernel_extent()) / stride + 1;
  }

  int input_extent(int output_extent) const {
    assert(output_extent > 0);
    return stride * (output_extent - 1) + dilated_kernel_extent();
  }

  // Return 'same' padding at min, max for a given input extent.
  std::pair<int, int> compute_same_padding(int input_extent) const {
    int output_extent = ceil_div(input_extent, stride);
    int total_padding = std::max(0, (output_extent - 1) * stride +
                                        dilated_kernel_extent() - input_extent);
    int padding_min = total_padding / 2;
    int padding_max = total_padding - padding_min;
    return {padding_min, padding_max};
  }
};

template <typename Rng>
StencilParams random_stencil_params(Rng& rng, int max_dilation = 2,
                                    int max_kernel_size = 7) {
  std::uniform_int_distribution<> size_dist{1, max_kernel_size};
  std::uniform_int_distribution<> dilation_dist{1, max_dilation};
  std::uniform_int_distribution<> stride_dist{1, 3};

  StencilParams result;
  result.size = size_dist(rng);
  result.dilation = dilation_dist(rng);
  result.stride = std::min(stride_dist(rng),
                           std::max(result.dilated_kernel_extent() - 1, 1));
  result.dilation = 1;
  result.stride = 1;
  return result;
}

template <typename T>
Tensor<T> make_stencil_dim(Tensor<T> x, const StencilParams& p) {
  return make_stencil_dim(x, p.dim, p.size, p.stride, p.dilation);
}

template <typename T>
void TestImpl(T, size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);

  for (auto _ : FuzzTest(std::chrono::milliseconds(200))) {
    for (size_t num_stencils = 1;
         num_stencils <= rank && num_stencils + rank <= YNN_MAX_TENSOR_RANK;
         ++num_stencils) {
      quantization_params quantization = random_quantization(type_of<T>(), rng);

      std::vector<int32_t> stencil_axes(rank);
      std::iota(stencil_axes.begin(), stencil_axes.end(), 0);
      std::shuffle(stencil_axes.begin(), stencil_axes.end(), rng);
      stencil_axes.resize(num_stencils);
      std::sort(stencil_axes.begin(), stencil_axes.end());
      std::vector<int32_t> new_axes(num_stencils);
      for (size_t i = 0; i < num_stencils; ++i) {
        // TODO: We should test more usage patterns than just inserting the new
        // dimension before the stencil dimension.
        new_axes[i] = stencil_axes[i] + i;
      }

      std::vector<StencilParams> stencils;
      stencils.reserve(num_stencils);
      for (size_t i = 0; i < num_stencils; ++i) {
        stencils.push_back(random_stencil_params(rng));
        stencils.back().dim = stencil_axes[i];
      }

      std::vector<size_t> stencil_dims(num_stencils);
      std::vector<size_t> stencil_strides(num_stencils);
      std::vector<size_t> stencil_dilations(num_stencils);
      for (size_t i = 0; i < num_stencils; ++i) {
        stencil_dims[i] = stencils[i].size;
        stencil_strides[i] = stencils[i].stride;
        stencil_dilations[i] = stencils[i].dilation;
      }

      std::reverse(stencils.begin(), stencils.end());

      // Define subgraph
      SubgraphBuilder subgraph(2);
      const uint32_t padding_id =
          bool_dist(rng) ? subgraph.DefineScalar<T>(0) : YNN_INVALID_VALUE_ID;
      subgraph.AddInput(type_of<T>(), rank, 0, quantization)
          .AddOutput(type_of<T>(), rank + num_stencils, 1, quantization)
          .AddStencilCopy(stencil_axes, new_axes, stencil_dims, stencil_strides,
                          stencil_dilations, 0, padding_id, 1);

      Runtime runtime(subgraph.GetSubgraph());
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      for (int reshape = 0; reshape < 2; ++reshape) {
        std::vector<size_t> shape = random_shape(rng, rank);
        for (StencilParams& stencil : stencils) {
          shape[stencil.dim] = stencil.input_extent(shape[stencil.dim]);
        }

        Tensor<T> input(shape);
        TypeGenerator<T> generator(quantization);
        input.generate([&]() { return generator(rng); });

        Tensor<T> expected = input;
        if (padding_id != YNN_INVALID_VALUE_ID) {
          std::vector<size_t> padding_min(input.rank());
          std::vector<size_t> padding_max(input.rank());
          for (const StencilParams& stencil : stencils) {
            std::tie(padding_min[stencil.dim], padding_max[stencil.dim]) =
                stencil.compute_same_padding(input.extent(stencil.dim));
          }
          expected = expected.pad(0, padding_min, padding_max);
        }
        for (const StencilParams& stencil : stencils) {
          expected = make_stencil_dim(expected, stencil);
        }
        // Make a deep copy so the expected result is contiguous.
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

class StencilCopy : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {
};

TEST_P(StencilCopy, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank); });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_test_rank = 4;

INSTANTIATE_TEST_SUITE_P(
    StencilCopy, StencilCopy,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, max_test_rank)),
    test_param_to_string<StencilCopy::ParamType>);

}  // namespace ynn
