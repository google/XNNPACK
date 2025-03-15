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
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

using testing::Combine;
using testing::ConvertGenerator;

namespace xnnpack {

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
void TestImpl(const Param& p) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<uint32_t> output_ids(p.num_outputs);
  std::iota(output_ids.begin(), output_ids.end(), 1);

  for (size_t axis = 0; axis < p.rank; ++axis) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(p.num_outputs + 1);
    subgraph.AddInputTensor(p.rank, xnn_datatype_of<T>(), quantization, 0);
    for (size_t i = 0; i < p.num_outputs; ++i) {
      subgraph.AddOutputTensor(p.rank, xnn_datatype_of<T>(), quantization,
                               i + 1);
    }
    subgraph.AddEvenSplit(axis, 0, output_ids).CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> expected_shape = random_shape(rng, p.rank);
      std::vector<size_t> input_shape = expected_shape;
      input_shape[axis] *= p.num_outputs;

      Tensor<T> input(input_shape, PaddingBytes{XNN_EXTRA_BYTES});
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Check reshaped shape is correct
      subgraph.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();
      for (uint32_t i = 0; i < p.num_outputs; ++i) {
        ASSERT_EQ(subgraph.GetExternalTensorShape(i + 1), expected_shape);
      }

      std::vector<Tensor<T>> outputs;
      for (size_t i = 0; i < p.num_outputs; ++i) {
        Tensor<T> output_i(expected_shape);
        outputs.push_back(std::move(output_i));
        subgraph.SetupExternalTensor(outputs[i].base(), i + 1);
      }
      // Run subgraph
      subgraph.SetupRuntime().InvokeRuntime();

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

template <typename T>
class EvenSplit : public ::testing::TestWithParam<Param> {};

using EvenSplitQS8 = EvenSplit<quantized<int8_t>>;
using EvenSplitQU8 = EvenSplit<quantized<uint8_t>>;
using EvenSplitF16 = EvenSplit<xnn_float16>;
using EvenSplitBF16 = EvenSplit<xnn_bfloat16>;
using EvenSplitF32 = EvenSplit<float>;

TEST_P(EvenSplitQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(EvenSplitQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(EvenSplitF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(EvenSplitBF16, test) { TestImpl<xnn_bfloat16>(GetParam()); }
TEST_P(EvenSplitF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
auto num_outputs_params = testing::Range(2, 5);
auto params =
    ConvertGenerator<Param::TupleT>(Combine(rank_params, num_outputs_params));
INSTANTIATE_TEST_SUITE_P(EvenSplit, EvenSplitQS8, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(EvenSplit, EvenSplitQU8, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(EvenSplit, EvenSplitF16, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(EvenSplit, EvenSplitBF16, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(EvenSplit, EvenSplitF32, params,
                         [](auto p) { return p.param.Name(); });

}  // namespace xnnpack
