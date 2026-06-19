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
  explicit Param(TupleT p) : rank(std::get<0>(p)), num_inputs(std::get<1>(p)) {}

  size_t rank;
  size_t num_inputs;

  std::string Name() const {
    return std::to_string(rank) + "_" + std::to_string(num_inputs);
  }
};

template <typename T>
void TestImpl(const Param& p) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<uint32_t> input_ids(p.num_inputs);
  std::iota(input_ids.begin(), input_ids.end(), 1);

  for (size_t axis = 0; axis < p.rank; ++axis) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(p.num_inputs + 1);
    subgraph.AddOutputTensor(p.rank, xnn_datatype_of<T>(), quantization, 0);
    for (size_t i = 0; i < p.num_inputs; ++i) {
      subgraph.AddInputTensor(p.rank, xnn_datatype_of<T>(), quantization,
                              i + 1);
    }
    subgraph.AddConcatenate(axis, input_ids, 0).CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, p.rank);
      std::vector<size_t> expected_shape = shape;
      expected_shape[axis] = 0;
      std::vector<Tensor<T>> inputs;
      for (size_t i = 0; i < p.num_inputs; ++i) {
        shape[axis] = random_shape(rng, 1)[0];
        expected_shape[axis] += shape[axis];

        Tensor<T> input_i(shape, xnnpack::XnnExtraBytes);
        DatatypeGenerator<T> generator(quantization);
        input_i.generate([&]() { return generator(rng); });
        inputs.push_back(std::move(input_i));

        subgraph.ReshapeExternalTensor(shape, inputs[i].base(), i + 1);
      }

      Tensor<T> output(expected_shape);
      // Check reshaped shape is correct
      subgraph.ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(0), output.extents());

      // Run subgraph
      subgraph.SetupExternalTensor(output.base(), 0)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      size_t offset = 0;
      for (const Tensor<T>& i : inputs) {
        Tensor<T> output_i =
            output.slice(axis, offset, offset + i.extents()[axis]).deep_copy();
        ASSERT_THAT(output_i, testing::ElementsAreArray(i));
        offset += i.extents()[axis];
      }
    }
  }
}

template <typename T>
class Concatenate : public ::testing::TestWithParam<Param> {};

using ConcatenateQS8 = Concatenate<quantized<int8_t>>;
using ConcatenateQU8 = Concatenate<quantized<uint8_t>>;
using ConcatenateF16 = Concatenate<xnn_float16>;
using ConcatenateBF16 = Concatenate<xnn_bfloat16>;
using ConcatenateF32 = Concatenate<float>;

TEST_P(ConcatenateQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(ConcatenateQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(ConcatenateF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(ConcatenateBF16, test) { TestImpl<xnn_bfloat16>(GetParam()); }
TEST_P(ConcatenateF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
auto num_inputs_params = testing::Range(2, 5);
auto params =
    ConvertGenerator<Param::TupleT>(Combine(rank_params, num_inputs_params));
INSTANTIATE_TEST_SUITE_P(Concatenate, ConcatenateQS8, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Concatenate, ConcatenateQU8, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Concatenate, ConcatenateF16, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Concatenate, ConcatenateBF16, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Concatenate, ConcatenateF32, params,
                         [](auto p) { return p.param.Name(); });

}  // namespace xnnpack
