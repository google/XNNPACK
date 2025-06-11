// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    std::vector<size_t> pre_padding = random_shape(rng, rank, 0, 3);
    std::vector<size_t> post_padding = random_shape(rng, rank, 0, 3);
    float pad_value = 1.0f;

    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddConstantPad(pre_padding, post_padding, pad_value, 0, 1)
        .CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      std::vector<size_t> output_shape(shape);
      for (size_t i = 0; i < rank; ++i) {
        output_shape[i] += pre_padding[i] + post_padding[i];
      }

      // Check reshape is correct
      subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      Tensor<T> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Make the expected output: fill a buffer with padding, and then copy
      // the unpadded area from the input.
      Tensor<T> expected(output_shape);
      expected.fill(quantize<T>(pad_value, quantization));
      expected.crop_padding(pre_padding, post_padding).assign(input);

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

template <typename T>
class ConstantPad : public ::testing::TestWithParam<int> {};

using ConstantPadQS8 = ConstantPad<quantized<int8_t>>;
using ConstantPadQU8 = ConstantPad<quantized<uint8_t>>;
using ConstantPadF16 = ConstantPad<xnn_float16>;
using ConstantPadBF16 = ConstantPad<xnn_bfloat16>;
using ConstantPadF32 = ConstantPad<float>;

TEST_P(ConstantPadQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(ConstantPadQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(ConstantPadF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(ConstantPadBF16, test) { TestImpl<xnn_bfloat16>(GetParam()); }
TEST_P(ConstantPadF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(ConstantPad, ConstantPadQS8, rank_params);
INSTANTIATE_TEST_SUITE_P(ConstantPad, ConstantPadQU8, rank_params);
INSTANTIATE_TEST_SUITE_P(ConstantPad, ConstantPadF16, rank_params);
INSTANTIATE_TEST_SUITE_P(ConstantPad, ConstantPadBF16, rank_params);
INSTANTIATE_TEST_SUITE_P(ConstantPad, ConstantPadF32, rank_params);

}  // namespace xnnpack
