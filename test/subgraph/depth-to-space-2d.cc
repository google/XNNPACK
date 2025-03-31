// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
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
Tensor<T> depth_to_space(Tensor<T> input, size_t block_size) {
  size_t b = input.extent(0);
  size_t h = input.extent(1);
  size_t w = input.extent(2);
  size_t c = input.extent(3);
  assert(c % (block_size * block_size) == 0);
  Tensor<T> output(
      {b, h * block_size, w * block_size, c / (block_size * block_size)});
  Tensor<T> output_reshaped = output.reshape(
      {b, h, block_size, w, block_size, c / (block_size * block_size)});
  Tensor<T> input_reshaped = input.reshape(
      {b, h, w, block_size, block_size, c / (block_size * block_size)});
  output_reshaped.assign(input_reshaped.transpose({0, 1, 3, 2, 4, 5}));
  return output;
}

template <typename T>
void TestImpl(size_t block_size) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(4, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(4, xnn_datatype_of<T>(), quantization, 1)
        .AddDepthToSpace2D(block_size, 0, 1)
        .CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, 4, 1, 3);
      shape[3] *= block_size * block_size;

      Tensor<T> input(shape, PaddingBytes{XNN_EXTRA_BYTES});
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      Tensor<T> expected = depth_to_space(input, block_size);

      // Check reshaped shape is correct
      subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), expected.extents());

      // Run subgraph
      Tensor<T> output(expected.extents());
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

template <typename T>
class DepthToSpace2D : public ::testing::TestWithParam<int> {};

using DepthToSpace2DQS8 = DepthToSpace2D<quantized<int8_t>>;
using DepthToSpace2DQU8 = DepthToSpace2D<quantized<uint8_t>>;
using DepthToSpace2DF16 = DepthToSpace2D<xnn_float16>;
using DepthToSpace2DF32 = DepthToSpace2D<float>;

TEST_P(DepthToSpace2DQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(DepthToSpace2DQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(DepthToSpace2DF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(DepthToSpace2DF32, test) { TestImpl<float>(GetParam()); }

auto block_size_params = testing::Range(2, 11);
INSTANTIATE_TEST_SUITE_P(DepthToSpace2D, DepthToSpace2DQS8, block_size_params);
INSTANTIATE_TEST_SUITE_P(DepthToSpace2D, DepthToSpace2DQU8, block_size_params);
INSTANTIATE_TEST_SUITE_P(DepthToSpace2D, DepthToSpace2DF16, block_size_params);
INSTANTIATE_TEST_SUITE_P(DepthToSpace2D, DepthToSpace2DF32, block_size_params);

}  // namespace xnnpack
