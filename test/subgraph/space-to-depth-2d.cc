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
Tensor<T> space_to_depth(Tensor<T> input, size_t block_size) {
  size_t b = input.extent(0);
  size_t h = input.extent(1);
  size_t w = input.extent(2);
  size_t c = input.extent(3);
  assert(h % block_size == 0);
  assert(w % block_size == 0);
  Tensor<T> output(
      {b, h / block_size, w / block_size, c * block_size * block_size});
  Tensor<T> output_reshaped = output.reshape(
      {b, h / block_size, w / block_size, block_size, block_size, c});
  Tensor<T> input_reshaped = input.reshape(
      {b, h / block_size, block_size, w / block_size, block_size, c});
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
        .AddSpaceToDepth2D(block_size, 0, 1)
        .CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, 4, 1, 3);
      shape[1] *= block_size;
      shape[2] *= block_size;

      Tensor<T> input(shape, XnnExtraBytes);
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      Tensor<T> expected = space_to_depth(input, block_size);

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
class SpaceToDepth2D : public ::testing::TestWithParam<int> {};

using SpaceToDepth2DQS8 = SpaceToDepth2D<quantized<int8_t>>;
using SpaceToDepth2DQU8 = SpaceToDepth2D<quantized<uint8_t>>;
using SpaceToDepth2DF16 = SpaceToDepth2D<xnn_float16>;
using SpaceToDepth2DF32 = SpaceToDepth2D<float>;

TEST_P(SpaceToDepth2DQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(SpaceToDepth2DQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(SpaceToDepth2DF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(SpaceToDepth2DF32, test) { TestImpl<float>(GetParam()); }

auto block_size_params = testing::Range(2, 11);
INSTANTIATE_TEST_SUITE_P(SpaceToDepth2D, SpaceToDepth2DQS8, block_size_params);
INSTANTIATE_TEST_SUITE_P(SpaceToDepth2D, SpaceToDepth2DQU8, block_size_params);
INSTANTIATE_TEST_SUITE_P(SpaceToDepth2D, SpaceToDepth2DF16, block_size_params);
INSTANTIATE_TEST_SUITE_P(SpaceToDepth2D, SpaceToDepth2DF32, block_size_params);

}  // namespace xnnpack
