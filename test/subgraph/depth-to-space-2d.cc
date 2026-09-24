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

      Tensor<T> input(shape, XnnExtraBytes);
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

#ifndef XNNPACK_USE_YNNPACK
TEST(DepthToSpace2D, ReshapeOverflowInputElements) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[4] = {1, 2, 2, 4};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[4] = {1, 4, 4, 1};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_depth_to_space_2d(subgraph, 2, input_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  runtime->values[input_id].shape.num_dims = 4;
  runtime->values[input_id].shape.dim[0] = SIZE_MAX;
  runtime->values[input_id].shape.dim[1] = 2;
  runtime->values[input_id].shape.dim[2] = 2;
  runtime->values[input_id].shape.dim[3] = 4;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_EQ(xnn_status_invalid_parameter, reshape_status);
}

TEST(DepthToSpace2D, ReshapeOverflowOutputSize) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[4] = {1, 2, 2, 4};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[4] = {1, 4, 4, 1};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_depth_to_space_2d(subgraph, 2, input_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  runtime->values[input_id].shape.num_dims = 4;
  runtime->values[input_id].shape.dim[0] = (SIZE_MAX / (4 * 4 * 1 * sizeof(float))) + 1;
  runtime->values[input_id].shape.dim[1] = 2;
  runtime->values[input_id].shape.dim[2] = 2;
  runtime->values[input_id].shape.dim[3] = 4;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_TRUE(reshape_status == xnn_status_out_of_memory ||
              reshape_status == xnn_status_invalid_parameter);
}
#endif  // XNNPACK_USE_YNNPACK

}  // namespace xnnpack
