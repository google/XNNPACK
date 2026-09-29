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
    size_t padding_rank = std::uniform_int_distribution<size_t>(1, rank)(rng);
    std::vector<size_t> pre_padding = random_shape(rng, padding_rank, 0, 3);
    std::vector<size_t> post_padding = random_shape(rng, padding_rank, 0, 3);
    float pad_value = 1.0f;

    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddConstantPad(pre_padding, post_padding, pad_value, 0, 1)
        .CreateRuntime();

    // The test code needs the padding to be the same rank as the input/output.
    pre_padding.resize(rank);
    post_padding.resize(rank);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      std::vector<size_t> output_shape(shape);
      for (size_t i = 0; i < padding_rank; ++i) {
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

#ifndef XNNPACK_USE_YNNPACK
TEST(ConstantPad, ReshapeOverflowInputElements) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[2] = {2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[2] = {4, 4};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  const size_t pre_paddings[2] = {1, 1};
  const size_t post_paddings[2] = {1, 1};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_static_constant_pad_v2(
          subgraph, 2, pre_paddings, post_paddings, 0.0f, input_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  runtime->values[input_id].shape.num_dims = 2;
  runtime->values[input_id].shape.dim[0] = SIZE_MAX;
  runtime->values[input_id].shape.dim[1] = 2;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_EQ(xnn_status_invalid_parameter, reshape_status);
}

TEST(ConstantPad, ReshapeOverflowOutputSize) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[2] = {2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[2] = {4, 4};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  const size_t pre_paddings[2] = {0, 0};
  const size_t post_paddings[2] = {0, 0};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_static_constant_pad_v2(
          subgraph, 2, pre_paddings, post_paddings, 0.0f, input_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  runtime->values[input_id].shape.num_dims = 2;
  runtime->values[input_id].shape.dim[0] = (SIZE_MAX / 4) + 1;
  runtime->values[input_id].shape.dim[1] = 1;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_TRUE(reshape_status == xnn_status_out_of_memory ||
              reshape_status == xnn_status_invalid_parameter);
}
#endif  // XNNPACK_USE_YNNPACK

}  // namespace xnnpack
