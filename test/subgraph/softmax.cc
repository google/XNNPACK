// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
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
Tensor<T> softmax(Tensor<T> x) {
  Tensor<T> y(x.extents());
  std::vector<size_t> batch_dims = x.extents();
  size_t channels = x.extents().back();
  batch_dims.pop_back();
  for (std::vector<size_t> i : EnumerateIndices(batch_dims)) {
    i.push_back(0);
    const T* x_i = &x(i);
    T* y_i = &y(i);
    // softmax(x) = softmax(x - C) for any C. To avoid computing exp of large
    // values which overflow, we use C = max(x).
    double max = *std::max_element(x_i, x_i + channels);
    double sum_exp = 0.0;
    for (size_t c = 0; c < channels; c++) {
      sum_exp += std::exp(static_cast<double>(x_i[c]) - max);
    }
    for (size_t c = 0; c < channels; c++) {
      y_i[c] = std::exp(static_cast<double>(x_i[c]) - max) / sum_exp;
    }
  }
  return y;
}

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  // Define subgraph
  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), 0)
      .AddOutputTensor(rank, xnn_datatype_of<T>(), 1)
      .AddSoftmax(0, 1);
  xnn_status status = subgraph.CreateRuntime();
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
    return;
  }

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    std::vector<size_t> shape = random_shape(rng, rank);

    Tensor<T> input(shape, xnnpack::XnnExtraBytes);
    DatatypeGenerator<T> generator(-20.0f, 20.0f);
    input.generate([&]() { return generator(rng); });

    Tensor<T> expected = softmax(input);

    // Check reshaped shape is correct
    subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
    ASSERT_EQ(subgraph.GetExternalTensorShape(1), expected.extents());

    // Run subgraph
    // Softmax reads from the output assuming XNN_EXTRA_BYTES exist.
    Tensor<T> output(expected.extents(), xnnpack::XnnExtraBytes);
    subgraph.SetupExternalTensor(output.base(), 1)
        .SetupRuntime()
        .InvokeRuntime();

    // Verify results.
    const float tolerance = sizeof(T) == 2 ? 1.0e-2f : 1.0e-4f;
    ASSERT_THAT(output,
                testing::Pointwise(testing::FloatNear(tolerance), expected));
  }
}

template <typename T>
class Softmax : public ::testing::TestWithParam<int> {};

using SoftmaxF16 = Softmax<xnn_float16>;
using SoftmaxF32 = Softmax<float>;

TEST_P(SoftmaxF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(SoftmaxF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxF32, rank_params);

#ifndef XNNPACK_USE_YNNPACK
TEST(Softmax, reshape_rejects_scalar_input) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(TensorShape(), xnn_datatype_fp32, 0)
      .AddOutputTensor(1, xnn_datatype_fp32, 1)
      .AddSoftmax(0, 1);
  if (subgraph.CreateRuntime() == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  // An external input may be reshaped to a scalar (rank 0) at runtime. The
  // reshape must reject it, not index shape.dim[num_dims - 1] with num_dims 0.
  float data = 0.0f;
  subgraph.ReshapeExternalTensor(TensorShape(), &data, 0).ReshapeRuntime();
  EXPECT_EQ(subgraph.Status(), xnn_status_invalid_parameter);
}

TEST(Softmax, define_rejects_input_output_datatype_mismatch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  // Input fp32, output fp16: the two datatypes are individually valid for
  // softmax, but they must match. Without the check the f32 kernel would
  // write 4-byte elements into a buffer sized for 2-byte fp16 elements.
  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(1, xnn_datatype_fp32, 0)
      .AddOutputTensor(1, xnn_datatype_fp16, 1);
  EXPECT_EQ(
      xnn_define_softmax(subgraph.Subgraph(), 0, 1, /*flags=*/0),
      xnn_status_invalid_parameter);
}

TEST(Softmax, ReshapeOverflowInputElements) {
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
  const size_t output_dims[2] = {2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_softmax(subgraph, input_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  const size_t large_dim = (size_t)1 << (sizeof(size_t) * 4);
  runtime->values[input_id].shape.num_dims = 2;
  runtime->values[input_id].shape.dim[0] = large_dim;
  runtime->values[input_id].shape.dim[1] = large_dim;

  EXPECT_EQ(xnn_reshape_runtime(runtime), xnn_status_invalid_parameter);
}

TEST(Softmax, ReshapeOverflowBatchDims) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[3] = {2, 2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 3, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[3] = {2, 2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 3, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_softmax(subgraph, input_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  const size_t large_dim = (size_t)1 << (sizeof(size_t) * 4);
  runtime->values[input_id].shape.num_dims = 3;
  runtime->values[input_id].shape.dim[0] = large_dim;
  runtime->values[input_id].shape.dim[1] = large_dim;
  runtime->values[input_id].shape.dim[2] = 2;

  EXPECT_EQ(xnn_reshape_runtime(runtime), xnn_status_invalid_parameter);
}

TEST(Softmax, ReshapeOverflowOutputSize) {
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
  const size_t output_dims[2] = {2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_softmax(subgraph, input_id, output_id, 0));

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
  runtime->values[input_id].shape.dim[0] = 1;
  runtime->values[input_id].shape.dim[1] = SIZE_MAX / 3;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_TRUE(reshape_status == xnn_status_out_of_memory ||
              reshape_status == xnn_status_invalid_parameter);
}
#else
// This is not an error in YNNPACK (and it doesn't crash either).
#endif  // XNNPACK_USE_YNNPACK

}  // namespace xnnpack
