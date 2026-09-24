// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/stencil.h"
#include "test/subgraph/subgraph-tester.h"

using testing::ElementsAreArray;

namespace xnnpack {

template <typename T>
Tensor<T> ReferenceImpl(Tensor<T> value, Tensor<int32_t> index,
                        const StencilParams& kh, const StencilParams& kw) {
  Tensor<T> output({value.extent(0), kh.input_extent(value.extent(1)),
                    kw.input_extent(value.extent(2)), value.extent(3)});

  assert(kw.padding() == 0);
  assert(kh.padding() == 0);

  output.fill(0);
  for (size_t i = 0; i < output.extent(0); i++) {
    for (size_t y = 0; y < value.extent(1); y++) {
      for (size_t x = 0; x < value.extent(2); x++) {
        for (size_t c = 0; c < output.extent(3); c++) {
          const uint32_t dy = index(i, y, x, c) % kh.size;
          const uint32_t dx = index(i, y, x, c) / kh.size;
          output(i, y * kh.size + dy, x * kw.size + dx, c) = value(i, y, x, c);
        }
      }
    }
  }
  return output;
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (int rep = 0; rep < 100; ++rep) {
    StencilParams kw = random_stencil_params(rng, /*max_dilation=*/1);
    StencilParams kh = random_stencil_params(rng, /*max_dilation=*/1);
    // argmax pooling is weird... stride = kernel extent.
    kw.stride = kw.size;
    kh.stride = kh.size;
    // And no padding
    kw.padding_min = 0;
    kw.padding_max = 0;
    kh.padding_min = 0;
    kh.padding_max = 0;

    // Define subgraph
    SubgraphTester subgraph(3);
    subgraph.AddInputTensor(4, xnn_datatype_of<T>(), 0)
        .AddInputTensor(4, xnn_datatype_int32, 1)
        .AddOutputTensor(4, xnn_datatype_of<T>(), 2)
        .AddUnpooling2D(kh.padding_min, kw.padding_max, kh.padding_max,
                        kw.padding_min, kh.size, kw.size, 0, 1, 2);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, 4);
      input_shape[0] = 1;
      input_shape[1] += kh.dilated_kernel_extent();
      input_shape[2] += kw.dilated_kernel_extent();
      input_shape[3] = 1;

      std::vector<size_t> output_shape = {
          input_shape[0],
          kh.input_extent(input_shape[1]),
          kw.input_extent(input_shape[2]),
          input_shape[3],
      };

      Tensor<T> value(input_shape, XnnExtraBytes);
      Tensor<int32_t> index(input_shape, XnnExtraBytes);
      DatatypeGenerator<T> value_gen(-10.0f, 20.0f);
      DatatypeGenerator<int32_t> index_gen(0, kh.size * kw.size - 1);
      value.generate([&]() { return value_gen(rng); });
      index.generate([&]() { return index_gen(rng); });

      subgraph.ReshapeExternalTensor(input_shape, value.base(), 0)
          .ReshapeExternalTensor(input_shape, index.base(), 1)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(2), output_shape)
          << "output_shape=" << index_to_string(output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;

      // Run subgraph
      Tensor<T> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), 2)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      Tensor<T> expected = ReferenceImpl(value, index, kh, kw);
      ASSERT_THAT(output, ElementsAreArray(expected))
          << "output_shape=" << index_to_string(output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;
    }
  }
}

TEST(Unpooling2DF32, test) { TestImpl<float>(); }

#ifndef XNNPACK_USE_YNNPACK
TEST(Unpooling2D, reshape_rejects_index_shape_mismatch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  SubgraphTester subgraph(3);
  subgraph.AddInputTensor(4, xnn_datatype_fp32, 0)
      .AddInputTensor(4, xnn_datatype_int32, 1)
      .AddOutputTensor(4, xnn_datatype_fp32, 2)
      .AddUnpooling2D(0, 0, 0, 0, 2, 2, 0, 1, 2);
  if (subgraph.CreateRuntime() == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  // The index input is consumed with strides taken from the value shape, so an
  // index tensor smaller than the value tensor reads past the end of the index
  // buffer. The reshape must reject the mismatch rather than carry it through.
  const std::vector<size_t> value_shape = {1, 4, 4, 8};
  const std::vector<size_t> index_shape = {1, 2, 2, 8};
  Tensor<float> value(value_shape, XnnExtraBytes);
  Tensor<int32_t> index(index_shape, XnnExtraBytes);

  subgraph.ReshapeExternalTensor(value_shape, value.base(), 0)
      .ReshapeExternalTensor(index_shape, index.base(), 1)
      .ReshapeRuntime();
  EXPECT_EQ(subgraph.Status(), xnn_status_invalid_parameter);
}

TEST(Unpooling2D, ReshapeOverflowInputElements) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[4] = {1, 2, 2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t index_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_int32, 4, input_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &index_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[4] = {1, 4, 4, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, output_dims, nullptr,
          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_unpooling_2d(
          subgraph, 0, 0, 0, 0, 2, 2, input_id, index_id, output_id, 0));

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
  runtime->values[input_id].shape.dim[3] = 2;

  runtime->values[index_id].shape.num_dims = 4;
  runtime->values[index_id].shape.dim[0] = SIZE_MAX;
  runtime->values[index_id].shape.dim[1] = 2;
  runtime->values[index_id].shape.dim[2] = 2;
  runtime->values[index_id].shape.dim[3] = 2;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_EQ(xnn_status_invalid_parameter, reshape_status);
}

TEST(Unpooling2D, ReshapeOverflowOutputSize) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[4] = {1, 2, 2, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t index_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_int32, 4, input_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &index_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[4] = {1, 4, 4, 2};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, output_dims, nullptr,
          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_unpooling_2d(
          subgraph, 0, 0, 0, 0, 2, 2, input_id, index_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  const size_t large_dim = (SIZE_MAX / (4 * 4 * 2 * sizeof(float))) + 1;
  runtime->values[input_id].shape.num_dims = 4;
  runtime->values[input_id].shape.dim[0] = large_dim;
  runtime->values[input_id].shape.dim[1] = 2;
  runtime->values[input_id].shape.dim[2] = 2;
  runtime->values[input_id].shape.dim[3] = 2;

  runtime->values[index_id].shape.num_dims = 4;
  runtime->values[index_id].shape.dim[0] = large_dim;
  runtime->values[index_id].shape.dim[1] = 2;
  runtime->values[index_id].shape.dim[2] = 2;
  runtime->values[index_id].shape.dim[3] = 2;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_TRUE(reshape_status == xnn_status_out_of_memory ||
              reshape_status == xnn_status_invalid_parameter);
}
#endif  // XNNPACK_USE_YNNPACK

}  // namespace xnnpack
