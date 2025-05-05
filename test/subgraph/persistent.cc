// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <numeric>
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

// We have two ways of implementing "persistent tensors":
// - Using XNN_VALUE_FLAG_PERSISTENT: allocate a tensor in the workspace, we own
//   the allocation.
// - Using a value that is both an input and an output, the caller owns the
//   allocation.
class Persistent : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(InputOutput, Persistent, testing::Bool());

TEST_P(Persistent, test) {
  const bool input_output = GetParam();

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<size_t> dims = {5};

  // Define subgraph
  SubgraphTester subgraph(5);
  const uint32_t input_id = 0;
  const uint32_t a_id = 1;
  const uint32_t b_id = 2;
  const uint32_t output_id = 3;
  uint32_t persistent_id = input_output ? 4 : XNN_INVALID_VALUE_ID;
  uint32_t persistent_a_id = XNN_INVALID_VALUE_ID;
  uint32_t input_b_id = XNN_INVALID_VALUE_ID;
  // This subgraph computes:
  //   (persistent) = (persistent) * (a) + (input) * (b)
  //   (output) = (persistent)
  // This allows us to conditionally read or write the persistent tensor.
  subgraph.AddInputTensor(dims, xnn_datatype_of<float>(), input_id)
      .AddInputTensor({}, xnn_datatype_of<float>(), a_id)
      .AddInputTensor({}, xnn_datatype_of<float>(), b_id)
      .AddOutputTensor(dims, xnn_datatype_of<float>(), output_id)
      .AddInternalDynamicTensorF32(dims, &persistent_a_id)
      .AddInternalDynamicTensorF32(dims, &input_b_id);

  if (input_output) {
    subgraph.AddDynamicTensor(
        dims, persistent_id, xnn_datatype_fp32,
        XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  } else {
    subgraph.AddInternalDynamicTensorF32(dims, &persistent_id,
                                         XNN_VALUE_FLAG_PERSISTENT);
  }

  subgraph.AddBinary(xnn_binary_multiply, nullptr, persistent_id, a_id,
                 persistent_a_id)
      .AddBinary(xnn_binary_multiply, nullptr, input_id, b_id, input_b_id)
      .AddBinary(xnn_binary_add, nullptr, persistent_a_id, input_b_id,
                 persistent_id)
      .AddCopy(persistent_id, output_id);
  ASSERT_EQ(xnn_status_success, subgraph.CreateRuntime());

  Tensor<float> input(dims, xnnpack::XnnExtraBytes);
  std::iota(input.begin(), input.end(), 0.0f);

  // If we are using an input-output tensor as the persistent tensor, we need
  // to provide the storage for it.
  Tensor<float> persistent(dims, xnnpack::XnnExtraBytes);
  persistent.fill(0.0f);
  if (input_output) {
    subgraph.ReshapeExternalTensor(dims, persistent.base(), persistent_id);
  }

  subgraph.ReshapeExternalTensor(dims, input.base(), input_id)
      .ReshapeRuntime();
  ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), dims);

  auto run = [&](float a, float b) {
    Tensor<float> output(dims);
    subgraph.SetupExternalTensor(output.base(), output_id)
        .SetupExternalTensor(&a, a_id)
        .SetupExternalTensor(&b, b_id)
        .SetupRuntime()
        .InvokeRuntime();
    return output;
  };

  // Initialize the persistent tensor.
  Tensor<float> init = run(0.0f, 1.0f);
  ASSERT_THAT(init, testing::ElementsAre(0.0f, 1.0f, 2.0f, 3.0f, 4.0f));

  // Try reading back the persistent tensor.
  Tensor<float> read = run(1.0f, 0.0f);
  ASSERT_THAT(read, testing::ElementsAre(0.0f, 1.0f, 2.0f, 3.0f, 4.0f));

  // Add two to the persistent tensor.
  std::fill(input.begin(), input.end(), 2.0f);
  Tensor<float> add = run(1.0f, 1.0f);
  ASSERT_THAT(add, testing::ElementsAre(2.0f, 3.0f, 4.0f, 5.0f, 6.0f));
}

TEST_P(Persistent, sliding_window) {
  const bool input_output = GetParam();

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  const size_t slices = 20;
  const size_t slice_size = 5;
  std::vector<size_t> dims = {slices, slice_size};
  std::vector<size_t> slice_dims = {1, slice_size};

  // Define subgraph
  SubgraphTester subgraph(3);
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t persistent_id = input_output ? 2 : XNN_INVALID_VALUE_ID;
  uint32_t prev_id = XNN_INVALID_VALUE_ID;
  subgraph.AddInputTensor(slice_dims, xnn_datatype_of<float>(), input_id)
      .AddOutputTensor(dims, xnn_datatype_of<float>(), output_id)
      .AddInternalDynamicTensorF32(dims, &prev_id);

  if (input_output) {
    subgraph.AddDynamicTensor(
        dims, persistent_id, xnn_datatype_fp32,
        XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  } else {
    subgraph.AddInternalDynamicTensorF32(dims, &persistent_id,
                                          XNN_VALUE_FLAG_PERSISTENT);
  }

  // Concatenate the new input with the persistent tensor, except for the last
  // slice.
  subgraph.AddSlice({0, 0}, {-1, 0}, {1, 1}, persistent_id, prev_id)
      .AddConcatenate(0, {input_id, prev_id}, persistent_id)
      .AddCopy(persistent_id, output_id);
  ASSERT_EQ(xnn_status_success, subgraph.CreateRuntime());

  // If we are using an input-output tensor as the persistent tensor, we need
  // to provide the storage for it.
  Tensor<float> persistent(dims, xnnpack::XnnExtraBytes);
  persistent.fill(0.0f);
  if (input_output) {
    subgraph.ReshapeExternalTensor(dims, persistent.base(), persistent_id);
  }

  Tensor<float> input(slice_dims, xnnpack::XnnExtraBytes);
  subgraph.ReshapeExternalTensor(slice_dims, input.base(), input_id)
      .ReshapeRuntime();
  ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), dims);

  Tensor<float> output(dims);
  subgraph.SetupExternalTensor(output.base(), output_id).SetupRuntime();

  // We should start out with a buffer of zeros.
  std::vector<float> expected(dims.front(), 0.0f);
  for (size_t t = 0; t < slices * 2; ++t) {
    std::fill(input.begin(), input.end(), t);
    subgraph.InvokeRuntime();

    for (size_t i = 0; i < slices; ++i) {
      for (size_t j = 0; j < slice_size; ++j) {
        ASSERT_EQ(output(i, j), doz(t, i));
      }
    }
  }
}

}  // namespace xnnpack
