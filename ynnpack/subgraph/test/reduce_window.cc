// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ::testing::Each;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::ElementsAreArray;

namespace ynn {

TEST(ReduceWindowTest, ReduceWindow1D) {
  SubgraphBuilder builder(/*external_value_count=*/2);
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  builder.AddInput(type_of<float>(), /*shape=*/1, input_id)
      .AddOutput(type_of<float>(), /*shape=*/1, output_id);

  const uint32_t padding_val_id = builder.DefineScalar(0.0f);

  uint32_t padded_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(type_of<float>(), /*shape=*/1, padded_id);

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(type_of<float>(), /*shape=*/2, stencil_id);

  builder.AddPad(/*axes=*/{0}, /*pre_paddings=*/{0}, /*post_paddings=*/{2},
                 /*input_id=*/input_id, /*padding_id=*/padding_val_id,
                 /*output_id=*/padded_id);

  builder.AddStencilCopy(/*stencil_axes=*/{0}, /*new_axes=*/{1},
                         /*stencil_dims=*/{3}, /*stencil_strides=*/{3},
                         /*stencil_dilations=*/{1}, /*input_id=*/padded_id,
                         /*padding_id=*/YNN_INVALID_VALUE_ID,
                         /*output_id=*/stencil_id);

  builder.AddReduce(ynn_reduce_sum, /*reduce_axes=*/{1},
                    /*input_a_id=*/stencil_id,
                    /*input_b_id=*/YNN_INVALID_VALUE_ID,
                    /*output_id=*/output_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const std::vector<size_t> input_shape = {10};
  Tensor<float> input(input_shape);
  input.fill(1.0f);
  runtime.ReshapeExternalTensor(input_shape, input.data(), input_id);
  runtime.ReshapeRuntime();

  const std::vector<size_t> output_shape = {4};
  Tensor<float> output(output_shape);
  runtime.SetupExternalTensor(output.data(), output_id).InvokeRuntime();

  const std::vector<float> expected = {3.0f, 3.0f, 3.0f, 1.0f};
  ASSERT_THAT(output, Pointwise(FloatNear(1e-4f), expected));
}

TEST(ReduceWindowTest, ReduceWindow2D) {
  const std::vector<size_t> input_shape = {4, 4};
  const std::vector<size_t> output_shape = {2, 2};

  SubgraphBuilder builder(/*external_value_count=*/2);
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  builder.AddInput(type_of<float>(), /*shape=*/2, input_id)
      .AddOutput(type_of<float>(), /*shape=*/2, output_id);

  const uint32_t padding_val_id = builder.DefineScalar(0.0f);

  uint32_t padded_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(type_of<float>(), /*shape=*/2, padded_id);

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(type_of<float>(), /*shape=*/4, stencil_id);

  builder.AddPad(/*axes=*/{0, 1}, /*pre_paddings=*/{0, 0},
                 /*post_paddings=*/{1, 1}, /*input_id=*/input_id,
                 /*padding_id=*/padding_val_id, /*output_id=*/padded_id);

  builder.AddStencilCopy(/*stencil_axes=*/{0, 1}, /*new_axes=*/{1, 3},
                         /*stencil_dims=*/{2, 2}, /*stencil_strides=*/{2, 2},
                         /*stencil_dilations=*/{1, 1}, /*input_id=*/padded_id,
                         /*padding_id=*/YNN_INVALID_VALUE_ID,
                         /*output_id=*/stencil_id);

  builder.AddReduce(ynn_reduce_sum, /*reduce_axes=*/{1, 3},
                    /*input_a_id=*/stencil_id,
                    /*input_b_id=*/YNN_INVALID_VALUE_ID,
                    /*output_id=*/output_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> input(input_shape);
  input.fill(1.0f);
  runtime.ReshapeExternalTensor(input_shape, input.data(), input_id);
  runtime.ReshapeRuntime();

  Tensor<float> output(output_shape);
  runtime.SetupExternalTensor(output.data(), output_id).InvokeRuntime();

  ASSERT_THAT(output, Each(FloatNear(4.0f, 1e-4f)));
}

TEST(ReduceWindowTest, MaxPooling2D) {
  SubgraphBuilder builder(/*external_value_count=*/2);
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  builder.AddInput(type_of<float>(), /*shape=*/2, input_id)
      .AddOutput(type_of<float>(), /*shape=*/2, output_id);

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(type_of<float>(), /*shape=*/4, stencil_id);

  builder.AddStencilCopy(/*stencil_axes=*/{0, 1}, /*new_axes=*/{2, 3},
                         /*stencil_dims=*/{2, 3}, /*stencil_strides=*/{1, 1},
                         /*stencil_dilations=*/{1, 1}, /*input_id=*/input_id,
                         /*padding_id=*/YNN_INVALID_VALUE_ID,
                         /*output_id=*/stencil_id);

  builder.AddReduce(ynn_reduce_max, /*reduce_axes=*/{2, 3},
                    /*input_a_id=*/stencil_id,
                    /*input_b_id=*/YNN_INVALID_VALUE_ID,
                    /*output_id=*/output_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // clang-format off
  float input[] = {
      0, 1, 2, 3,
      1, 2, 3, 4,
      2, 3, 4, 5,
      1, 2, 3, 4,
      0, 1, 2, 3,
  };
  // clang-format on
  runtime.ReshapeExternalTensor({5, 4}, input, input_id);
  runtime.ReshapeRuntime();

  float output[4 * 2];
  runtime.SetupExternalTensor(output, output_id).InvokeRuntime();

  // clang-format off
  const float expected[] = {
    3, 4,
    4, 5,
    4, 5,
    3, 4,
  };
  // clang-format on
  ASSERT_THAT(output, ElementsAreArray(expected));
}

}  // namespace ynn
