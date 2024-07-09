// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/allocation-type.h"
#include "xnnpack/node-type.h"
#include "xnnpack/subgraph.h"
#include "mock-allocator.h"
#include "replicable_random_device.h"
#include "runtime-tester.h"
#include "subgraph-tester.h"

namespace xnnpack {

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::Return;

TEST(SUBGRAPH_FP16, value_both_external_output_and_input) {
  SubgraphTester tester(4);
  std::array<size_t, 4> pre_paddings = {0,1,0,0};
  std::array<size_t, 4> post_paddings = {0,1,0,0};
  // external input[0]
  //      /
  // [constant pad]
  //     /
  //  external     dynamic[1]
  //  output[2]     /
  //           \   /
  //           [add]
  //             |
  //         external
  //         output[3]
  tester
      .AddInputTensorF32({1, 2, 2, 3}, 0)
      .AddDynamicTensorF32({1, 1, 1, 3}, 1)
      .AddOutputTensorF32({1, 4, 2, 3}, 2)
      .AddOutputTensorF32({1, 4, 2, 3}, 3)
      .AddConstantPad(pre_paddings.data(), post_paddings.data(), 0.0f, 0, 2)
      .AddAddition(2, 1, 3)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  //
  //   external input[0]
  //        |
  //    [convert]*
  //        |
  //     input[4]*
  //       /
  // [constant pad]
  //     /
  //   fp16 value[5]*
  //    |       \
  //  [convert]* \
  //    |         \
  //  external     \    dynamic[1] converted in-place
  //  output[2]     \     /
  //                 \   /
  //                 [add]
  //                   |
  //                fp16 value[6]*
  //                   |
  //                [convert]*
  //                   |
  //               external
  //               output[3]

  // We should have 3 convert nodes, one for external input, 2 for external
  // outputs, so 5 in total, including the pad and add in the original graph.
  ASSERT_EQ(tester.NumNodes(), 5);

  const xnn_node* output_convert_node = tester.Node(4);
  ASSERT_EQ(output_convert_node->type, xnn_node_type_convert);
  ASSERT_EQ(output_convert_node->compute_type, xnn_compute_type_fp16_to_fp32);

  // Check that Addition node refers to the FP16 value before conversion.
  const xnn_node* addition_node = tester.Node(3);
  ASSERT_EQ(addition_node->type, xnn_node_type_add2);
  ASSERT_EQ(addition_node->inputs[0], 5);
  ASSERT_EQ(addition_node->inputs[1], 1);
  ASSERT_EQ(tester.Value(5)->datatype, xnn_datatype_fp16);
  ASSERT_EQ(tester.Value(1)->datatype, xnn_datatype_fp16);

  ASSERT_EQ(tester.Node(2)->type, xnn_node_type_convert);
  ASSERT_EQ(tester.Node(2)->compute_type, xnn_compute_type_fp16_to_fp32);
  ASSERT_EQ(tester.Node(1)->type, xnn_node_type_static_constant_pad);
  ASSERT_EQ(tester.Node(0)->type, xnn_node_type_convert);
  ASSERT_EQ(tester.Node(0)->compute_type, xnn_compute_type_fp32_to_fp16);
}

TEST(SUBGRAPH_FP16, with_static_value) {
  SubgraphTester tester(3);
  float static_tensor_data[3 + XNN_EXTRA_BYTES / sizeof(float)] = {
    1.0f, 2.0f, 3.0f
  };
  // external input[0]   static[1]
  //               \     /
  //                \   /
  //                [add]
  //                  |
  //               external
  //               output[2]
  tester
      .AddInputTensorF32({1, 2, 2, 3}, 0)
      // Tensor #1 is both static and external
      .AddStaticTensorF32({1, 1, 1, 3}, TensorType::kDense, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                          static_tensor_data)
      .AddOutputTensorF32({1, 4, 2, 3}, 2)
      .AddAddition(0, 1, 2)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  // The static tensor data has been converted into a new buffer.
  //
  // external input[0]
  //        |
  //    [convert]*
  //        |
  //     input[3]*   static[1]* (converted into new buffer)
  //        \       /
  //         \     /
  //          [add]
  //            |
  //       fp16 value[4]*
  //            |
  //        [convert]*
  //            |
  //         external
  //         output[2]

  // We should have 3 nodes, the original add node, plus one convert node for
  // each of the external input and output.
  ASSERT_EQ(tester.NumNodes(), 3);

  // The static value should be converted to FP16
  const xnn_value* static_value = tester.Value(1);
  ASSERT_EQ(static_value->datatype, xnn_datatype_fp16);
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[0], fp16_ieee_from_fp32_value(1.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[1], fp16_ieee_from_fp32_value(2.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[2], fp16_ieee_from_fp32_value(3.0f));

  // Check that the output of convert is allocated in workspace.
  const xnn_value* convert_out = tester.Value(3);
  ASSERT_EQ(convert_out->allocation_type, xnn_allocation_type_workspace);
  // Check that external input remains external (bug in runtime changed its allocation type.
  // const xnn_value* input = tester.Value(0);
  // ASSERT_EQ(input->allocation_type, xnn_allocation_type_external);
}

TEST(SUBGRAPH_FP16, external_inputs_allocation_type_remains_external) {
  // external input[0]   static[1]
  //               \     /
  //                \   /
  //                [add]
  //                  |
  //               external
  //               output[2]
  RuntimeTester tester(3);
  tester
      .AddInputTensorF32({1, 2, 2, 3}, 0)
      .AddInputTensorF32({1, 2, 2, 3}, 1)
      .AddOutputTensorF32({1, 2, 2, 3}, 2)
      .AddAddition(0, 1, 2)
      .Optimize()
      .RewriteForFp16();

  xnn_runtime_t runtime = tester.Runtime();
  xnn_status status = xnn_create_runtime_v3(tester.Subgraph(), nullptr, nullptr, /*flags=*/0, &runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  // Check that both external inputs remain external inputs after rewriting for FP16.
  ASSERT_EQ(tester.Value(0)->allocation_type, xnn_allocation_type_external);
  ASSERT_EQ(tester.Value(1)->allocation_type, xnn_allocation_type_external);
}

TEST(SUBGRAPH_FP16, static_buffer_allocation_failure) {
  SubgraphTester tester(3);
  tester
      .AddInputTensorF32({1, 2, 2, 3}, 0)
      .AddStaticTensorF32({1, 1, 1, 3}, TensorType::kDense, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT)
      .AddOutputTensorF32({1, 4, 2, 3}, 2)
      .AddAddition(0, 1, 2)
      .Optimize();

  MockAllocator mock_allocator;
  std::unique_ptr<MockAllocator, decltype(&RestoreDefaultAllocator)>
      auto_mock_allocator(&mock_allocator, &RestoreDefaultAllocator);
  SetUpMockAllocator(&mock_allocator);

  // Make the allocation of the static fp16 tensor buffer
  // (of size 22 = 3 * 16bits + XNN_EXTRA_BYTES) fail.
  EXPECT_CALL(mock_allocator, allocate(_, _)).Times(AnyNumber());
  EXPECT_CALL(mock_allocator, allocate(_, 22)).WillOnce(Return(nullptr));

  tester.RewriteForFp16WithFailure();
}

TEST(SUBGRAPH_FP16, external_value_allocation_failure) {
  SubgraphTester tester(3);
  tester
      .AddInputTensorF32({1, 2, 2, 3}, 0)
      .AddStaticTensorF32({1, 1, 1, 3}, TensorType::kDense, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT)
      .AddOutputTensorF32({1, 4, 2, 3}, 2)
      .AddAddition(0, 1, 2)
      .Optimize();

  MockAllocator mock_allocator;
  std::unique_ptr<MockAllocator, decltype(&RestoreDefaultAllocator)>
      auto_mock_allocator(&mock_allocator, &RestoreDefaultAllocator);
  SetUpMockAllocator(&mock_allocator);

  // Make the allocation of the external values fail.
  EXPECT_CALL(mock_allocator, reallocate(_, tester.Subgraph()->values, _))
    .WillOnce(Return(nullptr));

  tester.RewriteForFp16WithFailure();
}

TEST(SUBGRAPH_FP16, convolution_weights_used_by_another_node) {
  SubgraphTester tester(7);

  float static_filter_data[6 + XNN_EXTRA_BYTES / sizeof(float)] = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
  };
  // external input[0]   bias [2]   static filter [1]      external input [6]
  //
  //               \     /        /          \            /
  //                \   /        /            \          /
  //                [convolution]          [subtract]
  //                  |                         |
  //                convolution out [3]        subtract out [5]
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t bias_id = 2;
  const uint32_t convolution_out_id = 3;
  const uint32_t out_id2 = 5;
  const uint32_t subtract_input_id = 6;
  tester
      .AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddStaticTensorF32({2, 1, 1, 3}, TensorType::kDense, filter_id, /*flags=*/0, static_filter_data)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({1, 5, 5, 2}, convolution_out_id)
      .AddInputTensorF32({1, 4, 2, 3}, subtract_input_id)
      .AddOutputTensorF32({2, 1, 1, 3}, out_id2)
      .AddConvolution2D(
          ConvolutionParams{
            Padding{0, 0, 0, 0},
            Kernel{3, 3},
            Subsampling{1, 1},
            Dilation{1, 1},
            /*groups=*/ 1,
            /*group_input_channels=*/3,
            /*group_output_channels*/2
          }, input_id, filter_id, bias_id, convolution_out_id)
      .AddSubtract(filter_id, subtract_input_id, out_id2)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  // The static filter data has been converted into a new buffer.
  //
  // external input[0]    bias [2]  filter [1]*        external input [6]
  //              \        /        /       \           /
  //        [convert]*    /        /         \      [convert]*
  //               \     /        /           \       /
  //                \   /        /             \     /
  //                [convolution]           [subtract]
  //                  |                          |
  //                [convert]*              [convert]*
  //                  |                          |
  //                convolution out [3]     subtract out [5]

  // We should have 6 nodes, the original convolution and subtraction node, a convert for the two external inputs, and a
  // convert for the two external outputs.
  ASSERT_EQ(tester.NumNodes(), 6);

  // The static value should be converted to FP16
  const xnn_value* static_value = tester.Value(filter_id);
  ASSERT_EQ(static_value->datatype, xnn_datatype_fp16);
  ASSERT_EQ(static_value->fp32_data, static_filter_data);
  // Weights are converted to fp16.
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[0], fp16_ieee_from_fp32_value(1.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[1], fp16_ieee_from_fp32_value(2.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[2], fp16_ieee_from_fp32_value(3.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[3], fp16_ieee_from_fp32_value(4.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[4], fp16_ieee_from_fp32_value(5.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[5], fp16_ieee_from_fp32_value(6.0f));
  // But original fp32 weights kept around.
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[0], 1.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[1], 2.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[2], 3.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[3], 4.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[4], 5.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[5], 6.0f);
}

TEST(SUBGRAPH_FP16, convolution_bias_used_by_another_node) {
  SubgraphTester tester(7);

  float static_bias_data[2 + XNN_EXTRA_BYTES / sizeof(float)] = {
    1.0f, 2.0f,
  };
  // external input[0]   bias [2]   static filter [1]      external input [6]
  //
  //               \     /        /          \            /
  //                \   /        /            \          /
  //                [convolution]          [subtract]
  //                  |                         |
  //                convolution out [3]        subtract out [5]
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t bias_id = 2;
  const uint32_t convolution_out_id = 3;
  const uint32_t out_id2 = 5;
  const uint32_t subtract_input_id = 6;
  tester
      .AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddStaticTensorF32({2, 1, 1, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id, /*flags=*/0, static_bias_data)
      .AddOutputTensorF32({1, 5, 5, 2}, convolution_out_id)
      .AddInputTensorF32({2}, subtract_input_id)
      .AddOutputTensorF32({2}, out_id2)
      .AddConvolution2D(
          ConvolutionParams{
            Padding{0, 0, 0, 0},
            Kernel{3, 3},
            Subsampling{1, 1},
            Dilation{1, 1},
            /*groups=*/ 1,
            /*group_input_channels=*/3,
            /*group_output_channels*/2
          }, input_id, filter_id, bias_id, convolution_out_id)
      .AddSubtract(bias_id, subtract_input_id, out_id2)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  // The static bias data has been converted into a new buffer.
  //
  // external input[0]    bias [2]  filter [1]*        external input [6]
  //              \        /        /       \           /
  //        [convert]*    /        /         \      [convert]*
  //               \     /        /           \       /
  //                \   /        /             \     /
  //                [convolution]           [subtract]
  //                  |                          |
  //                [convert]*              [convert]*
  //                  |                          |
  //                convolution out [3]     subtract out [5]

  // We should have 6 nodes, the original convolution and subtraction node, a convert for the two external inputs, and a
  // convert for the two external outputs.
  ASSERT_EQ(tester.NumNodes(), 6);

  // The static value should be converted to FP16
  const xnn_value* static_value = tester.Value(bias_id);
  ASSERT_EQ(static_value->datatype, xnn_datatype_fp16);
  ASSERT_EQ(static_value->fp32_data, static_bias_data);
  // Weights are converted to fp16.
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[0], fp16_ieee_from_fp32_value(1.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[1], fp16_ieee_from_fp32_value(2.0f));
  // But original fp32 weights kept around.
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[0], 1.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[1], 2.0f);
}

TEST(SUBGRAPH_FP16, fully_connected_qd8_f16_qc8w) {
  SubgraphTester tester(5);
  SubgraphTester reference_tester(5);

  int8_t static_filter_data[6 + XNN_EXTRA_BYTES / sizeof(int8_t)] = {
    1, 2, 3,
    -3, -2, -1,
  };
  float bias[2] = {1, 2};
  float kernel_scale[2] = {0.5f, 1.5f};
  // external input[0]   bias [2]   static filter [1]
  //        |              /        /
  //  [convert f32->qd8]  /        /
  //               \     /        /
  //                \   /        /
  //                [fully connected]
  //                  |
  //                fully connected out [3]
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t bias_id = 2;
  const uint32_t converted_input_id = 3;
  const uint32_t fully_connected_out_id = 4;
  tester
      .AddInputTensorF32({5, 3}, input_id)
      .AddDynamicallyQuantizedTensor({5, 3}, converted_input_id, /*flags=*/0)
      .AddStaticTensorQS8({2, 3}, TensorType::kDense, &kernel_scale[0], filter_id, /*flags=*/0, static_filter_data)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id, /*flags=*/0, &bias[0])
      .AddOutputTensorF32({5, 2}, fully_connected_out_id)
      .AddConvert(input_id, converted_input_id)
      .AddFullyConnected(converted_input_id, filter_id, bias_id, fully_connected_out_id)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  //
  // external input[0]    bias [2]  filter [1]*
  //           |                |     /
  //        [convert f32->f16]  |    /
  //              |             |   /
  //        [convert f16->qd8]* |  /
  //               \            | /
  //                \           |/
  //                [fully connected]
  //                  |
  //                [convert f16->f32]*
  //                  |
  //                fully connected out [3]

  reference_tester
      .AddInputTensorF32({5, 3}, input_id)
      .AddDynamicallyQuantizedTensor({5, 3}, converted_input_id, /*flags=*/0)
      .AddStaticTensorQS8({2, 3}, TensorType::kDense, &kernel_scale[0], filter_id, /*flags=*/0, static_filter_data)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id, /*flags=*/0, &bias[0])
      .AddOutputTensorF32({5, 2}, fully_connected_out_id)
      .AddConvert(input_id, converted_input_id)
      .AddFullyConnected(converted_input_id, filter_id, bias_id, fully_connected_out_id);

  // We should have 4 nodes, the original fully connected and conversion nodes, a convert for the external input,
  // and a convert for the external output.
  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.f, 1.f), std::ref(rng));
  std::vector<float> input(15 + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> reference_output(10), output(10);
  ASSERT_EQ(tester.NumNodes(), 4);


  xnn_runtime_t fp16_runtime_ptr = nullptr;
  xnn_status status = xnn_create_runtime(tester.Subgraph(), &fp16_runtime_ptr);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_fp16_runtime(fp16_runtime_ptr, xnn_delete_runtime);
  ASSERT_EQ(xnn_status_success, status);
  xnn_runtime_t fp32_runtime_ptr = nullptr;
  status = xnn_create_runtime(reference_tester.Subgraph(), &fp32_runtime_ptr);
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_fp32_runtime(fp32_runtime_ptr, xnn_delete_runtime);

  std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{fully_connected_out_id, output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(fp16_runtime_ptr, 2, external_values.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(fp16_runtime_ptr));

  std::array<xnn_external_value, 2> reference_external_values = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{fully_connected_out_id, reference_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(fp32_runtime_ptr, 2, reference_external_values.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(fp32_runtime_ptr));

  for (int i = 0; i < output.size(); ++i) {
    const float tolerance = std::max(std::abs(reference_output[i]) * 5e-2, 5e-2);
    ASSERT_NEAR(output[i], reference_output[i], tolerance);
  }
}

TEST(SUBGRAPH_FP16, fully_connected_weights_used_by_another_node) {
  SubgraphTester tester(7);

  float static_filter_data[6 + XNN_EXTRA_BYTES / sizeof(float)] = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
  };
  // external input[0]   bias [2]   static filter [1]      external input [6]
  //
  //               \     /        /          \            /
  //                \   /        /            \          /
  //                [fully connected]          [subtract]
  //                  |                         |
  //                fully connected out [3]        subtract out [5]
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t bias_id = 2;
  const uint32_t fully_connected_out_id = 3;
  const uint32_t out_id2 = 5;
  const uint32_t subtract_input_id = 6;
  tester
      .AddInputTensorF32({5, 3}, input_id)
      .AddStaticTensorF32({2, 3}, TensorType::kDense, filter_id, /*flags=*/0, static_filter_data)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({5, 2}, fully_connected_out_id)
      .AddInputTensorF32({2, 3}, subtract_input_id)
      .AddOutputTensorF32({2, 3}, out_id2)
      .AddFullyConnected(input_id, filter_id, bias_id, fully_connected_out_id)
      .AddSubtract(filter_id, subtract_input_id, out_id2)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  // The static filter data has been converted into a new buffer.
  //
  // external input[0]    bias [2]  filter [1]*        external input [6]
  //              \        /        /       \           /
  //        [convert]*    /        /         \      [convert]*
  //               \     /        /           \       /
  //                \   /        /             \     /
  //                [fully connected]           [subtract]
  //                  |                          |
  //                [convert]*              [convert]*
  //                  |                          |
  //                fully connected out [3]     subtract out [5]

  // We should have 6 nodes, the original fully connected and subtraction node, a convert for the two external inputs,
  // and a convert for the two external outputs.
  ASSERT_EQ(tester.NumNodes(), 6);

  // The static value should be converted to FP16
  const xnn_value* static_value = tester.Value(filter_id);
  ASSERT_EQ(static_value->datatype, xnn_datatype_fp16);
  ASSERT_EQ(static_value->fp32_data, static_filter_data);
  // Weights are converted to fp16.
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[0], fp16_ieee_from_fp32_value(1.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[1], fp16_ieee_from_fp32_value(2.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[2], fp16_ieee_from_fp32_value(3.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[3], fp16_ieee_from_fp32_value(4.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[4], fp16_ieee_from_fp32_value(5.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[5], fp16_ieee_from_fp32_value(6.0f));
  // But original fp32 weights kept around.
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[0], 1.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[1], 2.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[2], 3.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[3], 4.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[4], 5.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[5], 6.0f);
}

TEST(SUBGRAPH_FP16, fully_connected_bias_used_by_another_node) {
  SubgraphTester tester(7);

  float static_bias_data[2 + XNN_EXTRA_BYTES / sizeof(float)] = {
    1.0f, 2.0f,
  };
  // external input[0]   bias [2]   static filter [1]      external input [6]
  //
  //               \     /        /          \            /
  //                \   /        /            \          /
  //                [fully connected]          [subtract]
  //                  |                         |
  //                fully connected out [3]        subtract out [5]
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t bias_id = 2;
  const uint32_t fully_connected_out_id = 3;
  const uint32_t out_id2 = 5;
  const uint32_t subtract_input_id = 6;
  tester
      .AddInputTensorF32({5, 3}, input_id)
      .AddStaticTensorF32({2, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id, /*flags=*/0, static_bias_data)
      .AddOutputTensorF32({5, 2}, fully_connected_out_id)
      .AddInputTensorF32({2}, subtract_input_id)
      .AddOutputTensorF32({2}, out_id2)
      .AddFullyConnected(input_id, filter_id, bias_id, fully_connected_out_id)
      .AddSubtract(bias_id, subtract_input_id, out_id2)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  // The static bias data has been converted into a new buffer.
  //
  // external input[0]    bias [2]  filter [1]*        external input [6]
  //              \        /        /       \           /
  //        [convert]*    /        /         \      [convert]*
  //               \     /        /           \       /
  //                \   /        /             \     /
  //                [fully connected]           [subtract]
  //                  |                          |
  //                [convert]*              [convert]*
  //                  |                          |
  //                fully connected out [3]     subtract out [5]

  // We should have 6 nodes, the original fully connected and subtraction node, a convert for the two external inputs,
  // and a convert for the two external outputs.
  ASSERT_EQ(tester.NumNodes(), 6);

  // The static value should be converted to FP16
  const xnn_value* static_value = tester.Value(bias_id);
  ASSERT_EQ(static_value->datatype, xnn_datatype_fp16);
  ASSERT_EQ(static_value->fp32_data, static_bias_data);
  // Weights are converted to fp16.
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[0], fp16_ieee_from_fp32_value(1.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[1], fp16_ieee_from_fp32_value(2.0f));
  // But original fp32 weights kept around.
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[0], 1.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[1], 2.0f);
}

TEST(SUBGRAPH_FP16, prelu_slope_used_by_another_node) {
  SubgraphTester tester(5);

  float static_slope_data[2 + XNN_EXTRA_BYTES / sizeof(float)] = {
    1.0f, 2.0f,
  };
  // external input[0]   static slope [1]      external input [4]
  //
  //               \     /        \            /
  //                \   /          \          /
  //                [prelu]        [subtract]
  //                  |                    |
  //                prelu out [2]    subtract out [3]
  const uint32_t input_id = 0;
  const uint32_t slope_id = 1;
  const uint32_t prelu_out_id = 2;
  const uint32_t out_id2 = 3;
  const uint32_t subtract_input_id = 4;
  tester
      .AddInputTensorF32({5, 3, 3, 2}, input_id)
      .AddStaticTensorF32({2}, TensorType::kDense, slope_id, /*flags=*/0, static_slope_data)
      .AddOutputTensorF32({5, 3, 3, 2}, prelu_out_id)
      .AddInputTensorF32({2}, subtract_input_id)
      .AddOutputTensorF32({2}, out_id2)
      .AddPrelu(input_id, slope_id, prelu_out_id)
      .AddSubtract(slope_id, subtract_input_id, out_id2)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  // The static bias data has been converted into a new buffer.
  //
  // external input[0]    static slope [1]*        external input [4]
  //              \        /        \           /
  //        [convert]*    /          \      [convert]*
  //               \     /            \       /
  //                \   /              \     /
  //                [prelu]      [subtract]
  //                  |                      |
  //                [convert]*         [convert]*
  //                  |                      |
  //                prelu out [2]    subtract out [3]

  // We should have 6 nodes, the original prelu and subtraction node, a convert for the two external inputs, and a
  // convert for the two external outputs.
  ASSERT_EQ(tester.NumNodes(), 6);

  // The static value should be converted to FP16
  const xnn_value* static_value = tester.Value(slope_id);
  ASSERT_EQ(static_value->datatype, xnn_datatype_fp16);
  ASSERT_EQ(static_value->fp32_data, static_slope_data);
  // Weights are converted to fp16.
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[0], fp16_ieee_from_fp32_value(1.0f));
  ASSERT_EQ(static_cast<const uint16_t*>(static_value->data)[1], fp16_ieee_from_fp32_value(2.0f));
  // But original fp32 weights kept around.
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[0], 1.0f);
  ASSERT_EQ(static_cast<const float*>(static_value->fp32_data)[1], 2.0f);
}

TEST(SUBGRAPH_FP16_DYNAMIC_FULLY_CONNECTED, dynamic_weights_no_bias_weights_converted_to_fp16) {
  SubgraphTester tester(5);

  // external input[0]   external input [1]
  //              \       /
  //               \   [constant pad]
  //                \   /
  //                [fully connected]
  //                  |
  //                fully connected out [2]
  const uint32_t input_id = 0;
  const uint32_t input2_id = 1;
  const uint32_t weights_id = 3;
  const uint32_t fully_connected_out_id = 2;
  std::array<size_t, 4> pre_paddings = {1, 0, 0, 0};
  std::array<size_t, 4> post_paddings = {0, 0, 0, 0};
  tester
      .AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddInputTensorF32({1, 1, 1, 3}, input2_id)
      .AddOutputTensorF32({1, 5, 5, 2}, fully_connected_out_id)
      .AddDynamicTensorF32({2, 1, 1, 3}, weights_id)
      .AddConstantPad(pre_paddings.data(), post_paddings.data(), 0.0f, input2_id, weights_id)
      .AddFullyConnected(input_id, weights_id, /*bias_id=*/XNN_INVALID_VALUE_ID, fully_connected_out_id)
      .Optimize()
      .RewriteForFp16();

  const xnn_value* weights_value = tester.Value(weights_id);
  ASSERT_EQ(weights_value->datatype, xnn_datatype_fp16);
}

TEST(SUBGRAPH_FP16_DYNAMIC_FULLY_CONNECTED, dynamic_weights_static_bias_weights_converted_to_fp16) {
  SubgraphTester tester(5);

  // external input[0]   external input [1]
  //              \       /            static bias [4]
  //               \   [constant pad]   /
  //                \   /              /
  //                [fully connected]
  //                  |
  //                fully connected out [2]
  const uint32_t input_id = 0;
  const uint32_t input2_id = 1;
  const uint32_t weights_id = 3;
  const uint32_t bias_id = 4;
  const uint32_t fully_connected_out_id = 2;
  std::array<size_t, 4> pre_paddings = {1,0,0,0};
  std::array<size_t, 4> post_paddings = {0,0,0,0};
  tester
      .AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddInputTensorF32({1, 1, 1, 3}, input2_id)
      .AddOutputTensorF32({1, 5, 5, 2}, fully_connected_out_id)
      .AddDynamicTensorF32({2, 1, 1, 3}, weights_id)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id)
      .AddConstantPad(pre_paddings.data(), post_paddings.data(), 0.0f, input2_id, weights_id)
      .AddFullyConnected(input_id, weights_id, bias_id, fully_connected_out_id)
      .Optimize()
      .RewriteForFp16();

  const xnn_value* weights_value = tester.Value(weights_id);
  ASSERT_EQ(weights_value->datatype, xnn_datatype_fp16);
}

TEST(SUBGRAPH_FP16_DYNAMIC_FULLY_CONNECTED, static_weights_dynamic_bias_bias_converted_to_fp16) {
  SubgraphTester tester(5);

  // external input[0] static weights [4] external input [1]
  //              \         |              /
  //               \        |        [constant pad]
  //                \       |           /
  //                [fully connected]
  //                  |
  //                fully connected out [2]
  const uint32_t input_id = 0;
  const uint32_t input2_id = 1;
  const uint32_t weights_id = 3;
  const uint32_t bias_id = 4;
  const uint32_t fully_connected_out_id = 2;
  std::array<size_t, 4> pre_paddings = {1};
  std::array<size_t, 4> post_paddings = {0};
  tester
      .AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddInputTensorF32({1}, input2_id)
      .AddOutputTensorF32({1, 5, 5, 2}, fully_connected_out_id)
      .AddStaticTensorF32({2, 1, 1, 3}, TensorType::kDense, weights_id)
      .AddDynamicTensorF32({2}, bias_id)
      .AddConstantPad(pre_paddings.data(), post_paddings.data(), 0.0f, input2_id, bias_id)
      .AddFullyConnected(input_id, weights_id, bias_id, fully_connected_out_id)
      .Optimize()
      .RewriteForFp16();

  const xnn_value* bias_value = tester.Value(bias_id);
  ASSERT_EQ(bias_value->datatype, xnn_datatype_fp16);
}

TEST(SUBGRAPH_FP16_DYNAMIC_FULLY_CONNECTED, dynamic_weights_dynamic_bias_weights_and_bias_converted_to_fp16) {
  SubgraphTester tester(6);

  // external input[0]   external input [1]  external input [2]
  //              \       /                  /
  //               \   [constant pad] [constant pad]
  //                \   /                /
  //                [fully connected]
  //                  |
  //                fully connected out [5]
  const uint32_t input_id = 0;
  const uint32_t input2_id = 1;
  const uint32_t input3_id = 2;
  const uint32_t weights_id = 3;
  const uint32_t bias_id = 4;
  const uint32_t fully_connected_out_id = 5;

  std::array<size_t, 4> weights_pre_paddings = {1, 0, 0, 0};
  std::array<size_t, 4> weights_post_paddings = {0, 0, 0, 0};
  std::array<size_t, 4> bias_pre_paddings = {1};
  std::array<size_t, 4> bias_post_paddings = {0};

  tester
      .AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddInputTensorF32({1, 1, 1, 3}, input2_id)
      .AddInputTensorF32({1}, input3_id)
      .AddOutputTensorF32({1, 5, 5, 2}, fully_connected_out_id)
      .AddDynamicTensorF32({2, 1, 1, 3}, weights_id)
      .AddDynamicTensorF32({2}, bias_id)
      .AddConstantPad(weights_pre_paddings.data(), weights_post_paddings.data(), 0.0f, input2_id, weights_id)
      .AddConstantPad(bias_pre_paddings.data(), bias_post_paddings.data(), 0.0f, input3_id, bias_id)
      .AddFullyConnected(input_id, weights_id, bias_id, fully_connected_out_id)
      .Optimize()
      .RewriteForFp16();

  const xnn_value* weights_value = tester.Value(weights_id);
  ASSERT_EQ(weights_value->datatype, xnn_datatype_fp16);
  const xnn_value* bias_value = tester.Value(bias_id);
  ASSERT_EQ(bias_value->datatype, xnn_datatype_fp16);
}

}  // namespace xnnpack
