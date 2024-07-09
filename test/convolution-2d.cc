// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate, std::min.
#include <array>      // For std::array.
#include <cmath>      // For std::lrintf.
#include <cstddef>    // For size_t.
#include <cstdint>    // For uint32_t.
#include <limits>     // For std::numeric_limits.
#include <memory>     // For std::unique_ptr.
#include <random>     // For std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/requantization.h"
#include "xnnpack/subgraph.h"
#include "convolution-test-helpers.h"
#include "replicable_random_device.h"

namespace xnnpack {

template <class InputType, class KernelType = InputType, class BiasType = InputType, class OutputType = InputType> class ConvolutionTestBase : public ::testing::Test {
 protected:
  ConvolutionTestBase() {
    input_size_dist = std::uniform_int_distribution<uint32_t>(10, 15);
    kernel_size_dist = std::uniform_int_distribution<uint32_t>(1, 5);
    subsampling_dist = std::uniform_int_distribution<uint32_t>(1, 5);
    // max value of (dilation * kernel size) must be smaller than input size.
    dilation_dist = std::uniform_int_distribution<uint32_t>(1, 2);
    f32dist = std::uniform_real_distribution<float>(0.1f, 1.0f);
    scale_dist = std::uniform_real_distribution<float>(1.0f, 5.0f);
    i32dist = std::uniform_int_distribution<int32_t>(-10000, 10000);

    batch_size = input_size_dist(rng);
    input_height = input_size_dist(rng);
    input_width = input_size_dist(rng);
    kernel_height = kernel_size_dist(rng);
    kernel_width = kernel_size_dist(rng);
    subsampling_height = subsampling_dist(rng);
    subsampling_width = subsampling_dist(rng);
    dilation_height = dilation_dist(rng);
    dilation_width = dilation_dist(rng);
    groups = input_size_dist(rng);
    group_input_channels = input_size_dist(rng);
    group_output_channels = input_size_dist(rng);
    output_min = -std::numeric_limits<float>::infinity();
    output_max = std::numeric_limits<float>::infinity();
    output_height = xnn_compute_convolution_output_dimension(
        input_height, kernel_height, dilation_height, subsampling_height);
    output_width = xnn_compute_convolution_output_dimension(
        input_width, kernel_width, dilation_width, subsampling_width);

    input_dims = {
        {batch_size, input_height, input_width, groups * group_input_channels}};
    filter_dims = {{groups * group_output_channels, kernel_height, kernel_width,
                    group_input_channels}};
    bias_dims = {{groups * group_output_channels}};
    output_dims = {{batch_size, output_height, output_width,
                    groups * group_output_channels}};

    input = std::vector<InputType>(XNN_EXTRA_BYTES / sizeof(InputType) +
                                   batch_size * input_height * input_width *
                                       groups * group_input_channels);
    filter =
        std::vector<KernelType>(groups * group_output_channels * kernel_height *
                                kernel_width * group_input_channels);
    bias = std::vector<BiasType>(groups * group_output_channels);
    operator_output =
        std::vector<OutputType>(batch_size * output_height * output_width *
                                groups * group_output_channels);
    subgraph_output =
        std::vector<OutputType>(batch_size * output_height * output_width *
                                groups * group_output_channels);
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<uint32_t> input_size_dist;
  std::uniform_int_distribution<uint32_t> kernel_size_dist;
  std::uniform_int_distribution<uint32_t> subsampling_dist;
  std::uniform_int_distribution<uint32_t> dilation_dist;
  std::uniform_int_distribution<int32_t> i32dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;

  const uint32_t input_padding_top = 0;
  const uint32_t input_padding_right = 0;
  const uint32_t input_padding_bottom = 0;
  const uint32_t input_padding_left = 0;
  uint32_t batch_size;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t subsampling_height;
  uint32_t subsampling_width;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t groups;
  uint32_t group_input_channels;
  uint32_t group_output_channels;
  float output_min;
  float output_max;
  uint32_t output_height;
  uint32_t output_width;

  std::array<size_t, 4> input_dims;
  std::array<size_t, 4> filter_dims;
  std::array<size_t, 1> bias_dims;
  std::array<size_t, 4> output_dims;

  std::vector<InputType> input;
  std::vector<KernelType> filter;
  std::vector<BiasType> bias;
  std::vector<OutputType> operator_output;
  std::vector<OutputType> subgraph_output;
};

template <class InputType, class KernelType = InputType, class BiasType = InputType, class OutputType = InputType> class QuantizedConvolutionTestBase : public ConvolutionTestBase<InputType, KernelType, BiasType, OutputType> {
protected:
  QuantizedConvolutionTestBase()
  {
    i8dist =
        std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    w8dist = std::uniform_int_distribution<int32_t>(-std::numeric_limits<KernelType>::max(), std::numeric_limits<KernelType>::max());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    accumulators = std::vector<int32_t>(
      this->batch_size * this->output_height * this->output_width * this->groups * this->group_output_channels);
  }

  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;
  std::uniform_int_distribution<int32_t> w8dist;
  std::vector<int32_t> accumulators;
};

using ConvolutionTestQC8 = QuantizedConvolutionTestBase<int8_t, int8_t, int32_t,int8_t>;
using ConvolutionTestQD8F16QC8W = QuantizedConvolutionTestBase<float, int8_t, float, uint16_t>;
using ConvolutionTestQD8F32QC8W = QuantizedConvolutionTestBase<float, int8_t, float, float>;
using ConvolutionTestQS8 = QuantizedConvolutionTestBase<int8_t, int8_t, int32_t,int8_t>;
using ConvolutionTestQU8 = QuantizedConvolutionTestBase<uint8_t, uint8_t, int32_t,uint8_t>;
using ConvolutionTestF16 = ConvolutionTestBase<uint16_t, float, float>;
using ConvolutionTestF32 = ConvolutionTestBase<float>;

TEST_F(ConvolutionTestQC8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  std::vector<float> scale(groups * group_output_channels, 1.0f);
  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8, scale.data(), filter_dims.size(), 0, filter_dims.data(), filter.data(),
      /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint32, scale.data(), bias_dims.size(), 0, bias_dims.data(), bias.data(),
      /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qc8);
  ASSERT_EQ(node->params.convolution_2d.input_padding_top, input_padding_top);
  ASSERT_EQ(node->params.convolution_2d.input_padding_right, input_padding_right);
  ASSERT_EQ(node->params.convolution_2d.input_padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.convolution_2d.input_padding_left, input_padding_left);
  ASSERT_EQ(node->params.convolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.convolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.convolution_2d.subsampling_height, subsampling_height);
  ASSERT_EQ(node->params.convolution_2d.subsampling_width, subsampling_width);
  ASSERT_EQ(node->params.convolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.convolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.convolution_2d.groups, groups);
  ASSERT_EQ(node->params.convolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.convolution_2d.group_output_channels, group_output_channels);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], filter_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}


TEST_F(ConvolutionTestQD8F16QC8W, define)
{
  std::vector<float> requantization_scales(group_output_channels * groups, 1.0f);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_dynamically_quantized_tensor_value(
                          subgraph, xnn_datatype_qdint8, input_dims.size(), /*num_nonbatch_dims=*/1, input_dims.data(),
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_channelwise_quantized_tensor_value(
                          subgraph, xnn_datatype_qcint8, requantization_scales.data(), filter_dims.size(), /*channel_dim=*/0,
                          filter_dims.data(), filter.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qd8_to_fp16);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConvolutionTestQD8F16QC8W, internally_allocated_dynamic_quantization_parameters)
{
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  std::vector<uint16_t> convert_input(batch_size * input_height * input_width * groups * group_input_channels + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<int8_t> operator_dq_data(batch_size * input_height * input_width * groups * group_input_channels + XNN_EXTRA_BYTES);
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0xDEAD));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0xDEAD));
  std::vector<xnn_dynamic_quantization_params> quantization_params(batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);
  std::vector<float> kernel_scale(group_output_channels * groups);
  std::generate(kernel_scale.begin(), kernel_scale.end(), [&]() { return scale_dist(rng); });
  std::generate(filter.begin(), filter.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(convert_input.begin(), convert_input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t convolution_op = nullptr;
  const size_t quantized_batch_size = input_height * input_width * group_input_channels * groups;
  xnn_status status = xnn_create_convert_nc_f16_qd8(
    /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_convert_nc_f16_qd8(convert_op, batch_size, quantized_batch_size,
                                                               quantized_batch_size, quantized_batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_convert_nc_f16_qd8(convert_op, convert_input.data(),
                                                             operator_dq_data.data(), quantization_params.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels,
    kernel_scale.data(), filter.data(), bias.data(), output_min, output_max,
    /*flags=*/0, nullptr, nullptr, &convolution_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convolution_op);
  ASSERT_EQ( xnn_status_success, xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
                          convolution_op, batch_size, input_height, input_width, &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(convolution_op, /*workspace=*/nullptr, operator_dq_data.data(), operator_output.data(),
                                                      quantization_params.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_dynamically_quantized_tensor_value(
                          subgraph, xnn_datatype_qdint8, input_dims.size(), /*num_nonbatch_dims=*/3, input_dims.data(),
                          XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_channelwise_quantized_tensor_value(
                          subgraph, xnn_datatype_qcint8, kernel_scale.data(), filter_dims.size(), /*channel_dim=*/0,
                          filter_dims.data(), filter.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ( xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_define_convert(subgraph, input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, dq_quantized_id, kernel_id, bias_id, output_id,
      /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, convert_input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(ConvolutionTestQD8F32QC8W, define)
{
  std::vector<float> requantization_scales(group_output_channels * groups, 1.0f);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_dynamically_quantized_tensor_value(
                          subgraph, xnn_datatype_qdint8, input_dims.size(), /*num_nonbatch_dims=*/1, input_dims.data(),
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_channelwise_quantized_tensor_value(
                          subgraph, xnn_datatype_qcint8, requantization_scales.data(), filter_dims.size(), /*channel_dim=*/0,
                          filter_dims.data(), filter.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qd8_to_fp32);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}


TEST_F(ConvolutionTestQD8F32QC8W, internally_allocated_dynamic_quantization_parameters)
{
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  std::vector<float> convert_input(batch_size * input_height * input_width * groups * group_input_channels + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<int8_t> operator_dq_data(batch_size * input_height * input_width * groups * group_input_channels + XNN_EXTRA_BYTES);
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));
  std::vector<xnn_dynamic_quantization_params> quantization_params(batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);
  std::vector<float> kernel_scale(group_output_channels * groups);
  std::generate(kernel_scale.begin(), kernel_scale.end(), [&]() { return scale_dist(rng); });
  std::generate(filter.begin(), filter.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(convert_input.begin(), convert_input.end(), [&]() { return f32dist(rng); });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t convolution_op = nullptr;
  const size_t quantized_batch_size = input_height * input_width * group_input_channels * groups;
  xnn_status status = xnn_create_convert_nc_f32_qd8(
    /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_convert_nc_f32_qd8(convert_op, batch_size, quantized_batch_size,
                                                               quantized_batch_size, quantized_batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_convert_nc_f32_qd8(convert_op, convert_input.data(),
                                                             operator_dq_data.data(), quantization_params.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels,
    kernel_scale.data(), filter.data(), bias.data(), output_min, output_max,
    /*flags=*/0, nullptr, nullptr, &convolution_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convolution_op);
  ASSERT_EQ( xnn_status_success, xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
                          convolution_op, batch_size, input_height, input_width,
                          &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(convolution_op, /*workspace=*/nullptr, operator_dq_data.data(), operator_output.data(),
                                                      quantization_params.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_dynamically_quantized_tensor_value(
                          subgraph, xnn_datatype_qdint8, input_dims.size(), /*num_nonbatch_dims=*/3, input_dims.data(),
                          XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_channelwise_quantized_tensor_value(
                          subgraph, xnn_datatype_qcint8, kernel_scale.data(), filter_dims.size(), /*channel_dim=*/0,
                          filter_dims.data(), filter.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ( xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_define_convert(subgraph, input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, dq_quantized_id, kernel_id, bias_id, output_id,
      /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, convert_input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(ConvolutionTestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, filter_dims.size(), filter_dims.data(), filter.data(),
                          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint32, 0, 1.0f, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.convolution_2d.input_padding_top, input_padding_top);
  ASSERT_EQ(node->params.convolution_2d.input_padding_right, input_padding_right);
  ASSERT_EQ(node->params.convolution_2d.input_padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.convolution_2d.input_padding_left, input_padding_left);
  ASSERT_EQ(node->params.convolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.convolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.convolution_2d.subsampling_height, subsampling_height);
  ASSERT_EQ(node->params.convolution_2d.subsampling_width, subsampling_width);
  ASSERT_EQ(node->params.convolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.convolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.convolution_2d.groups, groups);
  ASSERT_EQ(node->params.convolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.convolution_2d.group_output_channels, group_output_channels);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], filter_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConvolutionTestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, 1.0f, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, 1.0f, filter_dims.size(), filter_dims.data(), filter.data(),
                          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint32, 0, 1.0f, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, 1.0f, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->params.convolution_2d.input_padding_top, input_padding_top);
  ASSERT_EQ(node->params.convolution_2d.input_padding_right, input_padding_right);
  ASSERT_EQ(node->params.convolution_2d.input_padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.convolution_2d.input_padding_left, input_padding_left);
  ASSERT_EQ(node->params.convolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.convolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.convolution_2d.subsampling_height, subsampling_height);
  ASSERT_EQ(node->params.convolution_2d.subsampling_width, subsampling_width);
  ASSERT_EQ(node->params.convolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.convolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.convolution_2d.groups, groups);
  ASSERT_EQ(node->params.convolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.convolution_2d.group_output_channels, group_output_channels);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], filter_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConvolutionTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, filter_dims.size(), filter_dims.data(), filter.data(), /*external_id=*/1,
      /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->params.convolution_2d.input_padding_top, input_padding_top);
  ASSERT_EQ(node->params.convolution_2d.input_padding_right, input_padding_right);
  ASSERT_EQ(node->params.convolution_2d.input_padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.convolution_2d.input_padding_left, input_padding_left);
  ASSERT_EQ(node->params.convolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.convolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.convolution_2d.subsampling_height, subsampling_height);
  ASSERT_EQ(node->params.convolution_2d.subsampling_width, subsampling_width);
  ASSERT_EQ(node->params.convolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.convolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.convolution_2d.groups, groups);
  ASSERT_EQ(node->params.convolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.convolution_2d.group_output_channels, group_output_channels);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], filter_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConvolutionTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, filter_dims.size(), filter_dims.data(), filter.data(), /*external_id=*/1,
      /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.convolution_2d.input_padding_top, input_padding_top);
  ASSERT_EQ(node->params.convolution_2d.input_padding_right, input_padding_right);
  ASSERT_EQ(node->params.convolution_2d.input_padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.convolution_2d.input_padding_left, input_padding_left);
  ASSERT_EQ(node->params.convolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.convolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.convolution_2d.subsampling_height, subsampling_height);
  ASSERT_EQ(node->params.convolution_2d.subsampling_width, subsampling_width);
  ASSERT_EQ(node->params.convolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.convolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.convolution_2d.groups, groups);
  ASSERT_EQ(node->params.convolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.convolution_2d.group_output_channels, group_output_channels);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], filter_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConvolutionTestQC8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::generate(filter.begin(), filter.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));
  std::vector<float> requantization_scales(groups * group_output_channels);
  const int8_t input_zero_point = i8dist(rng);
  const int8_t output_zero_point = i8dist(rng);
  const float input_scale = scale_dist(rng);
  const float output_scale = scale_dist(rng);
  const int8_t quantized_output_min = xnn_qs8_quantize(output_min, output_scale, output_zero_point);
  const int8_t quantized_output_max = xnn_qs8_quantize(output_max, output_scale, output_zero_point);

  compute_convolution_qs8_reference_results(
      batch_size,
      output_height,
      output_width,
      input_height,
      input_width,
      input_padding_top,
      input_padding_right,
      input_padding_bottom,
      input_padding_left,
      kernel_height,
      kernel_width,
      subsampling_height,
      subsampling_width,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      input_zero_point,
      input,
      filter,
      accumulators,
      /*has_bias=*/true,
      bias);

  // Compute renormalization parameters.
  for (size_t c = 0; c < groups * group_output_channels; c++) {
    int32_t accumulated_min = accumulators[c];
    int32_t accumulated_max = accumulators[c];
    for (size_t px = 0; px < batch_size * output_height * output_width; px++) {
      accumulated_min = std::min(accumulated_min, accumulators[px * groups * group_output_channels + c]);
      accumulated_max = std::max(accumulated_max, accumulators[px * groups * group_output_channels + c]);
    }

    float requantization_scale = 0x1.0p-32f;
    if (accumulated_max != 0) {
      requantization_scale = std::max(
        requantization_scale,
        float(int32_t(std::numeric_limits<int8_t>::max()) - int32_t(output_zero_point)) / float(accumulated_max));
    }
    if (accumulated_min != 0) {
      requantization_scale = std::max(
        requantization_scale,
        float(int32_t(std::numeric_limits<int8_t>::min()) - int32_t(output_zero_point)) / float(accumulated_min));
    }
    requantization_scale = std::min(requantization_scale, 0x1.FFFFFEp-1f);

    requantization_scales[c] = requantization_scale;
  }

  // Call operator API.
  const xnn_status status = xnn_create_convolution2d_nhwc_qs8_qc8w(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels, input_zero_point, input_scale,
    requantization_scales.data(), filter.data(), bias.data(), output_zero_point, output_scale, quantized_output_min,
    quantized_output_max, /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
   ASSERT_EQ(
    xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8_qc8w(
                          op, batch_size, input_height, input_width,
                          &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8_qc8w(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, input_zero_point, input_scale, input_dims.size(),
                          input_dims.data(), nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_channelwise_quantized_tensor_value(
                          subgraph, xnn_datatype_qcint8, requantization_scales.data(), filter_dims.size(), 0,
                          filter_dims.data(), filter.data(), /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_channelwise_quantized_tensor_value(
                          subgraph, xnn_datatype_qcint32, requantization_scales.data(), bias_dims.size(), 0,
                          bias_dims.data(), bias.data(), /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(ConvolutionTestQS8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::generate(filter.begin(), filter.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));
  const int8_t input_zero_point = -1;
  const float input_scale = scale_dist(rng);
  const float kernel_scale = scale_dist(rng);

  compute_convolution_qs8_reference_results(
      batch_size,
      output_height,
      output_width,
      input_height,
      input_width,
      input_padding_top,
      input_padding_right,
      input_padding_bottom,
      input_padding_left,
      kernel_height,
      kernel_width,
      subsampling_height,
      subsampling_width,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      input_zero_point,
      input,
      filter,
      accumulators,
      /*has_bias=*/true,
      bias);

  // Compute renormalization parameters.
  const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
  const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

  float output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
  int8_t output_zero_point = int8_t(std::max(
    std::min(
      lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
      long(std::numeric_limits<int8_t>::max())),
    long(std::numeric_limits<int8_t>::min())));
  const int8_t quantized_output_min = xnn_qs8_quantize(output_min, output_scale, output_zero_point);
  const int8_t quantized_output_max = xnn_qs8_quantize(output_max, output_scale, output_zero_point);

  // Call operator API.
  const xnn_status status = xnn_create_convolution2d_nhwc_qs8(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels, input_zero_point, input_scale,
    kernel_scale, filter.data(), bias.data(), output_zero_point, output_scale, quantized_output_min,
    quantized_output_max, /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8(
                          op, batch_size, input_height, input_width,
                          &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, input_zero_point, input_scale, input_dims.size(),
                          input_dims.data(), nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, kernel_scale, filter_dims.size(), filter_dims.data(),
                          filter.data(), /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint32, 0, kernel_scale, bias_dims.size(), bias_dims.data(),
                          bias.data(), /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(ConvolutionTestQU8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::generate(filter.begin(), filter.end(), [&]() { return u8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));
  const uint8_t input_zero_point = u8dist(rng);
  const uint8_t kernel_zero_point = 0;
  const float input_scale = scale_dist(rng);
  const float kernel_scale = scale_dist(rng);

  // Compute reference results, without renormalization.
  compute_convolution_qu8_reference_results(
      batch_size,
      output_height,
      output_width,
      input_height,
      input_width,
      input_padding_top,
      input_padding_right,
      input_padding_bottom,
      input_padding_left,
      kernel_height,
      kernel_width,
      subsampling_height,
      subsampling_width,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      input_zero_point,
      kernel_zero_point,
      input,
      filter,
      accumulators,
      /*has_bias=*/true,
      bias);

  // Compute renormalization parameters.
  const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
  const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

  const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
  const uint8_t output_zero_point = uint8_t(std::max(
    std::min(
      lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
      long(std::numeric_limits<uint8_t>::max())),
    long(std::numeric_limits<uint8_t>::min())));
  const uint8_t quantized_output_min = xnn_qu8_quantize(output_min, output_scale, output_zero_point);
  const uint8_t quantized_output_max = xnn_qu8_quantize(output_max, output_scale, output_zero_point);

  // Call operator API.
  const xnn_status status = xnn_create_convolution2d_nhwc_qu8(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels, input_zero_point, input_scale,
    kernel_zero_point, kernel_scale, filter.data(), bias.data(), output_zero_point, output_scale, quantized_output_min,
    quantized_output_max, /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_convolution2d_nhwc_qu8(
                          op, batch_size, input_height, input_width,
                          &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qu8(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, input_zero_point, input_scale, input_dims.size(),
                          input_dims.data(), nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, kernel_scale, filter_dims.size(), filter_dims.data(),
                          filter.data(), /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint32, 0, kernel_scale, bias_dims.size(), bias_dims.data(),
                          bias.data(), /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(ConvolutionTestF16, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(filter.begin(), filter.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  // Call operator API.
  const xnn_status status = xnn_create_convolution2d_nhwc_f16(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels, filter.data(), bias.data(),
    output_min, output_max,
    XNN_FLAG_FP32_STATIC_WEIGHTS, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_convolution2d_nhwc_f16(
                          op, batch_size, input_height, input_width,
                          &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f16(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, filter_dims.size(), filter_dims.data(), filter.data(),
                          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(ConvolutionTestF32, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(filter.begin(), filter.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_convolution2d_nhwc_f32(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels, filter.data(), bias.data(),
    output_min, output_max,
    /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_convolution2d_nhwc_f32(
                          op, batch_size, input_height, input_width,
                          &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, filter_dims.size(), filter_dims.data(), filter.data(),
                          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(ConvolutionTestF32, transient_indirection_buffer)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(filter.begin(), filter.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_convolution2d_nhwc_f32(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height, kernel_width,
    subsampling_height, subsampling_width, dilation_height, dilation_width, groups, group_input_channels,
    group_output_channels, groups * group_input_channels, groups * group_output_channels, filter.data(), bias.data(),
    output_min, output_max,
    /*flags=*/XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_convolution2d_nhwc_f32(
                          op, batch_size, input_height, input_width,
                          &workspace_size, &workspace_alignment,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  // workspace_size might be 0 if we hit the vmulcaddc path which does not require any indirection buffers.
  ASSERT_NE(workspace_size, SIZE_MAX);
  ASSERT_NE(workspace_alignment, SIZE_MAX);
  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(op, workspace.data(), input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, filter_dims.size(), filter_dims.data(), filter.data(),
                          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(ConvolutionTestF32, reshape_output)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t filter_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, filter_dims.size(), filter_dims.data(), filter.data(),
                          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, filter_id, bias_id, output_id,
      /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  input_dims[0] += 2;
  input_dims[1] += 3;
  input_dims[2] += 4;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  ASSERT_EQ(output_shape->dim[0], input_dims[0]);
  ASSERT_EQ(output_shape->dim[1], runtime->opdata[0].operator_objects[0]->output_height);
  ASSERT_EQ(output_shape->dim[2], runtime->opdata[0].operator_objects[0]->output_width);
  ASSERT_EQ(output_shape->dim[3], output_dims[3]);

  input_dims[0] -= 1;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
  ASSERT_EQ(output_shape->dim[0], input_dims[0]);
  ASSERT_EQ(output_shape->dim[1], runtime->opdata[0].operator_objects[0]->output_height);
  ASSERT_EQ(output_shape->dim[2], runtime->opdata[0].operator_objects[0]->output_width);
  ASSERT_EQ(output_shape->dim[3], output_dims[3]);

}
}  // namespace xnnpack
