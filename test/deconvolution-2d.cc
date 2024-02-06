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
#include <random>     // For std::random_device, std::mt19937, std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <fp16/fp16.h>
#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/requantization.h>
#include <xnnpack/subgraph.h>

template <class T, class KernelType = T, class BiasType = T> class DeconvolutionTestBase : public ::testing::Test {
protected:
  DeconvolutionTestBase()
  {
    random_device = std::make_unique<std::random_device>();
    rng = std::mt19937((*random_device)());
    input_size_dist = std::uniform_int_distribution<uint32_t>(10, 15);
    kernel_size_dist = std::uniform_int_distribution<uint32_t>(1, 5);
    stride_dist = std::uniform_int_distribution<uint32_t>(1, 3);
    f32dist = std::uniform_real_distribution<float>(0.1f, 1.0f);
    scale_dist = std::uniform_real_distribution<float>(1.0f, 5.0f);
    i32dist = std::uniform_int_distribution<int32_t>(-10000, 10000);

    batch_size = input_size_dist(rng);
    input_height = input_size_dist(rng);
    input_width = input_size_dist(rng);
    kernel_height = kernel_size_dist(rng);
    kernel_width = kernel_size_dist(rng);
    upsampling_height = stride_dist(rng);
    upsampling_width = stride_dist(rng);
    dilation_height = stride_dist(rng);
    dilation_width = stride_dist(rng);
    groups = input_size_dist(rng);
    group_input_channels = input_size_dist(rng);
    group_output_channels = input_size_dist(rng);
    output_min = -std::numeric_limits<float>::infinity();
    output_max = std::numeric_limits<float>::infinity();
    adjustment_height = 0;
    adjustment_width = 0;
    output_height = xnn_compute_deconvolution_output_dimension(
      input_height, padding_top + padding_bottom, adjustment_height, kernel_height, dilation_height, upsampling_height);
    output_width = xnn_compute_deconvolution_output_dimension(
      input_width, padding_left + padding_right, adjustment_width, kernel_width, dilation_width, upsampling_width);

    input_dims = {{batch_size, input_height, input_width, group_input_channels}};
    kernel_dims = {{groups * group_output_channels, kernel_height, kernel_width, group_input_channels}};
    bias_dims = {{groups * group_output_channels}};
    output_dims = {{batch_size, output_height, output_width, groups * group_output_channels}};

    input = std::vector<T>(
      XNN_EXTRA_BYTES / sizeof(T) + batch_size * input_height * input_width * groups * group_input_channels);
    kernel = std::vector<KernelType>(groups * group_output_channels * kernel_height * kernel_width * group_input_channels);
    bias = std::vector<BiasType>(groups * group_output_channels);
    operator_output = std::vector<T>(batch_size * output_height * output_width * groups * group_output_channels);
    subgraph_output = std::vector<T>(batch_size * output_height * output_width * groups * group_output_channels);
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_int_distribution<uint32_t> input_size_dist;
  std::uniform_int_distribution<uint32_t> kernel_size_dist;
  std::uniform_int_distribution<uint32_t> stride_dist;
  std::uniform_int_distribution<int32_t> i32dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;

  const uint32_t padding_top = 0;
  const uint32_t padding_right = 0;
  const uint32_t padding_bottom = 0;
  const uint32_t padding_left = 0;
  uint32_t batch_size;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t upsampling_height;
  uint32_t upsampling_width;
  uint32_t adjustment_height;
  uint32_t adjustment_width;
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
  std::array<size_t, 4> kernel_dims;
  std::array<size_t, 1> bias_dims;
  std::array<size_t, 4> output_dims;

  std::vector<T> input;
  std::vector<KernelType> kernel;
  std::vector<BiasType> bias;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

template <class T> class QuantizedDeconvolutionTestBase : public DeconvolutionTestBase<T, T, int32_t> {
protected:
  QuantizedDeconvolutionTestBase()
  {
    i8dist = std::uniform_int_distribution<int32_t>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    w8dist = std::uniform_int_distribution<int32_t>(-std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    accumulators = std::vector<int32_t>(
      this->batch_size * this->output_height * this->output_width * this->groups * this->group_output_channels);
  }

  void initialize_accumulators_from_bias()
  {
    for (size_t i = 0; i < this->batch_size; i++) {
      for (size_t oy = 0; oy < this->output_height; oy++) {
        for (size_t ox = 0; ox < this->output_width; ox++) {
          for (size_t g = 0; g < this->groups; g++) {
            for (size_t oc = 0; oc < this->group_output_channels; oc++) {
              accumulators
                [(((i * this->output_height + oy) * this->output_width + ox) * this->groups + g) *
                   this->group_output_channels +
                 oc] = this->bias[g * this->group_output_channels + oc];
            }
          }
        }
      }
    }
  }

  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;
  std::uniform_int_distribution<int32_t> w8dist;
  std::vector<int32_t> accumulators;
};

using DeconvolutionTestQS8 = QuantizedDeconvolutionTestBase<int8_t>;
using DeconvolutionTestQU8 = QuantizedDeconvolutionTestBase<uint8_t>;
using DeconvolutionTestF16 = DeconvolutionTestBase<uint16_t, float, float>;
using DeconvolutionTestF32 = DeconvolutionTestBase<float>;

TEST_F(DeconvolutionTestQS8, define)
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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, kernel_dims.size(), kernel_dims.data(), kernel.data(),
                          /*external_id=*/1, /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_deconvolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.deconvolution_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.deconvolution_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.deconvolution_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.deconvolution_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_height, upsampling_height);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_width, upsampling_width);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_height, adjustment_height);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_width, adjustment_width);
  ASSERT_EQ(node->params.deconvolution_2d.groups, groups);
  ASSERT_EQ(node->params.deconvolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.deconvolution_2d.group_output_channels, group_output_channels);
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

TEST_F(DeconvolutionTestQU8, define)
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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, 1.0f, kernel_dims.size(), kernel_dims.data(), kernel.data(),
                          /*external_id=*/1, /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_deconvolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->params.deconvolution_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.deconvolution_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.deconvolution_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.deconvolution_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_height, upsampling_height);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_width, upsampling_width);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_height, adjustment_height);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_width, adjustment_width);
  ASSERT_EQ(node->params.deconvolution_2d.groups, groups);
  ASSERT_EQ(node->params.deconvolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.deconvolution_2d.group_output_channels, group_output_channels);
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

TEST_F(DeconvolutionTestF16, define)
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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, kernel_dims.size(), kernel_dims.data(), kernel.data(), /*external_id=*/1,
      /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_deconvolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->params.deconvolution_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.deconvolution_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.deconvolution_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.deconvolution_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_height, upsampling_height);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_width, upsampling_width);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_height, adjustment_height);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_width, adjustment_width);
  ASSERT_EQ(node->params.deconvolution_2d.groups, groups);
  ASSERT_EQ(node->params.deconvolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.deconvolution_2d.group_output_channels, group_output_channels);
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

TEST_F(DeconvolutionTestF32, define)
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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, kernel_dims.size(), kernel_dims.data(), kernel.data(), /*external_id=*/1,
      /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_deconvolution_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.deconvolution_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.deconvolution_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.deconvolution_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.deconvolution_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_height, kernel_height);
  ASSERT_EQ(node->params.deconvolution_2d.kernel_width, kernel_width);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_height, upsampling_height);
  ASSERT_EQ(node->params.deconvolution_2d.upsampling_width, upsampling_width);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.deconvolution_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_height, adjustment_height);
  ASSERT_EQ(node->params.deconvolution_2d.adjustment_width, adjustment_width);
  ASSERT_EQ(node->params.deconvolution_2d.groups, groups);
  ASSERT_EQ(node->params.deconvolution_2d.group_input_channels, group_input_channels);
  ASSERT_EQ(node->params.deconvolution_2d.group_output_channels, group_output_channels);
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

TEST_F(DeconvolutionTestQS8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));
  const int8_t input_zero_point = 1;
  const float input_scale = scale_dist(rng);
  const float kernel_scale = scale_dist(rng);

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t y = oy + padding_top - ky * dilation_height;
          const size_t iy = y / upsampling_height;
          if (iy * upsampling_height == y && iy < input_height) {
            for (size_t kx = 0; kx < kernel_width; kx++) {
              const size_t x = ox + padding_left - kx * dilation_width;
              const size_t ix = x / upsampling_width;
              if (ix * upsampling_width == x && ix < input_width) {
                for (size_t g = 0; g < groups; g++) {
                  for (size_t oc = 0; oc < group_output_channels; oc++) {
                    for (size_t ic = 0; ic < group_input_channels; ic++) {
                      accumulators
                        [(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] +=
                        (int32_t(input[((i * input_height + iy) * input_width + ix) * g * group_input_channels + ic]) -
                         int32_t(input_zero_point)) *
                        int32_t(kernel
                                  [(((g * group_output_channels + oc) * kernel_height + ky) * kernel_width + kx) *
                                     group_input_channels +
                                   ic]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

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
  const xnn_status status = xnn_create_deconvolution2d_nhwc_qs8(
    padding_top, padding_right, padding_bottom, padding_left, kernel_height, kernel_width, upsampling_height,
    upsampling_width, dilation_height, dilation_width, groups, group_input_channels, group_output_channels,
    groups * group_input_channels, groups * group_output_channels, input_zero_point, input_scale, kernel_scale,
    kernel.data(), bias.data(), output_zero_point, output_scale, quantized_output_min, quantized_output_max,
    /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_deconvolution2d_nhwc_qs8(
                          op, batch_size, input_height, input_width, adjustment_height, adjustment_width,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(
    xnn_status_success, xnn_setup_deconvolution2d_nhwc_qs8(
                          op, input.data(), operator_output.data()));

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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, kernel_scale, kernel_dims.size(), kernel_dims.data(),
                          kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
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

TEST_F(DeconvolutionTestQU8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));
  const uint8_t input_zero_point = u8dist(rng);
  const uint8_t kernel_zero_point = 0;
  const float input_scale = scale_dist(rng);
  const float kernel_scale = scale_dist(rng);

  // Compute reference results, without renormalization.
  initialize_accumulators_from_bias();
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t y = oy + padding_top - ky * dilation_height;
          const size_t iy = y / upsampling_height;
          if (iy * upsampling_height == y && iy < input_height) {
            for (size_t kx = 0; kx < kernel_width; kx++) {
              const size_t x = ox + padding_left - kx * dilation_width;
              const size_t ix = x / upsampling_width;
              if (ix * upsampling_width == x && ix < input_width) {
                for (size_t g = 0; g < groups; g++) {
                  for (size_t oc = 0; oc < group_output_channels; oc++) {
                    for (size_t ic = 0; ic < group_input_channels; ic++) {
                      accumulators
                        [(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] +=
                        (int32_t(input[((i * input_height + iy) * input_width + ix) * g * group_input_channels + ic]) -
                         int32_t(input_zero_point)) *
                        (int32_t(kernel
                                   [(((g * group_output_channels + oc) * kernel_height + ky) * kernel_width + kx) *
                                      group_input_channels +
                                    ic]) -
                         int32_t(kernel_zero_point));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

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
  const xnn_status status = xnn_create_deconvolution2d_nhwc_qu8(
    padding_top, padding_right, padding_bottom, padding_left, kernel_height, kernel_width, upsampling_height,
    upsampling_width, dilation_height, dilation_width, groups, group_input_channels, group_output_channels,
    groups * group_input_channels, groups * group_output_channels, input_zero_point, input_scale, kernel_zero_point,
    kernel_scale, kernel.data(), bias.data(), output_zero_point, output_scale, quantized_output_min,
    quantized_output_max, /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_deconvolution2d_nhwc_qu8(
                          op, batch_size, input_height, input_width, adjustment_height, adjustment_width,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(
    xnn_status_success, xnn_setup_deconvolution2d_nhwc_qu8(
                          op, input.data(), operator_output.data()));

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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, kernel_scale, kernel_dims.size(), kernel_dims.data(),
                          kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
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

TEST_F(DeconvolutionTestF16, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), fp16_ieee_from_fp32_value(nanf("")));
  std::fill(subgraph_output.begin(), subgraph_output.end(), fp16_ieee_from_fp32_value(nanf("")));

  // Call operator API.
  const xnn_status status = xnn_create_deconvolution2d_nhwc_f16(
    padding_top, padding_right, padding_bottom, padding_left, kernel_height, kernel_width, upsampling_height,
    upsampling_width, dilation_height, dilation_width, groups, group_input_channels, group_output_channels,
    groups * group_input_channels, groups * group_output_channels, kernel.data(), bias.data(), output_min, output_max,
    XNN_FLAG_FP32_STATIC_WEIGHTS, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_deconvolution2d_nhwc_f16(
                          op, batch_size, input_height, input_width, adjustment_height, adjustment_width,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(
    xnn_status_success, xnn_setup_deconvolution2d_nhwc_f16(
                          op, input.data(), operator_output.data()));

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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, kernel_dims.size(), kernel_dims.data(), kernel.data(),
                          /*external_id=*/1, /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
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

TEST_F(DeconvolutionTestF32, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_deconvolution2d_nhwc_f32(
    padding_top, padding_right, padding_bottom, padding_left, kernel_height, kernel_width, upsampling_height,
    upsampling_width, dilation_height, dilation_width, groups, group_input_channels, group_output_channels,
    groups * group_input_channels, groups * group_output_channels, kernel.data(), bias.data(), output_min, output_max,
    /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_deconvolution2d_nhwc_f32(
                          op, batch_size, input_height, input_width, adjustment_height, adjustment_width,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(
    xnn_status_success, xnn_setup_deconvolution2d_nhwc_f32(
                          op, input.data(), operator_output.data()));

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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, kernel_dims.size(), kernel_dims.data(), kernel.data(),
                          /*external_id=*/1, /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
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

TEST_F(DeconvolutionTestF32, reshape_output)
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

  uint32_t kernel_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, kernel_dims.size(), kernel_dims.data(), kernel.data(),
                          /*external_id=*/1, /*flags=*/0, &kernel_id));

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
    xnn_define_deconvolution_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, adjustment_height, adjustment_width,
      kernel_height, kernel_width, upsampling_height, upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels, output_min, output_max, input_id, kernel_id, bias_id, output_id,
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
