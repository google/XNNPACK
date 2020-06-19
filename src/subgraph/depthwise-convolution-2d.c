// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_define_depthwise_convolution_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t depth_multiplier,
  size_t input_channels,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d));
    return xnn_status_uninitialized;
  }

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), kernel_width, kernel_height);
    return xnn_status_invalid_parameter;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " subsampling: subsampling dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), subsampling_width, subsampling_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (depth_multiplier == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 " depth multiplier: depth multiplier must be non-zero",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), depth_multiplier);
    return xnn_status_invalid_parameter;
  }

  if (input_channels == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu input channels: number of channels must be non-zero",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), input_channels);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const uint32_t supported_flags = XNN_FLAG_TENSORFLOW_SAME_PADDING;
  const uint32_t invalid_flags = flags & ~supported_flags;
  if (invalid_flags != 0) {
    xnn_log_error(
      "failed to define %s operator with 0x%08" PRIx32 " flags: invalid flags 0x%08" PRIx32,
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), flags, invalid_flags);
    return xnn_status_invalid_parameter;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && any_padding) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
      "TensorFlow SAME padding can't be combined with explicit padding specification",
      xnn_node_type_to_string(xnn_node_type_convolution_2d),
      input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
    return xnn_status_invalid_parameter;
  }

  // Convert TensorFlow SAME padding to explicit padding specification whenever possible
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && (subsampling_height | subsampling_width) == 1) {
    flags &= ~XNN_FLAG_TENSORFLOW_SAME_PADDING;
    const uint32_t padding_height = (kernel_height - 1) * dilation_height;
    const uint32_t padding_width = (kernel_width - 1) * dilation_width;
    input_padding_left = padding_width / 2;
    input_padding_top = padding_height / 2;
    input_padding_right = padding_width - input_padding_left;
    input_padding_bottom = padding_height - input_padding_top;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), input_id);
    return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), filter_id);
    return xnn_status_invalid_parameter;
  }

  if (bias_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with bias ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), bias_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_depthwise_convolution_2d), output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_depthwise_convolution_2d;
  node->params.depthwise_convolution_2d.input_padding_top = input_padding_top;
  node->params.depthwise_convolution_2d.input_padding_right = input_padding_right;
  node->params.depthwise_convolution_2d.input_padding_bottom = input_padding_bottom;
  node->params.depthwise_convolution_2d.input_padding_left = input_padding_left;
  node->params.depthwise_convolution_2d.kernel_height = kernel_height;
  node->params.depthwise_convolution_2d.kernel_width = kernel_width;
  node->params.depthwise_convolution_2d.subsampling_height = subsampling_height;
  node->params.depthwise_convolution_2d.subsampling_width = subsampling_width;
  node->params.depthwise_convolution_2d.dilation_height = dilation_height;
  node->params.depthwise_convolution_2d.dilation_width = dilation_width;
  node->params.depthwise_convolution_2d.depth_multiplier = depth_multiplier;
  node->params.depthwise_convolution_2d.input_channels = input_channels;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 3;
  node->inputs[0] = input_id;
  node->inputs[1] = filter_id;
  node->inputs[2] = bias_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
};
