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


enum xnn_status xnn_define_deconvolution_2d(
  xnn_subgraph_t subgraph,
  uint32_t padding_top,
  uint32_t padding_right,
  uint32_t padding_bottom,
  uint32_t padding_left,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t upsampling_height,
  uint32_t upsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
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
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d));
    return xnn_status_uninitialized;
  }

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), kernel_width, kernel_height);
    return xnn_status_invalid_parameter;
  }

  if (upsampling_width == 0 || upsampling_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " upsampling: upsampling dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), upsampling_width, upsampling_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 " groups: number of groups must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), groups);
    return xnn_status_invalid_parameter;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu input channels per group: number of channels must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), group_input_channels);
    return xnn_status_invalid_parameter;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu output channels per group: number of channels must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), group_output_channels);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* filter_value = &subgraph->values[filter_id];
  if (filter_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id, filter_value->type);
    return xnn_status_invalid_parameter;
  }

  if (filter_value->data == NULL) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": non-static Value",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id);
    return xnn_status_invalid_parameter;
  }

  switch (filter_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id,
        xnn_datatype_to_string(filter_value->datatype), filter_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (bias_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with bias ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* bias_value = &subgraph->values[bias_id];
  if (bias_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id, bias_value->type);
    return xnn_status_invalid_parameter;
  }

  if (bias_value->data == NULL) {
    xnn_log_error(
      "failed to define %s operator with bias ID #%" PRIu32 ": non-static Value",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id);
    return xnn_status_invalid_parameter;
  }

  switch (bias_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id,
        xnn_datatype_to_string(bias_value->datatype), bias_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_deconvolution_2d;
  node->params.deconvolution_2d.padding_top = padding_top;
  node->params.deconvolution_2d.padding_right = padding_right;
  node->params.deconvolution_2d.padding_bottom = padding_bottom;
  node->params.deconvolution_2d.padding_left = padding_left;
  node->params.deconvolution_2d.kernel_height = kernel_height;
  node->params.deconvolution_2d.kernel_width = kernel_width;
  node->params.deconvolution_2d.upsampling_height = upsampling_height;
  node->params.deconvolution_2d.upsampling_width = upsampling_width;
  node->params.deconvolution_2d.dilation_height = dilation_height;
  node->params.deconvolution_2d.dilation_width = dilation_width;
  node->params.deconvolution_2d.adjustment_height = adjustment_height;
  node->params.deconvolution_2d.adjustment_width = adjustment_width;
  node->params.deconvolution_2d.groups = groups;
  node->params.deconvolution_2d.group_input_channels = group_input_channels;
  node->params.deconvolution_2d.group_output_channels = group_output_channels;
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
