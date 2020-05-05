// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_create_subgraph(
    uint32_t external_value_ids,
    uint32_t flags,
    xnn_subgraph_t* subgraph_out)
{
  struct xnn_subgraph* subgraph = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create subgraph: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_out_of_memory;

  subgraph = xnn_allocate_zero_memory(sizeof(struct xnn_subgraph));
  if (subgraph == NULL) {
    xnn_log_error("failed to allocate %zu bytes for subgraph descriptor", sizeof(struct xnn_subgraph));
    goto error;
  }

  subgraph->external_value_ids = external_value_ids;

  subgraph->values = xnn_allocate_zero_memory(external_value_ids * sizeof(struct xnn_value));
  if (subgraph->values == NULL) {
    xnn_log_error("failed to allocate %zu bytes for subgraph values", external_value_ids * sizeof(struct xnn_value));
    goto error;
  }
  for (size_t i = 0; i < external_value_ids; i++) {
    subgraph->values[i].id = i;
  }
  subgraph->num_values = external_value_ids;
  subgraph->num_reserved_values = external_value_ids;

  *subgraph_out = subgraph;
  return xnn_status_success;

error:
  xnn_delete_subgraph(subgraph);
  return status;
}


struct xnn_value* xnn_subgraph_new_internal_value(xnn_subgraph_t subgraph)
{
  struct xnn_value* values = subgraph->values;
  const size_t size = subgraph->num_values;
  const size_t capacity = subgraph->num_reserved_values;
  if (capacity < size + 1) {
    const size_t new_capacity = max(min(capacity * 2, capacity + 512), capacity + 64);
    assert(new_capacity >= size + 1);
    values = xnn_reallocate_memory(values, new_capacity * sizeof(struct xnn_value));
    if (values == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subgraph values",
        capacity * sizeof(struct xnn_value));
      return values;
    }

    memset(values + size, 0, (new_capacity - size) * sizeof(struct xnn_value));
    subgraph->num_reserved_values = new_capacity;
    subgraph->values = values;
  }
  subgraph->num_values = size + 1;
  struct xnn_value* new_value = values + size;
  new_value->id = size;
  return new_value;
}

struct xnn_node* xnn_subgraph_new_node(xnn_subgraph_t subgraph)
{
  struct xnn_node* nodes = subgraph->nodes;
  const size_t size = subgraph->num_nodes;
  const size_t capacity = subgraph->num_reserved_nodes;

  if (capacity < size + 1) {
    const size_t new_capacity = max(min(capacity * 2, capacity + 512), capacity + 64);
    assert(new_capacity >= size + 1);
    nodes = xnn_reallocate_memory(nodes, new_capacity * sizeof(struct xnn_node));
    if (nodes == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subgraph nodes",
        capacity * sizeof(struct xnn_node));
      return nodes;
    }

    memset(nodes + size, 0, (new_capacity - size) * sizeof(struct xnn_node));
    subgraph->num_reserved_nodes = new_capacity;
    subgraph->nodes = nodes;
  }
  subgraph->num_nodes = size + 1;
  struct xnn_node* new_node = nodes + size;
  new_node->id = size;
  return new_node;
}

enum xnn_status xnn_define_convolution_2d(
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
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Convolution operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to define Convolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    return xnn_status_invalid_parameter;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to define Convolution operator with %" PRIu32 "x%" PRIu32 " subsampling: "
      "subsampling dimensions must be non-zero",
      subsampling_width, subsampling_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to define Convolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to define Convolution operator with %" PRIu32 " groups: number of groups must be non-zero", groups);
    return xnn_status_invalid_parameter;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to define Convolution operator with %zu input channels per group: "
      "number of channels must be non-zero",
      group_input_channels);
    return xnn_status_invalid_parameter;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to define Convolution operator with %zu output channels per group: "
      "number of channels must be non-zero",
      group_output_channels);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Convolution operator with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Convolution operator with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Convolution operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Convolution operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Convolution operator with filter ID #%" PRIu32 ": invalid Value ID",
      filter_id);
    return xnn_status_invalid_parameter;
  }

  if (bias_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Convolution operator with bias ID #%" PRIu32 ": invalid Value ID",
      bias_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Convolution operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_convolution_2d;
  node->params.convolution_2d.input_padding_top = input_padding_top;
  node->params.convolution_2d.input_padding_right = input_padding_right;
  node->params.convolution_2d.input_padding_bottom = input_padding_bottom;
  node->params.convolution_2d.input_padding_left = input_padding_left;
  node->params.convolution_2d.kernel_height = kernel_height;
  node->params.convolution_2d.kernel_width = kernel_width;
  node->params.convolution_2d.subsampling_height = subsampling_height;
  node->params.convolution_2d.subsampling_width = subsampling_width;
  node->params.convolution_2d.dilation_height = dilation_height;
  node->params.convolution_2d.dilation_width = dilation_width;
  node->params.convolution_2d.groups = groups;
  node->params.convolution_2d.group_input_channels = group_input_channels;
  node->params.convolution_2d.group_output_channels = group_output_channels;
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
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Deconvolution operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to define Deconvolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    return xnn_status_invalid_parameter;
  }

  if (upsampling_width == 0 || upsampling_height == 0) {
    xnn_log_error(
      "failed to define Deconvolution operator with %" PRIu32 "x%" PRIu32 " upsampling: "
      "upsampling dimensions must be non-zero",
      upsampling_width, upsampling_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to define Deconvolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to define Deconvolution operator with %" PRIu32 " groups: number of groups must be non-zero", groups);
    return xnn_status_invalid_parameter;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to define Deconvolution operator with %zu input channels per group: "
      "number of channels must be non-zero",
      group_input_channels);
    return xnn_status_invalid_parameter;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to define Deconvolution operator with %zu output channels per group: "
      "number of channels must be non-zero",
      group_output_channels);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Deconvolution operator with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Deconvolution operator with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Deconvolution operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Deconvolution operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Deconvolution operator with filter ID #%" PRIu32 ": invalid Value ID",
      filter_id);
    return xnn_status_invalid_parameter;
  }

  if (bias_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Deconvolution operator with bias ID #%" PRIu32 ": invalid Value ID",
      bias_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Deconvolution operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
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
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Depthwise Convolution operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    return xnn_status_invalid_parameter;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with %" PRIu32 "x%" PRIu32 " subsampling: "
      "subsampling dimensions must be non-zero",
      subsampling_width, subsampling_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (depth_multiplier == 0) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with %" PRIu32 " depth multiplier: "
      "depth multiplier must be non-zero",
      depth_multiplier);
    return xnn_status_invalid_parameter;
  }

  if (input_channels == 0) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with %zu input channels: "
      "number of channels must be non-zero",
      input_channels);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with filter ID #%" PRIu32 ": invalid Value ID",
      filter_id);
    return xnn_status_invalid_parameter;
  }

  if (bias_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with bias ID #%" PRIu32 ": invalid Value ID",
      bias_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Depthwise Convolution operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
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

enum xnn_status xnn_define_fully_connected(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Fully Connected operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Fully Connected operator with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Fully Connected operator with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Fully Connected operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Fully Connected operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Fully Connected operator with filter ID #%" PRIu32 ": invalid Value ID",
      filter_id);
    return xnn_status_invalid_parameter;
  }

  if (bias_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Fully Connected operator with bias ID #%" PRIu32 ": invalid Value ID",
      bias_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Fully Connected operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_fully_connected;
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
}

enum xnn_status xnn_define_average_pooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Average Pooling operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define Average Pooling operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to define Average Pooling operator with 1 pooling element: 1x1 pooling is meaningless");
    return xnn_status_invalid_parameter;
  }

  if (stride_height == 0 || stride_width == 0) {
    xnn_log_error(
      "failed to define Average Pooling operator with %" PRIu32 "x%" PRIu32 " stride: "
      "stride dimensions must be non-zero",
      stride_width, stride_height);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Average Pooling with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Average Pooling with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Average Pooling with [%.7g, %.7g] output range: lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to define Average Pooling operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      return xnn_status_invalid_parameter;
    }
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Average Pooling operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Average Pooling operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_average_pooling_2d;
  node->params.pooling_2d.input_padding_top = input_padding_top;
  node->params.pooling_2d.input_padding_right = input_padding_right;
  node->params.pooling_2d.input_padding_bottom = input_padding_bottom;
  node->params.pooling_2d.input_padding_left = input_padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->params.pooling_2d.stride_height = stride_height;
  node->params.pooling_2d.stride_width = stride_width;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_max_pooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Max Pooling operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define Max Pooling operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to define Max Pooling operator with 1 pooling element: 1x1 pooling is meaningless");
    return xnn_status_invalid_parameter;
  }

  if (stride_height == 0 || stride_width == 0) {
    xnn_log_error(
      "failed to define Max Pooling operator with %" PRIu32 "x%" PRIu32 " stride: "
      "stride dimensions must be non-zero",
      stride_width, stride_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_height == 0 || dilation_width == 0) {
    xnn_log_error(
      "failed to define Max Pooling operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Max Pooling with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Max Pooling with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Max Pooling with [%.7g, %.7g] output range: lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to define Max Pooling operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      return xnn_status_invalid_parameter;
    }
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Max Pooling operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Max Pooling operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_max_pooling_2d;
  node->params.pooling_2d.input_padding_top = input_padding_top;
  node->params.pooling_2d.input_padding_right = input_padding_right;
  node->params.pooling_2d.input_padding_bottom = input_padding_bottom;
  node->params.pooling_2d.input_padding_left = input_padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->params.pooling_2d.stride_height = stride_height;
  node->params.pooling_2d.stride_width = stride_width;
  node->params.pooling_2d.dilation_height = dilation_height;
  node->params.pooling_2d.dilation_width = dilation_width;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_argmax_pooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t input_id,
  uint32_t output_value_id,
  uint32_t output_index_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define ArgMax Pooling: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define ArgMax Pooling with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to define ArgMax Pooling with 1 pooling element: 1x1 pooling is meaningless");
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define ArgMax Pooling with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_value_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define ArgMax Pooling with output value ID #%" PRIu32 ": invalid Value ID",
      output_value_id);
    return xnn_status_invalid_parameter;
  }

  if (output_index_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define ArgMax Pooling with output index ID #%" PRIu32 ": invalid Value ID",
      output_index_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_argmax_pooling_2d;
  node->params.pooling_2d.input_padding_top = input_padding_top;
  node->params.pooling_2d.input_padding_right = input_padding_right;
  node->params.pooling_2d.input_padding_bottom = input_padding_bottom;
  node->params.pooling_2d.input_padding_left = input_padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 2;
  node->outputs[0] = output_value_id;
  node->outputs[1] = output_index_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_unpooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t padding_top,
  uint32_t padding_right,
  uint32_t padding_bottom,
  uint32_t padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t input_value_id,
  uint32_t input_index_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define UnPooling: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define UnPooling with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to define UnPooling with 1 pooling element: 1x1 pooling is meaningless");
    return xnn_status_invalid_parameter;
  }

  if (input_value_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define UnPooling with input value ID #%" PRIu32 ": invalid Value ID",
      input_value_id);
    return xnn_status_invalid_parameter;
  }

  if (input_index_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define UnPooling with input index ID #%" PRIu32 ": invalid Value ID",
      input_index_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define UnPooling with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_unpooling_2d;
  node->params.pooling_2d.input_padding_top = padding_top;
  node->params.pooling_2d.input_padding_right = padding_right;
  node->params.pooling_2d.input_padding_bottom = padding_bottom;
  node->params.pooling_2d.input_padding_left = padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->num_inputs = 2;
  node->inputs[0] = input_value_id;
  node->inputs[1] = input_index_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_add2(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Add2 operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Add2 operator with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Add2 operator with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Add2 operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input1_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Add2 operator with the first input ID #%" PRIu32 ": invalid Value ID",
      input1_id);
    return xnn_status_invalid_parameter;
  }

  if (input2_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Add2 operator with the second input ID #%" PRIu32 ": invalid Value ID",
      input2_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Add2 operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_add2;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2;
  node->inputs[0] = input1_id;
  node->inputs[1] = input2_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_multiply2(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Multiply2 operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define Multiply2 operator with NaN output lower bound: lower bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define Multiply2 operator with NaN output upper bound: upper bound must be non-NaN");
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define Multiply2 operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input1_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Multiply2 operator with the first input ID #%" PRIu32 ": invalid Value ID",
      input1_id);
    return xnn_status_invalid_parameter;
  }

  if (input2_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Multiply2 operator with the second input ID #%" PRIu32 ": invalid Value ID",
      input2_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Multiply2 operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_multiply2;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2;
  node->inputs[0] = input1_id;
  node->inputs[1] = input2_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_prelu(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t slope_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define PReLU operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define PReLU operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (slope_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define PReLU operator with slope ID #%" PRIu32 ": invalid Value ID",
      slope_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define PReLU operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_prelu;
  node->num_inputs = 2;
  node->inputs[0] = input_id;
  node->inputs[1] = slope_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_clamp(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Clamp operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Clamp operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Clamp operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_clamp;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_hardswish(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define HardSwish operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define HardSwish operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define HardSwish operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_hardswish;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_sigmoid(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define Sigmoid operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Sigmoid operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define Sigmoid operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_sigmoid;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_define_softmax(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define SoftMax operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define SoftMax operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define SoftMax operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_softmax;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}

enum xnn_status xnn_delete_subgraph(
  xnn_subgraph_t subgraph)
{
  if (subgraph != NULL) {
    memset(subgraph->nodes, 0, sizeof(struct xnn_node) * subgraph->num_nodes);
    xnn_release_memory(subgraph->nodes);

    memset(subgraph->values, 0, sizeof(struct xnn_value) * subgraph->num_values);
    xnn_release_memory(subgraph->values);

    memset(subgraph, 0, sizeof(struct xnn_subgraph));
    xnn_release_memory(subgraph);
  }
  return xnn_status_success;
}
