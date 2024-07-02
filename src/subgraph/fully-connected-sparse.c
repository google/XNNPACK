// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

static enum xnn_status create_fully_connected_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs >= 2);
  assert(node->num_inputs <= 3);
  const uint32_t filter_id = node->inputs[1];
  assert(filter_id != XNN_INVALID_VALUE_ID);
  assert(filter_id < num_values);

  assert(node->num_outputs == 1);

  size_t output_channels = values[node->inputs[1]].shape.dim[0];
  size_t input_channels = values[node->inputs[1]].shape.dim[1];

  const void* kernel_data = values[filter_id].fp32_data != NULL ? values[filter_id].fp32_data : values[filter_id].data;
  assert(kernel_data != NULL);

  const void* bias_data = NULL;
  if (node->num_inputs > 2) {
    const uint32_t bias_id = node->inputs[2];
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);

    bias_data = values[bias_id].fp32_data != NULL ? values[bias_id].fp32_data : values[bias_id].data;
    assert(bias_data != NULL);
  }

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
    {
      status = xnn_create_convolution2d_nchw_f16(
        /*input_padding_top=*/0,
        /*input_padding_right=*/0,
        /*input_padding_bottom=*/0,
        /*input_padding_left=*/0,
        /*kernel_height=*/1,
        /*kernel_width=*/1,
        /*subsampling_height=*/1,
        /*subsampling_width=*/1,
        /*dilation_height=*/1,
        /*dilation_width=*/1,
        /*groups=*/1,
        /*group_input_channels=*/input_channels,
        /*group_output_channels=*/output_channels,
        /*input_channel_stride=*/input_channels,
        /*output_channel_stride=*/output_channels,
        kernel_data,
        bias_data,
        node->activation.output_min,
        node->activation.output_max,
        node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS,
        code_cache,
        weights_cache,
        &opdata->operator_objects[0]);
      break;
    }
    case xnn_compute_type_fp32:
    {
      assert(values[filter_id].datatype == xnn_datatype_fp32);
      status = xnn_create_convolution2d_nchw_f32(
        /*input_padding_top=*/0,
        /*input_padding_right=*/0,
        /*input_padding_bottom=*/0,
        /*input_padding_left=*/0,
        /*kernel_height=*/1,
        /*kernel_width=*/1,
        /*subsampling_height=*/1,
        /*subsampling_width=*/1,
        /*dilation_height=*/1,
        /*dilation_width=*/1,
        /*groups=*/1,
        /*group_input_channels=*/input_channels,
        /*group_output_channels=*/output_channels,
        /*input_channel_stride=*/input_channels,
        /*output_channel_stride=*/output_channels,
        kernel_data,
        bias_data,
        node->activation.output_min,
        node->activation.output_max,
        node->flags,
        code_cache,
        weights_cache,
        &opdata->operator_objects[0]);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_fully_connected_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const uint32_t filter_id = opdata->inputs[0];
  assert(filter_id < num_values);
  const size_t input_channels = values[filter_id].shape.dim[1];
  const size_t num_input_elements = xnn_shape_multiply_all_dims(&values[input_id].shape);
  const size_t batch_size = num_input_elements / input_channels;
  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_convolution_nchw_f16:
      status = xnn_reshape_convolution2d_nchw_f16(
        opdata->operator_objects[0],
        batch_size,
        1, 1, NULL, NULL,
        threadpool);
      break;
    case xnn_operator_type_convolution_nchw_f32:
      status = xnn_reshape_convolution2d_nchw_f32(
        opdata->operator_objects[0],
        batch_size,
        1, 1, NULL, NULL,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }
  return resize_fully_connected_output_tensor(opdata, values, num_values, old_workspace_size, threadpool);
}

static enum xnn_status setup_fully_connected_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_convolution_nchw_f16:
      return xnn_setup_convolution2d_nchw_f16(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_convolution_nchw_f32:
      return xnn_setup_convolution2d_nchw_f32(
        opdata->operator_objects[0],
        input_data,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static inline enum xnn_compute_type validate_datatypes_with_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype kernel_datatype,
  enum xnn_datatype bias_datatype,
  enum xnn_datatype output_datatype)
{
  switch (kernel_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_fp16 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return xnn_compute_type_fp16;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

static inline enum xnn_compute_type validate_datatypes_without_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype kernel_datatype,
  enum xnn_datatype output_datatype)
{
  switch (kernel_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_fp16 && output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return xnn_compute_type_fp16;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

enum xnn_status xnn_define_fully_connected_sparse(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_fully_connected_sparse)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_min_max(xnn_node_type_fully_connected_sparse, output_min, output_max);
  if (status != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_fully_connected_sparse, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_fully_connected_sparse, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), filter_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* kernel_value = &subgraph->values[filter_id];
  if (kernel_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), filter_id, kernel_value->type);
    return xnn_status_invalid_parameter;
  }

  if (kernel_value->data == NULL) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": non-static Value",
      xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), filter_id);
    return xnn_status_invalid_parameter;
  }

  switch (kernel_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), filter_id,
        xnn_datatype_to_string(kernel_value->datatype), kernel_value->datatype);
      return xnn_status_invalid_parameter;
  }

  const struct xnn_value* bias_value = NULL;
  if (bias_id != XNN_INVALID_VALUE_ID) {
    if (bias_id >= subgraph->num_values) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": invalid Value ID",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), bias_id);
      return xnn_status_invalid_parameter;
    }

    bias_value = &subgraph->values[bias_id];
    if (bias_value->type != xnn_value_type_dense_tensor) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), bias_id, bias_value->type);
      return xnn_status_invalid_parameter;
    }

    if (bias_value->data == NULL) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": non-static Value",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), bias_id);
      return xnn_status_invalid_parameter;
    }

    switch (bias_value->datatype) {
      case xnn_datatype_fp16:
      case xnn_datatype_fp32:
        break;
      default:
        xnn_log_error(
          "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
          xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), bias_id,
          xnn_datatype_to_string(bias_value->datatype), bias_value->datatype);
        return xnn_status_invalid_parameter;
    }
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_fully_connected_sparse, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_fully_connected_sparse, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  if (bias_value != NULL) {
    compute_type = validate_datatypes_with_bias(
      input_value->datatype, kernel_value->datatype, bias_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", bias ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), bias (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), input_id, filter_id, bias_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(kernel_value->datatype),
        xnn_datatype_to_string(bias_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  } else {
    compute_type = validate_datatypes_without_bias(
      input_value->datatype, kernel_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_fully_connected_sparse), input_id, filter_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(kernel_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_fully_connected_sparse;
  node->compute_type = compute_type;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2 + (size_t) (bias_id != XNN_INVALID_VALUE_ID);
  node->inputs[0] = input_id;
  node->inputs[1] = filter_id;
  node->inputs[2] = bias_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_fully_connected_operator;
  node->reshape = reshape_fully_connected_operator;
  node->setup = setup_fully_connected_operator;

  return xnn_status_success;
}
