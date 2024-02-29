// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>


static enum xnn_status create_global_sum_pooling_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);
  assert(node->num_outputs == 1);

  enum xnn_status status;
  assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
  assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
  switch (node->compute_type) {
    case xnn_compute_type_fp32:
      status = xnn_create_global_sum_pooling_nwc_f32(
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp16:
      status = xnn_create_global_sum_pooling_nwc_f16(
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_global_sum_pooling_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const size_t num_input_dims = values[input_id].shape.num_dims;
  assert(num_input_dims >= 1);
  size_t batch_size, input_width, num_batch_dims;
  switch (opdata->type) {
    case xnn_node_type_global_sum_pooling_1d:
      num_batch_dims = num_input_dims - 2;
      batch_size = xnn_shape_multiply_batch_dims(&values[input_id].shape, 2);
      input_width = values[input_id].shape.dim[num_input_dims - 2];
      break;
    case xnn_node_type_global_sum_pooling_2d:
      num_batch_dims = num_input_dims - 3;
      batch_size = xnn_shape_multiply_batch_dims(&values[input_id].shape, 3);
      input_width = values[input_id].shape.dim[num_input_dims - 3] * values[input_id].shape.dim[num_input_dims - 2];
      break;
    default:
      XNN_UNREACHABLE;
  }
  const size_t channel_dim = values[input_id].shape.dim[num_input_dims - 1];
  enum xnn_status status = xnn_status_invalid_state;
  const size_t old_workspace_size = opdata->workspace_size;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_global_sum_pooling_nwc_f32:
      status = xnn_reshape_global_sum_pooling_nwc_f32(
        opdata->operator_objects[0],
        batch_size,
        input_width,
        /*channels=*/channel_dim,
        /*input_stride=*/channel_dim,
        /*output_stride=*/channel_dim,
        &opdata->workspace_size, &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_global_sum_pooling_nwc_f16:
      status = xnn_reshape_global_sum_pooling_nwc_f16(
        opdata->operator_objects[0],
        batch_size,
        input_width,
        /*channels=*/channel_dim,
        /*input_stride=*/channel_dim,
        /*output_stride=*/channel_dim,
        &opdata->workspace_size, &opdata->workspace_alignment,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  struct xnn_value* output_value = values + output_id;

  memcpy(&output_value->shape.dim[0], &values[input_id].shape.dim[0], num_batch_dims);
  if (opdata->operator_objects[0]->flags & XNN_FLAG_REDUCE_DIMS) {
    output_value->shape.dim[num_batch_dims] = channel_dim;
    output_value->shape.num_dims = num_batch_dims + 1;
  } else {
    output_value->shape.num_dims = num_input_dims;
    output_value->shape.dim[num_input_dims - 1] = channel_dim;
    switch (opdata->type) {
      case xnn_node_type_global_sum_pooling_1d:
        output_value->shape.dim[num_batch_dims] = 1;
        break;
      case xnn_node_type_global_sum_pooling_2d:
        output_value->shape.dim[num_batch_dims] = 1;
        output_value->shape.dim[num_batch_dims + 1] = 1;
        break;
      default:
        XNN_UNREACHABLE;
    }
  }
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_global_sum_pooling_operator(
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
    case xnn_operator_type_global_sum_pooling_nwc_f32:
      return xnn_setup_global_sum_pooling_nwc_f32(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_global_sum_pooling_nwc_f16:
      return xnn_setup_global_sum_pooling_nwc_f16(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status define_global_sum_pooling_nd(
  xnn_subgraph_t subgraph,
  enum xnn_node_type node_type,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(node_type)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_min_max(node_type, output_min, output_max);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_input_node_id(node_type, input_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(node_type, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
      break;
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(node_type, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(node_type, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output_value->datatype) {
    case xnn_datatype_fp16:
      compute_type = xnn_compute_type_fp16;
      break;
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_datatype_matches(
    node_type, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = node_type;
  node->compute_type = compute_type;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_global_sum_pooling_operator;
  node->reshape = reshape_global_sum_pooling_operator;
  node->setup = setup_global_sum_pooling_operator;

  return xnn_status_success;
}

enum xnn_status xnn_define_global_sum_pooling_1d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  return define_global_sum_pooling_nd(
    subgraph, xnn_node_type_global_sum_pooling_1d, output_min, output_max, input_id, output_id, flags);
}

enum xnn_status xnn_define_global_sum_pooling_2d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  return define_global_sum_pooling_nd(
    subgraph, xnn_node_type_global_sum_pooling_2d, output_min, output_max, input_id, output_id, flags);
}
