// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>


static enum xnn_status create_mean_operator(
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
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      status = xnn_create_mean_nd_f16(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      status = xnn_create_mean_nd_f32(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    const size_t num_reduction_axes = node->params.reduce.num_reduction_axes;
    opdata->num_reduction_axes = num_reduction_axes;
    memcpy(opdata->reduction_axes, node->params.reduce.reduction_axes, num_reduction_axes * sizeof(size_t));
  }
  return status;
}

static enum xnn_status reshape_mean_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const struct xnn_value* input_value = values + input_id;
  assert(input_value->type == xnn_value_type_dense_tensor);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_mean_nd_f16:
      status = xnn_reshape_mean_nd_f16(
        opdata->operator_objects[0],
        opdata->num_reduction_axes,
        opdata->reduction_axes,
        input_value->shape.num_dims,
        input_value->shape.dim,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_mean_nd_f32:
      status = xnn_reshape_mean_nd_f32(
        opdata->operator_objects[0],
        opdata->num_reduction_axes,
        opdata->reduction_axes,
        input_value->shape.num_dims,
        input_value->shape.dim,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  struct xnn_value* output_value = values + output_id;
  size_t input_num_dims = input_value->shape.num_dims;
  size_t num_reduction_axes = opdata->num_reduction_axes;
  if (opdata->operator_objects[0]->flags & XNN_FLAG_KEEP_DIMS) {
    output_value->shape.num_dims = input_value->shape.num_dims;
    for (size_t idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (size_t axis_idx = 0; axis_idx < num_reduction_axes; ++axis_idx) {
        if (opdata->reduction_axes[axis_idx] == idx) {
          is_axis = true;
          break;
        }
      }
      if (is_axis) {
        output_value->shape.dim[idx] = 1;
      } else {
        output_value->shape.dim[idx] = input_value->shape.dim[idx];
      }
    }
  } else {
    size_t num_skip_axis = 0;
    for (size_t idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (size_t axis_idx = 0; axis_idx < num_reduction_axes; ++axis_idx) {
        if (opdata->reduction_axes[axis_idx] == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_value->shape.dim[idx - num_skip_axis] = input_value->shape.dim[idx];
      }
    }
    output_value->shape.num_dims = input_value->shape.num_dims - num_skip_axis;
  }
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return status;
}

static enum xnn_status setup_mean_operator(
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
  assert(input_value->type == xnn_value_type_dense_tensor);
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_value* output_value = values + output_id;
  assert(output_value->type == xnn_value_type_dense_tensor);
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_mean_nd_f16:
      return xnn_setup_mean_nd_f16(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data, output_data);
    case xnn_operator_type_mean_nd_f32:
      return xnn_setup_mean_nd_f32(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data, output_data);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_static_mean(
  xnn_subgraph_t subgraph,
  size_t num_reduction_axes,
  const size_t* reduction_axes,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_static_mean)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_nth_input_node_id(xnn_node_type_static_mean, input_id, subgraph->num_values, 1);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_nth_input_type_dense(xnn_node_type_static_mean, input_id, input_value, 1);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with the first input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_mean), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_static_mean, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_static_mean, output_id, output_value);
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
        xnn_node_type_to_string(xnn_node_type_static_mean), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (num_reduction_axes > input_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with %zu reduction axes: "
      "the number of reduction axes must not exceed the number of input dimensions %zu",
      xnn_node_type_to_string(xnn_node_type_static_mean), num_reduction_axes, input_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (num_reduction_axes == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu reduction axes: the number of reduction axes must be non-zero",
      xnn_node_type_to_string(xnn_node_type_static_mean), num_reduction_axes);
    return xnn_status_invalid_parameter;
  }

  size_t last_axis = 0;
  for (size_t i = 0; i < num_reduction_axes; i++) {
    const size_t axis = reduction_axes[i];
    if (axis > input_value->shape.num_dims) {
      xnn_log_error(
        "failed to define %s operator with #%zu reduction axis of %zu: the index is out of bounds for a %zuD input shape",
        xnn_node_type_to_string(xnn_node_type_static_mean), i, axis, input_value->shape.num_dims);
      return xnn_status_invalid_parameter;
    }
    if (i != 0) {
      if (axis != last_axis + 1) {
        xnn_log_error(
          "failed to define %s operator with #%zu reduction axis of %zu: the axis is disjoint with #%zu reduction axis of %zu",
          xnn_node_type_to_string(xnn_node_type_static_mean), i, axis, i - 1, last_axis);
        return xnn_status_invalid_parameter;
      }
    }
    last_axis = axis;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_static_mean;
  node->compute_type = compute_type;
  node->params.reduce.num_reduction_axes = num_reduction_axes;
  memcpy(node->params.reduce.reduction_axes, reduction_axes, num_reduction_axes * sizeof(size_t));
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_mean_operator;
  node->reshape = reshape_mean_operator;
  node->setup = setup_mean_operator;

  return xnn_status_success;
}
