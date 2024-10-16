// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

static const size_t NCHW_AXES_MAPPING[4] = {0, 2, 3, 1};
static const size_t INVERSE_NCHW_AXES_MAPPING[4] = {0, 3, 2, 1};

static void rewrite_reduction_axes_for_nchw(size_t num_reduction_axes, size_t* reduction_axes) {
  bool mask[4] = {false};
  size_t original_reduction_axes[4];
  memcpy(original_reduction_axes, reduction_axes, num_reduction_axes * sizeof(size_t));

  for (size_t idx = 0; idx < num_reduction_axes; ++idx) {
    reduction_axes[idx] = NCHW_AXES_MAPPING[original_reduction_axes[idx]];
    mask[reduction_axes[idx]] = true;
  }

  size_t counter = 0;
  for (size_t idx = 0; idx < num_reduction_axes; ++idx) {
    while (mask[counter]) {
      reduction_axes[counter++] = idx;
      mask[idx] = false;
    }
  }
}

static enum xnn_status create_reduce_operator(
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
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const struct xnn_value *input_value = &values[input_id];

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  float scale = 0;
  int32_t input_zero_point = 0;
  int32_t output_zero_point = 0;

  switch (input_value->datatype) {
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      scale = values[input_id].quantization.scale / values[output_id].quantization.scale;
      input_zero_point = values[input_id].quantization.zero_point;
      output_zero_point = values[output_id].quantization.zero_point;
      break;
    default:
      break;
  }

  status = xnn_create_reduce_nd(
    xnn_node_type_to_reduce_operator(node->type),
    input_value->datatype,
    scale, input_zero_point, output_zero_point,
    node->flags,
    &opdata->operator_objects[0]);
  if (status == xnn_status_success) {
    const size_t num_reduction_axes = node->params.reduce.num_reduction_axes;
    opdata->num_reduction_axes = num_reduction_axes;
    memcpy(opdata->reduction_axes, node->params.reduce.reduction_axes, num_reduction_axes * sizeof(size_t));
  }
  return status;
}

static enum xnn_status reshape_reduce_operator(
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

  void* workspace_size = &opdata->workspace_size;
  void* workspace_alignment = &opdata->workspace_alignment;

  if(input_value->datatype == xnn_datatype_fp32) {
    workspace_size = NULL;
    workspace_alignment = NULL;
  }

  enum xnn_status status = xnn_status_invalid_state;

  size_t num_reduction_axes = opdata->num_reduction_axes;
  size_t input_num_dims = input_value->shape.num_dims;

  // This is the case when NHWC was rewritten to NCHW.
  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];
  size_t input_dims[XNN_MAX_TENSOR_DIMS];
  memcpy(reduction_axes, opdata->reduction_axes, num_reduction_axes * sizeof(size_t));
  memcpy(input_dims, input_value->shape.dim, input_num_dims * sizeof(size_t));
  if (input_value->shape.num_dims == 4 && input_value->layout == xnn_layout_type_nchw) {
    rewrite_reduction_axes_for_nchw(num_reduction_axes, reduction_axes);

    for (size_t idx = 0; idx < input_num_dims; ++idx) {
      input_dims[idx] = input_value->shape.dim[INVERSE_NCHW_AXES_MAPPING[idx]];
    }
  }

  status = xnn_reshape_reduce_nd(
      opdata->operator_objects[0],
      input_value->datatype,
      num_reduction_axes,
      reduction_axes,
      input_num_dims,
      input_dims,
      workspace_size,
      workspace_alignment,
      threadpool);

  struct xnn_value* output_value = values + output_id;
  if (opdata->operator_objects[0]->flags & XNN_FLAG_KEEP_DIMS) {
    output_value->shape.num_dims = input_num_dims;
    for (size_t input_idx = 0; input_idx < input_num_dims; ++input_idx) {
      bool is_axis = false;
      size_t mapped_input_idx = input_idx;
      size_t mapped_output_idx = input_idx;
      if (input_value->layout == xnn_layout_type_nchw) {
        mapped_input_idx = INVERSE_NCHW_AXES_MAPPING[input_idx];
      }
      if (output_value->layout == xnn_layout_type_nchw) {
        mapped_output_idx = NCHW_AXES_MAPPING[mapped_input_idx];
      }

      for (size_t axis_idx = 0; axis_idx < num_reduction_axes; ++axis_idx) {
        size_t reduction_axis = reduction_axes[axis_idx];
        if (output_value->layout == xnn_layout_type_nchw) {
          reduction_axis = NCHW_AXES_MAPPING[reduction_axis];
        }

        if (reduction_axis == mapped_output_idx) {
          is_axis = true;
          break;
        }
      }

      if (is_axis) {
        output_value->shape.dim[input_idx] = 1;
      } else {
        output_value->shape.dim[input_idx] = input_dims[input_idx];
      }
    }
  } else {
    size_t num_skip_axis = 0;
    for (size_t input_idx = 0; input_idx < input_num_dims; ++input_idx) {
      bool is_axis = false;
      size_t mapped_input_idx = input_idx;
      size_t mapped_output_idx = input_idx;
      if (input_value->layout == xnn_layout_type_nchw) {
        mapped_input_idx = INVERSE_NCHW_AXES_MAPPING[input_idx];
      }
      if (output_value->layout == xnn_layout_type_nchw) {
        mapped_output_idx = NCHW_AXES_MAPPING[mapped_input_idx];
      }

      for (size_t axis_idx = 0; axis_idx < num_reduction_axes; ++axis_idx) {
        if (reduction_axes[axis_idx] == mapped_output_idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_value->shape.dim[input_idx - num_skip_axis] = input_dims[input_idx];
      }
    }
    output_value->shape.num_dims = input_num_dims - num_skip_axis;
  }
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return status;
}

static enum xnn_status setup_reduce_operator(
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

  void* workspace = input_value->datatype != xnn_datatype_fp32 ? opdata->workspace : NULL;

  return xnn_setup_reduce_nd(opdata->operator_objects[0], workspace, input_data, output_data);
}

enum xnn_status xnn_define_static_reduce(
  xnn_subgraph_t subgraph,
  enum xnn_reduce_operator reduce_operator,
  size_t num_reduction_axes,
  const size_t* reduction_axes,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  const enum xnn_node_type node_type = xnn_reduce_operator_to_node_type(reduce_operator);
  if(node_type == xnn_node_type_invalid) {
    xnn_log_error("failed to define reduce operator: invalid operation %d.", reduce_operator);
    return xnn_status_invalid_parameter;
  }

  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(node_type)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_nth_input_node_id(node_type, input_id, subgraph->num_values, 1);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_nth_input_type_dense(node_type, input_id, input_value, 1);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with the first input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
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
    case xnn_datatype_qint8:
      compute_type = xnn_compute_type_qs8;
      break;
    case xnn_datatype_quint8:
      compute_type = xnn_compute_type_qu8;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (num_reduction_axes == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu reduction axes: the number of reduction axes must be non-zero",
      xnn_node_type_to_string(node_type), num_reduction_axes);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 1; i < num_reduction_axes; i++) {
    if (reduction_axes[i] <= reduction_axes[i - 1]) {
      xnn_log_error(
        "failed to define %s operator with #%zu reduction axis of %zu: the reduction "
        "axes must be in ascending order and unique",
        xnn_node_type_to_string(node_type), i, reduction_axes[i]);
      return xnn_status_invalid_parameter;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = node_type;
  node->compute_type = compute_type;
  node->params.reduce.num_reduction_axes = num_reduction_axes;
  memcpy(node->params.reduce.reduction_axes, reduction_axes, num_reduction_axes * sizeof(size_t));
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_reduce_operator;
  node->reshape = reshape_reduce_operator;
  node->setup = setup_reduce_operator;

  return xnn_status_success;
}
