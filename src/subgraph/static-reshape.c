// Copyright 2020 Google LLC
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

static enum xnn_status create_copy_operator(
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
      status = xnn_create_copy_nc_x16(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      status = xnn_create_copy_nc_x32(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_qs8:
    case xnn_compute_type_qu8:
      status = xnn_create_copy_nc_x8(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    opdata->num_reshape_dims = node->params.static_reshape.new_shape.num_dims;
    memcpy(opdata->reshape_dims, node->params.static_reshape.new_shape.dim, opdata->num_reshape_dims * sizeof(size_t));
  }
  return status;
}

static enum xnn_status resize_copy_output_tensor(
  const struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t old_workspace_size)
{
  const uint32_t input_id = opdata->inputs[0];
  const struct xnn_value* input = &values[input_id];

  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = (struct xnn_value*) &values[output_id];

  const size_t num_output_dims = opdata->num_reshape_dims;
  size_t output_axis_dynamic = XNN_MAX_TENSOR_DIMS;

  // Propagate output channels.
  output->shape.num_dims = num_output_dims;
  for (size_t dim_idx = 0; dim_idx < num_output_dims; ++dim_idx) {
    size_t hint_cur_dim = opdata->reshape_dims[dim_idx];
    if (hint_cur_dim == 0) {
      if (output_axis_dynamic < XNN_MAX_TENSOR_DIMS) {
        return xnn_status_invalid_parameter;
      }
      output_axis_dynamic = dim_idx;
      hint_cur_dim = 1;
    }
    output->shape.dim[dim_idx] = hint_cur_dim;
  }

  const size_t input_num_elements = xnn_shape_multiply_all_dims(&input->shape);
  if (output_axis_dynamic < XNN_MAX_TENSOR_DIMS) {
    const size_t output_num_elements = xnn_shape_multiply_all_dims(&output->shape);
    const size_t inferred_dim = input_num_elements / output_num_elements;
    if (inferred_dim * output_num_elements != input_num_elements) {
      xnn_log_error("Cannot infer output shape given input number of elements %zu, and output number of elements %zu",
                    input_num_elements, output_num_elements);
      return xnn_status_invalid_parameter;
    }
    // Infer dynamic dimension
    output->shape.dim[output_axis_dynamic] = inferred_dim;
  } else {
    const size_t output_num_elements = xnn_shape_multiply_all_dims(&output->shape);

    if (input_num_elements != output_num_elements) {
      xnn_log_error(
          "failed to reshape %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
          ": number of input elements, %zu, does not match number of output elements %zu",
          xnn_node_type_to_string(xnn_node_type_static_reshape), input_id, output_id, input_num_elements,
          output_num_elements);
        return xnn_status_invalid_parameter;
    }
  }

  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || old_workspace_size < opdata->workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }

  return xnn_status_success;
}

static enum xnn_status reshape_copy_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const size_t batch_size = xnn_shape_multiply_all_dims(&values[input_id].shape);

  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_copy_nc_x8:
      status = xnn_reshape_copy_nc_x8(
        opdata->operator_objects[0],
        batch_size,
        1 /* channels */, 1 /* input stride */, 1 /* output stride */,
        threadpool);
      break;
    case xnn_operator_type_copy_nc_x16:
      status = xnn_reshape_copy_nc_x16(
        opdata->operator_objects[0],
        batch_size,
        1 /* channels */, 1 /* input stride */, 1 /* output stride */,
        threadpool);
      break;
    case xnn_operator_type_copy_nc_x32:
      status = xnn_reshape_copy_nc_x32(
        opdata->operator_objects[0],
        batch_size,
        1 /* channels */, 1 /* input stride */, 1 /* output stride */,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status != xnn_status_success) {
    return status;
  }
  return resize_copy_output_tensor(opdata, values, old_workspace_size);
}

static enum xnn_status setup_copy_operator(
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
    case xnn_operator_type_copy_nc_x8:
      return xnn_setup_copy_nc_x8(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_copy_nc_x16:
      return xnn_setup_copy_nc_x16(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_copy_nc_x32:
      return xnn_setup_copy_nc_x32(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_static_reshape(
  xnn_subgraph_t subgraph,
  size_t num_dims,
  const size_t* new_shape,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_static_reshape)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_static_reshape, input_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_static_reshape, input_id, input_value);
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
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_reshape), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_static_reshape, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_static_reshape, output_id, output_value);
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
        xnn_node_type_to_string(xnn_node_type_static_reshape), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_datatype_matches(xnn_node_type_static_reshape, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_static_reshape, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to define %s operator with %zu-dimensional output shape: at most %zu dimensions are supported",
      xnn_node_type_to_string(xnn_node_type_static_reshape), num_dims, (size_t) XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.static_reshape.new_shape.num_dims = num_dims;
  if (num_dims != 0) {
    memcpy(&node->params.static_reshape.new_shape.dim, new_shape, num_dims * sizeof(size_t));
  }

  node->type = xnn_node_type_static_reshape;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_copy_operator;
  node->reshape = reshape_copy_operator;
  node->setup = setup_copy_operator;

  return xnn_status_success;
}
