// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/reshape-helpers.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

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
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const struct xnn_value *input_value = &values[input_id];
  switch (xnn_datatype_size_bits(input_value->datatype)) {
    case 8:
      status = xnn_create_copy_nc_x8(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case 16:
      status = xnn_create_copy_nc_x16(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case 32:
      status = xnn_create_copy_nc_x32(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    if (node->params.static_reshape.new_shape.num_dims > 0) {
      opdata->num_reshape_dims = node->params.static_reshape.new_shape.num_dims;
      memcpy(opdata->reshape_dims, node->params.static_reshape.new_shape.dim, opdata->num_reshape_dims * sizeof(size_t));
    }
    opdata->axis = node->params.static_reshape.axis;
    opdata->operator_objects[0]->copy.subtype = node->type;
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
      xnn_log_error(
          "Cannot infer output shape given input number of elements %zu, and "
          "output number of elements %zu",
          input_num_elements, output_num_elements);
      return xnn_status_invalid_parameter;
    }
    // Infer dynamic dimension
    output->shape.dim[output_axis_dynamic] = inferred_dim;
  } else {
    const size_t output_num_elements = xnn_shape_multiply_all_dims(&output->shape);

    if (input_num_elements != output_num_elements) {
      xnn_log_error("failed to reshape %s operator with input ID #%" PRIu32
                    " and output ID #%" PRIu32
                    ": number of input elements, %zu, does not match number of "
                    "output elements %zu",
                    xnn_node_type_to_string(xnn_node_type_static_reshape),
                    input_id, output_id, input_num_elements,
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

static enum xnn_status resize_expand_dims_output_tensor(
  const struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t old_workspace_size)
{
  const uint32_t input_id = opdata->inputs[0];
  const struct xnn_value* input = &values[input_id];

  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = (struct xnn_value*) &values[output_id];

  const struct xnn_shape* input_shape = &input->shape;
  struct xnn_shape* output_shape = &output->shape;
  const size_t *new_axes = opdata->reshape_dims;
  const size_t num_output_dims = input_shape->num_dims + opdata->num_reshape_dims;
  if (num_output_dims > XNN_MAX_TENSOR_DIMS) {
    return xnn_status_invalid_parameter;
  }
  size_t input_iter = 0;
  size_t axes_iter = 0;

  output_shape->num_dims = num_output_dims;
  if (opdata->num_reshape_dims == 0) {
    memcpy(&output->shape.dim[0], &input->shape.dim[0], num_output_dims * sizeof(size_t));
    return xnn_status_success;
  }
  for (int i = 0; i < num_output_dims; ++i) {
    if (new_axes[axes_iter] == i) {
      output_shape->dim[i] = 1;
      ++axes_iter;
    } else {
      output_shape->dim[i] = input_shape->dim[input_iter++];
    }
  }

  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || old_workspace_size < opdata->workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }

  return xnn_status_success;
}

static enum xnn_status resize_fuse_dims_output_tensor(
    const struct xnn_operator_data* opdata, struct xnn_value* values,
    size_t old_workspace_size) {
  const uint32_t input_id = opdata->inputs[0];
  const struct xnn_value* input = &values[input_id];

  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = (struct xnn_value*)&values[output_id];

  const size_t first_dim = opdata->reshape_dims[0];
  const size_t num_dims = opdata->num_reshape_dims;

  const struct xnn_shape* input_shape = &input->shape;
  struct xnn_shape* output_shape = &output->shape;

  if (input_shape->num_dims < first_dim + num_dims) {
    xnn_log_error(
        "failed to fuse dims in %s operator with input ID #%" PRIu32
        " and output ID #%" PRIu32
        ": number of input dimensions, "
        "%zu, is smaller than the fused dimensions, %zu-%zu",
        xnn_node_type_to_string(xnn_node_type_static_reshape), input_id,
        output_id, input_shape->num_dims, first_dim, first_dim + num_dims);
    return xnn_status_invalid_parameter;
  }

  for (size_t k = 0; k < first_dim; k++) {
    output_shape->dim[k] = input_shape->dim[k];
  }
  output_shape->dim[first_dim] = 1;
  for (size_t k = first_dim; k < first_dim + num_dims; k++) {
    output_shape->dim[first_dim] *= input_shape->dim[k];
  }
  for (size_t k = first_dim + num_dims; k < input_shape->num_dims; k++) {
    output_shape->dim[k - num_dims + 1] = input_shape->dim[k];
  }
  output_shape->num_dims = input_shape->num_dims - num_dims + 1;

  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || old_workspace_size < opdata->workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }

  return xnn_status_success;
}

static enum xnn_status resize_split_dims_output_tensor(
    const struct xnn_operator_data* opdata, struct xnn_value* values,
    size_t old_workspace_size) {
  const uint32_t input_id = opdata->inputs[0];
  const struct xnn_value* input = &values[input_id];

  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = (struct xnn_value*)&values[output_id];

  const size_t axis = opdata->axis;
  const size_t num_dims = opdata->num_reshape_dims;
  const size_t* splits = opdata->reshape_dims;

  const struct xnn_shape* input_shape = &input->shape;
  struct xnn_shape* output_shape = &output->shape;

  if (input_shape->num_dims - 1 + num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error("failed to split dims in %s operator with input ID #%" PRIu32
                  " and output ID #%" PRIu32
                  ": number of output dimensions, "
                  "%zu, is larger than the maximum number of dimensions, %zu",
                  xnn_node_type_to_string(xnn_node_type_static_reshape),
                  input_id, output_id, input_shape->num_dims - 1 + num_dims,
                  (size_t)XNN_MAX_TENSOR_DIMS);
    return xnn_status_invalid_parameter;
  }

  size_t count = 1;
  for (size_t k = 0; k < num_dims; k++) {
    if (splits[k]) {
      count *= splits[k];
    }
  }
  size_t remainder = input_shape->dim[axis] / count;
  if (remainder * count != input_shape->dim[axis]) {
    xnn_log_error("failed to split dims in %s operator with input ID #%" PRIu32
                  " and output ID #%" PRIu32
                  ": product of defined splits, %zu, does not divide the split "
                  "input dimension, %zu",
                  xnn_node_type_to_string(xnn_node_type_static_reshape),
                  input_id, output_id, count, input_shape->dim[axis]);
    return xnn_status_invalid_parameter;
  }

  for (size_t k = 0; k < axis; k++) {
    output_shape->dim[k] = input_shape->dim[k];
  }
  for (size_t k = 0; k < num_dims; k++) {
    if (splits[k]) {
      output_shape->dim[axis + k] = splits[k];
    } else {
      output_shape->dim[axis + k] = remainder;
      remainder = 1;
    }
  }
  for (size_t k = axis + 1; k < input_shape->num_dims; k++) {
    output_shape->dim[k + num_dims - 1] = input_shape->dim[k];
  }
  output_shape->num_dims = input_shape->num_dims + num_dims - 1;

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
        /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1,
        threadpool);
      break;
    case xnn_operator_type_copy_nc_x16:
      status = xnn_reshape_copy_nc_x16(
        opdata->operator_objects[0],
        batch_size,
        /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1,
        threadpool);
      break;
    case xnn_operator_type_copy_nc_x32:
      status = xnn_reshape_copy_nc_x32(
        opdata->operator_objects[0],
        batch_size,
        /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status != xnn_status_success) {
    return status;
  }
  switch (opdata->type) {
    case xnn_node_type_static_reshape:
      return resize_copy_output_tensor(opdata, values, old_workspace_size);
    case xnn_node_type_static_expand_dims:
      return resize_expand_dims_output_tensor(opdata, values, old_workspace_size);
    case xnn_node_type_fuse_dims:
      return resize_fuse_dims_output_tensor(opdata, values,
                                                old_workspace_size);
    case xnn_node_type_split_dims:
      return resize_split_dims_output_tensor(opdata, values,
                                             old_workspace_size);
    case xnn_node_type_copy:
      return resize_unary_elementwise_output_tensor(opdata, values, num_values, old_workspace_size, threadpool);
    default:
      XNN_UNREACHABLE;
  }
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

enum xnn_status define_copy_node(xnn_subgraph_t subgraph, size_t num_dims,
                                 const size_t* new_shape, size_t axis,
                                 enum xnn_node_type node_type,
                                 uint32_t input_id, uint32_t output_id,
                                 uint32_t flags) {
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(node_type)) != xnn_status_success) {
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

  if (!xnn_datatype_is_byte_addressable(input_value->datatype)) {
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

  status = xnn_subgraph_check_datatype_matches(node_type, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_quantization_parameter_matches(
      node_type, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.static_reshape.new_shape.num_dims = num_dims;
  if (num_dims != 0) {
    memcpy(&node->params.static_reshape.new_shape.dim, new_shape, num_dims * sizeof(size_t));
  }
  node->params.static_reshape.axis = axis;

  node->type = node_type;
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

enum xnn_status xnn_define_static_reshape(
  xnn_subgraph_t subgraph,
  size_t num_dims,
  const size_t* new_shape,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to define %s operator with %zu-dimensional output shape: at most %zu dimensions are supported",
      xnn_node_type_to_string(xnn_node_type_static_reshape), num_dims, (size_t) XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }
  return define_copy_node(subgraph, num_dims, new_shape, /*axis=*/0,
                          xnn_node_type_static_reshape, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_static_expand_dims(
  xnn_subgraph_t subgraph,
  size_t num_new_axes,
  const size_t* new_axes,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  return define_copy_node(subgraph, num_new_axes, new_axes, /*axis=*/0,
                          xnn_node_type_static_expand_dims, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_fuse_dims(
    xnn_subgraph_t subgraph, size_t axis, size_t axes_count,
    uint32_t input_id, uint32_t output_id, uint32_t flags) {
  if (axis + axes_count > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
        "failed to define %s operator with %zu-dimensional input shape: at "
        "most %zu dimensions are supported",
        xnn_node_type_to_string(xnn_node_type_fuse_dims),
        axis + axes_count, (size_t)XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }
  size_t dims[XNN_MAX_TENSOR_DIMS];
  for (size_t k = 0; k < axes_count; k++) {
    dims[k] = axis + k;
  }
  return define_copy_node(subgraph, /*num_dims=*/axes_count, /*new_shape=*/dims,
                          /*axis=*/0, xnn_node_type_fuse_dims,
                          input_id, output_id, flags);
}

enum xnn_status xnn_define_split_dim(xnn_subgraph_t subgraph,
                                             size_t axis, size_t num_splits,
                                             const size_t* splits,
                                             uint32_t input_id,
                                             uint32_t output_id,
                                             uint32_t flags) {
  if (axis + num_splits > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
        "failed to define %s operator with %zu-dimensional output shape: at "
        "most %zu dimensions are supported",
        xnn_node_type_to_string(xnn_node_type_fuse_dims),
        axis + num_splits, (size_t)XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }
  return define_copy_node(subgraph, /*num_dims=*/num_splits,
                          /*new_shape=*/splits, axis,
                          xnn_node_type_split_dims, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_copy(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  return define_copy_node(subgraph, /*num_dims=*/0, /*new_shape=*/NULL,
                          /*axis=*/0, xnn_node_type_copy, input_id, output_id,
                          flags);
}

