// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>  // For size_t.
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

static enum xnn_status create_even_split_operator_helper(
    const uint32_t output_id,
    const struct xnn_node* node,
    struct xnn_operator_data* opdata,
    const enum xnn_datatype datatype,
    size_t index)
{
  if (output_id == XNN_INVALID_VALUE_ID) {
    // Node's output value has been optimized away, don't even create operator object.
    return xnn_status_success;
  }

  switch (xnn_datatype_size_bits(datatype)) {
    case 8:
      return xnn_create_copy_nc_x8(
          node->flags, &opdata->operator_objects[index]);
    case 16:
      return xnn_create_copy_nc_x16(
          node->flags, &opdata->operator_objects[index]);
    case 32:
      return xnn_create_copy_nc_x32(
          node->flags, &opdata->operator_objects[index]);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_even_split_operator(
  const struct xnn_node* node,
  const struct xnn_runtime_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  xnn_weights_cache_t weights_cache)
{
  size_t num_splits = opdata->num_outputs;
  assert(node->num_inputs == 1);
  assert(node->num_outputs == num_splits);
  enum xnn_datatype datatype = values[opdata->inputs[0]].datatype;

  int operator_index = 0;
  const int32_t axis = node->params.even_split.axis;
  opdata->axis = axis;
  enum xnn_status status;
  for (size_t i = 0; i < num_splits; ++i) {
    if (values[opdata->outputs[i]].type == xnn_value_type_invalid) continue;
    assert(operator_index < XNN_MAX_OPERATOR_OBJECTS);
    status = create_even_split_operator_helper(opdata->outputs[i], node, opdata, datatype, operator_index);
    ++operator_index;
    if (status != xnn_status_success) {
      return status;
    }
  }

  return status;
}

static enum xnn_status reshape_even_split_operator_helper(
  const struct xnn_runtime_value* values,
  const uint32_t num_values,
  struct xnn_operator_data* opdata,
  size_t operator_index,
  size_t output_index,
  size_t num_splits,
  int32_t axis,
  size_t batch_size,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const uint32_t output_id = opdata->outputs[output_index];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  if (values[output_id].allocation_type == xnn_allocation_type_invalid) {
    // output_id was removed during optimization.
    return xnn_status_success;
  }
  const size_t input_stride = xnn_shape_multiply_trailing_dims(&values[input_id].shape, axis);
  assert(input_stride % num_splits == 0);
  const size_t channels = input_stride / num_splits;
  const size_t output_stride = channels;

  switch (opdata->operator_objects[operator_index]->type) {
    case xnn_operator_type_copy_nc_x16:
      return xnn_reshape_copy_nc_x16(
        opdata->operator_objects[operator_index], batch_size, channels, input_stride, output_stride, threadpool);
    case xnn_operator_type_copy_nc_x32:
      return xnn_reshape_copy_nc_x32(
        opdata->operator_objects[operator_index], batch_size, channels, input_stride, output_stride, threadpool);
    case xnn_operator_type_copy_nc_x8:
      return xnn_reshape_copy_nc_x8(
        opdata->operator_objects[operator_index], batch_size, channels, input_stride, output_stride, threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status reshape_even_split_operator(
  struct xnn_operator_data* opdata,
  struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  enum xnn_status status = xnn_status_success;

  assert(opdata->num_inputs == 1);
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const struct xnn_runtime_value* input_value = values + input_id;

  int32_t axis = opdata->axis;
  if (axis < 0) {
    axis += input_value->shape.num_dims;
  }
  // Check that the split dimension can be evenly split into outputs.
  if (axis >= input_value->shape.num_dims) {
    xnn_log_error(
      "failed to reshape Even Split operator with the input ID #%" PRIu32
      ": split dimension (%d) exceeds the number of dimensions (%zu)",
      input_id, axis, input_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }
  size_t batch_size = xnn_shape_multiply_leading_dims(&input_value->shape, axis);

  size_t num_splits = opdata->num_outputs;
  const size_t axis_elements = input_value->shape.dim[axis] / num_splits;
  const size_t old_workspace_size = opdata->workspace_size;
  bool reallocation_required = false;
  int operator_index = 0;
  for (size_t i = 0; i < num_splits; ++i) {
    const uint32_t output_id = opdata->outputs[i];
    if (values[output_id].type == xnn_value_type_invalid)  continue;
    status = reshape_even_split_operator_helper(values, num_values, opdata, operator_index, i, num_splits, axis, batch_size, threadpool);
    ++operator_index;
    if (status != xnn_status_success) {
      return status;
    }
    const uint32_t output_n_id = opdata->outputs[i];
    assert(output_n_id != XNN_INVALID_VALUE_ID);
    assert(output_n_id < num_values);
    struct xnn_runtime_value* output_n_value = values + output_n_id;
    if (output_n_value->allocation_type == xnn_allocation_type_invalid) {
      // output_id was removed during optimization.
      continue;
    }
    memcpy(output_n_value->shape.dim, input_value->shape.dim, input_value->shape.num_dims * sizeof(size_t));
    output_n_value->shape.num_dims = input_value->shape.num_dims;
    output_n_value->shape.dim[axis] = axis_elements;
    const size_t new_size = xnn_runtime_tensor_get_size(output_n_value);
    if (new_size > output_n_value->size) {
      output_n_value->size = new_size;
      reallocation_required = true;
    }
  }
  if (reallocation_required || opdata->workspace_size > old_workspace_size) {
    return xnn_status_reallocation_required;
  }
  return status;
}

static enum xnn_status setup_even_split_operator_helper(
  const struct xnn_runtime_value* values,
  const uint32_t num_values,
  const struct xnn_operator_data* opdata,
  size_t output_index,
  size_t operator_index,
  const void* input_data,
  pthreadpool_t threadpool)
{
  const uint32_t output_id = opdata->outputs[output_index];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  if (values[output_id].allocation_type == xnn_allocation_type_invalid) {
    // output_id was removed during optimization.
    return xnn_status_success;
  }

  const size_t channels = opdata->operator_objects[operator_index]->channels;

  assert(output_id < num_values);
  const struct xnn_runtime_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[operator_index]->type) {
    case xnn_operator_type_copy_nc_x16:
      return xnn_setup_copy_nc_x16(
        opdata->operator_objects[operator_index], (const uint16_t*) input_data + output_index * channels,
        output_data);
    case xnn_operator_type_copy_nc_x32:
      return xnn_setup_copy_nc_x32(
        opdata->operator_objects[operator_index], (const uint32_t*) input_data + output_index * channels,
        output_data);
    case xnn_operator_type_copy_nc_x8:
      return xnn_setup_copy_nc_x8(
        opdata->operator_objects[operator_index], (const uint8_t*) input_data + output_index * channels,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status setup_even_split_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const struct xnn_runtime_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  enum xnn_status status = xnn_status_success;

  size_t num_splits = opdata->num_outputs;
  int operator_index = 0;
  for (size_t i = 0; i < num_splits; ++i) {
    const uint32_t output_id = opdata->outputs[i];
    if (values[output_id].type == xnn_value_type_invalid)  continue;
    status = setup_even_split_operator_helper(values, num_values, opdata, i, operator_index, input_data, threadpool);
    ++operator_index;
    if (status != xnn_status_success) {
      return status;
    }
  }

  return status;
}

enum xnn_status check_output_value(
  xnn_subgraph_t subgraph,
  int32_t split_dim,
  uint32_t input_id,
  uint32_t output_id,
  const char* nth,
  enum xnn_node_type node_type)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];
  const struct xnn_value* output_value = &subgraph->values[output_id];
  enum xnn_status status;

  status = xnn_subgraph_check_output_node_id(node_type, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_type_dense(node_type, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_datatype_matches(node_type, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_status_success;
}

static enum xnn_status check_datatype_copyable(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  const char* nth,
  enum xnn_node_type node_type)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];
  const struct xnn_value* output_value = &subgraph->values[output_id];

  enum xnn_status status = xnn_subgraph_check_datatype_matches(node_type, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }
  return xnn_subgraph_check_quantization_parameter_matches(node_type, input_id, input_value, output_id, output_value);
}

enum xnn_status xnn_define_even_split(
  xnn_subgraph_t subgraph,
  int32_t split_dim,
  uint32_t input_id,
  size_t num_outputs,
  const uint32_t* output_ids,
  uint32_t flags)
{
  assert(num_outputs >= 1);
  assert(num_outputs <= XNN_MAX_OUTPUTS);

  enum xnn_node_type node_type = xnn_node_type_even_split;
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(node_type)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(node_type, input_id, subgraph->num_values)) != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(node_type, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  for (int i = 0; i < num_outputs; ++i) {
    status = check_output_value(subgraph, split_dim, input_id, output_ids[i], "Nth", node_type);
    if (status != xnn_status_success) {
      return status;
    }
  }

  if (num_outputs > XNN_MAX_OUTPUTS) {
    xnn_log_error(
      "failed to define %s operator with %zu inputs: number of inputs (%zu) exceeds the supported maximum (%zu)",
      xnn_node_type_to_string(node_type), num_outputs, num_outputs, (size_t) XNN_MAX_OUTPUTS);
    return xnn_status_invalid_parameter;
  }

  for (int i = 0; i < num_outputs; ++i) {
    check_datatype_copyable(subgraph, input_id, output_ids[i], "Nth", node_type);
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.even_split.axis = split_dim;
  node->type = node_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = num_outputs;
  for(int i=0;i<num_outputs;++i){
    node->outputs[i]=output_ids[i];
  }
  node->create = create_even_split_operator;
  node->reshape = reshape_even_split_operator;
  node->setup = setup_even_split_operator;
  node->flags = flags;

  return xnn_status_success;
}
