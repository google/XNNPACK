// Copyright 2022 Google LLC
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
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

static enum xnn_status create_concatenate_operator_helper(
  const struct xnn_node *node,
  struct xnn_operator_data *opdata,
  const enum xnn_datatype datatype,
  size_t index)
{
  switch (xnn_datatype_size_bits(datatype)) {
    case 8:
      return xnn_create_copy_nc_x8(node->flags, &opdata->operator_objects[index]);
    case 16:
      return xnn_create_copy_nc_x16(node->flags, &opdata->operator_objects[index]);
    case 32:
      return xnn_create_copy_nc_x32(node->flags, &opdata->operator_objects[index]);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_concatenate_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  size_t num_inputs = opdata->num_inputs;
  enum xnn_status status;
  const int32_t axis = node->params.concatenate.axis;
  opdata->axis = axis;
  const uint32_t input1_id = opdata->inputs[0];
  assert(input1_id < num_values);
  const struct xnn_value *input1_value = &values[input1_id];
  for (size_t i = 0; i < num_inputs; ++i) {
    status = create_concatenate_operator_helper(node, opdata, input1_value->datatype, i);
    if (status != xnn_status_success) {
      return status;
    }
  }

  return status;
}

static enum xnn_status reshape_concatenate_operator_helper(
  const struct xnn_operator_data *opdata,
  size_t index,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool)
{
  switch (opdata->operator_objects[index]->type) {
    case xnn_operator_type_copy_nc_x16:
      return xnn_reshape_copy_nc_x16(
        opdata->operator_objects[index],
        opdata->batch_size,
        channels, input_stride, output_stride,
        threadpool);
    case xnn_operator_type_copy_nc_x32:
      return xnn_reshape_copy_nc_x32(
        opdata->operator_objects[index],
        opdata->batch_size,
        channels, input_stride, output_stride,
        threadpool);
    case xnn_operator_type_copy_nc_x8:
      return xnn_reshape_copy_nc_x8(
        opdata->operator_objects[index],
        opdata->batch_size,
        channels, input_stride, output_stride,
        threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status reshape_concatenate_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  enum xnn_status status;

  size_t num_inputs = opdata->num_inputs;
  assert(num_inputs <= XNN_MAX_OPERATOR_OBJECTS);
  uint32_t input_id[XNN_MAX_OPERATOR_OBJECTS];
  for (size_t i = 0; i < num_inputs; ++i) {
    input_id[i] = opdata->inputs[i];
    assert(input_id[i] != XNN_INVALID_VALUE_ID);
    assert(input_id[i] < num_values);
  }

  size_t input_channels[XNN_MAX_OPERATOR_OBJECTS];
  for (size_t i = 0; i < num_inputs; ++i) {
    input_channels[i] = 1;
  }

  int32_t axis = opdata->axis;
  if (axis < 0) {
    axis += values[input_id[0]].shape.num_dims;
  }
  size_t output_stride = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    for (size_t j = axis; j < values[input_id[0]].shape.num_dims; j++) {
      input_channels[i] *= values[input_id[i]].shape.dim[j];
    }
    output_stride += input_channels[i];
  }

  assert(opdata->num_outputs == 1);
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input0_value = values + input_id[0];
  struct xnn_value* output_value = values + output_id;

  output_value->shape.num_dims = input0_value->shape.num_dims;
  if (axis >= output_value->shape.num_dims) {
    xnn_log_error(
      "failed to reshape reshape operator operator with the output ID #%" PRIu32
      ": axis (%d) exceeds the number of dimensions (%zu)",
      output_id, axis, input0_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  memcpy(output_value->shape.dim, input0_value->shape.dim, input0_value->shape.num_dims * sizeof(size_t));
  size_t concatenated_elements = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    concatenated_elements += values[input_id[i]].shape.dim[axis];
  }
  output_value->shape.dim[axis] = concatenated_elements;
  opdata->batch_size = xnn_shape_multiply_leading_dims(&output_value->shape, axis);
  const size_t old_workspace_size = opdata->workspace_size;
  for (size_t i = 0; i < num_inputs; ++i) {
    status = reshape_concatenate_operator_helper(opdata, i, input_channels[i], input_channels[i], output_stride, threadpool);
    if (status != xnn_status_success) {
      return status;
    }
  }
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_concatenate_operator_helper(
  const void* input_data,
  void* output_data,
  const struct xnn_operator_data *opdata,
  size_t index,
  pthreadpool_t threadpool)
{
  // The output pointer of this operator is the sum of all channels of the earlier operators.
  size_t channels = 0;
  for (size_t i = 0; i < index; i++) {
    if (opdata->operator_objects[i]->state == xnn_run_state_skip) {
      continue;
    }
    channels += opdata->operator_objects[i]->channels;
  }

  switch (opdata->operator_objects[index]->type) {
    case xnn_operator_type_copy_nc_x16:
      return xnn_setup_copy_nc_x16(
        opdata->operator_objects[index],
        input_data,
        (uint16_t*) output_data + channels);
    case xnn_operator_type_copy_nc_x32:
      return xnn_setup_copy_nc_x32(
        opdata->operator_objects[index],
        input_data,
        (uint32_t*) output_data + channels);
    case xnn_operator_type_copy_nc_x8:
      return xnn_setup_copy_nc_x8(
        opdata->operator_objects[index],
        input_data,
        (uint8_t*) output_data + channels);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status setup_concatenate_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  size_t num_inputs = opdata->num_inputs;
  uint32_t input_id[XNN_MAX_OPERATOR_OBJECTS];
  for (size_t i = 0; i < num_inputs; ++i) {
    input_id[i] = opdata->inputs[i];
    assert(input_id[i] != XNN_INVALID_VALUE_ID);
    assert(input_id[i] < num_values);
  }

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input_value[XNN_MAX_OPERATOR_OBJECTS];
  const void * input_data[XNN_MAX_OPERATOR_OBJECTS];
  for (size_t i = 0; i < num_inputs; ++i) {
    input_value[i] = values + input_id[i];
    input_data[i] = input_value[i]->data;
    assert(input_data[i] != NULL);
  }

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  enum xnn_status status;
  for (size_t i = 0; i < num_inputs; ++i) {
    status = setup_concatenate_operator_helper(input_data[i], output_data, opdata, i, threadpool);
    if (status != xnn_status_success) {
      return status;
    }
  }
  return xnn_status_success;
}

enum xnn_status check_input_value(
  xnn_subgraph_t subgraph,
  int32_t axis,
  uint32_t input_id,
  uint32_t output_id,
  size_t nth,
  enum xnn_node_type node_type)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_nth_input_node_id(node_type, input_id, subgraph->num_values, nth)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(node_type, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
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

enum xnn_status xnn_define_concatenate(
  xnn_subgraph_t subgraph,
  int32_t axis,
  size_t num_inputs,
  const uint32_t* inputs,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_concatenate)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_concatenate, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];

  status = xnn_subgraph_check_output_type_dense(xnn_node_type_concatenate, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  for (size_t i = 0; i < num_inputs; i++) {
    status = check_input_value(subgraph, axis, inputs[i], output_id, i + 1, xnn_node_type_concatenate);
    if (status != xnn_status_success) {
      return status;
    }
  }

  if (num_inputs > XNN_MAX_OPERATOR_OBJECTS) {
    xnn_log_error(
      "failed to define %s operator with %zu inputs: number of inputs (%zu) exceeds the supported maximum (%zu)",
      xnn_node_type_to_string(xnn_node_type_concatenate), num_inputs, num_inputs, (size_t) XNN_MAX_OPERATOR_OBJECTS);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < num_inputs; i++) {
    status = check_datatype_copyable(subgraph, inputs[i], output_id, "ith", xnn_node_type_concatenate);
    if (status != xnn_status_success) {
      return status;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.concatenate.axis = axis;
  node->type = xnn_node_type_concatenate;
  node->num_inputs = num_inputs;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_concatenate_operator;
  node->reshape = reshape_concatenate_operator;
  node->setup = setup_concatenate_operator;

  for (size_t i = 0; i < num_inputs; ++i) {
    node->inputs[i] = inputs[i];
  }

  return xnn_status_success;
  }
