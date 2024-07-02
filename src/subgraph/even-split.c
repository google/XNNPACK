// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>  // For size_t.
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/allocation-type.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

static enum xnn_status create_even_split_operator_helper(
    const uint32_t output_id,
    const struct xnn_node* node,
    struct xnn_operator_data* opdata,
    size_t index)
{
  if (output_id == XNN_INVALID_VALUE_ID) {
    // Node's output value has been optimized away, don't even create operator object.
    return xnn_status_success;
  }

  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      return xnn_create_copy_nc_x16(
          node->flags, &opdata->operator_objects[index]);
    case xnn_compute_type_fp32:
      return xnn_create_copy_nc_x32(
          node->flags, &opdata->operator_objects[index]);
    case xnn_compute_type_qs8:
    case xnn_compute_type_qu8:
      return xnn_create_copy_nc_x8(
          node->flags, &opdata->operator_objects[index]);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_even_split_n_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  size_t num_splits,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);
  assert(node->num_outputs == num_splits);
  uint32_t output_id[XNN_MAX_OPERATOR_OBJECTS];
  for (size_t i = 0; i < num_splits; ++i) {
    output_id[i] = opdata->outputs[i];
    assert(output_id[i] != XNN_INVALID_VALUE_ID);
    assert(output_id[i] < num_values);
    if (values[output_id[i]].type == xnn_value_type_invalid) {
      output_id[i] = XNN_INVALID_VALUE_ID;
    }
  }

  const int32_t axis = node->params.even_split.axis;
  opdata->axis = axis;
  enum xnn_status status;
  for (size_t i = 0; i < num_splits; ++i) {
    status = create_even_split_operator_helper(output_id[i], node, opdata, i);
    if (status != xnn_status_success) {
      return status;
    }
  }

  return status;
}

static enum xnn_status create_even_split2_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  return create_even_split_n_operator(node, values, num_values, opdata, code_cache, /*num_splits=*/2, weights_cache);
}

static enum xnn_status create_even_split3_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  return create_even_split_n_operator(node, values, num_values, opdata, code_cache, /*num_splits=*/3, weights_cache);
}

static enum xnn_status create_even_split4_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  return create_even_split_n_operator(node, values, num_values, opdata, code_cache, /*num_splits=*/4, weights_cache);
}

static enum xnn_status reshape_even_split_operator_helper(
  const struct xnn_value* values,
  const uint32_t num_values,
  struct xnn_operator_data* opdata,
  size_t index,
  size_t num_splits,
  int32_t axis,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const uint32_t output_id = opdata->outputs[index];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  if (values[output_id].allocation_type == xnn_allocation_type_invalid) {
    assert(opdata->operator_objects[index] == NULL);
    // output_id was removed during optimization.
    return xnn_status_success;
  }
  const size_t input_stride = xnn_shape_multiply_trailing_dims(&values[input_id].shape, axis);
  assert(input_stride % num_splits == 0);
  const size_t channels = input_stride / num_splits;
  const size_t output_stride = channels;

  switch (opdata->operator_objects[index]->type) {
    case xnn_operator_type_copy_nc_x16:
      return xnn_reshape_copy_nc_x16(
        opdata->operator_objects[index], opdata->batch_size, channels, input_stride, output_stride, threadpool);
    case xnn_operator_type_copy_nc_x32:
      return xnn_reshape_copy_nc_x32(
        opdata->operator_objects[index], opdata->batch_size, channels, input_stride, output_stride, threadpool);
    case xnn_operator_type_copy_nc_x8:
      return xnn_reshape_copy_nc_x8(
        opdata->operator_objects[index], opdata->batch_size, channels, input_stride, output_stride, threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status reshape_even_split_n_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  size_t num_splits,
  pthreadpool_t threadpool)
{
  enum xnn_status status = xnn_status_success;

  assert(opdata->num_inputs == 1);
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const struct xnn_value* input_value = values + input_id;

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
  opdata->batch_size = xnn_shape_multiply_leading_dims(&input_value->shape, axis);

  const size_t axis_elements = input_value->shape.dim[axis] / num_splits;
  const size_t old_workspace_size = opdata->workspace_size;
  bool reallocation_required = false;
  for (size_t i = 0; i < num_splits; ++i) {
    status = reshape_even_split_operator_helper(values, num_values, opdata, i, num_splits, axis, threadpool);
    if (status != xnn_status_success) {
      return status;
    }
    const uint32_t output_n_id = opdata->outputs[i];
    assert(output_n_id != XNN_INVALID_VALUE_ID);
    assert(output_n_id < num_values);
    struct xnn_value* output_n_value = values + output_n_id;
    if (output_n_value->allocation_type == xnn_allocation_type_invalid) {
      // output_id was removed during optimization.
      continue;
    }
    memcpy(output_n_value->shape.dim, input_value->shape.dim, input_value->shape.num_dims * sizeof(size_t));
    output_n_value->shape.num_dims = input_value->shape.num_dims;
    output_n_value->shape.dim[axis] = axis_elements;
    const size_t new_size = xnn_tensor_get_size(output_n_value);
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

static enum xnn_status reshape_even_split2_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return reshape_even_split_n_operator(opdata, values, num_values, /*num_splits=*/2, threadpool);
}

static enum xnn_status reshape_even_split3_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return reshape_even_split_n_operator(opdata, values, num_values, /*num_splits=*/3, threadpool);
}

static enum xnn_status reshape_even_split4_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return reshape_even_split_n_operator(opdata, values, num_values, /*num_splits=*/4, threadpool);
}

static enum xnn_status setup_even_split_operator_helper(
  const struct xnn_value* values,
  const uint32_t num_values,
  const struct xnn_operator_data* opdata,
  size_t index,
  const void* input_data,
  pthreadpool_t threadpool)
{
  const uint32_t output_id = opdata->outputs[index];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  if (values[output_id].allocation_type == xnn_allocation_type_invalid) {
    assert(opdata->operator_objects[index] == NULL);
    // output_id was removed during optimization.
    return xnn_status_success;
  }

  const size_t channels = opdata->operator_objects[index]->channels;

  assert(output_id < num_values);
  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[index]->type) {
    case xnn_operator_type_copy_nc_x16:
      return xnn_setup_copy_nc_x16(
        opdata->operator_objects[index], (const uint16_t*) input_data + index * channels,
        output_data);
    case xnn_operator_type_copy_nc_x32:
      return xnn_setup_copy_nc_x32(
        opdata->operator_objects[index], (const uint32_t*) input_data + index * channels,
        output_data);
    case xnn_operator_type_copy_nc_x8:
      return xnn_setup_copy_nc_x8(
        opdata->operator_objects[index], (const uint8_t*) input_data + index * channels,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status setup_even_split_n_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  size_t num_splits,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const struct xnn_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  enum xnn_status status = xnn_status_success;

  for (size_t i = 0; i < num_splits; ++i) {
    status = setup_even_split_operator_helper(values, num_values, opdata, i, input_data, threadpool);
    if (status != xnn_status_success) {
      return status;
    }
  }

  return status;
}

static enum xnn_status setup_even_split2_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return setup_even_split_n_operator(opdata, values, num_values, /*num_splits=*/2, threadpool);;
}

static enum xnn_status setup_even_split3_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return setup_even_split_n_operator(opdata, values, num_values, /*num_splits=*/3, threadpool);;
}

static enum xnn_status setup_even_split4_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return setup_even_split_n_operator(opdata, values, num_values, /*num_splits=*/4, threadpool);;
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

enum xnn_status check_output_compute_type(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  const char* nth,
  enum xnn_node_type node_type)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];
  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (input_value->quantization.zero_point != output_value->quantization.zero_point) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching quantization zero point across the input (%" PRId32 ") and the %s output (%" PRId32 ")",
      xnn_node_type_to_string(node_type), input_id, output_id,
      input_value->quantization.zero_point, nth, output_value->quantization.zero_point);
    return xnn_status_invalid_parameter;
  }
  if (input_value->quantization.scale != output_value->quantization.scale) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching quantization scale across the input (%.7g) and the %s output (%.7g)",
      xnn_node_type_to_string(node_type), input_id, output_id, input_value->quantization.scale,
      nth, output_value->quantization.scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_define_even_split_n(
  enum xnn_node_type node_type,
  xnn_subgraph_t subgraph,
  int32_t split_dim,
  uint32_t input_id,
  size_t num_outputs,
  const uint32_t* output_ids,
  uint32_t flags)
{
  assert(num_outputs > 1);
  assert(num_outputs < 5);

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

  status = check_output_value(subgraph, split_dim, input_id, output_ids[0], "first", node_type);
  if (status != xnn_status_success) {
    return status;
  }
  status = check_output_value(subgraph, split_dim, input_id, output_ids[1], "second", node_type);
  if (status != xnn_status_success) {
    return status;
  }

  if (num_outputs > 2) {
    status = check_output_value(subgraph, split_dim, input_id, output_ids[2], "third", node_type);
    if (status != xnn_status_success) {
      return status;
    }
  }
  if (num_outputs > 3) {
    status = check_output_value(subgraph, split_dim, input_id, output_ids[3], "fourth", node_type);
    if (status != xnn_status_success) {
      return status;
    }
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (input_value->datatype) {
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
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), input_id, xnn_datatype_to_string(input_value->datatype),
        input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (compute_type == xnn_compute_type_qs8 || compute_type == xnn_compute_type_qu8) {
    check_output_compute_type(subgraph, input_id, output_ids[0], "first", node_type);
    check_output_compute_type(subgraph, input_id, output_ids[1], "second", node_type);
    if (num_outputs > 2) {
      check_output_compute_type(subgraph, input_id, output_ids[2], "third", node_type);
    }
    if (num_outputs > 3) {
      check_output_compute_type(subgraph, input_id, output_ids[3], "fourth", node_type);
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.even_split.axis = split_dim;
  node->type = node_type;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = num_outputs;
  node->outputs[0] = output_ids[0];
  node->outputs[1] = output_ids[1];
  switch (num_outputs) {
    case 2:
      node->create = create_even_split2_operator;
      node->reshape = reshape_even_split2_operator;
      node->setup = setup_even_split2_operator;
      break;
    case 3:
      node->outputs[2] = output_ids[2];
      node->create = create_even_split3_operator;
      node->reshape = reshape_even_split3_operator;
      node->setup = setup_even_split3_operator;
      break;
    case 4:
      node->outputs[2] = output_ids[2];
      node->outputs[3] = output_ids[3];
      node->create = create_even_split4_operator;
      node->reshape = reshape_even_split4_operator;
      node->setup = setup_even_split4_operator;
      break;
    default:
      XNN_UNREACHABLE;
  }
  node->flags = flags;

  return xnn_status_success;
};

enum xnn_status xnn_define_even_split2(
  xnn_subgraph_t subgraph,
  int32_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t flags)
{
  const uint32_t output_ids[2] = { output1_id, output2_id };
  return xnn_define_even_split_n(
    xnn_node_type_even_split2, subgraph, split_dim, input_id, XNN_COUNT_OF(output_ids), output_ids, flags);
}

enum xnn_status xnn_define_even_split3(
  xnn_subgraph_t subgraph,
  int32_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t output3_id,
  uint32_t flags)
{
  const uint32_t output_ids[3] = { output1_id, output2_id, output3_id };
  return xnn_define_even_split_n(
    xnn_node_type_even_split3, subgraph, split_dim, input_id, XNN_COUNT_OF(output_ids), output_ids, flags);
}

enum xnn_status xnn_define_even_split4(
  xnn_subgraph_t subgraph,
  int32_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t output3_id,
  uint32_t output4_id,
  uint32_t flags)
{
  const uint32_t output_ids[4] = { output1_id, output2_id, output3_id, output4_id };
  return xnn_define_even_split_n(
    xnn_node_type_even_split4, subgraph, split_dim, input_id, XNN_COUNT_OF(output_ids), output_ids, flags);
}
