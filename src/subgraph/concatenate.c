// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>

static enum xnn_status create_concatenate_operator_helper(
  const struct xnn_node *node,
  struct xnn_operator_data *opdata,
  size_t index)
{
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      return xnn_create_copy_nc_x16(node->flags, &opdata->operator_objects[index]);
    case xnn_compute_type_fp32:
      return xnn_create_copy_nc_x32(node->flags, &opdata->operator_objects[index]);
    case xnn_compute_type_qs8:
    case xnn_compute_type_qu8:
      return xnn_create_copy_nc_x8(node->flags, &opdata->operator_objects[index]);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_concatenate_n_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  size_t num_inputs,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  enum xnn_status status;
  const size_t axis = node->params.concatenate.axis;
  opdata->axis = axis;
  for (size_t i = 0; i < num_inputs; ++i) {
    status = create_concatenate_operator_helper(node, opdata, i);
    if (status != xnn_status_success) {
      return status;
    }
  }

  return status;
}

static enum xnn_status create_concatenate2_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  return create_concatenate_n_operator(node, values, num_values, /*num_inputs=*/2, opdata, code_cache, weights_cache);
}

static enum xnn_status create_concatenate3_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  return create_concatenate_n_operator(node, values, num_values, /*num_inputs=*/3, opdata, code_cache, weights_cache);
}

static enum xnn_status create_concatenate4_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  return create_concatenate_n_operator(node, values, num_values, /*num_inputs=*/4, opdata, code_cache, weights_cache);
}

static enum xnn_status create_concatenate5_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  return create_concatenate_n_operator(node, values, num_values, /*num_inputs=*/5, opdata, code_cache, weights_cache);
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

static enum xnn_status reshape_concatenate_n_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  size_t num_inputs,
  pthreadpool_t threadpool)
{
  enum xnn_status status;

  assert(opdata->num_inputs == num_inputs);
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

  const size_t axis = opdata->axis;
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
      ": axis (%zu) exceeds the number of dimensions (%zu)",
      output_id, axis, input0_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  memcpy(output_value->shape.dim, input0_value->shape.dim, input0_value->shape.num_dims * sizeof(size_t));
  size_t concatenated_elements = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    concatenated_elements += values[input_id[i]].shape.dim[axis];
  }
  output_value->shape.dim[axis] = concatenated_elements;
  opdata->batch_size = xnn_shape_multiply_leading_dims(&output_value->shape, opdata->axis);
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

static enum xnn_status reshape_concatenate2_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return reshape_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/2, threadpool);
}

static enum xnn_status reshape_concatenate3_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return reshape_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/3, threadpool);
}

static enum xnn_status reshape_concatenate4_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return reshape_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/4, threadpool);
}

static enum xnn_status reshape_concatenate5_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return reshape_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/5, threadpool);
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

static enum xnn_status setup_concatenate_n_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  size_t num_inputs,
  pthreadpool_t threadpool)
{
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

static enum xnn_status setup_concatenate2_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return setup_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/2, threadpool);
}

static enum xnn_status setup_concatenate3_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return setup_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/3, threadpool);
}

static enum xnn_status setup_concatenate4_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return setup_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/4, threadpool);
}

static enum xnn_status setup_concatenate5_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  return setup_concatenate_n_operator(opdata, values, num_values, /*num_inputs=*/5, threadpool);
}

enum xnn_status check_input_value(
  xnn_subgraph_t subgraph,
  size_t axis,
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

enum xnn_status check_input_compute_type(
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
        ": mismatching quantization zero point across the %s input (%" PRId32 ") and the output (%" PRId32 ")",
        xnn_node_type_to_string(node_type), input_id, output_id,
        nth, input_value->quantization.zero_point, output_value->quantization.zero_point);
    return xnn_status_invalid_parameter;
  }
  if (input_value->quantization.scale != output_value->quantization.scale) {
    xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatching quantization scale across the %s input (%.7g) and the output (%.7g)",
        xnn_node_type_to_string(node_type), input_id, output_id,
        nth, input_value->quantization.scale, output_value->quantization.scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_define_concatenate_n(
  enum xnn_node_type node_type,
  xnn_subgraph_t subgraph,
  size_t axis,
  size_t num_inputs,
  uint32_t* input_ids,
  uint32_t output_id,
  uint32_t flags)
{
  assert(num_inputs >= 2);
  assert(num_inputs <= 5);

  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(node_type)) != xnn_status_success) {
    return status;
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

  for (size_t i = 0; i < num_inputs; i++) {
    status = check_input_value(subgraph, axis, input_ids[i], output_id, i+1, node_type);
    if (status != xnn_status_success) {
      return status;
    }
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

  if (compute_type == xnn_compute_type_qs8 || compute_type == xnn_compute_type_qu8) {
    status = check_input_compute_type(subgraph, input_ids[0], output_id, "first", node_type);
    if (status != xnn_status_success) {
      return status;
    }
    status = check_input_compute_type(subgraph, input_ids[1], output_id, "second", node_type);
    if (status != xnn_status_success) {
      return status;
    }
  }
  if (num_inputs > 2) {
    status = check_input_compute_type(subgraph, input_ids[2], output_id, "third", node_type);
    if (status != xnn_status_success) {
      return status;
    }
  }
  if (num_inputs > 3) {
    status = check_input_compute_type(subgraph, input_ids[3], output_id, "fourth", node_type);
    if (status !=  xnn_status_success) {
      return status;
    }
  }
  if (num_inputs > 4) {
    status = check_input_compute_type(subgraph, input_ids[4], output_id, "fifth", node_type);
    if (status !=  xnn_status_success) {
      return status;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.concatenate.axis = axis;
  node->type = node_type;
  node->compute_type = compute_type;
  node->num_inputs = num_inputs;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  switch (num_inputs) {
    case 2:
      node->create = create_concatenate2_operator;
      node->reshape = reshape_concatenate2_operator;
      node->setup = setup_concatenate2_operator;
      break;
    case 3:
      node->create = create_concatenate3_operator;
      node->reshape = reshape_concatenate3_operator;
      node->setup = setup_concatenate3_operator;
      break;
    case 4:
      node->create = create_concatenate4_operator;
      node->reshape = reshape_concatenate4_operator;
      node->setup = setup_concatenate4_operator;
      break;
    case 5:
      node->create = create_concatenate5_operator;
      node->reshape = reshape_concatenate5_operator;
      node->setup = setup_concatenate5_operator;
      break;
    default:
      XNN_UNREACHABLE;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    node->inputs[i] = input_ids[i];
  }

  return xnn_status_success;
}

enum xnn_status xnn_define_concatenate2(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags)
{
  uint32_t input_ids[2] = { input1_id, input2_id };
  return xnn_define_concatenate_n(
    xnn_node_type_concatenate2, subgraph, axis, XNN_COUNT_OF(input_ids), input_ids, output_id, flags);
}

enum xnn_status xnn_define_concatenate3(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t input3_id,
  uint32_t output_id,
  uint32_t flags)
{
  uint32_t input_ids[3] = { input1_id, input2_id, input3_id };
  return xnn_define_concatenate_n(
    xnn_node_type_concatenate3, subgraph, axis, XNN_COUNT_OF(input_ids), input_ids, output_id, flags);
}

enum xnn_status xnn_define_concatenate4(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t input3_id,
  uint32_t input4_id,
  uint32_t output_id,
  uint32_t flags)
{
  uint32_t input_ids[4] = { input1_id, input2_id, input3_id, input4_id };
  return xnn_define_concatenate_n(
    xnn_node_type_concatenate4, subgraph, axis, XNN_COUNT_OF(input_ids), input_ids, output_id, flags);
}

enum xnn_status xnn_define_concatenate5(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t input3_id,
  uint32_t input4_id,
  uint32_t input5_id,
  uint32_t output_id,
  uint32_t flags)
{
  uint32_t input_ids[5] = { input1_id, input2_id, input3_id, input4_id, input5_id };
  return xnn_define_concatenate_n(
    xnn_node_type_concatenate5, subgraph, axis, XNN_COUNT_OF(input_ids), input_ids, output_id, flags);
}
