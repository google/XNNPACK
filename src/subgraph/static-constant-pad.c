// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/operator.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>


static enum xnn_status create_constant_pad_operator(
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
      status = xnn_create_constant_pad_nd_x16(
        &node->params.static_pad.padding_value,
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      status = xnn_create_constant_pad_nd_x32(
        &node->params.static_pad.padding_value,
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_qs8:
    case xnn_compute_type_qu8:
      status = xnn_create_constant_pad_nd_x8(
        &node->params.static_pad.padding_value,
        node->flags,
        &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    memcpy(opdata->pre_paddings, node->params.static_pad.pre_paddings, sizeof(size_t) * XNN_MAX_TENSOR_DIMS);
    memcpy(opdata->post_paddings, node->params.static_pad.post_paddings, sizeof(size_t) * XNN_MAX_TENSOR_DIMS);
  }
  return status;
}

static enum xnn_status reshape_constant_pad_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  enum xnn_status status = xnn_status_invalid_state;
  const size_t old_workspace_size = opdata->workspace_size;
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  struct xnn_value* input_value = values + input_id;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_constant_pad_nd_x8:
      status = xnn_reshape_constant_pad_nd_x8(
        opdata->operator_objects[0],
        input_value->shape.num_dims,
        input_value->shape.dim,
        opdata->pre_paddings,
        opdata->post_paddings,
        threadpool);
      break;
    case xnn_operator_type_constant_pad_nd_x16:
      status = xnn_reshape_constant_pad_nd_x16(
        opdata->operator_objects[0],
        input_value->shape.num_dims,
        input_value->shape.dim,
        opdata->pre_paddings,
        opdata->post_paddings,
        threadpool);
      break;
    case xnn_operator_type_constant_pad_nd_x32:
      status = xnn_reshape_constant_pad_nd_x32(
        opdata->operator_objects[0],
        input_value->shape.num_dims,
        input_value->shape.dim,
        opdata->pre_paddings,
        opdata->post_paddings,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);
  struct xnn_value* output_value = values + output_id;
  output_value->shape.num_dims = input_value->shape.num_dims;
  for (size_t i = 0; i < input_value->shape.num_dims; ++i) {
    output_value->shape.dim[i] = input_value->shape.dim[i] + opdata->pre_paddings[i] + opdata->post_paddings[i];
  }
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_constant_pad_operator(
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
    case xnn_operator_type_constant_pad_nd_x8:
      return xnn_setup_constant_pad_nd_x8(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_constant_pad_nd_x16:
      return xnn_setup_constant_pad_nd_x16(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_constant_pad_nd_x32:
      return xnn_setup_constant_pad_nd_x32(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_static_constant_pad(
  xnn_subgraph_t subgraph,
  const size_t* pre_paddings,
  const size_t* post_paddings,
  float padding_value,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_static_constant_pad)) != xnn_status_success) {
    return status;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_static_constant_pad), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_static_constant_pad, input_id, input_value);
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
        xnn_node_type_to_string(xnn_node_type_static_constant_pad), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_static_constant_pad, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_static_constant_pad, output_id, output_value);
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
        xnn_node_type_to_string(xnn_node_type_static_constant_pad), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_datatype_matches(
    xnn_node_type_static_constant_pad, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_static_constant_pad, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  const size_t num_dims = subgraph->values[input_id].shape.num_dims;
  memcpy(&node->params.static_pad.pre_paddings, pre_paddings, num_dims * sizeof(size_t));
  memcpy(&node->params.static_pad.post_paddings, post_paddings, num_dims * sizeof(size_t));
  switch (output_value->datatype) {
    case xnn_datatype_fp32:
      node->params.static_pad.padding_value = float_as_uint32(padding_value);
      break;
    case xnn_datatype_fp16:
      node->params.static_pad.padding_value = fp16_ieee_from_fp32_value(padding_value);
      break;
    case xnn_datatype_qint8:
    {
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      node->params.static_pad.padding_value = xnn_qs8_quantize(padding_value, output_scale, output_zero_point);
      break;
    }
    case xnn_datatype_quint8:
    {
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      node->params.static_pad.padding_value = xnn_qu8_quantize(padding_value, output_scale, output_zero_point);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  node->type = xnn_node_type_static_constant_pad;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_constant_pad_operator;
  node->reshape = reshape_constant_pad_operator;
  node->setup = setup_constant_pad_operator;

  return xnn_status_success;
}
