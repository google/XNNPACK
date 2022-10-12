// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/operator.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <fp16.h>

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
  const struct xnn_caches* caches)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  enum xnn_status status;
  switch (node->compute_type) {
#ifndef XNN_NO_F16_OPERATORS
    case xnn_compute_type_fp16:
      status = xnn_create_constant_pad_nd_x16(
        &node->params.static_pad.padding_value,
        node->flags,
        &opdata->operator_objects[0]);
      break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_compute_type_fp32:
      status = xnn_create_constant_pad_nd_x32(
        &node->params.static_pad.padding_value,
        node->flags,
        &opdata->operator_objects[0]);
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_compute_type_qs8:
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_compute_type_qu8:
#endif  // !defined(XNN_NO_QU8_OPERATORS)
#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
      status = xnn_create_constant_pad_nd_x8(
        &node->params.static_pad.padding_value,
        node->flags,
        &opdata->operator_objects[0]);
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    opdata->shape1 = values[input_id].shape;
    memcpy(opdata->pre_paddings, node->params.static_pad.pre_paddings, sizeof(size_t) * XNN_MAX_TENSOR_DIMS);
    memcpy(opdata->post_paddings, node->params.static_pad.post_paddings, sizeof(size_t) * XNN_MAX_TENSOR_DIMS);
    opdata->inputs[0] = input_id;
    opdata->outputs[0] = output_id;
  }
  return status;
}

static enum xnn_status setup_constant_pad_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  const struct xnn_blob* output_blob = blobs + output_id;
  void* output_data = output_blob->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    case xnn_operator_type_constant_pad_nd_x8:
      return xnn_setup_constant_pad_nd_x8(
        opdata->operator_objects[0],
        opdata->shape1.num_dims,
        opdata->shape1.dim,
        opdata->pre_paddings,
        opdata->post_paddings,
        input_data,
        output_data,
        threadpool);
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
#ifndef XNN_NO_F16_OPERATORS
    case xnn_operator_type_constant_pad_nd_x16:
      return xnn_setup_constant_pad_nd_x16(
        opdata->operator_objects[0],
        opdata->shape1.num_dims,
        opdata->shape1.dim,
        opdata->pre_paddings,
        opdata->post_paddings,
        input_data,
        output_data,
        threadpool);
      break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_operator_type_constant_pad_nd_x32:
      return xnn_setup_constant_pad_nd_x32(
        opdata->operator_objects[0],
        opdata->shape1.num_dims,
        opdata->shape1.dim,
        opdata->pre_paddings,
        opdata->post_paddings,
        input_data,
        output_data,
        threadpool);
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
    case xnn_datatype_fp32:
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
#endif  // !defined(XNN_NO_QU8_OPERATORS)
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
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
      compute_type = xnn_compute_type_qs8;
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
      compute_type = xnn_compute_type_qu8;
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
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

#if !defined(XNN_NO_QU8_OPERATORS) || !defined(XNN_NO_QS8_OPERATORS)
  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_static_constant_pad, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }
#endif  // !defined(XNN_NO_QU8_OPERATORS) || !defined(XNN_NO_QS8_OPERATORS)

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
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
    {
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      node->params.static_pad.padding_value = xnn_qs8_quantize(padding_value, output_scale, output_zero_point);
      break;
    }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
    {
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      node->params.static_pad.padding_value = xnn_qu8_quantize(padding_value, output_scale, output_zero_point);
      break;
    }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
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
  node->setup = setup_constant_pad_operator;

  return xnn_status_success;
}
