// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>



static enum xnn_status create_convert_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const size_t num_input_dims = values[input_id].shape.num_dims;
  const size_t channel_dim = num_input_dims == 0 ? 1 : values[input_id].shape.dim[num_input_dims - 1];

  enum xnn_status status = xnn_status_uninitialized;
  switch (node->compute_type) {
    case xnn_compute_type_fp32_to_fp16:
      status = xnn_create_convert_nc_f32_f16(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        node->flags,
        &opdata->operator_object);
      break;
    case xnn_compute_type_fp32_to_qs8:
      status = xnn_create_convert_nc_f32_qs8(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        values[output_id].quantization.scale,
        (int8_t) values[output_id].quantization.zero_point,
        INT8_MIN, INT8_MAX,
        node->flags,
        &opdata->operator_object);
      break;
    case xnn_compute_type_fp32_to_qu8:
      status = xnn_create_convert_nc_f32_qu8(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        values[output_id].quantization.scale,
        (uint8_t) values[output_id].quantization.zero_point,
        0, UINT8_MAX,
        node->flags,
        &opdata->operator_object);
      break;
    case xnn_compute_type_fp16_to_fp32:
      status = xnn_create_convert_nc_f16_f32(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        node->flags,
        &opdata->operator_object);
      break;
    case xnn_compute_type_qs8_to_fp32:
      status = xnn_create_convert_nc_qs8_f32(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        values[input_id].quantization.scale,
        (int8_t) values[input_id].quantization.zero_point,
        node->flags,
        &opdata->operator_object);
      break;
    case xnn_compute_type_qu8_to_fp32:
      status = xnn_create_convert_nc_qu8_f32(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        values[input_id].quantization.scale,
        (uint8_t) values[input_id].quantization.zero_point,
        node->flags,
        &opdata->operator_object);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    opdata->batch_size = xnn_shape_multiply_non_channel_dims(&values[input_id].shape);
    opdata->inputs[0] = input_id;
    opdata->outputs[0] = output_id;
  }
  return status;
}

static enum xnn_status setup_convert_operator(
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

  switch (opdata->operator_object->type) {
    case xnn_operator_type_convert_nc_f32_f16:
      return xnn_setup_convert_nc_f32_f16(
        opdata->operator_object,
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_convert_nc_f32_qs8:
      return xnn_setup_convert_nc_f32_qs8(
        opdata->operator_object,
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_convert_nc_f32_qu8:
      return xnn_setup_convert_nc_f32_qu8(
        opdata->operator_object,
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_convert_nc_f16_f32:
      return xnn_setup_convert_nc_f16_f32(
        opdata->operator_object,
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_convert_nc_qs8_f32:
      return xnn_setup_convert_nc_qs8_f32(
        opdata->operator_object,
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_convert_nc_qu8_f32:
      return xnn_setup_convert_nc_qu8_f32(
        opdata->operator_object,
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

static inline enum xnn_compute_type validate_datatypes(
  enum xnn_datatype input_datatype,
  enum xnn_datatype output_datatype)
{
  switch (input_datatype) {
    case xnn_datatype_fp32:
      switch (output_datatype) {
        case xnn_datatype_fp16:
          return xnn_compute_type_fp32_to_fp16;
        case xnn_datatype_qint8:
          return xnn_compute_type_fp32_to_qs8;
        case xnn_datatype_quint8:
          return xnn_compute_type_fp32_to_qu8;
        default:
          break;
      }
      break;
    case xnn_datatype_fp16:
      if (output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp16_to_fp32;
      }
      break;
    case xnn_datatype_qint8:
      if (output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qs8_to_fp32;
      }
      break;
    case xnn_datatype_quint8:
      if (output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qu8_to_fp32;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

void xnn_init_convert_node(
  struct xnn_node* node,
  enum xnn_compute_type compute_type,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  node->type = xnn_node_type_convert;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_convert_operator;
  node->setup = setup_convert_operator;
}

enum xnn_status xnn_define_convert(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized",
      xnn_node_type_to_string(xnn_node_type_convert));
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_convert), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_convert), input_id, input_value->type);
    return xnn_status_invalid_parameter;
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
        xnn_node_type_to_string(xnn_node_type_convert), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_convert), output_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_convert), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_convert), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = validate_datatypes(input_value->datatype, output_value->datatype);
  if (compute_type == xnn_compute_type_invalid) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching datatypes across input (%s) and output (%s)",
      xnn_node_type_to_string(xnn_node_type_convert), input_id, output_id,
      xnn_datatype_to_string(input_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  xnn_init_convert_node(node, compute_type, input_id, output_id, flags);
  return xnn_status_success;
}
