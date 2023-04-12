// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>


static enum xnn_status create_fully_connected_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  const struct xnn_caches* caches)
{
  assert(node->num_inputs >= 2);
  assert(node->num_inputs <= 3);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const uint32_t kernel_id = node->inputs[1];
  assert(kernel_id != XNN_INVALID_VALUE_ID);
  assert(kernel_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const size_t num_input_elements = xnn_shape_multiply_all_dims(&values[node->inputs[0]].shape);
  size_t output_channels, input_channels;
  if (node->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    input_channels = values[node->inputs[1]].shape.dim[0];
    output_channels = values[node->inputs[1]].shape.dim[1];
  } else {
    output_channels = values[node->inputs[1]].shape.dim[0];
    input_channels = values[node->inputs[1]].shape.dim[1];
  }

  const void* kernel_data = values[kernel_id].fp32_data != NULL ? values[kernel_id].fp32_data : values[kernel_id].data;
  bool has_non_static_weights = (kernel_data == NULL);

  const void* bias_data = NULL;
  if (node->num_inputs > 2) {
    const uint32_t bias_id = node->inputs[2];
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);

    bias_data = values[bias_id].fp32_data != NULL ? values[bias_id].fp32_data : values[bias_id].data;
    has_non_static_weights |= (bias_data == NULL);
  }

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      assert(kernel_data != NULL);
      assert(!has_non_static_weights);
      status = xnn_create_fully_connected_nc_f16(
        input_channels,
        output_channels,
        input_channels /* input stride */,
        output_channels /* output stride */,
        kernel_data,
        bias_data,
        node->activation.output_min,
        node->activation.output_max,
        node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS,
        caches,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      if (has_non_static_weights) {
        status = xnn_create_dynamic_fully_connected_nc_f32(
          node->activation.output_min,
          node->activation.output_max,
          node->flags /* flags */,
          &opdata->operator_objects[0]);
      } else {
        status = xnn_create_fully_connected_nc_f32(
          input_channels,
          output_channels,
          input_channels /* input stride */,
          output_channels /* output stride */,
          kernel_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags /* flags */,
          caches,
          &opdata->operator_objects[0]);
      }
      break;
    case xnn_compute_type_qs8:
    {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qs8(
        input_channels,
        output_channels,
        input_channels /* input stride */,
        output_channels /* output stride */,
        (int8_t) values[input_id].quantization.zero_point,
        values[input_id].quantization.scale,
        values[kernel_id].quantization.scale,
        kernel_data,
        bias_data,
        (int8_t) output_zero_point,
        output_scale, output_min, output_max,
        node->flags /* flags */,
        caches,
        &opdata->operator_objects[0]);
      break;
    }
    case xnn_compute_type_qu8:
    {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const uint8_t output_min = xnn_qu8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const uint8_t output_max = xnn_qu8_quantize(node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qu8(
        input_channels,
        output_channels,
        input_channels /* input stride */,
        output_channels /* output stride */,
        (uint8_t) values[input_id].quantization.zero_point,
        values[input_id].quantization.scale,
        (uint8_t) values[kernel_id].quantization.zero_point,
        values[kernel_id].quantization.scale,
        kernel_data,
        bias_data,
        (uint8_t) output_zero_point,
        output_scale, output_min, output_max,
        node->flags /* flags */,
        caches,
        &opdata->operator_objects[0]);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    opdata->batch_size = num_input_elements / input_channels;
    opdata->input_channels = input_channels;
    opdata->output_channels = output_channels;
    opdata->inputs[0] = input_id;
    opdata->inputs[1] = has_non_static_weights ? kernel_id : XNN_INVALID_VALUE_ID;
    opdata->inputs[2] = has_non_static_weights && node->num_inputs == 3 ? node->inputs[2] : XNN_INVALID_VALUE_ID;
    opdata->outputs[0] = output_id;
  }
  return status;
}

static enum xnn_status setup_fully_connected_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const uint32_t kernel_id = opdata->inputs[1];
  const uint32_t bias_id = opdata->inputs[2];

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  const void* kernel_data = NULL;
  if (kernel_id != XNN_INVALID_VALUE_ID) {
    assert(kernel_id < num_blobs);
    const struct xnn_blob* kernel_blob = blobs + kernel_id;
    kernel_data = kernel_blob->data;
    assert(kernel_data != NULL);
  }

  const void* bias_data = NULL;
  if (bias_id != XNN_INVALID_VALUE_ID) {
    assert(bias_id < num_blobs);
    const struct xnn_blob* bias_blob = blobs + bias_id;
    bias_data = bias_blob->data;
    assert(bias_data != NULL);
  }

  const struct xnn_blob* output_blob = blobs + output_id;
  void* output_data = output_blob->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_dynamic_fully_connected_nc_f32:
      assert(kernel_data != NULL);
      return xnn_setup_dynamic_fully_connected_nc_f32(
        opdata->operator_objects[0],
        opdata->batch_size,
        opdata->input_channels, opdata->output_channels,
        opdata->input_channels, opdata->output_channels,
        input_data, kernel_data, bias_data, output_data,
        threadpool);
    case xnn_operator_type_fully_connected_nc_f16:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f16(
        opdata->operator_objects[0],
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_fully_connected_nc_f32:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f32(
        opdata->operator_objects[0],
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_fully_connected_nc_qs8:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qs8(
        opdata->operator_objects[0],
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    case xnn_operator_type_fully_connected_nc_qu8:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qu8(
        opdata->operator_objects[0],
        opdata->batch_size,
        input_data,
        output_data,
        threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

static inline enum xnn_compute_type validate_datatypes_with_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype kernel_datatype,
  enum xnn_datatype bias_datatype,
  enum xnn_datatype output_datatype)
{
  switch (kernel_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_fp32;
      }
      break;
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_qint8)
      {
        return xnn_compute_type_qs8;
      }
      break;
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_quint8)
      {
        return xnn_compute_type_qu8;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

static inline enum xnn_compute_type validate_datatypes_without_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype kernel_datatype,
  enum xnn_datatype output_datatype)
{
  switch (kernel_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      }
      break;
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 && output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qs8;
      }
      break;
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 && output_datatype == xnn_datatype_quint8) {
        return xnn_compute_type_qu8;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

enum xnn_status xnn_define_fully_connected(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t kernel_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_fully_connected)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_min_max(xnn_node_type_fully_connected, output_min, output_max);
  if (status != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_fully_connected, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_fully_connected, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (kernel_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_fully_connected), kernel_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* kernel_value = &subgraph->values[kernel_id];
  if (kernel_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_fully_connected), kernel_id, kernel_value->type);
    return xnn_status_invalid_parameter;
  }

  // Non-static kernel is supported, but only for some data types
  switch (kernel_value->datatype) {
    case xnn_datatype_fp32:
      break;  // non-static kernel is supported
    default:
      if (kernel_value->data == NULL) {
        xnn_log_error(
          "failed to define %s operator with filter ID #%" PRIu32 ": non-static Value",
          xnn_node_type_to_string(xnn_node_type_fully_connected), kernel_id);
        return xnn_status_invalid_parameter;
      }
      break;
  }

  switch (kernel_value->datatype) {
    case xnn_datatype_fp32:
      break;
    case xnn_datatype_qint8:
      if (kernel_value->quantization.zero_point != 0) {
        xnn_log_error(
          "failed to define %s operator with filter ID #%" PRIu32 ": unsupported quantization zero point %" PRId32 " for datatype %s",
          xnn_node_type_to_string(xnn_node_type_convolution_2d), kernel_id,
          kernel_value->quantization.zero_point, xnn_datatype_to_string(kernel_value->datatype));
      }
      break;
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), kernel_id,
        xnn_datatype_to_string(kernel_value->datatype), kernel_value->datatype);
      return xnn_status_invalid_parameter;
  }

  const struct xnn_value* bias_value = NULL;
  if (bias_id != XNN_INVALID_VALUE_ID) {
    if (bias_id >= subgraph->num_values) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": invalid Value ID",
        xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id);
      return xnn_status_invalid_parameter;
    }

    bias_value = &subgraph->values[bias_id];
    if (bias_value->type != xnn_value_type_dense_tensor) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id, bias_value->type);
      return xnn_status_invalid_parameter;
    }

    // Non-static bias is supported, but only for some data types
    switch (bias_value->datatype) {
      case xnn_datatype_fp32:
        break;  // non-static bias is supported
      default:
        if (bias_value->data == NULL) {
          xnn_log_error(
            "failed to define %s operator with bias ID #%" PRIu32 ": non-static Value",
            xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id);
          return xnn_status_invalid_parameter;
        }
        break;
    }

    switch (bias_value->datatype) {
      case xnn_datatype_fp32:
      case xnn_datatype_qint32:
        break;
      default:
        xnn_log_error(
          "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
          xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id,
          xnn_datatype_to_string(bias_value->datatype), bias_value->datatype);
        return xnn_status_invalid_parameter;
    }
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_fully_connected, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_fully_connected, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  if (bias_value != NULL) {
    compute_type = validate_datatypes_with_bias(
      input_value->datatype, kernel_value->datatype, bias_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", bias ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), bias (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id, kernel_id, bias_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(kernel_value->datatype),
        xnn_datatype_to_string(bias_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  } else {
    compute_type = validate_datatypes_without_bias(
      input_value->datatype, kernel_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id, kernel_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(kernel_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_fully_connected;
  node->compute_type = compute_type;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2 + (size_t) (bias_id != XNN_INVALID_VALUE_ID);
  node->inputs[0] = input_id;
  node->inputs[1] = kernel_id;
  node->inputs[2] = bias_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_fully_connected_operator;
  node->setup = setup_fully_connected_operator;

  return xnn_status_success;
}
