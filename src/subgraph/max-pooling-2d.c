// Copyright 2020 Google LLC
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


static enum xnn_status create_max_pooling_operator(
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

  const size_t channel_dim = values[input_id].shape.dim[3];
  assert(channel_dim == values[output_id].shape.dim[3]);

  enum xnn_status status;
  switch (node->compute_type) {
#ifndef XNN_NO_F16_OPERATORS
    case xnn_compute_type_fp16:
      status = xnn_create_max_pooling2d_nhwc_f16(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        node->activation.output_min,
        node->activation.output_max,
        node->flags,
        &opdata->operator_object);
      break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_compute_type_fp32:
      status = xnn_create_max_pooling2d_nhwc_f32(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        node->activation.output_min,
        node->activation.output_max,
        node->flags,
        &opdata->operator_object);
      break;
#ifndef XNN_NO_S8_OPERATORS
    case xnn_compute_type_qs8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const int8_t output_min =
        (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
      const int8_t output_max =
        (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
      status = xnn_create_max_pooling2d_nhwc_s8(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        output_min,
        output_max,
        node->flags,
        &opdata->operator_object);
      break;
    }
#endif  // !defined(XNN_NO_S8_OPERATORS)
#ifndef XNN_NO_U8_OPERATORS
    case xnn_compute_type_qu8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const uint8_t output_min =
        (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
      const uint8_t output_max =
        (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
      status = xnn_create_max_pooling2d_nhwc_u8(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        output_min,
        output_max,
        node->flags,
        &opdata->operator_object);
      break;
    }
#endif  // !defined(XNN_NO_U8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    opdata->batch_size = values[input_id].shape.dim[0];
    opdata->input_height = values[input_id].shape.dim[1];
    opdata->input_width = values[input_id].shape.dim[2];
    opdata->inputs[0] = input_id;
    opdata->outputs[0] = output_id;
  }
  return status;
}

static enum xnn_status setup_max_pooling_operator(
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
#ifndef XNN_NO_F16_OPERATORS
    case xnn_operator_type_max_pooling_nhwc_f16:
      return xnn_setup_max_pooling2d_nhwc_f16(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_height,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_operator_type_max_pooling_nhwc_f32:
      return xnn_setup_max_pooling2d_nhwc_f32(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_height,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
#ifndef XNN_NO_S8_OPERATORS
    case xnn_operator_type_max_pooling_nhwc_s8:
      return xnn_setup_max_pooling2d_nhwc_s8(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_height,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
#endif  // !defined(XNN_NO_S8_OPERATORS)
#ifndef XNN_NO_U8_OPERATORS
    case xnn_operator_type_max_pooling_nhwc_u8:
      return xnn_setup_max_pooling2d_nhwc_u8(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_height,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
#endif  // !defined(XNN_NO_U8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_max_pooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d));
    return xnn_status_uninitialized;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to define %s operator with 1 pooling element: 1x1 pooling is meaningless",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d));
    return xnn_status_invalid_parameter;
  }

  if (stride_height == 0 || stride_width == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " stride: stride dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), stride_width, stride_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_height == 0 || dilation_width == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define %s with NaN output lower bound: lower bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define %s with NaN output upper bound: upper bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define %s with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to define %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d),
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      return xnn_status_invalid_parameter;
    }
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
#ifndef XNN_NO_S8_OPERATORS
    case xnn_datatype_qint8:
#endif  // !defined(XNN_NO_S8_OPERATORS)
#ifndef XNN_NO_U8_OPERATORS
    case xnn_datatype_quint8:
#endif  // !defined(XNN_NO_U8_OPERATORS)
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), output_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output_value->datatype) {
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
#ifndef XNN_NO_S8_OPERATORS
    case xnn_datatype_qint8:
      compute_type = xnn_compute_type_qs8;
      break;
#endif  // !defined(XNN_NO_S8_OPERATORS)
#ifndef XNN_NO_U8_OPERATORS
    case xnn_datatype_quint8:
      compute_type = xnn_compute_type_qu8;
      break;
#endif  // !defined(XNN_NO_U8_OPERATORS)
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (input_value->datatype != output_value->datatype) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching datatypes across input (%s) and output (%s)",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), input_id, output_id,
      xnn_datatype_to_string(input_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }

#if !defined(XNN_NO_S8_OPERATORS) || !defined(XNN_NO_U8_OPERATORS)
  if (output_value->datatype == xnn_datatype_qint8 || output_value->datatype == xnn_datatype_quint8) {
    if (input_value->quantization.zero_point != output_value->quantization.zero_point) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatching zero point quantization parameter across input (%"PRId32") and output (%"PRId32")",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d), input_id, output_id,
        input_value->quantization.zero_point, output_value->quantization.zero_point);
      return xnn_status_invalid_parameter;
    }
    if (input_value->quantization.scale != output_value->quantization.scale) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatching zero point quantization parameter across input (%.7g) and output (%.7g)",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d), input_id, output_id,
        input_value->quantization.scale, output_value->quantization.scale);
      return xnn_status_invalid_parameter;
    }
  }
#endif  // !defined(XNN_NO_S8_OPERATORS) || !defined(XNN_NO_U8_OPERATORS)

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_max_pooling_2d;
  node->compute_type = compute_type;
  node->params.pooling_2d.padding_top = input_padding_top;
  node->params.pooling_2d.padding_right = input_padding_right;
  node->params.pooling_2d.padding_bottom = input_padding_bottom;
  node->params.pooling_2d.padding_left = input_padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->params.pooling_2d.stride_height = stride_height;
  node->params.pooling_2d.stride_width = stride_width;
  node->params.pooling_2d.dilation_height = dilation_height;
  node->params.pooling_2d.dilation_width = dilation_width;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_max_pooling_operator;
  node->setup = setup_max_pooling_operator;

  return xnn_status_success;
}
