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


static enum xnn_status create_max_pooling_operator(
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
        &opdata->operator_objects[0]);
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
        &opdata->operator_objects[0]);
      break;
#ifndef XNN_NO_S8_OPERATORS
    case xnn_compute_type_qs8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(node->activation.output_max, output_scale, output_zero_point);
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
        &opdata->operator_objects[0]);
      break;
    }
#endif  // !defined(XNN_NO_S8_OPERATORS)
#ifndef XNN_NO_U8_OPERATORS
    case xnn_compute_type_qu8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const uint8_t output_min = xnn_qu8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const uint8_t output_max = xnn_qu8_quantize(node->activation.output_max, output_scale, output_zero_point);
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
        &opdata->operator_objects[0]);
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

  switch (opdata->operator_objects[0]->type) {
#ifndef XNN_NO_F16_OPERATORS
    case xnn_operator_type_max_pooling_nhwc_f16:
      return xnn_setup_max_pooling2d_nhwc_f16(
        opdata->operator_objects[0],
        opdata->batch_size,
        opdata->input_height,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_operator_type_max_pooling_nhwc_f32:
      return xnn_setup_max_pooling2d_nhwc_f32(
        opdata->operator_objects[0],
        opdata->batch_size,
        opdata->input_height,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
#ifndef XNN_NO_S8_OPERATORS
    case xnn_operator_type_max_pooling_nhwc_s8:
      return xnn_setup_max_pooling2d_nhwc_s8(
        opdata->operator_objects[0],
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
        opdata->operator_objects[0],
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
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_max_pooling_2d)) != xnn_status_success) {
    return status;
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

  if (stride_height > pooling_height) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 " stride height: must be less than pooling height %" PRIu32,
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), stride_height, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (stride_width > pooling_width) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 " stride width: must be less than pooling width %" PRIu32,
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), stride_width, pooling_width);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_min_max(xnn_node_type_max_pooling_2d, output_min, output_max);
  if (status != xnn_status_success) {
    return status;
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

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_max_pooling_2d, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_max_pooling_2d, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
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

  status = xnn_subgraph_check_output_node_id(xnn_node_type_max_pooling_2d, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_max_pooling_2d, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
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

  status = xnn_subgraph_check_datatype_matches(
    xnn_node_type_max_pooling_2d, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

#if !defined(XNN_NO_S8_OPERATORS) || !defined(XNN_NO_U8_OPERATORS)
  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_max_pooling_2d, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
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
