// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


static enum xnn_status create_deconvolution_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata)
{
  assert(node->num_inputs >= 2);
  assert(node->num_inputs <= 3);
  const bool use_bias = node->num_inputs >= 3;

  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const uint32_t filter_id = node->inputs[1];
  assert(filter_id != XNN_INVALID_VALUE_ID);
  assert(filter_id < num_values);

  const void* bias_data = NULL;
  if (use_bias) {
    const uint32_t bias_id = node->inputs[2];
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);

    bias_data = values[bias_id].data;
    assert(bias_data != NULL);
  }

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const void* filter_data = values[filter_id].data;
  assert(filter_data != NULL);

  enum xnn_status status = xnn_status_uninitialized;
  switch (node->compute_type) {
#ifndef XNN_NO_F16_OPERATORS
    case xnn_compute_type_fp16:
      status = xnn_create_deconvolution2d_nhwc_f16(
          node->params.deconvolution_2d.padding_top,
          node->params.deconvolution_2d.padding_right,
          node->params.deconvolution_2d.padding_bottom,
          node->params.deconvolution_2d.padding_left,
          node->params.deconvolution_2d.kernel_height,
          node->params.deconvolution_2d.kernel_width,
          node->params.deconvolution_2d.upsampling_height,
          node->params.deconvolution_2d.upsampling_width,
          node->params.deconvolution_2d.dilation_height,
          node->params.deconvolution_2d.dilation_width,
          node->params.deconvolution_2d.groups,
          node->params.deconvolution_2d.group_input_channels,
          node->params.deconvolution_2d.group_output_channels,
          node->params.deconvolution_2d.group_input_channels * node->params.deconvolution_2d.groups /* input_pixel_stride */,
          node->params.deconvolution_2d.group_output_channels * node->params.deconvolution_2d.groups /* output_pixel_stride */,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS,
          &opdata->operator_object);
      break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_compute_type_fp32:
      status = xnn_create_deconvolution2d_nhwc_f32(
          node->params.deconvolution_2d.padding_top,
          node->params.deconvolution_2d.padding_right,
          node->params.deconvolution_2d.padding_bottom,
          node->params.deconvolution_2d.padding_left,
          node->params.deconvolution_2d.kernel_height,
          node->params.deconvolution_2d.kernel_width,
          node->params.deconvolution_2d.upsampling_height,
          node->params.deconvolution_2d.upsampling_width,
          node->params.deconvolution_2d.dilation_height,
          node->params.deconvolution_2d.dilation_width,
          node->params.deconvolution_2d.groups,
          node->params.deconvolution_2d.group_input_channels,
          node->params.deconvolution_2d.group_output_channels,
          node->params.deconvolution_2d.group_input_channels * node->params.deconvolution_2d.groups /* input_pixel_stride */,
          node->params.deconvolution_2d.group_output_channels * node->params.deconvolution_2d.groups /* output_pixel_stride */,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &opdata->operator_object);
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_compute_type_qs8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const int8_t output_min =
        (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
      const int8_t output_max =
        (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
      status = xnn_create_deconvolution2d_nhwc_qs8(
          node->params.deconvolution_2d.padding_top,
          node->params.deconvolution_2d.padding_right,
          node->params.deconvolution_2d.padding_bottom,
          node->params.deconvolution_2d.padding_left,
          node->params.deconvolution_2d.kernel_height,
          node->params.deconvolution_2d.kernel_width,
          node->params.deconvolution_2d.upsampling_height,
          node->params.deconvolution_2d.upsampling_width,
          node->params.deconvolution_2d.dilation_height,
          node->params.deconvolution_2d.dilation_width,
          node->params.deconvolution_2d.groups,
          node->params.deconvolution_2d.group_input_channels,
          node->params.deconvolution_2d.group_output_channels,
          node->params.deconvolution_2d.group_input_channels * node->params.deconvolution_2d.groups /* input_pixel_stride */,
          node->params.deconvolution_2d.group_output_channels * node->params.deconvolution_2d.groups /* output_pixel_stride */,
          (int8_t) values[input_id].quantization.zero_point,
          values[input_id].quantization.scale,
          values[filter_id].quantization.scale,
          filter_data,
          bias_data,
          output_zero_point,
          output_scale,
          output_min,
          output_max,
          node->flags,
          &opdata->operator_object);
      break;
    }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_compute_type_qu8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const uint8_t output_min =
        (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
      const uint8_t output_max =
        (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
      status = xnn_create_deconvolution2d_nhwc_qu8(
          node->params.deconvolution_2d.padding_top,
          node->params.deconvolution_2d.padding_right,
          node->params.deconvolution_2d.padding_bottom,
          node->params.deconvolution_2d.padding_left,
          node->params.deconvolution_2d.kernel_height,
          node->params.deconvolution_2d.kernel_width,
          node->params.deconvolution_2d.upsampling_height,
          node->params.deconvolution_2d.upsampling_width,
          node->params.deconvolution_2d.dilation_height,
          node->params.deconvolution_2d.dilation_width,
          node->params.deconvolution_2d.groups,
          node->params.deconvolution_2d.group_input_channels,
          node->params.deconvolution_2d.group_output_channels,
          node->params.deconvolution_2d.group_input_channels * node->params.deconvolution_2d.groups /* input_pixel_stride */,
          node->params.deconvolution_2d.group_output_channels * node->params.deconvolution_2d.groups /* output_pixel_stride */,
          (uint8_t) values[input_id].quantization.zero_point,
          values[input_id].quantization.scale,
          (uint8_t) values[filter_id].quantization.zero_point,
          values[filter_id].quantization.scale,
          filter_data,
          bias_data,
          output_zero_point,
          output_scale,
          output_min,
          output_max,
          node->flags,
          &opdata->operator_object);
      break;
    }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
  if (status == xnn_status_success) {
    opdata->batch_size = values[input_id].shape.dim[0];
    opdata->input_height = values[input_id].shape.dim[1];
    opdata->input_width = values[input_id].shape.dim[2];
    opdata->adjustment_height = node->params.deconvolution_2d.adjustment_height;
    opdata->adjustment_width = node->params.deconvolution_2d.adjustment_width;
    opdata->inputs[0] = input_id;
    opdata->outputs[0] = output_id;
  }
  return status;
}

static enum xnn_status setup_deconvolution_operator(
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
    case xnn_operator_type_deconvolution_nhwc_f16:
      return xnn_setup_deconvolution2d_nhwc_f16(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          opdata->adjustment_height,
          opdata->adjustment_width,
          input_data,
          output_data,
          threadpool);
      break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_operator_type_deconvolution_nhwc_f32:
      return xnn_setup_deconvolution2d_nhwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          opdata->adjustment_height,
          opdata->adjustment_width,
          input_data,
          output_data,
          threadpool);
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_operator_type_deconvolution_nhwc_qs8:
      return xnn_setup_deconvolution2d_nhwc_qs8(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          opdata->adjustment_height,
          opdata->adjustment_width,
          input_data,
          output_data,
          threadpool);
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_operator_type_deconvolution_nhwc_qu8:
      return xnn_setup_deconvolution2d_nhwc_qu8(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          opdata->adjustment_height,
          opdata->adjustment_width,
          input_data,
          output_data,
          threadpool);
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

static inline enum xnn_compute_type validate_datatypes_with_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype filter_datatype,
  enum xnn_datatype bias_datatype,
  enum xnn_datatype output_datatype)
{
  switch (filter_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_fp32;
      }
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_qint8)
      {
        return xnn_compute_type_qs8;
      }
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_quint8)
      {
        return xnn_compute_type_qu8;
      }
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

static inline enum xnn_compute_type validate_datatypes_without_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype filter_datatype,
  enum xnn_datatype output_datatype)
{
  switch (filter_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      }
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 && output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qs8;
      }
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 && output_datatype == xnn_datatype_quint8) {
        return xnn_compute_type_qu8;
      }
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

enum xnn_status xnn_define_deconvolution_2d(
  xnn_subgraph_t subgraph,
  uint32_t padding_top,
  uint32_t padding_right,
  uint32_t padding_bottom,
  uint32_t padding_left,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t upsampling_height,
  uint32_t upsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d));
    return xnn_status_uninitialized;
  }

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), kernel_width, kernel_height);
    return xnn_status_invalid_parameter;
  }

  if (upsampling_width == 0 || upsampling_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " upsampling: upsampling dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), upsampling_width, upsampling_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 " groups: number of groups must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), groups);
    return xnn_status_invalid_parameter;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu input channels per group: number of channels must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), group_input_channels);
    return xnn_status_invalid_parameter;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu output channels per group: number of channels must be non-zero",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), group_output_channels);
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id, input_value->type);
    return xnn_status_invalid_parameter;
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
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* filter_value = &subgraph->values[filter_id];
  if (filter_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id, filter_value->type);
    return xnn_status_invalid_parameter;
  }

  if (filter_value->data == NULL) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": non-static Value",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id);
    return xnn_status_invalid_parameter;
  }

  switch (filter_value->datatype) {
    case xnn_datatype_fp32:
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
      if (filter_value->quantization.zero_point != 0) {
        xnn_log_error(
          "failed to define %s operator with filter ID #%" PRIu32 ": unsupported quantization zero point %" PRId32 " for datatype %s",
          xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id,
          filter_value->quantization.zero_point, xnn_datatype_to_string(filter_value->datatype));
      }
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), filter_id,
        xnn_datatype_to_string(filter_value->datatype), filter_value->datatype);
      return xnn_status_invalid_parameter;
  }

  const struct xnn_value* bias_value = NULL;

  if (bias_id != XNN_INVALID_VALUE_ID) {
    if (bias_id >= subgraph->num_values) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": invalid Value ID",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id);
      return xnn_status_invalid_parameter;
    }

    bias_value = &subgraph->values[bias_id];
    if (bias_value->type != xnn_value_type_dense_tensor) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id, bias_value->type);
      return xnn_status_invalid_parameter;
    }

    if (bias_value->data == NULL) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": non-static Value",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id);
      return xnn_status_invalid_parameter;
    }

    switch (bias_value->datatype) {
      case xnn_datatype_fp32:
#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
      case xnn_datatype_qint32:
#endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
        break;
      default:
        xnn_log_error(
          "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
          xnn_node_type_to_string(xnn_node_type_deconvolution_2d), bias_id,
          xnn_datatype_to_string(bias_value->datatype), bias_value->datatype);
        return xnn_status_invalid_parameter;
    }
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (output_value->datatype) {
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
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  if (bias_value != NULL) {
    compute_type = validate_datatypes_with_bias(
      input_value->datatype, filter_value->datatype, bias_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", bias ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), bias (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id, filter_id, bias_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(filter_value->datatype),
        xnn_datatype_to_string(bias_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  } else {
    compute_type = validate_datatypes_without_bias(
      input_value->datatype, filter_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_deconvolution_2d), input_id, filter_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(filter_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_deconvolution_2d;
  node->compute_type = compute_type;
  node->params.deconvolution_2d.padding_top = padding_top;
  node->params.deconvolution_2d.padding_right = padding_right;
  node->params.deconvolution_2d.padding_bottom = padding_bottom;
  node->params.deconvolution_2d.padding_left = padding_left;
  node->params.deconvolution_2d.kernel_height = kernel_height;
  node->params.deconvolution_2d.kernel_width = kernel_width;
  node->params.deconvolution_2d.upsampling_height = upsampling_height;
  node->params.deconvolution_2d.upsampling_width = upsampling_width;
  node->params.deconvolution_2d.dilation_height = dilation_height;
  node->params.deconvolution_2d.dilation_width = dilation_width;
  node->params.deconvolution_2d.adjustment_height = adjustment_height;
  node->params.deconvolution_2d.adjustment_width = adjustment_width;
  node->params.deconvolution_2d.groups = groups;
  node->params.deconvolution_2d.group_input_channels = group_input_channels;
  node->params.deconvolution_2d.group_output_channels = group_output_channels;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2 + (size_t) (bias_value != NULL);
  node->inputs[0] = input_id;
  node->inputs[1] = filter_id;
  node->inputs[2] = bias_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_deconvolution_operator;
  node->setup = setup_deconvolution_operator;

  return xnn_status_success;
};
