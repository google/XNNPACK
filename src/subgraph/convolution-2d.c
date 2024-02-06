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


static enum xnn_status create_convolution_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs >= 2);
  assert(node->num_inputs <= 3);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const uint32_t filter_id = node->inputs[1];
  assert(filter_id != XNN_INVALID_VALUE_ID);
  assert(filter_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const void* filter_data = values[filter_id].fp32_data != NULL ? values[filter_id].fp32_data : values[filter_id].data;
  assert(filter_data != NULL);

  const void* bias_data = NULL;
  if (node->num_inputs > 2) {
    const uint32_t bias_id = node->inputs[2];
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);

    bias_data = values[bias_id].fp32_data != NULL ? values[bias_id].fp32_data : values[bias_id].data;
    assert(bias_data != NULL);
  }

  enum xnn_status status;
  if (values[output_id].layout == xnn_layout_type_nchw) {
    switch (node->compute_type) {
      case xnn_compute_type_fp16:
        status = xnn_create_convolution2d_nchw_f16(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags | (values[input_id].layout == xnn_layout_type_nhwc ? XNN_FLAG_INPUT_NHWC : 0) | XNN_FLAG_FP32_STATIC_WEIGHTS,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_fp32:
        status = xnn_create_convolution2d_nchw_f32(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags | (values[input_id].layout == xnn_layout_type_nhwc ? XNN_FLAG_INPUT_NHWC : 0),
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      default:
        XNN_UNREACHABLE;
    }
  } else {
    assert(values[input_id].layout == xnn_layout_type_nhwc);
    assert(values[output_id].layout == xnn_layout_type_nhwc);
    switch (node->compute_type) {
      case xnn_compute_type_fp32:
        status = xnn_create_convolution2d_nhwc_f32(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_fp16:
        status = xnn_create_convolution2d_nhwc_f16(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_qd8_to_fp16:
      {
        status = xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          /*input_channel_stride=*/node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups,
          /*output_channel_stride=*/node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups,
          values[filter_id].quantization.channelwise_scale,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      }
      case xnn_compute_type_qd8_to_fp32:
      {
        status = xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          /*input_channel_stride=*/node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups,
          /*output_channel_stride=*/node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups,
          values[filter_id].quantization.channelwise_scale,
          filter_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      }
      case xnn_compute_type_qs8:
      {
        const float output_scale = values[output_id].quantization.scale;
        const int32_t output_zero_point = values[output_id].quantization.zero_point;
        const int8_t output_min = xnn_qs8_quantize(node->activation.output_min, output_scale, output_zero_point);
        const int8_t output_max = xnn_qs8_quantize(node->activation.output_max, output_scale, output_zero_point);
        status = xnn_create_convolution2d_nhwc_qs8(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          (int8_t) values[input_id].quantization.zero_point,
          values[input_id].quantization.scale,
          values[filter_id].quantization.scale,
          filter_data,
          bias_data,
          (int8_t) output_zero_point,
          output_scale, output_min, output_max,
          node->flags,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      }
      case xnn_compute_type_qc8:
      {
        const float output_scale = values[output_id].quantization.scale;
        const int32_t output_zero_point = values[output_id].quantization.zero_point;
        const int8_t output_min = xnn_qs8_quantize(node->activation.output_min, output_scale, output_zero_point);
        const int8_t output_max = xnn_qs8_quantize(node->activation.output_max, output_scale, output_zero_point);
        status = xnn_create_convolution2d_nhwc_qs8_qc8w(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          (int8_t) values[input_id].quantization.zero_point,
          values[input_id].quantization.scale,
          values[filter_id].quantization.channelwise_scale,
          filter_data,
          bias_data,
          (int8_t) output_zero_point,
          output_scale, output_min, output_max,
          node->flags,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      }
      case xnn_compute_type_qu8:
      {
        const float output_scale = values[output_id].quantization.scale;
        const int32_t output_zero_point = values[output_id].quantization.zero_point;
        const uint8_t output_min = xnn_qu8_quantize(node->activation.output_min, output_scale, output_zero_point);
        const uint8_t output_max = xnn_qu8_quantize(node->activation.output_max, output_scale, output_zero_point);
        status = xnn_create_convolution2d_nhwc_qu8(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          (uint8_t) values[input_id].quantization.zero_point,
          values[input_id].quantization.scale,
          (uint8_t) values[filter_id].quantization.zero_point,
          values[filter_id].quantization.scale,
          filter_data,
          bias_data,
          (uint8_t) output_zero_point,
          output_scale, output_min, output_max,
          node->flags,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
        break;
      }
      default:
        XNN_UNREACHABLE;
    }
  }
  return status;
}

static enum xnn_status reshape_convolution_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);

  const size_t batch_size = values[input_id].shape.dim[0];
  const size_t input_height = values[input_id].shape.dim[1];
  const size_t input_width = values[input_id].shape.dim[2];

  size_t output_height, output_width;
  enum xnn_status status = xnn_status_invalid_state;
  const size_t old_workspace_size = opdata->workspace_size;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_convolution_nchw_f16:
      status = xnn_reshape_convolution2d_nchw_f16(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nchw_f32:
      status = xnn_reshape_convolution2d_nchw_f32(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nhwc_f32:
      status = xnn_reshape_convolution2d_nhwc_f32(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nhwc_f16:
      status = xnn_reshape_convolution2d_nhwc_f16(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nhwc_qd8_f16_qc8w:
      status = xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nhwc_qd8_f32_qc8w:
      status = xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nhwc_qc8:
      status = xnn_reshape_convolution2d_nhwc_qs8_qc8w(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nhwc_qs8:
      status = xnn_reshape_convolution2d_nhwc_qs8(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_convolution_nhwc_qu8:
      status = xnn_reshape_convolution2d_nhwc_qu8(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        &output_height,
        &output_width,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status != xnn_status_success) {
    return status;
  }

  const size_t output_pixel_stride = opdata->operator_objects[0]->output_pixel_stride;
  struct xnn_value* output_value = values + output_id;
  output_value->shape.dim[0] = batch_size;
  output_value->shape.dim[1] = output_height;
  output_value->shape.dim[2] = output_width;
  output_value->shape.dim[3] = output_pixel_stride;

  output_value->shape.num_dims = 4;
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_convolution_operator(
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
    case xnn_operator_type_convolution_nchw_f16:
      return xnn_setup_convolution2d_nchw_f16(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_convolution_nchw_f32:
      return xnn_setup_convolution2d_nchw_f32(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_convolution_nhwc_f32:
      return xnn_setup_convolution2d_nhwc_f32(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_convolution_nhwc_f16:
      return xnn_setup_convolution2d_nhwc_f16(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_convolution_nhwc_qc8:
      return xnn_setup_convolution2d_nhwc_qs8_qc8w(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_convolution_nhwc_qd8_f16_qc8w:
      {
        const void* quantization_params = input_value->quantization.dynamic_params;
        assert(quantization_params != NULL);
        return xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
          opdata->operator_objects[0],
          opdata->workspace,
          input_data,
          output_data,
          quantization_params);
      }
      break;
    case xnn_operator_type_convolution_nhwc_qd8_f32_qc8w:
      {
        const void* quantization_params = input_value->quantization.dynamic_params;
        assert(quantization_params != NULL);
        return xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
          opdata->operator_objects[0],
          opdata->workspace,
          input_data,
          output_data,
          quantization_params);
      }
      break;
    case xnn_operator_type_convolution_nhwc_qs8:
      return xnn_setup_convolution2d_nhwc_qs8(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_convolution_nhwc_qu8:
      return xnn_setup_convolution2d_nhwc_qu8(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
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
      } else if (input_datatype == xnn_datatype_fp16 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return xnn_compute_type_fp16;
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
    case xnn_datatype_qcint8:
      if (input_datatype == xnn_datatype_qint8 &&
          bias_datatype == xnn_datatype_qcint32 &&
          output_datatype == xnn_datatype_qint8)
      {
        return xnn_compute_type_qc8;
      } else if (input_datatype == xnn_datatype_qdint8 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp16)
      {
        return xnn_compute_type_qd8_to_fp16;
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
  enum xnn_datatype filter_datatype,
  enum xnn_datatype output_datatype)
{
  switch (filter_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_fp16 && output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return xnn_compute_type_fp16;
      }
      break;
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 && output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qs8;
      }
      break;
    case xnn_datatype_qcint8:
      if (input_datatype == xnn_datatype_qint8 && output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qc8;
      } else if (input_datatype == xnn_datatype_qdint8 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 && output_datatype == xnn_datatype_fp16) {
        return xnn_compute_type_qd8_to_fp16;
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

enum xnn_status xnn_define_convolution_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
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
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_convolution_2d)) != xnn_status_success) {
    return status;
  }

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), kernel_width, kernel_height);
    return xnn_status_invalid_parameter;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " subsampling: subsampling dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), subsampling_width, subsampling_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 " groups: number of groups must be non-zero",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), groups);
    return xnn_status_invalid_parameter;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu input channels per group: number of channels must be non-zero",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), group_input_channels);
    return xnn_status_invalid_parameter;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu output channels per group: number of channels must be non-zero",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), group_output_channels);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_min_max(xnn_node_type_convolution_2d, output_min, output_max);
  if (status != xnn_status_success) {
    return status;
  }

  const uint32_t supported_flags = XNN_FLAG_TENSORFLOW_SAME_PADDING | XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
  const uint32_t invalid_flags = flags & ~supported_flags;
  if (invalid_flags != 0) {
    xnn_log_error(
      "failed to define %s operator with 0x%08" PRIx32 " flags: invalid flags 0x%08" PRIx32,
      xnn_node_type_to_string(xnn_node_type_convolution_2d), flags, invalid_flags);
    return xnn_status_invalid_parameter;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && any_padding) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
      "TensorFlow SAME padding can't be combined with explicit padding specification",
      xnn_node_type_to_string(xnn_node_type_convolution_2d),
      input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
    return xnn_status_invalid_parameter;
  }

  // Convert TensorFlow SAME padding to explicit padding specification whenever possible
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && (subsampling_height | subsampling_width) == 1) {
    flags &= ~XNN_FLAG_TENSORFLOW_SAME_PADDING;
    const uint32_t padding_height = (kernel_height - 1) * dilation_height;
    const uint32_t padding_width = (kernel_width - 1) * dilation_width;
    input_padding_left = padding_width / 2;
    input_padding_top = padding_height / 2;
    input_padding_right = padding_width - input_padding_left;
    input_padding_bottom = padding_height - input_padding_top;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_convolution_2d, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_convolution_2d, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    case xnn_datatype_qdint8:
      if (input_value->quantization.num_nonbatch_dims >= input_value->shape.num_dims) {
        xnn_log_error(
          "failed to define %s operator with input ID #%" PRIu32 ": num_nonbatch_dims (%zu) must be "
          "< num_dims (%zu)",
          xnn_node_type_to_string(xnn_node_type_convolution_2d), input_id,
          input_value->quantization.num_nonbatch_dims, input_value->shape.num_dims);
        return xnn_status_invalid_parameter;
      }
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_convolution_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), filter_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* filter_value = &subgraph->values[filter_id];
  if (filter_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), filter_id, filter_value->type);
    return xnn_status_invalid_parameter;
  }

  if (filter_value->data == NULL) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": non-static Value",
      xnn_node_type_to_string(xnn_node_type_convolution_2d), filter_id);
    return xnn_status_invalid_parameter;
  }

  switch (filter_value->datatype) {
    case xnn_datatype_fp32:
    case xnn_datatype_fp16:
      break;
    case xnn_datatype_qint8:
      if (filter_value->quantization.zero_point != 0) {
        xnn_log_error(
          "failed to define %s operator with filter ID #%" PRIu32 ": unsupported quantization zero point %" PRId32 " for datatype %s",
          xnn_node_type_to_string(xnn_node_type_convolution_2d), filter_id,
          filter_value->quantization.zero_point, xnn_datatype_to_string(filter_value->datatype));
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_qcint8:
      break;
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_convolution_2d), filter_id,
        xnn_datatype_to_string(filter_value->datatype), filter_value->datatype);
      return xnn_status_invalid_parameter;
  }

  const struct xnn_value* bias_value = NULL;
  if (bias_id != XNN_INVALID_VALUE_ID) {
    if (bias_id >= subgraph->num_values) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": invalid Value ID",
        xnn_node_type_to_string(xnn_node_type_convolution_2d), bias_id);
      return xnn_status_invalid_parameter;
    }

    bias_value = &subgraph->values[bias_id];
    if (bias_value->type != xnn_value_type_dense_tensor) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
        xnn_node_type_to_string(xnn_node_type_convolution_2d), bias_id, bias_value->type);
      return xnn_status_invalid_parameter;
    }

    if (bias_value->data == NULL) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": non-static Value",
        xnn_node_type_to_string(xnn_node_type_convolution_2d), bias_id);
      return xnn_status_invalid_parameter;
    }

    switch (bias_value->datatype) {
      case xnn_datatype_fp16:
      case xnn_datatype_fp32:
      case xnn_datatype_qint32:
      case xnn_datatype_qcint32:
        break;
      default:
        xnn_log_error(
          "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
          xnn_node_type_to_string(xnn_node_type_convolution_2d), bias_id,
          xnn_datatype_to_string(bias_value->datatype), bias_value->datatype);
        return xnn_status_invalid_parameter;
    }
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_convolution_2d, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_convolution_2d, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  if (filter_value->shape.dim[0] != group_output_channels * groups) {
    xnn_log_error(
        "failed to define %s operator with filter output channels %zu, groups #%" PRIu32 " and group_output_channels %zu:"
        "mismatching shapes, filter output channels must be equal to groups * group_output_channels.",
        xnn_node_type_to_string(xnn_node_type_convolution_2d), filter_value->shape.dim[0], groups, group_output_channels);
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
        xnn_node_type_to_string(xnn_node_type_convolution_2d), output_id,
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
        xnn_node_type_to_string(xnn_node_type_convolution_2d), input_id, filter_id, bias_id, output_id,
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
        xnn_node_type_to_string(xnn_node_type_convolution_2d), input_id, filter_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(filter_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  }

  if (filter_value->datatype == xnn_datatype_qcint8) {
    if (filter_value->quantization.channel_dimension != 0) {
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": invalid channel dimension %zu",
        xnn_node_type_to_string(xnn_node_type_convolution_2d), input_id, filter_value->quantization.channel_dimension);
      return xnn_status_invalid_parameter;
    }

    if (bias_value != NULL) {
      assert(bias_value->datatype == xnn_datatype_qcint32 || bias_value->datatype == xnn_datatype_fp32);
      if (bias_value->datatype == xnn_datatype_qcint32 && bias_value->quantization.channel_dimension != 0) {
        xnn_log_error(
          "failed to define %s operator with bias ID #%" PRIu32 ": invalid channel dimension %zu",
          xnn_node_type_to_string(xnn_node_type_convolution_2d), bias_id, bias_value->quantization.channel_dimension);
        return xnn_status_invalid_parameter;
      }
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_convolution_2d;
  node->compute_type = compute_type;
  node->params.convolution_2d.input_padding_top = input_padding_top;
  node->params.convolution_2d.input_padding_right = input_padding_right;
  node->params.convolution_2d.input_padding_bottom = input_padding_bottom;
  node->params.convolution_2d.input_padding_left = input_padding_left;
  node->params.convolution_2d.kernel_height = kernel_height;
  node->params.convolution_2d.kernel_width = kernel_width;
  node->params.convolution_2d.subsampling_height = subsampling_height;
  node->params.convolution_2d.subsampling_width = subsampling_width;
  node->params.convolution_2d.dilation_height = dilation_height;
  node->params.convolution_2d.dilation_width = dilation_width;
  node->params.convolution_2d.groups = groups;
  node->params.convolution_2d.group_input_channels = group_input_channels;
  node->params.convolution_2d.group_output_channels = group_output_channels;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2 + (size_t) (bias_id != XNN_INVALID_VALUE_ID);
  node->inputs[0] = input_id;
  node->inputs[1] = filter_id;
  node->inputs[2] = bias_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_convolution_operator;
  node->reshape = reshape_convolution_operator;
  node->setup = setup_convolution_operator;

  return xnn_status_success;
};
