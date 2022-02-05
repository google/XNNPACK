// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/indirection.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params.h>


static inline size_t compute_output_dimension(
    size_t input_dimension,
    size_t output_padding_dimension,
    size_t adjustment_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t stride_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(
    stride_dimension * (input_dimension - 1) + adjustment_dimension + effective_kernel_dimension,
    output_padding_dimension);
}

static enum xnn_status create_deconvolution2d_nhwc(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_conv_goki_w_function pack_conv_goki_w,
    xnn_pack_deconv_goki_w_function pack_deconv_goki_w,
    const void* packing_params,
    int input_padding_byte,
    int packed_weights_padding_byte,
    const void* params,
    size_t params_size,
    const struct gemm_parameters* gemm_parameters,
    const struct gemm_fused_ukernels* gemm_ukernels,
    enum xnn_operator_type operator_type,
    xnn_operator_t* deconvolution_op_out)
{
  xnn_operator_t deconvolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), kernel_width, kernel_height);
    goto error;
  }

  if (stride_width == 0 || stride_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " stride: stride dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), stride_width, stride_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 " groups: number of groups must be non-zero",
      xnn_operator_type_to_string(operator_type), groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu input channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu output channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_pixel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      input_pixel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_pixel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      output_pixel_stride, groups, group_output_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  deconvolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (deconvolution_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const uint32_t mr = gemm_parameters->mr;
  const uint32_t nr = gemm_parameters->nr;
  const uint32_t kr = UINT32_C(1) << gemm_parameters->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_parameters->log2_sr;

  const uint32_t n_stride = round_up(group_output_channels, nr);
  const uint32_t k_stride = round_up_po2(group_input_channels, kr * sr);
  const uint32_t kernel_size = kernel_height * kernel_width;
  enum xnn_ukernel_type ukernel_type = xnn_ukernel_type_igemm;
  size_t packed_group_weights_size = (((kernel_size * k_stride) << log2_filter_element_size) + bias_element_size) * n_stride;
  if (max(stride_height, stride_width) > 1 && max(dilation_height, dilation_width) == 1 && stride_width <= kernel_width && stride_height <= kernel_height) {
    ukernel_type = xnn_ukernel_type_subconv2d;
    const size_t subkernels = stride_height * stride_width;
    packed_group_weights_size = n_stride *
      (((kernel_size * k_stride) << log2_filter_element_size) + bias_element_size * subkernels);

    const size_t subconvolution_buffer_size = sizeof(struct subconvolution_params) * subkernels;
    deconvolution_op->subconvolution_buffer = xnn_allocate_zero_memory(subconvolution_buffer_size);
    if (deconvolution_op->subconvolution_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator subconvolution buffer",
        subconvolution_buffer_size, xnn_operator_type_to_string(operator_type));
      goto error;
    }

    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
      for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
        const size_t subkernel_height = divide_round_up(kernel_height - offset_y, stride_height);
        const size_t subkernel_width = divide_round_up(kernel_width - offset_x, stride_width);
        const size_t subkernel_size = subkernel_height * subkernel_width;

        subconvolution_params->indirection_x_stride = sizeof(void*) * subkernel_size;
        subconvolution_params->w_stride = bias_element_size + ((k_stride * subkernel_size) << log2_filter_element_size);
        subconvolution_params++;
      }
    }
  }
  deconvolution_op->packed_weights = xnn_allocate_simd_memory(packed_group_weights_size * groups);
  if (deconvolution_op->packed_weights == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      packed_group_weights_size * groups, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  memset(deconvolution_op->packed_weights, packed_weights_padding_byte, packed_group_weights_size * groups);

  switch (ukernel_type) {
    case xnn_ukernel_type_igemm:
      pack_conv_goki_w(
        groups, group_output_channels, kernel_size, group_input_channels,
        nr, kr, sr,
        kernel, bias, deconvolution_op->packed_weights,
        0 /* extra bytes */,
        packing_params);
      break;
    case xnn_ukernel_type_subconv2d:
      pack_deconv_goki_w(
        groups, group_output_channels, kernel_height, kernel_width, group_input_channels,
        stride_height, stride_width,
        nr, kr, sr,
        kernel, bias, deconvolution_op->packed_weights, deconvolution_op->subconvolution_buffer,
        packing_params);
      break;
    default:
      XNN_UNREACHABLE;
  }

  const size_t zero_size = (k_stride << log2_input_element_size) + XNN_EXTRA_BYTES;
  deconvolution_op->zero_buffer = xnn_allocate_simd_memory(zero_size);
  if (deconvolution_op->zero_buffer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator zero padding",
      zero_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  memset(deconvolution_op->zero_buffer, input_padding_byte, zero_size);

  deconvolution_op->padding_top = output_padding_top;
  deconvolution_op->padding_right = output_padding_right;
  deconvolution_op->padding_bottom = output_padding_bottom;
  deconvolution_op->padding_left = output_padding_left;

  deconvolution_op->kernel_height = kernel_height;
  deconvolution_op->kernel_width = kernel_width;
  deconvolution_op->stride_height = stride_height;
  deconvolution_op->stride_width = stride_width;
  deconvolution_op->dilation_height = dilation_height;
  deconvolution_op->dilation_width = dilation_width;
  deconvolution_op->groups = groups;
  deconvolution_op->group_input_channels = group_input_channels;
  deconvolution_op->group_output_channels = group_output_channels;
  deconvolution_op->input_pixel_stride = input_pixel_stride;
  deconvolution_op->output_pixel_stride = output_pixel_stride;

  memcpy(&deconvolution_op->params, params, params_size);
  deconvolution_op->type = operator_type;
  deconvolution_op->ukernel.type = ukernel_type;
  deconvolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
    .general_case = gemm_ukernels->igemm,
    .gemm_case = gemm_ukernels->gemm,
    .mr = mr,
    .nr = nr,
    .kr = kr,
    .sr = sr,
  };

  deconvolution_op->state = xnn_run_state_invalid;

  *deconvolution_op_out = deconvolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(deconvolution_op);
  return status;
}

enum xnn_status xnn_create_deconvolution2d_nhwc_qs8(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    int8_t input_zero_point,
    float input_scale,
    float kernel_scale,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* deconvolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  union xnn_qs8_conv_minmax_params params;
  if XNN_LIKELY(xnn_params.qs8.gemm.init.qs8 != NULL) {
    xnn_params.qs8.gemm.init.qs8(&params,
      requantization_scale, output_zero_point, output_min, output_max);
  }
  const struct xnn_qs8_packing_params packing_params = {
    .input_zero_point = input_zero_point,
  };
  return create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    0 /* log2(sizeof(input element)) = log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    (xnn_pack_conv_goki_w_function) xnn_pack_qs8_conv_goki_w,
    (xnn_pack_deconv_goki_w_function) xnn_pack_qs8_deconv_goki_w,
    &packing_params, input_zero_point /* input padding byte */, 0 /* packed weights padding byte */,
    &params, sizeof(params),
    &xnn_params.qs8.gemm, &xnn_params.qs8.gemm.minmax,
    xnn_operator_type_deconvolution_nhwc_qs8,
    deconvolution_op_out);
}

enum xnn_status xnn_create_deconvolution2d_nhwc_qu8(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* deconvolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  union xnn_qu8_conv_minmax_params params;
  if XNN_LIKELY(xnn_params.qu8.gemm.init.qu8 != NULL) {
    xnn_params.qu8.gemm.init.qu8(&params,
      kernel_zero_point, requantization_scale, output_zero_point, output_min, output_max);
  }
  const struct xnn_qu8_packing_params packing_params = {
    .input_zero_point = input_zero_point,
    .kernel_zero_point = kernel_zero_point,
  };
  return create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    (xnn_pack_conv_goki_w_function) xnn_pack_qu8_conv_goki_w,
    (xnn_pack_deconv_goki_w_function) xnn_pack_qu8_deconv_goki_w,
    &packing_params, input_zero_point /* input padding byte */, kernel_zero_point /* packed weights padding byte */,
    &params, sizeof(params),
    &xnn_params.qu8.gemm, &xnn_params.qu8.gemm.minmax,
    xnn_operator_type_deconvolution_nhwc_qu8,
    deconvolution_op_out);
}

enum xnn_status xnn_create_deconvolution2d_nhwc_f16(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const void* kernel,
    const void* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* deconvolution_op_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_F16) != XNN_INIT_FLAG_F16) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16));
    return xnn_status_unsupported_hardware;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  const uint16_t output_min_as_half = fp16_ieee_from_fp32_value(output_min);
  const uint16_t output_max_as_half = fp16_ieee_from_fp32_value(output_max);
  output_min = fp16_ieee_to_fp32_value(output_min_as_half);
  output_max = fp16_ieee_to_fp32_value(output_max_as_half);
  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct gemm_parameters* gemm_parameters = &xnn_params.f16.gemm;
  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_parameters->minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_parameters->linear.gemm.function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_parameters->linear;
  }

  union xnn_f16_scaleminmax_params params;
  if XNN_LIKELY(xnn_params.f16.gemm.init.f16 != NULL) {
    gemm_parameters->init.f16(&params, UINT16_C(0x3C00) /* 1.0 */, output_min_as_half, output_max_as_half);
  }

  xnn_pack_conv_goki_w_function pack_conv_goki_w = (xnn_pack_conv_goki_w_function) xnn_pack_f16_conv_goki_w;
  xnn_pack_deconv_goki_w_function pack_deconv_goki_w = (xnn_pack_deconv_goki_w_function) xnn_pack_f16_deconv_goki_w;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_conv_goki_w = (xnn_pack_conv_goki_w_function) xnn_pack_f32_to_f16_conv_goki_w;
    pack_deconv_goki_w = (xnn_pack_deconv_goki_w_function) xnn_pack_f32_to_f16_deconv_goki_w;
  }

  return create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    1 /* log2(sizeof(input element)) = log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(filter element)) = log2(sizeof(uint16_t)) */,
    sizeof(uint16_t) /* sizeof(bias element) */,
    pack_conv_goki_w,
    pack_deconv_goki_w,
    NULL /* packing params */, 0 /* input padding byte */, 0 /* packed weights padding byte */,
    &params, sizeof(params),
    gemm_parameters, gemm_ukernels,
    xnn_operator_type_deconvolution_nhwc_f16,
    deconvolution_op_out);
}

enum xnn_status xnn_create_deconvolution2d_nhwc_f32(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* deconvolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct gemm_parameters* gemm_parameters = &xnn_params.f32.gemm;
  if (gemm_parameters->nr > group_output_channels) {
    // Default micro-kernel is suboptimal. Try to find a better micro-kernel.
    if (xnn_params.f32.gemm2.minmax.igemm.function[XNN_UARCH_DEFAULT] != NULL) {
      gemm_parameters = &xnn_params.f32.gemm2;
    }
  }
  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_parameters->minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_parameters->linear.gemm.function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_parameters->linear;
  }

  union xnn_f32_minmax_params params;
  if XNN_LIKELY(xnn_params.f32.gemm.init.f32 != NULL) {
    gemm_parameters->init.f32(&params, output_min, output_max);
  }
  return create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    (xnn_pack_conv_goki_w_function) xnn_pack_f32_conv_goki_w,
    (xnn_pack_deconv_goki_w_function) xnn_pack_f32_deconv_goki_w,
    NULL /* packing params */, 0 /* input padding byte */, 0 /* packed weights padding byte */,
    &params, sizeof(params),
    gemm_parameters, gemm_ukernels,
    xnn_operator_type_deconvolution_nhwc_f32,
    deconvolution_op_out);
}

static enum xnn_status setup_conv_path(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  size_t output_height,
  size_t output_width,
  void* output,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  size_t params_size,
  size_t num_threads)
{
  assert(deconvolution_op->ukernel.type == xnn_ukernel_type_igemm);

  const size_t kernel_height = deconvolution_op->kernel_height;
  const size_t kernel_width = deconvolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;

  const size_t groups = deconvolution_op->groups;
  const size_t output_size = output_height * output_width;
  const size_t mr = deconvolution_op->ukernel.igemm.mr;
  const size_t tiled_output_size = round_up(output_size, mr);
  const size_t indirection_buffer_size = sizeof(void*) * kernel_size * tiled_output_size;

  if (input_height != deconvolution_op->last_input_height ||
      input_width != deconvolution_op->last_input_width)
  {
    const void** indirection_buffer = (const void**) xnn_reallocate_memory(deconvolution_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator indirection buffer",
        indirection_buffer_size, xnn_operator_type_to_string(deconvolution_op->type));
      return xnn_status_out_of_memory;
    }
    deconvolution_op->indirection_buffer = indirection_buffer;
    deconvolution_op->last_input = input;
    deconvolution_op->last_input_height = input_height;
    deconvolution_op->last_input_width = input_width;

    xnn_indirection_init_deconv2d(deconvolution_op, mr, log2_input_element_size);
  }

  const size_t group_input_channels = deconvolution_op->group_input_channels;
  const size_t group_output_channels = deconvolution_op->group_output_channels;
  const uint32_t nr = deconvolution_op->ukernel.igemm.nr;
  const size_t w_stride = bias_element_size +
    (round_up_po2(group_input_channels, deconvolution_op->ukernel.igemm.kr * deconvolution_op->ukernel.igemm.sr) * kernel_size << log2_filter_element_size);
  deconvolution_op->context.igemm = (struct igemm_context) {
      .ks = kernel_size,
      .ks_scaled = kernel_size * mr * sizeof(void*),
      .kc = group_input_channels << log2_input_element_size,
      .w_stride = w_stride,
      .indirect_a = deconvolution_op->indirection_buffer,
      .a_offset = (size_t) ((uintptr_t) input - (uintptr_t) deconvolution_op->last_input),
      .zero = deconvolution_op->zero_buffer,
      .packed_w = deconvolution_op->packed_weights,
      .c = deconvolution_op->output,
      .cm_stride = deconvolution_op->output_pixel_stride << log2_output_element_size,
      .cn_stride = nr << log2_output_element_size,
      .ga_stride = group_input_channels << log2_input_element_size,
      .gw_stride = w_stride * round_up(group_output_channels, nr),
      .gc_stride = group_output_channels << log2_output_element_size,
      .ba_stride = input_height * input_width * deconvolution_op->input_pixel_stride << log2_input_element_size,
      .bc_stride = output_size * deconvolution_op->output_pixel_stride << log2_output_element_size,
      .log2_csize = log2_output_element_size,
      .ukernel = deconvolution_op->ukernel.igemm.general_case,
  };
  if (output_size == 1 && deconvolution_op->ukernel.igemm.mr1_case.function[XNN_UARCH_DEFAULT] != NULL) {
    deconvolution_op->context.igemm.ukernel = deconvolution_op->ukernel.igemm.mr1_case;
  }
  memcpy(&deconvolution_op->context.igemm.params, params, params_size);

  size_t nc = group_output_channels;
  if (num_threads > 1) {
    const size_t num_other_tiles = groups * batch_size * divide_round_up(output_size, mr);
    const size_t target_tiles_per_thread = 5;
    const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
    if (max_nc < nc) {
      nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
    }
  }
  if (groups == 1) {
    if (batch_size > 1) {
      deconvolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
      deconvolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_igemm;
      deconvolution_op->compute.range[0] = batch_size;
      deconvolution_op->compute.range[1] = output_size;
      deconvolution_op->compute.range[2] = group_output_channels;
    } else {
      deconvolution_op->compute.type = xnn_parallelization_type_2d_tile_2d;
      deconvolution_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_igemm;
      deconvolution_op->compute.range[0] = output_size;
      deconvolution_op->compute.range[1] = group_output_channels;
    }
    deconvolution_op->compute.tile[0] = mr;
    deconvolution_op->compute.tile[1] = nc;
  } else {
    if (batch_size > 1) {
      deconvolution_op->compute.type = xnn_parallelization_type_4d_tile_2d;
      deconvolution_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_igemm;
      deconvolution_op->compute.range[0] = batch_size;
      deconvolution_op->compute.range[1] = groups;
      deconvolution_op->compute.range[2] = output_size;
      deconvolution_op->compute.range[3] = group_output_channels;
    } else {
      deconvolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
      deconvolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_igemm;
      deconvolution_op->compute.range[0] = groups;
      deconvolution_op->compute.range[1] = output_size;
      deconvolution_op->compute.range[2] = group_output_channels;
    }
    deconvolution_op->compute.tile[0] = mr;
    deconvolution_op->compute.tile[1] = nc;
  }
  deconvolution_op->state = xnn_run_state_ready;
  return xnn_status_success;
}

static enum xnn_status setup_subconv2d_path(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  size_t output_height,
  size_t output_width,
  void* output,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  size_t params_size,
  size_t num_threads,
  bool use_gemm)
{
  assert(deconvolution_op->ukernel.type == xnn_ukernel_type_subconv2d);

  const size_t kernel_height = deconvolution_op->kernel_height;
  const size_t kernel_width = deconvolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t stride_height = deconvolution_op->stride_height;
  const size_t stride_width = deconvolution_op->stride_width;

  const size_t groups = deconvolution_op->groups;
  const size_t output_size = output_height * output_width;
  const size_t mr = deconvolution_op->ukernel.igemm.mr;

  const size_t input_pixel_stride = deconvolution_op->input_pixel_stride << log2_input_element_size;
  const size_t output_pixel_stride = deconvolution_op->output_pixel_stride << log2_output_element_size;

  const bool any_size_change =
    input_height != deconvolution_op->last_input_height ||
    input_width != deconvolution_op->last_input_width ||
    output_height != deconvolution_op->last_output_height ||
    output_width != deconvolution_op->last_output_width;

  if (any_size_change || output != deconvolution_op->last_output) {
    // Initialize subconvolution parameters which depend on output dimensions or MR.
    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    const size_t modulo_padding_top = deconvolution_op->padding_top % stride_height;
    const size_t modulo_padding_left = deconvolution_op->padding_left % stride_width;
    for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
      for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
        const size_t output_x_start = subtract_modulo(offset_x, modulo_padding_left, stride_width);
        const size_t output_y_start = subtract_modulo(offset_y, modulo_padding_top, stride_height);
        subconvolution_params->scaled_kernel_size = mr * subconvolution_params->indirection_x_stride;
        subconvolution_params->slice_width = divide_round_up(output_width - output_x_start, stride_width);
        subconvolution_params->slice_height = divide_round_up(output_height - output_y_start, stride_height);
        subconvolution_params->output =
          (void*) ((uintptr_t) output + ((output_y_start * output_width + output_x_start) * output_pixel_stride));
        ++subconvolution_params;
      }
    }
    deconvolution_op->last_output = output;
  }

  if (any_size_change) {
    if (!use_gemm) {
      const size_t indirection_buffer_size = sizeof(void*) *
        kernel_size * output_height * stride_width * round_up(divide_round_up(output_width, stride_width), mr);

      const void** indirection_buffer =
        (const void**) xnn_reallocate_memory(deconvolution_op->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator indirection buffer",
          indirection_buffer_size, xnn_operator_type_to_string(deconvolution_op->type));
        return xnn_status_out_of_memory;
      }
      deconvolution_op->indirection_buffer = indirection_buffer;
      deconvolution_op->last_input = input;

      xnn_indirection_init_subconv2d(deconvolution_op, mr, log2_input_element_size);
    }
    deconvolution_op->last_input_height = input_height;
    deconvolution_op->last_input_width = input_width;
    deconvolution_op->last_output_height = output_height;
    deconvolution_op->last_output_width = output_width;
  }

  const size_t group_input_channels = deconvolution_op->group_input_channels;
  const size_t group_output_channels = deconvolution_op->group_output_channels;
  const uint32_t nr = deconvolution_op->ukernel.igemm.nr;
  const uint32_t kr = deconvolution_op->ukernel.igemm.kr;
  const uint32_t sr = deconvolution_op->ukernel.igemm.sr;
  const size_t w_stride = stride_height * stride_width * bias_element_size +
    (round_up_po2(group_input_channels, kr * sr) * kernel_size << log2_filter_element_size);
  if (use_gemm) {
    deconvolution_op->context.subgemm = (struct subgemm_context) {
        .subconvolution_params = deconvolution_op->subconvolution_buffer,
        .kc = group_input_channels << log2_input_element_size,
        .a = input,
        .ax_stride = input_pixel_stride,
        .ay_stride = input_width * input_pixel_stride,
        .cx_stride = stride_width * output_pixel_stride,
        .cy_stride = stride_height * output_width * output_pixel_stride,
        .cn_stride = nr << log2_output_element_size,
        .ga_stride = group_input_channels << log2_input_element_size,
        .gw_stride = w_stride * round_up(group_output_channels, nr),
        .gc_stride = group_output_channels << log2_output_element_size,
        .ba_stride = input_height * input_width * input_pixel_stride,
        .bc_stride = output_size * output_pixel_stride,
        .log2_csize = log2_output_element_size,
        .ukernel = deconvolution_op->ukernel.igemm.gemm_case,
    };
    memcpy(&deconvolution_op->context.subgemm.params, params, params_size);
  } else {
    deconvolution_op->context.subconv = (struct subconv_context) {
        .subconvolution_params = deconvolution_op->subconvolution_buffer,
        .kc = group_input_channels << log2_input_element_size,
        .a_offset = (size_t) ((uintptr_t) input - (uintptr_t) deconvolution_op->last_input),
        .zero = deconvolution_op->zero_buffer,
        .cx_stride = stride_width * output_pixel_stride,
        .cy_stride = stride_height * output_width * output_pixel_stride,
        .cn_stride = nr << log2_output_element_size,
        .ga_stride = group_input_channels << log2_input_element_size,
        .gw_stride = w_stride * round_up(group_output_channels, nr),
        .gc_stride = group_output_channels << log2_output_element_size,
        .ba_stride = input_height * input_width * input_pixel_stride,
        .bc_stride = output_size * output_pixel_stride,
        .log2_csize = log2_output_element_size,
        .ukernel = deconvolution_op->ukernel.igemm.general_case,
    };
    memcpy(&deconvolution_op->context.subconv.params, params, params_size);
  }

  const size_t output_height_positions = divide_round_up(output_height, stride_height);
  const size_t output_width_positions = divide_round_up(output_width, stride_width);

  size_t nc = group_output_channels;
  if (num_threads > 1) {
    const size_t num_other_tiles = groups * stride_height * stride_width *
      output_height_positions * divide_round_up(output_width_positions, mr);
    const size_t target_tiles_per_thread = 5;
    const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
    if (max_nc < nc) {
      nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
    }
  }

  if (groups == 1) {
    deconvolution_op->compute.type = xnn_parallelization_type_5d_tile_2d;
    deconvolution_op->compute.task_5d_tile_2d = use_gemm ?
      (pthreadpool_task_5d_tile_2d_t) xnn_compute_subgemm2d : (pthreadpool_task_5d_tile_2d_t) xnn_compute_subconv2d;
    deconvolution_op->compute.range[0] = batch_size;
    deconvolution_op->compute.range[1] = stride_height * stride_width;
    deconvolution_op->compute.range[2] = divide_round_up(output_height, stride_height);
    deconvolution_op->compute.range[3] = divide_round_up(output_width, stride_width);
    deconvolution_op->compute.range[4] = group_output_channels;
    deconvolution_op->compute.tile[0] = mr;
    deconvolution_op->compute.tile[1] = nc;
  } else {
    deconvolution_op->compute.type = xnn_parallelization_type_6d_tile_2d;
    deconvolution_op->compute.task_6d_tile_2d = use_gemm ?
      (pthreadpool_task_6d_tile_2d_t) xnn_compute_grouped_subgemm2d : (pthreadpool_task_6d_tile_2d_t) xnn_compute_grouped_subconv2d;
    deconvolution_op->compute.range[0] = batch_size;
    deconvolution_op->compute.range[1] = groups;
    deconvolution_op->compute.range[2] = stride_height * stride_width;
    deconvolution_op->compute.range[3] = divide_round_up(output_height, stride_height);
    deconvolution_op->compute.range[4] = divide_round_up(output_width, stride_width);
    deconvolution_op->compute.range[5] = group_output_channels;
    deconvolution_op->compute.tile[0] = mr;
    deconvolution_op->compute.tile[1] = nc;
  }

  deconvolution_op->state = xnn_run_state_ready;
  return xnn_status_success;
}

static enum xnn_status setup_deconvolution2d_nhwc(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  const void* input,
  void* output,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  size_t params_size,
  size_t num_threads)
{
  deconvolution_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(deconvolution_op->type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (adjustment_height >= deconvolution_op->stride_height) {
    xnn_log_error(
      "failed to setup %s operator with %" PRIu32 " height adjustment: "
      "height adjustment must be smaller than height stride (%" PRIu32 ")",
      xnn_operator_type_to_string(deconvolution_op->type), adjustment_height, deconvolution_op->stride_height);
    return xnn_status_invalid_parameter;
  }

  if (adjustment_width >= deconvolution_op->stride_width) {
    xnn_log_error(
      "failed to setup %s operator with %" PRIu32 " width adjustment: "
      "width adjustment must be smaller than width stride (%" PRIu32 ")",
      xnn_operator_type_to_string(deconvolution_op->type), adjustment_width, deconvolution_op->stride_width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    deconvolution_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  deconvolution_op->batch_size = batch_size;
  deconvolution_op->input_height = input_height;
  deconvolution_op->input_width = input_width;
  deconvolution_op->input = input;
  deconvolution_op->output = output;

  deconvolution_op->output_height = compute_output_dimension(
      input_height, deconvolution_op->padding_top + deconvolution_op->padding_bottom,
      adjustment_height, deconvolution_op->kernel_height, deconvolution_op->dilation_height, deconvolution_op->stride_height);
  deconvolution_op->output_width = deconvolution_op->output_width = compute_output_dimension(
      input_width, deconvolution_op->padding_left + deconvolution_op->padding_right,
      adjustment_width, deconvolution_op->kernel_width, deconvolution_op->dilation_width, deconvolution_op->stride_width);

  switch (deconvolution_op->ukernel.type) {
    case xnn_ukernel_type_igemm:
      return setup_conv_path(
        deconvolution_op,
        batch_size,
        input_height, input_width, input,
        deconvolution_op->output_height, deconvolution_op->output_width, output,
        log2_input_element_size, log2_filter_element_size, bias_element_size, log2_output_element_size,
        params, params_size, num_threads);
    case xnn_ukernel_type_subconv2d:
    {
      const bool no_padding = (deconvolution_op->padding_top | deconvolution_op->padding_right | deconvolution_op->padding_bottom | deconvolution_op->padding_left) == 0;
      const bool no_adjustment = (adjustment_height | adjustment_width) == 0;
      const bool use_gemm = no_padding && no_adjustment &&
        deconvolution_op->kernel_height == deconvolution_op->stride_height &&
        deconvolution_op->kernel_width == deconvolution_op->stride_width &&
        deconvolution_op->ukernel.igemm.gemm_case.function[XNN_UARCH_DEFAULT] != NULL;
      return setup_subconv2d_path(
        deconvolution_op,
        batch_size,
        input_height, input_width, input,
        deconvolution_op->output_height, deconvolution_op->output_width, output,
        log2_input_element_size, log2_filter_element_size, bias_element_size, log2_output_element_size,
        params, params_size, num_threads, use_gemm);
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_qs8(
    xnn_operator_t deconvolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    uint32_t adjustment_height,
    uint32_t adjustment_width,
    const int8_t* input,
    int8_t* output,
    pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_qs8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    input, output,
    0 /* log2(sizeof(input element)) = log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(int8_t)) */,
    &deconvolution_op->params.qs8_conv_minmax, sizeof(deconvolution_op->params.qs8_conv_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_qu8(
    xnn_operator_t deconvolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    uint32_t adjustment_height,
    uint32_t adjustment_width,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_qu8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    input, output,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(uint8_t)) */,
    &deconvolution_op->params.qu8_conv_minmax, sizeof(deconvolution_op->params.qu8_conv_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_f16(
    xnn_operator_t deconvolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    uint32_t adjustment_height,
    uint32_t adjustment_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_f16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    input, output,
    1 /* log2(sizeof(input element)) = log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(filter element)) = log2(sizeof(uint16_t)) */,
    sizeof(uint16_t) /* sizeof(bias element) */,
    1 /* log2(sizeof(output element)) = log2(sizeof(uint16_t)) */,
    &deconvolution_op->params.f16_scaleminmax, sizeof(deconvolution_op->params.f16_scaleminmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_f32(
    xnn_operator_t deconvolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    uint32_t adjustment_height,
    uint32_t adjustment_width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    input, output,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    &deconvolution_op->params.f32_minmax, sizeof(deconvolution_op->params.f32_minmax),
    pthreadpool_get_threads_count(threadpool));
}
