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

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/indirection.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
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

enum xnn_status xnn_create_deconvolution2d_nhwc_q8(
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
  xnn_operator_t deconvolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Deconvolution operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    goto error;
  }

  if (stride_width == 0 || stride_height == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "x%" PRIu32 " stride: stride dimensions must be non-zero",
      stride_width, stride_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 " groups: number of groups must be non-zero", groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %zu input channels per group: "
      "number of channels must be non-zero",
      group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %zu output channels per group: "
      "number of channels must be non-zero",
      group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_pixel_stride < input_channels) {
    xnn_log_error(
      "failed to create Deconvolution operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      input_pixel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_pixel_stride < output_channels) {
    xnn_log_error(
      "failed to create Deconvolution operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      output_pixel_stride, groups, group_output_channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create Deconvolution operator with %.7g input scale: scale must be finite, normalized, and positive",
      input_scale);
    goto error;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create Deconvolution operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      kernel_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create Deconvolution operator with %.7g output scale: scale must be finite, normalized, and positive",
      output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Deconvolution operator with [%" PRIu8 ", %" PRIu8 "] output range: "
      "range min must be below range max",
      output_min, output_max);
    goto error;
  }

  const bool any_padding = (output_padding_left | output_padding_top | output_padding_right | output_padding_bottom) != 0;
  if (any_padding && (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
      "TensorFlow SAME padding can't be combined with explicit padding specification",
      output_padding_top, output_padding_left, output_padding_bottom, output_padding_right);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  const float deconvolution_scale = input_scale * kernel_scale / output_scale;
  if (deconvolution_scale >= 1.0f) {
    xnn_log_error(
      "failed to create Deconvolution operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "Deconvolution operator scale %.7g is greater or equal to 1.0",
      input_scale, kernel_scale, output_scale, deconvolution_scale);
    goto error;
  }

  status = xnn_status_out_of_memory;

  deconvolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (deconvolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Deconvolution operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  const uint32_t mr = xnn_params.q8.gemm.mr;
  const uint32_t nr = xnn_params.q8.gemm.nr;
  const uint32_t kr = UINT32_C(1) << xnn_params.q8.gemm.log2_kr;
  const struct xnn_hmp_igemm_ukernel igemm_ukernel = xnn_params.q8.gemm.minmax.igemm;
  const struct xnn_hmp_gemm_ukernel gemm_ukernel = xnn_params.q8.gemm.minmax.gemm;

  const uint32_t n_stride = round_up(group_output_channels, nr);
  const uint32_t k_stride = round_up_po2(group_input_channels, kr);
  const uint32_t kernel_size = kernel_height * kernel_width;
  enum xnn_ukernel_type ukernel_type = xnn_ukernel_type_igemm;
  size_t packed_group_weights_size = (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) * n_stride;
  if (max(stride_height, stride_width) > 1 && max(dilation_height, dilation_width) == 1 && stride_width <= kernel_width && stride_height <= kernel_height) {
    ukernel_type = xnn_ukernel_type_subconv2d;
    const size_t subkernels = stride_height * stride_width;
    packed_group_weights_size = n_stride *
      (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t) * subkernels);

    const size_t subconvolution_buffer_size = sizeof(struct subconvolution_params) * subkernels;
    deconvolution_op->subconvolution_buffer = xnn_allocate_zero_memory(subconvolution_buffer_size);
    if (deconvolution_op->subconvolution_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subconvolution buffer", subconvolution_buffer_size);
      goto error;
    }

    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
      for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
        const size_t subkernel_height = divide_round_up(kernel_height - offset_y, stride_height);
        const size_t subkernel_width = divide_round_up(kernel_width - offset_x, stride_width);
        const size_t subkernel_size = subkernel_height * subkernel_width;

        subconvolution_params->indirection_x_stride = sizeof(void*) * subkernel_size;
        subconvolution_params->w_stride = sizeof(int32_t) + k_stride * subkernel_size * sizeof(uint8_t);
        subconvolution_params++;
      }
    }
  }
  deconvolution_op->packed_weights = xnn_allocate_simd_memory(packed_group_weights_size * groups);
  if (deconvolution_op->packed_weights == NULL) {
    xnn_log_error("failed to allocate %zu bytes for packed weights", packed_group_weights_size * groups);
    goto error;
  }
  memset(deconvolution_op->packed_weights, kernel_zero_point, packed_group_weights_size * groups);

  switch (ukernel_type) {
    case xnn_ukernel_type_igemm:
      xnn_pack_q8_conv_goki_w(
        groups, group_output_channels, kernel_size, group_input_channels,
        nr, kr,
        input_zero_point, kernel_zero_point,
        kernel, bias, deconvolution_op->packed_weights);
      break;
    case xnn_ukernel_type_subconv2d:
      xnn_pack_q8_deconv_goki_w(
        groups, group_output_channels, kernel_height, kernel_width, group_input_channels,
        stride_height, stride_width,
        nr, kr,
        input_zero_point, kernel_zero_point,
        kernel, bias, deconvolution_op->packed_weights, deconvolution_op->subconvolution_buffer);
      break;
    default:
      XNN_UNREACHABLE;
  }

  size_t zero_size = sizeof(uint8_t) * k_stride + XNN_EXTRA_BYTES;
  void* zero_buffer = xnn_allocate_simd_memory(zero_size);
  if (zero_buffer == NULL) {
    xnn_log_error("failed to allocate %zu bytes for zero padding", zero_size);
    goto error;
  }
  memset(zero_buffer, input_zero_point, zero_size);
  deconvolution_op->zero_buffer = zero_buffer;

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

  deconvolution_op->kernel_zero_point = kernel_zero_point;

  deconvolution_op->q8_gemm_params =
    xnn_init_q8_gemm_params(
      input_zero_point, kernel_zero_point,
      deconvolution_scale, output_zero_point, output_min, output_max);

  deconvolution_op->type = xnn_operator_type_deconvolution_nhwc_q8;
  deconvolution_op->ukernel.type = ukernel_type;
  deconvolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
    .general_case = igemm_ukernel,
    .gemm_case = gemm_ukernel,
    .mr = mr,
    .nr = nr,
    .kr = kr,
  };

  if (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    if ((stride_height | stride_width) == 1) {
      // Padding can be computed statically
      const uint32_t padding_height = (kernel_height - 1) * dilation_height;
      const uint32_t padding_width = (kernel_width - 1) * dilation_width;

      const uint32_t padding_top = padding_height / 2;
      const uint32_t padding_left = padding_width / 2;

      deconvolution_op->padding_top = padding_top;
      deconvolution_op->padding_left = padding_left;
      deconvolution_op->padding_bottom = padding_height - padding_top;
      deconvolution_op->padding_right = padding_width - padding_left;
    } else {
      deconvolution_op->flags = XNN_FLAG_TENSORFLOW_SAME_PADDING;
    }
  }

  deconvolution_op->state = xnn_run_state_invalid;

  *deconvolution_op_out = deconvolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(deconvolution_op);
  return status;
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
  xnn_operator_t deconvolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Deconvolution operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    goto error;
  }

  if (stride_width == 0 || stride_height == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "x%" PRIu32 " stride: stride dimensions must be non-zero",
      stride_width, stride_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 " groups: number of groups must be non-zero", groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %zu input channels per group: "
      "number of channels must be non-zero",
      group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %zu output channels per group: "
      "number of channels must be non-zero",
      group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_pixel_stride < input_channels) {
    xnn_log_error(
      "failed to create Deconvolution operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      input_pixel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_pixel_stride < output_channels) {
    xnn_log_error(
      "failed to create Deconvolution operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      output_pixel_stride, groups, group_output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create Deconvolution operator with NaN output lower bound: lower bound must be non-NaN");
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create Deconvolution operator with NaN output upper bound: upper bound must be non-NaN");
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Deconvolution operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    goto error;
  }

  const bool any_padding = (output_padding_left | output_padding_top | output_padding_right | output_padding_bottom) != 0;
  if (any_padding && (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    xnn_log_error(
      "failed to create Deconvolution operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
      "TensorFlow SAME padding can't be combined with explicit padding specification",
      output_padding_top, output_padding_left, output_padding_bottom, output_padding_right);
    goto error;
  }

  status = xnn_status_out_of_memory;

  deconvolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (deconvolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Deconvolution operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  const struct gemm_parameters* gemm_params = &xnn_params.f32.gemm;
  if (gemm_params->nr > group_output_channels) {
    // Default micro-kernel is suboptimal. Try to find a better micro-kernel.
    if (xnn_params.f32.gemm2.minmax.igemm.function[XNN_UARCH_DEFAULT] != NULL) {
      gemm_params = &xnn_params.f32.gemm2;
    }
  }
  const uint32_t mr = gemm_params->mr;
  const uint32_t nr = gemm_params->nr;
  const uint32_t kr = UINT32_C(1) << gemm_params->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_params->log2_sr;
  const struct gemm_fused_ukernels* ukernels = &gemm_params->minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_params->linear.gemm.function[XNN_UARCH_DEFAULT] != NULL) {
    ukernels = &gemm_params->linear;
  }
  struct xnn_hmp_igemm_ukernel igemm_ukernel = ukernels->igemm;
  struct xnn_hmp_gemm_ukernel gemm_ukernel = ukernels->gemm;

  const uint32_t n_stride = round_up(group_output_channels, nr);
  const uint32_t k_stride = round_up_po2(group_input_channels, kr);
  const uint32_t kernel_size = kernel_height * kernel_width;
  enum xnn_ukernel_type ukernel_type = xnn_ukernel_type_igemm;
  size_t packed_group_weights_size = (sizeof(float) * kernel_size * k_stride + sizeof(float)) * n_stride;
  if (max(stride_height, stride_width) > 1 && max(dilation_height, dilation_width) == 1 && stride_width <= kernel_width && stride_height <= kernel_height) {
    ukernel_type = xnn_ukernel_type_subconv2d;
    const size_t subkernels = stride_height * stride_width;
    packed_group_weights_size = n_stride *
      (sizeof(float) * kernel_size * k_stride + sizeof(float) * subkernels);

    const size_t subconvolution_buffer_size = sizeof(struct subconvolution_params) * subkernels;
    deconvolution_op->subconvolution_buffer = xnn_allocate_zero_memory(subconvolution_buffer_size);
    if (deconvolution_op->subconvolution_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subconvolution buffer", subconvolution_buffer_size);
      goto error;
    }

    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
      for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
        const size_t subkernel_height = divide_round_up(kernel_height - offset_y, stride_height);
        const size_t subkernel_width = divide_round_up(kernel_width - offset_x, stride_width);
        const size_t subkernel_size = subkernel_height * subkernel_width;

        subconvolution_params->indirection_x_stride = sizeof(void*) * subkernel_size;
        subconvolution_params->w_stride = sizeof(float) + k_stride * subkernel_size * sizeof(float);
        subconvolution_params++;
      }
    }
  }
  deconvolution_op->packed_weights = xnn_allocate_simd_memory(packed_group_weights_size * groups);
  if (deconvolution_op->packed_weights == NULL) {
    xnn_log_error("failed to allocate %zu bytes for packed weights", packed_group_weights_size * groups);
    goto error;
  }
  memset(deconvolution_op->packed_weights, 0, packed_group_weights_size * groups);

  switch (ukernel_type) {
    case xnn_ukernel_type_igemm:
      xnn_pack_f32_conv_goki_w(
        groups, group_output_channels, kernel_size, group_input_channels,
        nr, kr, sr,
        kernel, bias, deconvolution_op->packed_weights);
      break;
    case xnn_ukernel_type_subconv2d:
      xnn_pack_f32_deconv_goki_w(
        groups, group_output_channels, kernel_height, kernel_width, group_input_channels,
        stride_height, stride_width,
        nr, kr, sr,
        kernel, bias, deconvolution_op->packed_weights, deconvolution_op->subconvolution_buffer);
      break;
    default:
      XNN_UNREACHABLE;
  }

  const size_t zero_size = k_stride * sizeof(float) + XNN_EXTRA_BYTES;
  void* zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
  if (zero_buffer == NULL) {
    xnn_log_error("failed to allocate %zu bytes for zero padding", zero_size);
    goto error;
  }
  deconvolution_op->zero_buffer = zero_buffer;

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

  deconvolution_op->f32_minmax_params = xnn_init_f32_minmax_params(output_min, output_max);

  deconvolution_op->type = xnn_operator_type_deconvolution_nhwc_f32;
  deconvolution_op->ukernel.type = ukernel_type;
  deconvolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
    .general_case = igemm_ukernel,
    .gemm_case = gemm_ukernel,
    .mr = mr,
    .nr = nr,
    .kr = kr,
  };

  if (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    if ((stride_height | stride_width) == 1) {
      // Padding can be computed statically
      const uint32_t padding_height = (kernel_height - 1) * dilation_height;
      const uint32_t padding_width = (kernel_width - 1) * dilation_width;

      const uint32_t padding_top = padding_height / 2;
      const uint32_t padding_left = padding_width / 2;

      deconvolution_op->padding_top = padding_top;
      deconvolution_op->padding_left = padding_left;
      deconvolution_op->padding_bottom = padding_height - padding_top;
      deconvolution_op->padding_right = padding_width - padding_left;
    } else {
      deconvolution_op->flags = XNN_FLAG_TENSORFLOW_SAME_PADDING;
    }
  }

  deconvolution_op->state = xnn_run_state_invalid;

  *deconvolution_op_out = deconvolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(deconvolution_op);
  return status;
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
      xnn_log_error("failed to allocate %zu bytes for indirection buffer", indirection_buffer_size);
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
    (round_up_po2(group_input_channels, deconvolution_op->ukernel.igemm.kr) * kernel_size << log2_filter_element_size);
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
  memcpy(&deconvolution_op->context.igemm.params, params, sizeof(deconvolution_op->context.igemm.params));

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
    deconvolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
    deconvolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_igemm;
    deconvolution_op->compute.range[0] = batch_size;
    deconvolution_op->compute.range[1] = output_size;
    deconvolution_op->compute.range[2] = group_output_channels;
    deconvolution_op->compute.tile[0] = mr;
    deconvolution_op->compute.tile[1] = nc;
  } else {
    deconvolution_op->compute.type = xnn_parallelization_type_4d_tile_2d;
    deconvolution_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_igemm;
    deconvolution_op->compute.range[0] = batch_size;
    deconvolution_op->compute.range[1] = groups;
    deconvolution_op->compute.range[2] = output_size;
    deconvolution_op->compute.range[3] = group_output_channels;
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
        xnn_log_error("failed to allocate %zu bytes for indirection buffer", indirection_buffer_size);
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
  const size_t w_stride = stride_height * stride_width * bias_element_size +
    (round_up_po2(group_input_channels, kr) * kernel_size << log2_filter_element_size);
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
    memcpy(&deconvolution_op->context.subgemm.params, params, sizeof(deconvolution_op->context.subgemm.params));
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
    memcpy(&deconvolution_op->context.subconv.params, params, sizeof(deconvolution_op->context.subconv.params));
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

static enum xnn_status setup_deconvolution2d(
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
  size_t num_threads)
{
  deconvolution_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Deconvolution operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup Deconvolution with %zux%zu input: input dimensions must be non-zero",
      input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (adjustment_height >= deconvolution_op->stride_height) {
    xnn_log_error(
      "failed to setup Deconvolution with %" PRIu32 " height adjustment: "
      "height adjustment must be smaller than height stride (%" PRIu32 ")",
      adjustment_height, deconvolution_op->stride_height);
    return xnn_status_invalid_parameter;
  }

  if (adjustment_width >= deconvolution_op->stride_width) {
    xnn_log_error(
      "failed to setup Deconvolution with %" PRIu32 " width adjustment: "
      "width adjustment must be smaller than width stride (%" PRIu32 ")",
      adjustment_width, deconvolution_op->stride_width);
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

  if (deconvolution_op->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    // Recompute padding for the input size.
    const uint32_t dilated_kernel_height_minus_1 = (deconvolution_op->kernel_height - 1) * deconvolution_op->dilation_height;
    const uint32_t dilated_kernel_width_minus_1 = (deconvolution_op->kernel_width - 1) * deconvolution_op->dilation_width;

    const size_t total_padding_height = doz(dilated_kernel_height_minus_1, (input_height - 1) % deconvolution_op->stride_height);
    const size_t total_padding_width = doz(dilated_kernel_width_minus_1, (input_width - 1) % deconvolution_op->stride_width);

    const uint32_t padding_top = deconvolution_op->padding_top = total_padding_height / 2;
    const uint32_t padding_left = deconvolution_op->padding_left = total_padding_width / 2;
    deconvolution_op->padding_bottom = total_padding_height - padding_top;
    deconvolution_op->padding_right = total_padding_width - padding_left;
  }

  const size_t output_height = deconvolution_op->output_height = compute_output_dimension(
    input_height, deconvolution_op->padding_top + deconvolution_op->padding_bottom,
    adjustment_height, deconvolution_op->kernel_height, deconvolution_op->dilation_height, deconvolution_op->stride_height);
  const size_t output_width = deconvolution_op->output_width = compute_output_dimension(
    input_width, deconvolution_op->padding_left + deconvolution_op->padding_right,
    adjustment_width, deconvolution_op->kernel_width, deconvolution_op->dilation_width, deconvolution_op->stride_width);

  switch (deconvolution_op->ukernel.type) {
    case xnn_ukernel_type_igemm:
      return setup_conv_path(
        deconvolution_op,
        batch_size,
        input_height, input_width, input,
        output_height, output_width, output,
        log2_input_element_size, log2_filter_element_size, bias_element_size, log2_output_element_size,
        params, num_threads);
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
        output_height, output_width, output,
        log2_input_element_size, log2_filter_element_size, bias_element_size, log2_output_element_size,
        params, num_threads, use_gemm);
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_q8(
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
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_q8) {
    xnn_log_error("failed to setup Deconvolution (NHWC, Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_deconvolution2d(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    input, output,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(uint8_t)) */,
    &deconvolution_op->q8_gemm_params,
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
    xnn_log_error("failed to setup Deconvolution (NHWC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_deconvolution2d(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    input, output,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    &deconvolution_op->f32_minmax_params,
    pthreadpool_get_threads_count(threadpool));
}
