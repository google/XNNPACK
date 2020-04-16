// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/compute.h>
#include <xnnpack/indirection.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t subsampling_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(padded_input_dimension, effective_kernel_dimension) / subsampling_dimension + 1;
}

static inline size_t compute_output_dimension_with_tf_same_padding(
    size_t input_dimension,
    size_t subsampling_dimension)
{
  return divide_round_up(input_dimension, subsampling_dimension);
}

static const struct dwconv_parameters* find_dwigemm_ukernel(
    size_t kernel_size,
    const struct dwconv_parameters* ukernel,
    size_t num_ukernels)
{
  while (num_ukernels-- != 0) {
    if (ukernel->primary_tile == kernel_size) {
      return ukernel;
    }
    ukernel++;
  }
  return NULL;
}

enum xnn_status xnn_create_convolution2d_nhwc_q8(
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
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Convolution operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " subsampling: "
      "subsampling dimensions must be non-zero",
      subsampling_width, subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 " groups: number of groups must be non-zero", groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %zu input channels per group: "
      "number of channels must be non-zero",
      group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %zu output channels per group: "
      "number of channels must be non-zero",
      group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_pixel_stride < input_channels) {
    xnn_log_error(
      "failed to create Convolution operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%zu)",
      input_pixel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_pixel_stride < output_channels) {
    xnn_log_error(
      "failed to create Convolution operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      output_pixel_stride, groups, group_output_channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create Convolution operator with %.7g input scale: scale must be finite, normalized, and positive",
      input_scale);
    goto error;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create Convolution operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      kernel_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create Convolution operator with %.7g output scale: scale must be finite, normalized, and positive",
      output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Convolution operator with [%" PRIu8 ", %" PRIu8 "] output range: "
      "range min must be below range max",
      output_min, output_max);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create Depthwise Convolution operator with %zu input channels per group: "
      "Depthwise Convolution must have exactly 1 input channel per group",
      group_input_channels);
    goto error;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to create Convolution operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      goto error;
    }
  }

  status = xnn_status_unsupported_parameter;

  const float convolution_scale = input_scale * kernel_scale / output_scale;
  if (convolution_scale >= 1.0f) {
    xnn_log_error(
      "failed to create Convolution operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "convolution scale %.7g is greater or equal to 1.0",
      input_scale, kernel_scale, output_scale, convolution_scale);
    goto error;
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Convolution operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  const size_t kernel_size = kernel_height * kernel_width;

  enum xnn_ukernel_type ukernel_type = xnn_ukernel_type_none;
  const struct dwconv_parameters* dwconv_parameters = NULL;
  if (group_input_channels == 1 && group_output_channels == 1 && groups > 1 &&
      (dwconv_parameters = find_dwigemm_ukernel(kernel_size, xnn_params.q8.dwconv, XNN_MAX_Q8_DWCONV_UKERNELS)) != NULL)
  {
    ukernel_type = xnn_ukernel_type_dwconv;
  } else if (kernel_size == 1 && subsampling_height == 1 && subsampling_width == 1 && !any_padding) {
    ukernel_type = xnn_ukernel_type_gemm;
  } else {
    ukernel_type = xnn_ukernel_type_igemm;
  }

  size_t zero_size = 0;
  switch (ukernel_type) {
    case xnn_ukernel_type_dwconv:
    {
      assert(dwconv_parameters != NULL);
      assert(dwconv_parameters->primary_tile == kernel_size);

      const size_t c_stride = round_up_po2(groups, dwconv_parameters->channel_tile);
      const size_t packed_weights_size = (sizeof(uint8_t) * kernel_size + sizeof(int32_t)) * c_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_weights_size);
        goto error;
      }

      if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
        xnn_pack_q8_dwconv_hwg_w(
          kernel_height, kernel_width,
          groups, dwconv_parameters->channel_tile,
          input_zero_point, kernel_zero_point,
          kernel, bias, convolution_op->packed_weights);
      } else {
        xnn_pack_q8_dwconv_ghw_w(
          kernel_height, kernel_width,
          groups, dwconv_parameters->channel_tile,
          input_zero_point, kernel_zero_point,
          kernel, bias, convolution_op->packed_weights);
      }

      convolution_op->ukernel.dwconv = (struct xnn_ukernel_dwconv) {
        .unipass_function = dwconv_parameters->minmax.unipass,
        .primary_tile = dwconv_parameters->primary_tile,
        .incremental_tile = dwconv_parameters->incremental_tile,
      };

      zero_size = sizeof(uint8_t) * c_stride + XNN_EXTRA_BYTES;
      break;
    }
    case xnn_ukernel_type_gemm:
    case xnn_ukernel_type_igemm:
    {
      const uint32_t nr = xnn_params.q8.gemm.nr;
      const uint32_t kr = UINT32_C(1) << xnn_params.q8.gemm.log2_kr;
      const size_t n_stride = round_up(group_output_channels, nr);
      const size_t k_stride = round_up_po2(group_input_channels, kr);

      const size_t packed_group_weights_size =
        (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) * n_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_group_weights_size * groups);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_group_weights_size * groups);
        goto error;
      }
      memset(convolution_op->packed_weights, kernel_zero_point, packed_group_weights_size * groups);

      switch (ukernel_type) {
        case xnn_ukernel_type_gemm:
          xnn_pack_q8_gemm_goi_w(
              groups, group_output_channels, group_input_channels,
              nr, kr,
              input_zero_point, kernel_zero_point,
              kernel, bias, convolution_op->packed_weights);
          convolution_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
            .mr = xnn_params.q8.gemm.mr,
            .nr = nr,
            .kr = kr,
            .general_case = xnn_params.q8.gemm.minmax.gemm,
          };
          break;
        case xnn_ukernel_type_igemm:
          if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
            xnn_pack_q8_conv_kgo_w(
                groups, group_output_channels, kernel_size,
                nr, kr,
                input_zero_point, kernel_zero_point,
                kernel, bias, convolution_op->packed_weights);
          } else {
            xnn_pack_q8_conv_goki_w(
                groups, group_output_channels, kernel_size, group_input_channels,
                nr, kr,
                input_zero_point, kernel_zero_point,
                kernel, bias, convolution_op->packed_weights);
          }
          convolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
            .mr = xnn_params.q8.gemm.mr,
            .nr = nr,
            .kr = kr,
            .general_case = xnn_params.q8.gemm.minmax.igemm,
          };
          break;
        default:
          XNN_UNREACHABLE;
      }

      zero_size = sizeof(uint8_t) * k_stride + XNN_EXTRA_BYTES;
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  const bool tf_same_padding = (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && kernel_size != 1;
  if (any_padding || tf_same_padding) {
    void* zero_buffer = xnn_allocate_simd_memory(zero_size);
    if (zero_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for zero padding", zero_size);
      goto error;
    }
    memset(zero_buffer, input_zero_point, zero_size);
    convolution_op->zero_buffer = zero_buffer;
  }

  convolution_op->padding_top = input_padding_top;
  convolution_op->padding_right = input_padding_right;
  convolution_op->padding_bottom = input_padding_bottom;
  convolution_op->padding_left = input_padding_left;

  convolution_op->kernel_height = kernel_height;
  convolution_op->kernel_width = kernel_width;
  convolution_op->stride_height = subsampling_height;
  convolution_op->stride_width = subsampling_width;
  convolution_op->dilation_height = dilation_height;
  convolution_op->dilation_width = dilation_width;
  convolution_op->groups = groups;
  convolution_op->group_input_channels = group_input_channels;
  convolution_op->group_output_channels = group_output_channels;
  convolution_op->input_pixel_stride = input_pixel_stride;
  convolution_op->output_pixel_stride = output_pixel_stride;

  convolution_op->kernel_zero_point = kernel_zero_point;

  convolution_op->q8_gemm_params =
    xnn_init_q8_gemm_params(
      input_zero_point, kernel_zero_point,
      convolution_scale, output_zero_point, output_min, output_max);

  convolution_op->type = xnn_operator_type_convolution_nhwc_q8;
  convolution_op->ukernel.type = ukernel_type;
  if (tf_same_padding) {
    convolution_op->flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  convolution_op->state = xnn_run_state_invalid;

  *convolution_op_out = convolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(convolution_op);
  return status;
}

enum xnn_status xnn_create_convolution2d_nhwc_f32(
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
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Convolution operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " subsampling: "
      "subsampling dimensions must be non-zero",
      subsampling_width, subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 " groups: number of groups must be non-zero", groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %zu input channels per group: "
      "number of channels must be non-zero",
      group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %zu output channels per group: "
      "number of channels must be non-zero",
      group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_pixel_stride < input_channels) {
    xnn_log_error(
      "failed to create Convolution operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%zu)",
      input_pixel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_pixel_stride < output_channels) {
    xnn_log_error(
      "failed to create Convolution operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      output_pixel_stride, groups, group_output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create Convolution operator with NaN output lower bound: lower bound must be non-NaN");
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create Convolution operator with NaN output upper bound: upper bound must be non-NaN");
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Convolution operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create Depthwise Convolution operator with %zu input channels per group: "
      "Depthwise Convolution must have exactly 1 input channel per group",
      group_input_channels);
    goto error;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to create Convolution operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      goto error;
    }
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Convolution operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  const size_t kernel_size = kernel_height * kernel_width;

  enum xnn_ukernel_type ukernel_type = xnn_ukernel_type_none;
  const struct dwconv_parameters* dwconv_parameters = NULL;
  const bool unit_subsampling = (subsampling_width | subsampling_height) == 1;
  if (group_input_channels == 1 && group_output_channels == 1 && kernel_size == 1 && unit_subsampling && !any_padding) {
    ukernel_type = xnn_ukernel_type_vmulcaddc;
  } else if (group_input_channels == 1 && group_output_channels == 1 && (dwconv_parameters =
               find_dwigemm_ukernel(kernel_size, xnn_params.f32.dwconv, XNN_MAX_F32_DWCONV_UKERNELS)) != NULL)
  {
    ukernel_type = xnn_ukernel_type_dwconv;
  } else if (kernel_size == 1 && unit_subsampling && !any_padding) {
    ukernel_type = xnn_ukernel_type_gemm;
  } else {
    ukernel_type = xnn_ukernel_type_igemm;
  }
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);

  size_t zero_size = 0;
  switch (ukernel_type) {
    case xnn_ukernel_type_vmulcaddc:
    {
      const size_t c_stride = round_up_po2(groups, xnn_params.f32.vmulcaddc.channel_tile);
      const size_t packed_weights_size = 2 * sizeof(float) * c_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_weights_size);
        goto error;
      }

      xnn_pack_f32_vmulcaddc_w(
        groups, xnn_params.f32.vmulcaddc.channel_tile,
        kernel, bias, convolution_op->packed_weights);

      convolution_op->ukernel.vmulcaddc = (struct xnn_ukernel_vmulcaddc) {
        .function = xnn_params.f32.vmulcaddc.ukernel,
        .mr = xnn_params.f32.vmulcaddc.row_tile,
      };
      break;
    }
    case xnn_ukernel_type_dwconv:
    {
      assert(dwconv_parameters != NULL);
      assert(dwconv_parameters->primary_tile == kernel_size);

      const size_t c_stride = round_up_po2(groups, dwconv_parameters->channel_tile);
      const size_t packed_weights_size = (kernel_size + 1) * sizeof(float) * c_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_weights_size);
        goto error;
      }

      if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
        xnn_pack_f32_dwconv_hwg_w(
          kernel_height, kernel_width,
          groups, dwconv_parameters->channel_tile,
          kernel, bias, convolution_op->packed_weights);
      } else {
        xnn_pack_f32_dwconv_ghw_w(
          kernel_height, kernel_width,
          groups, dwconv_parameters->channel_tile,
          kernel, bias, convolution_op->packed_weights);
      }

      const union dwconv_fused_ukernels* ukernels = &dwconv_parameters->minmax;
      if (linear_activation && dwconv_parameters->linear.unipass != NULL) {
        ukernels = &dwconv_parameters->linear;
      }
      convolution_op->ukernel.dwconv = (struct xnn_ukernel_dwconv) {
        .unipass_function = ukernels->unipass,
        .primary_tile = dwconv_parameters->primary_tile,
        .incremental_tile = dwconv_parameters->incremental_tile,
      };

      zero_size = sizeof(float) * c_stride;
      break;
    }
    case xnn_ukernel_type_gemm:
    case xnn_ukernel_type_igemm:
    {
      const uint32_t nr = xnn_params.f32.gemm.nr;
      const uint32_t kr = UINT32_C(1) << xnn_params.f32.gemm.log2_kr;
      const uint32_t sr = UINT32_C(1) << xnn_params.f32.gemm.log2_sr;
      const size_t n_stride = round_up(group_output_channels, nr);
      const size_t k_stride = round_up_po2(group_input_channels, kr);

      const size_t packed_group_weights_size = (kernel_size * k_stride + 1) * sizeof(float) * n_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_group_weights_size * groups);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_group_weights_size * groups);
        goto error;
      }
      memset(convolution_op->packed_weights, 0, packed_group_weights_size * groups);

      const struct gemm_fused_ukernels* ukernels = &xnn_params.f32.gemm.minmax;
      if (linear_activation && xnn_params.f32.gemm.linear.gemm.function[XNN_UARCH_DEFAULT] != NULL) {
        ukernels = &xnn_params.f32.gemm.linear;
      }
      switch (ukernel_type) {
        case xnn_ukernel_type_gemm:
          xnn_pack_f32_gemm_goi_w(
              groups, group_output_channels, group_input_channels,
              nr, kr, sr,
              kernel, bias, convolution_op->packed_weights);
          convolution_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
            .mr = xnn_params.f32.gemm.mr,
            .nr = nr,
            .kr = kr,
            .general_case = ukernels->gemm,
            .mr1_case = ukernels->gemm1,
          };
          break;
        case xnn_ukernel_type_igemm:
          if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
            xnn_pack_f32_conv_kgo_w(
              groups, group_output_channels, kernel_size,
              nr, kr,
              kernel, bias, convolution_op->packed_weights);
          } else {
            xnn_pack_f32_conv_goki_w(
              groups, group_output_channels, kernel_size, group_input_channels,
              nr, kr, sr,
              kernel, bias, convolution_op->packed_weights);
          }
          convolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
            .mr = xnn_params.f32.gemm.mr,
            .nr = nr,
            .kr = kr,
            .general_case = ukernels->igemm,
            .mr1_case = ukernels->igemm1,
          };
          break;
        default:
          XNN_UNREACHABLE;
      }

      zero_size = sizeof(float) * k_stride;
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  const bool tf_same_padding = (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && kernel_size != 1;
  if (any_padding || tf_same_padding) {
    void* zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
    if (zero_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for zero padding", zero_size);
      goto error;
    }
    convolution_op->zero_buffer = zero_buffer;
  }

  convolution_op->padding_top = input_padding_top;
  convolution_op->padding_right = input_padding_right;
  convolution_op->padding_bottom = input_padding_bottom;
  convolution_op->padding_left = input_padding_left;

  convolution_op->kernel_height = kernel_height;
  convolution_op->kernel_width = kernel_width;
  convolution_op->stride_height = subsampling_height;
  convolution_op->stride_width = subsampling_width;
  convolution_op->dilation_height = dilation_height;
  convolution_op->dilation_width = dilation_width;
  convolution_op->groups = groups;
  convolution_op->group_input_channels = group_input_channels;
  convolution_op->group_output_channels = group_output_channels;
  convolution_op->input_pixel_stride = input_pixel_stride;
  convolution_op->output_pixel_stride = output_pixel_stride;

  convolution_op->f32_minmax_params = xnn_init_f32_minmax_params(output_min, output_max);

  convolution_op->type = xnn_operator_type_convolution_nhwc_f32;
  convolution_op->ukernel.type = ukernel_type;
  if (tf_same_padding) {
    convolution_op->flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  convolution_op->state = xnn_run_state_invalid;

  *convolution_op_out = convolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(convolution_op);
  return status;
}

static enum xnn_status setup_convolution2d_nhwc(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  size_t num_threads)
{
  convolution_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Convolution operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup Convolution operator with %zux%zu input: input dimensions must be non-zero",
      input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    convolution_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  convolution_op->batch_size = batch_size;
  convolution_op->input_height = input_height;
  convolution_op->input_width = input_width;
  convolution_op->input = input;

  if (convolution_op->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    convolution_op->output_height = compute_output_dimension_with_tf_same_padding(
        input_height, convolution_op->stride_height);
    convolution_op->output_width = compute_output_dimension_with_tf_same_padding(
        input_width, convolution_op->stride_width);

    const uint32_t effective_kernel_height = (convolution_op->kernel_height - 1) * convolution_op->dilation_height + 1;
    const uint32_t effective_kernel_width = (convolution_op->kernel_width - 1) * convolution_op->dilation_width + 1;
    const size_t total_padding_height =
      (convolution_op->output_height - 1) * convolution_op->stride_height + effective_kernel_height - input_height;
    const size_t total_padding_width =
      (convolution_op->output_width - 1) * convolution_op->stride_width + effective_kernel_width - input_width;
    convolution_op->padding_top = total_padding_height / 2;
    convolution_op->padding_left = total_padding_width / 2;
    convolution_op->padding_bottom = total_padding_height - convolution_op->padding_top;
    convolution_op->padding_right = total_padding_width - convolution_op->padding_left;
  } else {
    convolution_op->output_height = compute_output_dimension(
        convolution_op->padding_top + input_height + convolution_op->padding_bottom,
        convolution_op->kernel_height,
        convolution_op->dilation_height,
        convolution_op->stride_height);
    convolution_op->output_width = compute_output_dimension(
        convolution_op->padding_left + input_width + convolution_op->padding_right,
        convolution_op->kernel_width,
        convolution_op->dilation_width,
        convolution_op->stride_width);
  }
  convolution_op->output = output;

  switch (convolution_op->ukernel.type) {
    case xnn_ukernel_type_gemm:
    {
      // Convolution maps directly to GEMM and doesn't use indirection buffer.

      const size_t output_height = convolution_op->output_height;
      const size_t output_width = convolution_op->output_width;
      const size_t output_size = output_height * output_width;
      const size_t batch_output_size = batch_size * output_size;

      const size_t groups = convolution_op->groups;
      const size_t group_input_channels = convolution_op->group_input_channels;
      const size_t w_stride = (round_up_po2(group_input_channels, convolution_op->ukernel.gemm.kr) << log2_filter_element_size) + bias_element_size;
      const size_t group_output_channels = convolution_op->group_output_channels;

      uint32_t mr = convolution_op->ukernel.gemm.mr;
      const uint32_t nr = convolution_op->ukernel.gemm.nr;
      struct xnn_hmp_gemm_ukernel gemm_ukernel = convolution_op->ukernel.gemm.general_case;
      if (batch_output_size == 1 && convolution_op->ukernel.gemm.mr1_case.function[XNN_UARCH_DEFAULT] != NULL) {
        mr = 1;
        gemm_ukernel = convolution_op->ukernel.gemm.mr1_case;
      }

      convolution_op->context.gemm = (struct gemm_context) {
          .k_scaled = group_input_channels << log2_input_element_size,
          .a = input,
          .a_stride = convolution_op->input_pixel_stride << log2_input_element_size,
          .packed_w = convolution_op->packed_weights,
          .w_stride = w_stride,
          .wg_stride = w_stride * round_up(group_output_channels, nr),
          .c = output,
          .cm_stride = convolution_op->output_pixel_stride << log2_output_element_size,
          .cn_stride = nr << log2_output_element_size,
          .cg_stride = group_output_channels << log2_output_element_size,
          .log2_csize = log2_output_element_size,
          .ukernel = gemm_ukernel,
      };
      memcpy(&convolution_op->context.gemm.params, params, sizeof(convolution_op->context.gemm.params));

      size_t nc = group_output_channels;
      if (num_threads > 1) {
        const size_t num_other_tiles = groups * divide_round_up(batch_output_size, mr);
        const size_t target_tiles_per_thread = 5;
        const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
        if (max_nc < nc) {
          nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
        }
      }
      if (groups == 1) {
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
            convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d_with_uarch;
            convolution_op->compute.task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_gemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d;
            convolution_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
          }
        #else
          convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d;
          convolution_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
        #endif
        convolution_op->compute.range[0] = batch_output_size;
        convolution_op->compute.range[1] = group_output_channels;
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      } else {
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d_with_uarch;
            convolution_op->compute.task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_gemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
            convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
          }
        #else
          convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
          convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
        #endif
        convolution_op->compute.range[0] = groups;
        convolution_op->compute.range[1] = batch_output_size;
        convolution_op->compute.range[2] = group_output_channels;
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      }
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_ukernel_type_igemm:
    {
      const size_t groups = convolution_op->groups;
      const size_t kernel_height = convolution_op->kernel_height;
      const size_t kernel_width = convolution_op->kernel_width;
      const size_t kernel_size = kernel_height * kernel_width;
      const size_t output_height = convolution_op->output_height;
      const size_t output_width = convolution_op->output_width;
      const size_t output_size = output_height * output_width;

      uint32_t mr = convolution_op->ukernel.igemm.mr;
      const uint32_t nr = convolution_op->ukernel.igemm.nr;
      struct xnn_hmp_igemm_ukernel igemm_ukernel = convolution_op->ukernel.igemm.general_case;
      if (output_size == 1 && convolution_op->ukernel.igemm.mr1_case.function[XNN_UARCH_DEFAULT] != NULL) {
        mr = 1;
        igemm_ukernel = convolution_op->ukernel.igemm.mr1_case;
      }

      const size_t tiled_output_size = round_up(output_size, mr);
      const size_t indirection_buffer_size = sizeof(void*) * kernel_size * tiled_output_size;

      if (input_height != convolution_op->last_input_height ||
          input_width != convolution_op->last_input_width)
      {
        const void** indirection_buffer = (const void**) xnn_reallocate_memory((void*) convolution_op->indirection_buffer, indirection_buffer_size);
        if (indirection_buffer == NULL) {
          xnn_log_error("failed to allocate %zu bytes for indirection buffer", indirection_buffer_size);
          return xnn_status_out_of_memory;
        }
        convolution_op->indirection_buffer = indirection_buffer;
        convolution_op->last_input = input;
        convolution_op->last_input_height = input_height;
        convolution_op->last_input_width = input_width;

        xnn_indirection_init_conv2d(convolution_op, mr, log2_input_element_size);
      }

      const size_t group_input_channels = convolution_op->group_input_channels;
      const size_t w_stride = (round_up_po2(group_input_channels, convolution_op->ukernel.igemm.kr) * kernel_size << log2_filter_element_size) + bias_element_size;
      const size_t group_output_channels = convolution_op->group_output_channels;
      convolution_op->context.igemm = (struct igemm_context) {
          .ks = kernel_size,
          .ks_scaled = kernel_size * mr * sizeof(void*),
          .kc = group_input_channels << log2_input_element_size,
          .w_stride = w_stride,
          .indirect_a = convolution_op->indirection_buffer,
          .a_offset = (size_t) ((uintptr_t) input - (uintptr_t) convolution_op->last_input),
          .zero = convolution_op->zero_buffer,
          .packed_w = convolution_op->packed_weights,
          .c = convolution_op->output,
          .cm_stride = convolution_op->output_pixel_stride << log2_output_element_size,
          .cn_stride = nr << log2_output_element_size,
          .ga_stride = group_input_channels << log2_input_element_size,
          .gw_stride = w_stride * round_up(group_output_channels, nr),
          .gc_stride = group_output_channels << log2_output_element_size,
          .ba_stride = input_height * input_width * convolution_op->input_pixel_stride << log2_input_element_size,
          .bc_stride = output_size * convolution_op->output_pixel_stride << log2_output_element_size,
          .log2_csize = log2_output_element_size,
          .ukernel = igemm_ukernel,
      };
      memcpy(&convolution_op->context.igemm.params, params, sizeof(convolution_op->context.igemm.params));

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
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d_with_uarch;
            convolution_op->compute.task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_igemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
            convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_igemm;
          }
        #else
          convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
          convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_igemm;
        #endif
        convolution_op->compute.range[0] = batch_size;
        convolution_op->compute.range[1] = output_size;
        convolution_op->compute.range[2] = group_output_channels;
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      } else {
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
            convolution_op->compute.type = xnn_parallelization_type_4d_tile_2d_with_uarch;
            convolution_op->compute.task_4d_tile_2d_with_id = (pthreadpool_task_4d_tile_2d_with_id_t) xnn_compute_hmp_grouped_igemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_4d_tile_2d;
            convolution_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_igemm;
          }
        #else
          convolution_op->compute.type = xnn_parallelization_type_4d_tile_2d;
          convolution_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_igemm;
        #endif
        convolution_op->compute.range[0] = batch_size;
        convolution_op->compute.range[1] = groups;
        convolution_op->compute.range[2] = output_size;
        convolution_op->compute.range[3] = group_output_channels;
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      }
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_ukernel_type_dwconv:
    {
      size_t valid_batch_size = 0;
      if (input == convolution_op->last_input &&
          input_height == convolution_op->last_input_height &&
          input_width == convolution_op->last_input_width)
      {
        valid_batch_size = convolution_op->valid_batch_size;
        if (batch_size <= valid_batch_size) {
          convolution_op->compute.range[0] = batch_size * convolution_op->output_height;
          convolution_op->context.dwconv.output = output;
          convolution_op->state = xnn_run_state_ready;
          return xnn_status_success;
        }
      }

      const size_t kernel_height = convolution_op->kernel_height;
      const size_t kernel_width = convolution_op->kernel_width;
      const size_t kernel_size = kernel_height * kernel_width;
      const size_t output_height = convolution_op->output_height;
      const size_t output_width = convolution_op->output_width;
      const size_t step_width = convolution_op->dilation_width == 1 ? convolution_op->stride_width : kernel_width;
      const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;
      const size_t indirection_buffer_size = sizeof(void*) * batch_size * output_height * step_height;

      const void** indirection_buffer =
        (const void**) xnn_reallocate_memory((void*) convolution_op->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error("failed to allocate %zu bytes for indirection buffer", indirection_buffer_size);
        return xnn_status_out_of_memory;
      }
      convolution_op->indirection_buffer = indirection_buffer;

      xnn_indirection_init_dwconv2d(convolution_op, valid_batch_size, step_height, step_width, log2_input_element_size);

      const size_t groups = convolution_op->groups;
      convolution_op->context.dwconv = (struct dwconv_context) {
          .groups = groups,
          .indirection_buffer = convolution_op->indirection_buffer,
          .indirection_buffer_row_stride = step_height,
          .indirection_buffer_col_stride = kernel_height * step_width * sizeof(void*),
          .packed_weights = convolution_op->packed_weights,
          .output = convolution_op->output,
          .output_width = output_width,
          .output_row_stride = output_width * convolution_op->output_pixel_stride << log2_output_element_size,
          .output_col_increment = (convolution_op->output_pixel_stride - groups) << log2_output_element_size,
          .unipass_ukernel = convolution_op->ukernel.dwconv.unipass_function,
      };
      memcpy(&convolution_op->context.dwconv.params, params, sizeof(convolution_op->context.dwconv.params));

      convolution_op->compute.type = xnn_parallelization_type_1d;
      convolution_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_dwconv_unipass;
      convolution_op->compute.range[0] = batch_size * output_height;
      convolution_op->state = xnn_run_state_ready;

      convolution_op->last_input = input;
      convolution_op->last_input_height = input_height;
      convolution_op->last_input_width = input_width;
      convolution_op->valid_batch_size = max(valid_batch_size, batch_size);

      return xnn_status_success;
    }
    case xnn_ukernel_type_vmulcaddc:
    {
      const size_t batch_output_size = batch_size * convolution_op->output_height * convolution_op->output_width;

      convolution_op->context.vmulcaddc = (struct vmulcaddc_context) {
          .n = convolution_op->groups << log2_input_element_size,
          .x = input,
          .x_stride = convolution_op->input_pixel_stride << log2_input_element_size,
          .w = convolution_op->packed_weights,
          .y = output,
          .y_stride = convolution_op->output_pixel_stride << log2_output_element_size,
          .ukernel = convolution_op->ukernel.vmulcaddc.function,
      };
      memcpy(&convolution_op->context.vmulcaddc.params, params, sizeof(convolution_op->context.vmulcaddc.params));

      size_t mc = batch_output_size;
      if (num_threads > 1) {
        const size_t target_tiles_per_thread = 5;
        const size_t max_mc = divide_round_up(batch_output_size, num_threads * target_tiles_per_thread);
        if (max_mc < mc) {
          const uint32_t mr = convolution_op->ukernel.vmulcaddc.mr;
          mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
        }
      }
      convolution_op->compute.type = xnn_parallelization_type_1d_tile_1d;
      convolution_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_vmulcaddc;
      convolution_op->compute.range[0] = batch_output_size;
      convolution_op->compute.tile[0] = mc;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_convolution2d_nhwc_q8(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nhwc_q8) {
    xnn_log_error("failed to setup Convolution (NHWC, Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nhwc(
    convolution_op,
    batch_size, input_height, input_width,
    input, output,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(uint8_t)) */,
    &convolution_op->q8_gemm_params,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convolution2d_nhwc_f32(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nhwc_f32) {
    xnn_log_error("failed to setup Convolution (NHWC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nhwc(
    convolution_op,
    batch_size, input_height, input_width,
    input, output,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    &convolution_op->f32_minmax_params,
    pthreadpool_get_threads_count(threadpool));
}
