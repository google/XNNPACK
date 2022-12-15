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

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/compute.h>
#include <xnnpack/indirection.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/pack.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>

enum xnn_status xnn_create_convolution2d_nchw_f16(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    const void* kernel,
    const void* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_caches_t caches,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  const uint32_t datatype_init_flags = XNN_INIT_FLAG_F16 | XNN_INIT_FLAG_F16_NATIVE;
  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), kernel_width, kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " subsampling: subsampling dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), subsampling_width, subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 " groups: number of groups must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu input channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu output channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16),
      input_channel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16),
      output_channel_stride, groups, group_output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
    goto error;
  }

  const uint16_t fp16_output_min = fp16_ieee_from_fp32_value(output_min);
  const uint16_t fp16_output_max = fp16_ieee_from_fp32_value(output_max);
  const float rounded_output_min = fp16_ieee_to_fp32_value(fp16_output_min);
  const float rounded_output_max = fp16_ieee_to_fp32_value(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), rounded_output_min, rounded_output_max);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create depthwise %s operator with %zu input channels per group: "
      "depthwise convolution must have exactly 1 input channel per group",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16), group_input_channels);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  enum xnn_microkernel_type ukernel_type;
  struct dwconv2d_chw_parameters* dwconv2d_parameters = NULL;
  // Supported cases:
  // + 1x1 convolution (no groups)
  // + 3x3 stride-2 with 3 input channels and NHWC input layout
  // + 3x3 stride-2 depthwise convolution with horizontal padding 1 & no vertical padding
  // + 3x3 stride-1 depthwise convolution with horizontal padding 1 & no vertical padding
  // + 5x5 stride-2 depthwise convolution with horizontal padding 2 & no vertical padding
  // + 5x5 stride-1 depthwise convolution with horizontal padding 2 & no vertical padding
  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  const bool is_1x1 = kernel_width == 1 && kernel_height == 1 && subsampling_height == 1 && subsampling_width == 1;
  const bool is_3x3 = kernel_width == 3 && kernel_height == 3 && dilation_height == 1 && dilation_width == 1;
  const bool is_5x5 = kernel_width == 5 && kernel_height == 5 && dilation_height == 1 && dilation_width == 1;
  const bool nhwc_input = (flags & XNN_FLAG_INPUT_NHWC) != 0;
  if (is_1x1 && !any_padding && !nhwc_input && groups == 1) {
    ukernel_type = xnn_microkernel_type_spmm;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    input_padding_top == 1 && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    nhwc_input && groups == 1)
  {
    ukernel_type = xnn_microkernel_type_conv2d_hwc2chw;
  } else if (is_3x3 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 1 && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f16.dwconv2d_chw_3x3;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 0 || input_padding_top == 1) && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f16.dwconv2d_chw_3x3s2;
  } else if (is_5x5 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 2 && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f16.dwconv2d_chw_5x5;
  } else if (is_5x5 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 1 || input_padding_top == 2) && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f16.dwconv2d_chw_5x5s2;
  } else {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel, %"PRIu32 "x%" PRIu32 " subsampling, %"PRIu32 "x%" PRIu32 " dilation"
      ", %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding, %" PRIu32 "x%zu input channels, and %" PRIu32 "x%zu output channels: "
      "only selected convolution parameters are supported",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16),
      kernel_width, kernel_height, subsampling_width, subsampling_height, dilation_width, dilation_height,
      input_padding_top, input_padding_left, input_padding_bottom, input_padding_right,
      groups, group_input_channels, groups, group_output_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
    goto error;
  }

  if (caches != NULL && ukernel_type != xnn_microkernel_type_spmm) {
    convolution_op->weights_cache = caches->weights_cache;
  }

  switch (ukernel_type) {
    case xnn_microkernel_type_spmm:
    {
      assert(kernel_height == 1);
      assert(kernel_width == 1);
      assert(groups == 1);

      // Count number of non-zero values.
      size_t num_nonzeros[5];
      if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
        xnn_analyze_f32_spmm(group_output_channels, group_input_channels, kernel, num_nonzeros);
      } else {
        xnn_analyze_f16_spmm(group_output_channels, group_input_channels, kernel, num_nonzeros);
      }
      size_t num_nonzeroes = num_nonzeros[0];
      const size_t num_output_channel_blocks = group_output_channels;
      const size_t num_nonzero_values = num_nonzeroes;
      const size_t num_nonzero_blocks = num_nonzeroes;
      const struct spmm_parameters* spmm_parameters = &xnn_params.f16.spmm;

      // Sparse representation of weights consists of four components:
      // 1. An array of int32_t values storing scaled [by sizeof(input element)] difference between input channels
      //    corresponding to successive non-zero blocks.  Used by setup to compute (array 2).
      // 2. An array of int32_t values storing increment for input pointer after each processed tile. This array is
      //    derived from scaled difference in array 1 using parameters to setup function.
      // 3. An array of uint32_t values storing the number of non-zero kernel elements per each output channel.
      // 4. An array of fp16 values storing all bias elements (group_output_channels) and non-zero kernel elements.
      //    All elements within non-zero block are assumed to be non-zero.

      const size_t packed_weights_size =
        (num_nonzero_blocks * 2) * sizeof(int32_t) +
        (num_output_channel_blocks * sizeof(uint32_t) +
        group_output_channels + num_nonzero_values) * sizeof(uint16_t) + XNN_EXTRA_BYTES;

      convolution_op->packed_weights.pointer = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights.pointer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_weights_size, xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
        goto error;
      }
      convolution_op->num_nonzero_values = num_nonzero_values;
      convolution_op->num_nonzero_blocks = num_nonzero_blocks;
      convolution_op->num_output_channel_blocks = num_output_channel_blocks;

      int32_t* input_channel_diffs = (int32_t*) convolution_op->packed_weights.pointer;
      int32_t* input_increments = (int32_t*) (input_channel_diffs + num_nonzero_blocks);
      uint32_t* output_channel_nonzeros = (uint32_t*) (input_increments + num_nonzero_blocks);
      uint16_t* nonzero_values = (uint16_t*) (output_channel_nonzeros + num_output_channel_blocks);

      memset(output_channel_nonzeros, 0, num_output_channel_blocks * sizeof(uint32_t));

      // TODO(fbarchard): Support block encoding
      const size_t output_channels_block_size = 1;

      size_t first_ic = 0;
      if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
        status = xnn_pack_f32_to_f16_spmm(
            group_output_channels,
            output_channels_block_size,
            group_input_channels,
            kernel,
            bias,
            input_channel_diffs,
            output_channel_nonzeros,
            nonzero_values,
            &first_ic);
      } else {
        status = xnn_pack_f16_spmm(
            group_output_channels,
            output_channels_block_size,
            group_input_channels,
            (const uint16_t*) kernel,
            (const uint16_t*) bias,
            input_channel_diffs,
            output_channel_nonzeros,
            nonzero_values,
            &first_ic);
      }
      if (status != xnn_status_success) {
        goto error;
      }
      convolution_op->first_input_channel = first_ic;

      convolution_op->ukernel.spmm = (struct xnn_ukernel_spmm) {
        .function = spmm_parameters->ukernel,
        .mr = spmm_parameters->mr,
      };
      spmm_parameters->init.f16(&convolution_op->params.f16_minmax, fp16_output_min, fp16_output_max);

      break;
    }
    case xnn_microkernel_type_conv2d_hwc2chw:
    {
      assert(groups == 1);

      const size_t packed_group_output_channels =
        round_up(group_output_channels, xnn_params.f16.conv_hwc2chw_3x3c3s2.output_channel_tile);
      const size_t packed_weights_size = groups * packed_group_output_channels *
        (group_input_channels * kernel_height * kernel_width + 1 /* bias */) * sizeof(uint16_t);
      size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
      void* weights_ptr = xnn_get_pointer_to_write_weights(
          convolution_op, aligned_total_weights_size, 0);
      if (weights_ptr == NULL) {
        xnn_log_error("failed to reserve or allocate %zu bytes for %s operator conv2d_hwc2chw packed weights",
                      aligned_total_weights_size,
                      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
        goto error;
      }
      xnn_pack_dconv_oki_w_fn xnn_pack_dconv_oki_w = (xnn_pack_dconv_oki_w_fn) xnn_pack_f16_dconv_oki_w;
      if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
        xnn_pack_dconv_oki_w = (xnn_pack_dconv_oki_w_fn) xnn_pack_f32_to_f16_dconv_oki_w;
      }
      xnn_pack_dconv_oki_w(
        group_output_channels,
        group_input_channels,
        xnn_params.f16.conv_hwc2chw_3x3c3s2.output_channel_tile,
        kernel_height, kernel_width,
        kernel, bias, weights_ptr, NULL);

      if (use_weights_cache(convolution_op)) {
        convolution_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
            convolution_op->weights_cache, weights_ptr, aligned_total_weights_size);
      }

      convolution_op->ukernel.conv2d = (struct xnn_ukernel_conv2d) {
        .hwc2chw_fn = xnn_params.f16.conv_hwc2chw_3x3c3s2.ukernel_with_symm_padding,
        .output_height_tile = xnn_params.f16.conv_hwc2chw_3x3c3s2.output_height_tile,
        .output_channel_tile = xnn_params.f16.conv_hwc2chw_3x3c3s2.output_channel_tile,
      };
      xnn_params.f16.conv_hwc2chw_3x3c3s2.init.f16(&convolution_op->params.f16_minmax, fp16_output_min, fp16_output_max);

      break;
    }

    case xnn_microkernel_type_dwconv:
    {
      assert(dwconv2d_parameters != NULL);
      assert(group_input_channels == 1);
      assert(group_output_channels == 1);

      const size_t packed_weights_size = groups * (kernel_height * kernel_width + 1 /* bias */) * sizeof(uint16_t);
      size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
      void* weights_ptr = xnn_get_pointer_to_write_weights(
          convolution_op, aligned_total_weights_size, 0);
      if (weights_ptr == NULL) {
        xnn_log_error("failed to reserve or allocate %zu bytes for %s operator dwconv packed weights",
                      aligned_total_weights_size,
                      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f16));
        goto error;
      }

      xnn_pack_chw_dwconv_hwg_w_fn pack_chw_dwconv_hwg_w = (xnn_pack_chw_dwconv_hwg_w_fn) xnn_pack_f16_chw_dwconv_hwg_w;
      xnn_pack_chw_dwconv_ghw_w_fn pack_chw_dwconv_ghw_w = (xnn_pack_chw_dwconv_ghw_w_fn) xnn_pack_f16_chw_dwconv_ghw_w;
      if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
        pack_chw_dwconv_hwg_w = (xnn_pack_chw_dwconv_hwg_w_fn) xnn_pack_f32_to_f16_chw_dwconv_hwg_w;
        pack_chw_dwconv_ghw_w = (xnn_pack_chw_dwconv_ghw_w_fn) xnn_pack_f32_to_f16_chw_dwconv_ghw_w;
      }

      if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
        pack_chw_dwconv_hwg_w(
          kernel_height * kernel_width, groups,
          kernel, bias, weights_ptr, NULL);
      } else {
        pack_chw_dwconv_ghw_w(
          kernel_height * kernel_width, groups,
          kernel, bias, weights_ptr, NULL);
      }

      if (use_weights_cache(convolution_op)) {
        convolution_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
            convolution_op->weights_cache, weights_ptr, aligned_total_weights_size);
      }

      convolution_op->ukernel.dwconv2d = (struct xnn_ukernel_dwconv2d) {
        .chw_fn = dwconv2d_parameters->ukernel,
        .update_params = (xnn_update_chw_params_fn) dwconv2d_parameters->update.f16,
        .output_width_tile = dwconv2d_parameters->output_width_tile,
      };
      dwconv2d_parameters->init.f16(&convolution_op->params.f16_chw, 0, fp16_output_min, fp16_output_max);

      break;
    }
    default:
      XNN_UNREACHABLE;
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
  convolution_op->input_pixel_stride = input_channel_stride;
  convolution_op->output_pixel_stride = output_channel_stride;

  convolution_op->type = xnn_operator_type_convolution_nchw_f16;
  convolution_op->ukernel.type = ukernel_type;
  convolution_op->flags = flags;

  convolution_op->state = xnn_run_state_invalid;
  *convolution_op_out = convolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(convolution_op);
  return status;
}


enum xnn_status xnn_create_convolution2d_nchw_f32(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_caches_t caches,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  const uint32_t datatype_init_flags = XNN_INIT_FLAG_F32;
  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), kernel_width, kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " subsampling: subsampling dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), subsampling_width, subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 " groups: number of groups must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu input channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu output channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32),
      input_channel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32),
      output_channel_stride, groups, group_output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), output_min, output_max);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create depthwise %s operator with %zu input channels per group: "
      "depthwise convolution must have exactly 1 input channel per group",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32), group_input_channels);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  enum xnn_microkernel_type ukernel_type;
  struct dwconv2d_chw_parameters* dwconv2d_parameters = NULL;
  // Supported cases:
  // + 1x1 convolution (no groups)
  // + 3x3 stride-2 with 3 input channels and NHWC input layout
  // + 3x3 stride-2 depthwise convolution with horizontal padding 1 & no vertical padding
  // + 3x3 stride-1 depthwise convolution with horizontal padding 1 & no vertical padding
  // + 5x5 stride-2 depthwise convolution with horizontal padding 2 & no vertical padding
  // + 5x5 stride-1 depthwise convolution with horizontal padding 2 & no vertical padding
  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  const bool is_1x1 = kernel_width == 1 && kernel_height == 1 && subsampling_height == 1 && subsampling_width == 1;
  const bool is_3x3 = kernel_width == 3 && kernel_height == 3 && dilation_height == 1 && dilation_width == 1;
  const bool is_5x5 = kernel_width == 5 && kernel_height == 5 && dilation_height == 1 && dilation_width == 1;
  const bool nhwc_input = (flags & XNN_FLAG_INPUT_NHWC) != 0;
  if (is_1x1 && !any_padding && !nhwc_input && groups == 1) {
    ukernel_type = xnn_microkernel_type_spmm;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    input_padding_top == 1 && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    nhwc_input && groups == 1)
  {
    ukernel_type = xnn_microkernel_type_conv2d_hwc2chw;
  } else if (is_3x3 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 1 && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f32.dwconv2d_chw_3x3;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 0 || input_padding_top == 1) && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f32.dwconv2d_chw_3x3s2;
  } else if (is_5x5 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 2 && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f32.dwconv2d_chw_5x5;
  } else if (is_5x5 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 1 || input_padding_top == 2) && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &xnn_params.f32.dwconv2d_chw_5x5s2;
  } else {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel, %"PRIu32 "x%" PRIu32 " subsampling, %"PRIu32 "x%" PRIu32 " dilation"
      ", %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding, %" PRIu32 "x%zu input channels, and %" PRIu32 "x%zu output channels: "
      "only selected convolution parameters are supported",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32),
      kernel_width, kernel_height, subsampling_width, subsampling_height, dilation_width, dilation_height,
      input_padding_top, input_padding_left, input_padding_bottom, input_padding_right,
      groups, group_input_channels, groups, group_output_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    goto error;
  }

  if (caches != NULL && ukernel_type != xnn_microkernel_type_spmm) {
    convolution_op->weights_cache = caches->weights_cache;
  }

  switch (ukernel_type) {
    case xnn_microkernel_type_spmm:
    {
      assert(kernel_height == 1);
      assert(kernel_width == 1);
      assert(groups == 1);

      // Count number of non-zero values.
      size_t num_nonzeros[5];
      xnn_analyze_f32_spmm(group_output_channels, group_input_channels, kernel, num_nonzeros);
      size_t num_nonzeroes = num_nonzeros[0];
      size_t num_nonzero_blocks2 = num_nonzeros[1];
      size_t num_nonzero_blocks4 = num_nonzeros[2];
      size_t num_block2_nonzeroes = num_nonzeros[3];
      size_t num_block4_nonzeroes = num_nonzeros[4];

      // Select block encoding when 2 or 4 channels have non-zero values.
      size_t output_channels_block_size = 1;
      size_t num_output_channel_blocks = group_output_channels;
      size_t num_nonzero_values = num_nonzeroes;
      size_t num_nonzero_blocks = num_nonzeroes;
      const struct spmm_parameters* spmm_parameters = &xnn_params.f32.spmm;
      if (num_block4_nonzeroes * 5 >= num_nonzero_blocks4 * 18 && xnn_params.f32.spmm4.ukernel != NULL) {
        // 4-channel blocks have 90%+ non-zeroes

        output_channels_block_size = 4;
        num_output_channel_blocks = num_output_channel_blocks / 4 + num_output_channel_blocks % 4;
        spmm_parameters = &xnn_params.f32.spmm4;
        // Non-zeroes which don't fit into whole 4-channel blocks, processed one-by-one
        const size_t num_remaining_nonzeroes = num_nonzeroes - num_block4_nonzeroes;
        num_nonzero_values = num_nonzero_blocks4 * 4 + num_remaining_nonzeroes;
        num_nonzero_blocks = num_nonzero_blocks4 + num_remaining_nonzeroes;
      } else if (num_block2_nonzeroes * 5 >= num_nonzero_blocks2 * 9 && xnn_params.f32.spmm2.ukernel != NULL) {
        // 2-channel blocks have 90%+ non-zeroes

        output_channels_block_size = 2;
        num_output_channel_blocks = num_output_channel_blocks / 2 + num_output_channel_blocks % 2;
        spmm_parameters = &xnn_params.f32.spmm2;
        // Non-zeroes which don't fit into whole 2-channel blocks, processed one-by-one
        const size_t num_remaining_nonzeroes = num_nonzeroes - num_block2_nonzeroes;
        num_nonzero_values = num_nonzero_blocks2 * 2 + num_remaining_nonzeroes;
        num_nonzero_blocks = num_nonzero_blocks2 + num_remaining_nonzeroes;
      }

      // Sparse representation of weights consists of four components:
      // 1. An array of int32_t values storing scaled [by sizeof(input element)] difference between input channels
      //    corresponding to successive non-zero blocks.  Used by setup to compute (array 2).
      // 2. An array of int32_t values storing increment for input pointer after each processed tile. This array is
      //    derived from scaled difference in array 1 using parameters to setup function.
      // 3. An array of uint32_t values storing the number of non-zero kernel elements per each output channel.
      // 4. An array of float values storing non-zero kernel elements, and all (group_output_channels) bias elements.
      //    All elements within non-zero block are assumed to be non-zero.

      const size_t packed_weights_size =
        (num_nonzero_blocks * 2) * sizeof(int32_t) +
        (num_output_channel_blocks * sizeof(uint32_t) +
        num_nonzero_values + group_output_channels) * sizeof(float) + XNN_EXTRA_BYTES;

      convolution_op->packed_weights.pointer = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights.pointer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_weights_size, xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
        goto error;
      }
      convolution_op->num_nonzero_values = num_nonzero_values;
      convolution_op->num_nonzero_blocks = num_nonzero_blocks;
      convolution_op->num_output_channel_blocks = num_output_channel_blocks;

      int32_t* input_channel_diffs = (int32_t*) convolution_op->packed_weights.pointer;
      int32_t* input_increments = (int32_t*) (input_channel_diffs + num_nonzero_blocks);
      uint32_t* output_channel_nonzeros = (uint32_t*) (input_increments + num_nonzero_blocks);
      float* nonzero_values = (float*) (output_channel_nonzeros + num_output_channel_blocks);

      memset(output_channel_nonzeros, 0, num_output_channel_blocks * sizeof(uint32_t));

      size_t first_ic = 0;
      status = xnn_pack_f32_spmm(
          group_output_channels,
          output_channels_block_size,
          group_input_channels,
          kernel,
          bias,
          input_channel_diffs,
          output_channel_nonzeros,
          nonzero_values,
          &first_ic);
      if (status != xnn_status_success) {
        goto error;
      }

      convolution_op->first_input_channel = first_ic;

      convolution_op->ukernel.spmm = (struct xnn_ukernel_spmm) {
        .function = spmm_parameters->ukernel,
        .mr = spmm_parameters->mr,
      };
      spmm_parameters->init.f32(&convolution_op->params.f32_minmax, output_min, output_max);

      break;
    }
    case xnn_microkernel_type_conv2d_hwc2chw:
    {
      assert(groups == 1);

      const size_t packed_group_output_channels =
        round_up(group_output_channels, xnn_params.f32.conv_hwc2chw_3x3c3s2.output_channel_tile);
      const size_t packed_weights_size = groups * packed_group_output_channels *
        (group_input_channels * kernel_height * kernel_width + 1 /* bias */) * sizeof(float);
      size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
      void* weights_ptr = xnn_get_pointer_to_write_weights(
          convolution_op, aligned_total_weights_size, 0);
      if (weights_ptr == NULL) {
        xnn_log_error("failed to reserve or allocate %zu bytes for %s operator conv2d_hwc2chw packed weights",
                      aligned_total_weights_size,
                      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
        goto error;
      }

      xnn_pack_f32_dconv_oki_w(
        group_output_channels,
        group_input_channels,
        xnn_params.f32.conv_hwc2chw_3x3c3s2.output_channel_tile,
        kernel_height, kernel_width,
        kernel, bias, weights_ptr, NULL);

      if (use_weights_cache(convolution_op)) {
        convolution_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
            convolution_op->weights_cache, weights_ptr, aligned_total_weights_size);
      }

      convolution_op->ukernel.conv2d = (struct xnn_ukernel_conv2d) {
        .hwc2chw_fn = xnn_params.f32.conv_hwc2chw_3x3c3s2.ukernel_with_symm_padding,
        .output_height_tile = xnn_params.f32.conv_hwc2chw_3x3c3s2.output_height_tile,
        .output_channel_tile = xnn_params.f32.conv_hwc2chw_3x3c3s2.output_channel_tile,
      };
      xnn_params.f32.conv_hwc2chw_3x3c3s2.init.f32(&convolution_op->params.f32_minmax, output_min, output_max);

      break;
    }
    case xnn_microkernel_type_dwconv:
    {
      assert(dwconv2d_parameters != NULL);
      assert(group_input_channels == 1);
      assert(group_output_channels == 1);

      const size_t packed_weights_size = groups * (kernel_height * kernel_width + 1 /* bias */) * sizeof(float);
      size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
      void* weights_ptr = xnn_get_pointer_to_write_weights(
          convolution_op, aligned_total_weights_size, 0);
      if (weights_ptr == NULL) {
        xnn_log_error("failed to reserve or allocate %zu bytes for %s operator dwconv packed weights",
                      aligned_total_weights_size,
                      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
        goto error;
      }

      if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
        xnn_pack_f32_chw_dwconv_hwg_w(
          kernel_height * kernel_width, groups,
          kernel, bias, weights_ptr, NULL);
      } else {
        xnn_pack_f32_chw_dwconv_ghw_w(
          kernel_height * kernel_width, groups,
          kernel, bias, weights_ptr, NULL);
      }

      if (use_weights_cache(convolution_op)) {
        convolution_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
            convolution_op->weights_cache, weights_ptr, aligned_total_weights_size);
      }

      convolution_op->ukernel.dwconv2d = (struct xnn_ukernel_dwconv2d) {
        .chw_fn = dwconv2d_parameters->ukernel,
        .update_params = (xnn_update_chw_params_fn) dwconv2d_parameters->update.f32,
        .output_width_tile = dwconv2d_parameters->output_width_tile,
      };
      dwconv2d_parameters->init.f32(&convolution_op->params.f32_chw, 0, output_min, output_max);

      break;
    }
    default:
      XNN_UNREACHABLE;
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
  convolution_op->input_pixel_stride = input_channel_stride;
  convolution_op->output_pixel_stride = output_channel_stride;

  convolution_op->type = xnn_operator_type_convolution_nchw_f32;
  convolution_op->ukernel.type = ukernel_type;
  convolution_op->flags = flags;

  convolution_op->state = xnn_run_state_invalid;

  *convolution_op_out = convolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(convolution_op);
  return status;
}

static enum xnn_status setup_convolution2d_nchw(
  xnn_operator_t convolution_op,
  enum xnn_operator_type expected_operator_type,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  uint32_t datatype_init_flags,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  void* chw_params,
  size_t num_threads)
{
  if (convolution_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }
  convolution_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error(
      "failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_unsupported_hardware;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(convolution_op->type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    convolution_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  if (convolution_op->weights_cache != NULL && !xnn_weights_cache_is_finalized(convolution_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_state;
  }

  convolution_op->batch_size = batch_size;
  convolution_op->input_height = input_height;
  convolution_op->input_width = input_width;
  convolution_op->input = input;
  convolution_op->output = output;

  const size_t output_height = xnn_compute_convolution_output_dimension(
      convolution_op->padding_top + input_height + convolution_op->padding_bottom,
      convolution_op->kernel_height,
      convolution_op->dilation_height,
      convolution_op->stride_height);
  const size_t output_width = xnn_compute_convolution_output_dimension(
      convolution_op->padding_left + input_width + convolution_op->padding_right,
      convolution_op->kernel_width,
      convolution_op->dilation_width,
      convolution_op->stride_width);

  const size_t input_batch_stride = (input_height * input_width * convolution_op->input_pixel_stride) << log2_input_element_size;
  const size_t output_batch_stride = (output_height * output_width * convolution_op->output_pixel_stride) << log2_output_element_size;
  switch (convolution_op->ukernel.type) {
    case xnn_microkernel_type_spmm:
    {
      const size_t num_nonzero_values = convolution_op->num_nonzero_values;
      const size_t num_nonzero_blocks = convolution_op->num_nonzero_blocks;
      const size_t num_output_channel_blocks = convolution_op->num_output_channel_blocks;

      convolution_op->num_nonzero_values = num_nonzero_values;
      convolution_op->num_nonzero_blocks = num_nonzero_blocks;
      convolution_op->num_output_channel_blocks = num_output_channel_blocks;

      const int32_t* input_channel_diffs = (const int32_t*) packed_weights(convolution_op);
      int32_t* input_increments = ((int32_t*) packed_weights(convolution_op)) + num_nonzero_blocks;
      const uint32_t* output_channel_nonzeros = (uint32_t*) (input_increments + num_nonzero_blocks);
      const void* nonzero_values = (const void*) (output_channel_nonzeros + num_output_channel_blocks);

      // Scale input_channel_diffs by input_size to compute input_increments;
      const size_t input_size = input_height * input_width;
      for (size_t i = 0; i < num_nonzero_blocks; i++) {
        const int32_t diff = input_channel_diffs[i];
        const int64_t increment = (int64_t) diff * input_size;
        if ((int64_t) (int32_t) increment != increment) {
          xnn_log_error(
            "failed to setup %s operator with sparse kernel representation: input increment exceeds int32_t range",
            xnn_operator_type_to_string(convolution_op->type));
          return xnn_status_unsupported_parameter;
        }
        input_increments[i] = (int32_t) increment;
      }

      convolution_op->context.spmm = (struct spmm_context) {
          .n = convolution_op->group_output_channels,
          .scaled_m = input_size << log2_input_element_size,
          .input = (const void*) ((uintptr_t) input + (convolution_op->first_input_channel * input_size << log2_input_element_size)),
          .nonzero_weights = nonzero_values,
          .input_increments = input_increments,
          .output_channel_nonzeros = output_channel_nonzeros,
          .output = output,
          .batched_input_stride = input_batch_stride,
          .batched_output_stride = output_batch_stride,
          .ukernel = convolution_op->ukernel.spmm.function,
      };
      memcpy(&convolution_op->context.spmm.params, params, sizeof(convolution_op->context.spmm.params));

      const size_t mr = convolution_op->ukernel.spmm.mr;
      #if XNN_TEST_MODE
        const size_t mc = mr;
      #else
        size_t mc = input_size;
        if (num_threads > 1) {
          const size_t target_tiles_per_thread = 5;
          const size_t max_mc = divide_round_up(input_size, num_threads * target_tiles_per_thread);
          if (max_mc < mc) {
            mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
          }
        }
      #endif
      convolution_op->compute.type = xnn_parallelization_type_2d_tile_1d;
      convolution_op->compute.task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_spmm;
      convolution_op->compute.range[0] = batch_size;
      convolution_op->compute.range[1] = input_size << log2_input_element_size;
      convolution_op->compute.tile[0] = mc << log2_input_element_size;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_microkernel_type_conv2d_hwc2chw:
    {
      const size_t zero_size = (input_width * convolution_op->group_input_channels << log2_input_element_size) + XNN_EXTRA_BYTES;

      // Note: zero buffer must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(convolution_op->zero_buffer);
      convolution_op->zero_buffer = xnn_allocate_simd_memory(zero_size);
      if (convolution_op->zero_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator zero padding",
          sizeof(struct xnn_operator), xnn_operator_type_to_string(convolution_op->type));
        return xnn_status_out_of_memory;
      }
      memset(convolution_op->zero_buffer, 0, zero_size);

      convolution_op->context.conv2d = (struct conv2d_context) {
        .input_height = input_height,
        .input_width = input_width,
        .input = input,
        .input_batch_stride = input_batch_stride,
        .zero = convolution_op->zero_buffer,
        .packed_weights = packed_weights(convolution_op),
        .output = output,
        .output_batch_stride = output_batch_stride,
        .input_padding_top = convolution_op->padding_top,
        .output_channels = convolution_op->group_output_channels,
        .output_height_stride = output_width << log2_output_element_size,
        .output_channel_stride = output_height * output_width << log2_output_element_size,
        .hwc2chw_ukernel = convolution_op->ukernel.conv2d.hwc2chw_fn,
      };
      memcpy(&convolution_op->context.conv2d.params, params, sizeof(convolution_op->context.conv2d.params));

      const size_t output_height_tile = convolution_op->ukernel.conv2d.output_height_tile;
      #if XNN_TEST_MODE
        size_t output_height_slice = output_height_tile;
      #else
        size_t output_height_slice = output_height;
        if (num_threads > 1) {
          const size_t target_tiles_per_thread = 5;
          const size_t max_output_height_slice = divide_round_up(output_height, num_threads * target_tiles_per_thread);
          if (max_output_height_slice < output_height_slice) {
            output_height_slice = min(output_height_slice,
              divide_round_up(output_height_slice, max_output_height_slice * output_height_tile) * output_height_tile);
          }
        }
      #endif
      convolution_op->compute.type = xnn_parallelization_type_2d_tile_1d;
      convolution_op->compute.task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_conv2d_hwc2chw;
      convolution_op->compute.range[0] = batch_size;
      convolution_op->compute.range[1] = output_height;
      convolution_op->compute.tile[0] = output_height_slice;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_microkernel_type_dwconv:
    {
      const size_t zero_size = (input_width << log2_input_element_size) + 2 * XNN_EXTRA_BYTES;

      // Note: zero buffer must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(convolution_op->zero_buffer);
      convolution_op->zero_buffer = xnn_allocate_simd_memory(zero_size);
      if (convolution_op->zero_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator zero padding",
          sizeof(struct xnn_operator), xnn_operator_type_to_string(convolution_op->type));
        return xnn_status_out_of_memory;
      }
      memset(convolution_op->zero_buffer, 0, zero_size);

      if (convolution_op->ukernel.dwconv2d.update_params != NULL) {
        convolution_op->ukernel.dwconv2d.update_params(chw_params, (uint32_t) input_width);
      }
      convolution_op->context.dwconv2d = (struct dwconv2d_context) {
        .input_height = input_height,
        .input_width = input_width << log2_input_element_size,
        .input = input,
        .zero = convolution_op->zero_buffer,
        .input_padding_top = convolution_op->padding_top,
        .input_channel_stride = input_height * input_width << log2_input_element_size,
        .input_batch_stride = input_batch_stride,
        .packed_weights = packed_weights(convolution_op),
        .weights_channel_stride = bias_element_size +
          (convolution_op->kernel_height * convolution_op->kernel_width << log2_filter_element_size),
        .output = output,
        .output_channel_stride = output_height * output_width << log2_output_element_size,
        .output_batch_stride = output_batch_stride,
        .chw_ukernel = convolution_op->ukernel.dwconv2d.chw_fn,
      };
      memcpy(&convolution_op->context.dwconv2d.params, chw_params, sizeof(convolution_op->context.dwconv2d.params));

      convolution_op->compute.type = xnn_parallelization_type_2d;
      convolution_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_dwconv2d_chw;
      convolution_op->compute.range[0] = batch_size;
      convolution_op->compute.range[1] = convolution_op->groups;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_convolution2d_nchw_f32(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_convolution2d_nchw(
    convolution_op,
    xnn_operator_type_convolution_nchw_f32,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_F32,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    &convolution_op->params.f32_minmax,
    &convolution_op->params.f32_chw,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convolution2d_nchw_f16(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_convolution2d_nchw(
    convolution_op,
    xnn_operator_type_convolution_nchw_f16,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_F16 | XNN_INIT_FLAG_F16_NATIVE,
    1 /* log2(sizeof(input element)) = log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(filter element)) = log2(sizeof(uint16_t)) */,
    sizeof(uint16_t) /* sizeof(bias element) */,
    1 /* log2(sizeof(output element)) = log2(sizeof(uint16_t)) */,
    &convolution_op->params.f16_minmax,
    &convolution_op->params.f16_chw,
    pthreadpool_get_threads_count(threadpool));
}
