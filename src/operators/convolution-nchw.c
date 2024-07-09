// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/compute.h"
#include "xnnpack/config-types.h"
#include "xnnpack/config.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/pack.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static enum xnn_status create_spmm_path(
    const uint32_t kernel_height,
    const uint32_t kernel_width,
    const uint32_t groups,
    const size_t group_input_channels,
    const size_t group_output_channels,
    const void* kernel,
    const void* bias,
    const uint32_t log2_filter_element_size,
    const xnn_analyze_spmm_w_fn xnn_analyze_spmm,
    const xnn_pack_spmm_w_fn xnn_pack_spmm,
    const struct xnn_spmm_config* spmm_config,
    const struct xnn_spmm_config* spmm2_config,
    const struct xnn_spmm_config* spmm4_config,
    const enum xnn_operator_type operator_type,
    const xnn_operator_t convolution_op)
{
  assert(spmm_config != NULL);
  assert(kernel_height == 1);
  assert(kernel_width == 1);
  assert(groups == 1);

  // Count number of non-zero values.
  struct xnn_spmm_packing_params spmm_packing_params;

  xnn_analyze_spmm(group_output_channels, group_input_channels, kernel, &spmm_packing_params);

  size_t num_nonzeroes = spmm_packing_params.num_nonzeroes;
  size_t num_nonzero_blocks2 = spmm_packing_params.num_nonzero_blocks2;
  size_t num_nonzero_blocks4 = spmm_packing_params.num_nonzero_blocks4;
  size_t num_block2_nonzeroes = spmm_packing_params.num_block2_nonzeroes;
  size_t num_block4_nonzeroes = spmm_packing_params.num_block4_nonzeroes;

  // Select block encoding when 2 or 4 channels have non-zero values.
  size_t output_channels_block_size = 1;
  size_t num_output_channel_blocks = group_output_channels;
  size_t num_nonzero_values = num_nonzeroes;
  size_t num_nonzero_blocks = num_nonzeroes;
  if (num_block4_nonzeroes * 5 >= num_nonzero_blocks4 * 18 && spmm4_config != NULL && spmm4_config->ukernel != NULL) {
    // 4-channel blocks have 90%+ non-zeroes

    output_channels_block_size = 4;
    num_output_channel_blocks = num_output_channel_blocks / 4 + num_output_channel_blocks % 4;
    spmm_config = spmm4_config;
    // Non-zeroes which don't fit into whole 4-channel blocks, processed one-by-one
    const size_t num_remaining_nonzeroes = num_nonzeroes - num_block4_nonzeroes;
    num_nonzero_values = num_nonzero_blocks4 * 4 + num_remaining_nonzeroes;
    num_nonzero_blocks = num_nonzero_blocks4 + num_remaining_nonzeroes;
  } else if (num_block2_nonzeroes * 5 >= num_nonzero_blocks2 * 9 && spmm2_config != NULL && spmm2_config->ukernel != NULL) {
    // 2-channel blocks have 90%+ non-zeroes

    output_channels_block_size = 2;
    num_output_channel_blocks = num_output_channel_blocks / 2 + num_output_channel_blocks % 2;
    spmm_config = spmm2_config;
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
  // 4. An array of float or fp16 values storing all bias elements (group_output_channels) and non-zero kernel elements.
  //    All elements within non-zero block are assumed to be non-zero.

  const size_t packed_weights_size =
    num_nonzero_blocks * 2 * sizeof(int32_t) +
    num_output_channel_blocks * sizeof(uint32_t) +
    ((group_output_channels + num_nonzero_values) << log2_filter_element_size) + XNN_EXTRA_BYTES;

  convolution_op->packed_weights.pointer = xnn_allocate_simd_memory(packed_weights_size);
  if (convolution_op->packed_weights.pointer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      packed_weights_size, xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
    packed_weights_size, xnn_operator_type_to_string(operator_type));

  convolution_op->num_nonzero_values = num_nonzero_values;
  convolution_op->num_nonzero_blocks = num_nonzero_blocks;
  convolution_op->num_output_channel_blocks = num_output_channel_blocks;

  int32_t* input_channel_diffs = (int32_t*) convolution_op->packed_weights.pointer;
  int32_t* input_increments = (int32_t*) (input_channel_diffs + num_nonzero_blocks);
  uint32_t* output_channel_nonzeros = (uint32_t*) (input_increments + num_nonzero_blocks);
  void* nonzero_values = (void*) (output_channel_nonzeros + num_output_channel_blocks);

  memset(output_channel_nonzeros, 0, num_output_channel_blocks * sizeof(uint32_t));

  size_t first_ic = 0;
  enum xnn_status status = xnn_pack_spmm(
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
    .function = spmm_config->ukernel,
    .mr = spmm_config->mr,
  };
  return xnn_status_success;

error:
  xnn_release_simd_memory(convolution_op->packed_weights.pointer);
  return status;
}

static enum xnn_status create_conv2d_hwc2chw_path(
    const uint32_t kernel_height,
    const uint32_t kernel_width,
    const uint32_t groups,
    const size_t group_input_channels,
    const size_t group_output_channels,
    const size_t output_height_tile,
    const size_t output_channel_tile,
    const void* kernel,
    const void* bias,
    const uint32_t log2_filter_element_size,
    const xnn_pack_dconv_oki_w_fn xnn_pack_dconv_oki_w,
    const xnn_conv_hwc2chw_ukernel_fn conv_hwc2chw_ukernel,
    const enum xnn_operator_type operator_type,
    const xnn_operator_t convolution_op)
{
  assert(conv_hwc2chw_ukernel != NULL);

  const size_t packed_group_output_channels = round_up(group_output_channels, output_channel_tile);
  const size_t packed_weights_size = (groups * packed_group_output_channels *
    (group_input_channels * kernel_height * kernel_width + 1 /* bias */)) << log2_filter_element_size;
  const size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(convolution_op, aligned_total_weights_size, 0);
  if (weights_ptr == NULL) {
    xnn_log_error("failed to reserve or allocate %zu bytes for %s operator conv2d_hwc2chw packed weights",
                  aligned_total_weights_size,
                  xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
    aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  xnn_pack_dconv_oki_w(
    group_output_channels,
    group_input_channels,
    output_channel_tile,
    kernel_height, kernel_width,
    kernel, bias, weights_ptr, NULL);

  if (use_weights_cache(convolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = group_input_channels ^ group_output_channels ^ output_channel_tile;
    cache_key.kernel = kernel;
    cache_key.bias = bias;
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr, aligned_total_weights_size);
  }

  convolution_op->ukernel.conv2d = (struct xnn_ukernel_conv2d) {
    .hwc2chw_fn = conv_hwc2chw_ukernel,
    .output_height_tile = output_height_tile,
    .output_channel_tile = output_channel_tile,
  };
  return xnn_status_success;
}

static enum xnn_status create_dwconv_path(
    const uint32_t kernel_height,
    const uint32_t kernel_width,
    const uint32_t groups,
    const void* kernel,
    const void* bias,
    const uint32_t flags,
    const uint32_t log2_filter_element_size,
    const xnn_pack_chw_dwconv_hwg_w_fn pack_chw_dwconv_hwg_w,
    const xnn_pack_chw_dwconv_ghw_w_fn pack_chw_dwconv_ghw_w,
    const xnn_update_chw_params_fn update_chw_params,
    const size_t output_width_tile,
    const xnn_dwconv2d_chw_ukernel_fn dwconv_ukernel,
    const enum xnn_operator_type operator_type,
    const xnn_operator_t convolution_op)
{
  assert(dwconv_ukernel != NULL);

  const size_t packed_weights_size = (groups * (kernel_height * kernel_width + 1 /* bias */)) << log2_filter_element_size;
  const size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size, 0);
  if (weights_ptr == NULL) {
    xnn_log_error("failed to reserve or allocated %zu bytes for %s operator dwconv packed weights",
                  aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  uint32_t cache_seed = kernel_height ^ kernel_width ^ groups;
  if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
    pack_chw_dwconv_hwg_w(
      kernel_height * kernel_width, groups,
      kernel, bias, weights_ptr, NULL);
  } else {
    cache_seed = ~cache_seed;
    pack_chw_dwconv_ghw_w(
      kernel_height * kernel_width, groups,
      kernel, bias, weights_ptr, NULL);
  }

  if (use_weights_cache(convolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = cache_seed;
    cache_key.kernel = kernel;
    cache_key.bias = bias;
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr, aligned_total_weights_size);
  }

  convolution_op->ukernel.dwconv2d = (struct xnn_ukernel_dwconv2d) {
    .chw_fn = dwconv_ukernel,
    .update_params = (xnn_update_chw_params_fn) update_chw_params,
    .output_width_tile = output_width_tile,
  };

  return xnn_status_success;
}

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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_HALF;
  const enum xnn_operator_type operator_type = xnn_operator_type_convolution_nchw_f16;

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

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " subsampling: subsampling dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), subsampling_width, subsampling_height);
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
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      input_channel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      output_channel_stride, groups, group_output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const uint16_t fp16_output_min = fp16_ieee_from_fp32_value(output_min);
  const uint16_t fp16_output_max = fp16_ieee_from_fp32_value(output_max);
  const float rounded_output_min = fp16_ieee_to_fp32_value(fp16_output_min);
  const float rounded_output_max = fp16_ieee_to_fp32_value(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(operator_type), rounded_output_min, rounded_output_max);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create depthwise %s operator with %zu input channels per group: "
      "depthwise convolution must have exactly 1 input channel per group",
      xnn_operator_type_to_string(operator_type), group_input_channels);
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  const struct xnn_spmm_config* spmm_config = xnn_init_f16_spmm_config();
  if (spmm_config == NULL) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const struct xnn_dwconv2d_chw_config* dwconv2d_chw_config = xnn_init_f16_dwconv2d_chw_config();
  if (dwconv2d_chw_config == NULL) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  enum xnn_microkernel_type ukernel_type;
  const struct xnn_dwconv2d_chw_parameters* dwconv2d_parameters = NULL;
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
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_3x3;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 0 || input_padding_top == 1) && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_3x3s2;
  } else if (is_5x5 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 2 && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_5x5;
  } else if (is_5x5 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 1 || input_padding_top == 2) && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_5x5s2;
  } else {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel, %"PRIu32 "x%" PRIu32 " subsampling, %"PRIu32 "x%" PRIu32 " dilation"
      ", %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding, %" PRIu32 "x%zu input channels, and %" PRIu32 "x%zu output channels: "
      "only selected convolution parameters are supported",
      xnn_operator_type_to_string(operator_type),
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
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  if (ukernel_type != xnn_microkernel_type_spmm) {
    convolution_op->weights_cache = weights_cache;
  }

  switch (ukernel_type) {
    case xnn_microkernel_type_spmm:
    {
      xnn_analyze_spmm_w_fn xnn_analyze_spmm;
      xnn_pack_spmm_w_fn xnn_pack_spmm;
      if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
        xnn_analyze_spmm = (xnn_analyze_spmm_w_fn) xnn_analyze_f32_spmm_w;
        xnn_pack_spmm = (xnn_pack_spmm_w_fn) xnn_pack_f32_to_f16_spmm_w;
      } else {
        xnn_analyze_spmm = (xnn_analyze_spmm_w_fn) xnn_analyze_f16_spmm_w;
        xnn_pack_spmm = (xnn_pack_spmm_w_fn) xnn_pack_f16_spmm_w;
      }

      spmm_config->init.f16(&convolution_op->params.f16_minmax, fp16_output_min, fp16_output_max);

      status = create_spmm_path(
          kernel_height, kernel_width, groups,
          group_input_channels, group_output_channels,
          kernel, bias, log2_filter_element_size,
          xnn_analyze_spmm, xnn_pack_spmm,
          spmm_config, NULL, NULL,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_conv2d_hwc2chw:
    {
      assert(groups == 1);
      xnn_pack_dconv_oki_w_fn xnn_pack_dconv_oki_w = (xnn_pack_dconv_oki_w_fn) xnn_pack_f16_dconv_oki_w;
      if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
        xnn_pack_dconv_oki_w = (xnn_pack_dconv_oki_w_fn) xnn_pack_f32_to_f16_dconv_oki_w;
      }

      const struct xnn_conv_hwc2chw_config* conv_hwc2chw_config = xnn_init_f16_conv_hwc2chw_3x3c3s2_config();
      if (conv_hwc2chw_config == NULL) {
        status = xnn_status_unsupported_hardware;
        xnn_log_error("failed to create %s operator: operations on data type are not supported",
                      xnn_operator_type_to_string(operator_type));
        goto error;
      }

      conv_hwc2chw_config->init.f16(&convolution_op->params.f16_minmax, fp16_output_min, fp16_output_max);

      status = create_conv2d_hwc2chw_path(
          kernel_height, kernel_width, groups,
          group_input_channels,
          group_output_channels,
          conv_hwc2chw_config->output_height_tile,
          conv_hwc2chw_config->output_channel_tile,
          kernel, bias, log2_filter_element_size,
          xnn_pack_dconv_oki_w,
          conv_hwc2chw_config->ukernel_with_symm_padding,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_dwconv:
    {
      xnn_pack_chw_dwconv_hwg_w_fn pack_chw_dwconv_hwg_w = (xnn_pack_chw_dwconv_hwg_w_fn) xnn_pack_f16_chw_dwconv_hwg_w;
      xnn_pack_chw_dwconv_ghw_w_fn pack_chw_dwconv_ghw_w = (xnn_pack_chw_dwconv_ghw_w_fn) xnn_pack_f16_chw_dwconv_ghw_w;
      if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
        pack_chw_dwconv_hwg_w = (xnn_pack_chw_dwconv_hwg_w_fn) xnn_pack_f32_to_f16_chw_dwconv_hwg_w;
        pack_chw_dwconv_ghw_w = (xnn_pack_chw_dwconv_ghw_w_fn) xnn_pack_f32_to_f16_chw_dwconv_ghw_w;
      }

      dwconv2d_parameters->init.f16(&convolution_op->params.f16_chw, 0, fp16_output_min, fp16_output_max);

      status = create_dwconv_path(
          kernel_height, kernel_width, groups,
          kernel, bias, flags, log2_filter_element_size,
          pack_chw_dwconv_hwg_w,
          pack_chw_dwconv_ghw_w,
          (xnn_update_chw_params_fn) dwconv2d_parameters->update.f16,
          dwconv2d_parameters->output_width_tile,
          dwconv2d_parameters->ukernel,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
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

  convolution_op->type = operator_type;
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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_FLOAT;
  const enum xnn_operator_type operator_type = xnn_operator_type_convolution_nchw_f32;

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

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " subsampling: subsampling dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), subsampling_width, subsampling_height);
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
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      input_channel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      output_channel_stride, groups, group_output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(operator_type), output_min, output_max);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create depthwise %s operator with %zu input channels per group: "
      "depthwise convolution must have exactly 1 input channel per group",
      xnn_operator_type_to_string(operator_type), group_input_channels);
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  const struct xnn_dwconv2d_chw_config* dwconv2d_chw_config = xnn_init_f32_dwconv2d_chw_config();
  if (dwconv2d_chw_config == NULL) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  enum xnn_microkernel_type ukernel_type;
  const struct xnn_dwconv2d_chw_parameters* dwconv2d_parameters = NULL;
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
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_3x3;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 0 || input_padding_top == 1) && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_3x3s2;
  } else if (is_5x5 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 2 && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_5x5;
  } else if (is_5x5 && subsampling_height == 2 && subsampling_width == 2 &&
    (input_padding_top == 1 || input_padding_top == 2) && input_padding_left == 2 && input_padding_bottom == 2 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
    dwconv2d_parameters = &dwconv2d_chw_config->dwconv2d_chw_5x5s2;
  } else {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel, %"PRIu32 "x%" PRIu32 " subsampling, %"PRIu32 "x%" PRIu32 " dilation"
      ", %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding, %" PRIu32 "x%zu input channels, and %" PRIu32 "x%zu output channels: "
      "only selected convolution parameters are supported",
      xnn_operator_type_to_string(operator_type),
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
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  if (ukernel_type != xnn_microkernel_type_spmm) {
    convolution_op->weights_cache = weights_cache;
  }

  const struct xnn_spmm_config* spmm_config = xnn_init_f32_spmm_config();
  if (spmm_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    return xnn_status_unsupported_hardware;
  }
  const struct xnn_spmm_config* spmm2_config = xnn_init_f32_spmm2_config();
  if (spmm2_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    return xnn_status_unsupported_hardware;
  }
  const struct xnn_spmm_config* spmm4_config = xnn_init_f32_spmm4_config();
  if (spmm4_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nchw_f32));
    return xnn_status_unsupported_hardware;
  }

  switch (ukernel_type) {
    case xnn_microkernel_type_spmm:
    {
      const xnn_analyze_spmm_w_fn xnn_analyze_spmm = (xnn_analyze_spmm_w_fn) xnn_analyze_f32_spmm_w;
      const xnn_pack_spmm_w_fn xnn_pack_spmm = (xnn_pack_spmm_w_fn) xnn_pack_f32_spmm_w;

      spmm_config->init.f32(&convolution_op->params.f32_minmax, output_min, output_max);

      status = create_spmm_path(
          kernel_height, kernel_width, groups,
          group_input_channels, group_output_channels,
          kernel, bias, log2_filter_element_size,
          xnn_analyze_spmm, xnn_pack_spmm,
          spmm_config, spmm2_config, spmm4_config,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_conv2d_hwc2chw:
    {
      const struct xnn_conv_hwc2chw_config* conv_hwc2chw_config = xnn_init_f32_conv_hwc2chw_3x3c3s2_config();
      if (conv_hwc2chw_config == NULL) {
        status = xnn_status_unsupported_hardware;
        xnn_log_error("failed to create %s operator: operations on data type are not supported",
                      xnn_operator_type_to_string(operator_type));
        goto error;
      }

      conv_hwc2chw_config->init.f32(&convolution_op->params.f32_minmax, output_min, output_max);

      status = create_conv2d_hwc2chw_path(
          kernel_height, kernel_width, groups,
          group_input_channels,
          group_output_channels,
          conv_hwc2chw_config->output_height_tile,
          conv_hwc2chw_config->output_channel_tile,
          kernel, bias, log2_filter_element_size,
          (xnn_pack_dconv_oki_w_fn) xnn_pack_f32_dconv_oki_w,
          conv_hwc2chw_config->ukernel_with_symm_padding,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_dwconv:
    {
      dwconv2d_parameters->init.f32(&convolution_op->params.f32_chw, 0, output_min, output_max);

      status = create_dwconv_path(
          kernel_height, kernel_width, groups,
          kernel, bias, flags, log2_filter_element_size,
          (xnn_pack_chw_dwconv_hwg_w_fn) xnn_pack_f32_chw_dwconv_hwg_w,
          (xnn_pack_chw_dwconv_ghw_w_fn) xnn_pack_f32_chw_dwconv_ghw_w,
          (xnn_update_chw_params_fn) dwconv2d_parameters->update.f32,
          dwconv2d_parameters->output_width_tile,
          dwconv2d_parameters->ukernel,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
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

  convolution_op->type = operator_type;
  convolution_op->ukernel.type = ukernel_type;
  convolution_op->flags = flags;

  convolution_op->state = xnn_run_state_invalid;

  *convolution_op_out = convolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(convolution_op);
  return status;
}

static enum xnn_status reshape_convolution2d_nchw(
  xnn_operator_t convolution_op,
  enum xnn_operator_type expected_operator_type,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  void* chw_params,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (convolution_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }
  convolution_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(convolution_op->type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    convolution_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  convolution_op->batch_size = batch_size;
  convolution_op->input_height = input_height;
  convolution_op->input_width = input_width;

  const size_t output_height = xnn_compute_convolution_output_dimension(
      convolution_op->padding_top + input_height + convolution_op->padding_bottom,
      convolution_op->kernel_height,
      convolution_op->dilation_height,
      convolution_op->stride_height);
  if (output_height_out != NULL) {
    *output_height_out = output_height;
  }
  const size_t output_width = xnn_compute_convolution_output_dimension(
      convolution_op->padding_left + input_width + convolution_op->padding_right,
      convolution_op->kernel_width,
      convolution_op->dilation_width,
      convolution_op->stride_width);
  if (output_width_out != NULL) {
    *output_width_out = output_width;
  }

  const size_t input_batch_stride = (input_height * input_width * convolution_op->input_pixel_stride) << log2_input_element_size;
  const size_t output_batch_stride = (output_height * output_width * convolution_op->output_pixel_stride) << log2_output_element_size;
  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
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
            "failed to reshape %s operator with sparse kernel representation: input increment exceeds int32_t range",
            xnn_operator_type_to_string(convolution_op->type));
          return xnn_status_unsupported_parameter;
        }
        input_increments[i] = (int32_t) increment;
      }

      convolution_op->context.spmm = (struct spmm_context) {
          .n = convolution_op->group_output_channels,
          .scaled_m = input_size << log2_input_element_size,
          .nonzero_weights = nonzero_values,
          .input_increments = input_increments,
          .output_channel_nonzeros = output_channel_nonzeros,
          .batched_input_stride = input_batch_stride,
          .batched_output_stride = output_batch_stride,
          .ukernel = convolution_op->ukernel.spmm.function,
      };
      memcpy(&convolution_op->context.spmm.params, params, sizeof(convolution_op->context.spmm.params));

      const size_t mr = convolution_op->ukernel.spmm.mr;
      size_t mc = input_size;
      if (num_threads > 1) {
        const size_t target_tiles_per_thread = 5;
        const size_t max_mc = divide_round_up(input_size, num_threads * target_tiles_per_thread);
        if (max_mc < mc) {
          mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
        }
      }
      convolution_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
      convolution_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_spmm;
      convolution_op->compute[0].range[0] = batch_size;
      convolution_op->compute[0].range[1] = input_size << log2_input_element_size;
      convolution_op->compute[0].tile[0] = mc << log2_input_element_size;
      convolution_op->state = xnn_run_state_needs_setup;

      return xnn_status_success;
    }
    case xnn_microkernel_type_conv2d_hwc2chw:
    {
      const size_t zero_size = (input_width * convolution_op->group_input_channels << log2_input_element_size) + XNN_EXTRA_BYTES;

      // Note: zero buffer must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(convolution_op->zero_buffer);
      convolution_op->zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
      if (convolution_op->zero_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator zero padding",
          sizeof(struct xnn_operator), xnn_operator_type_to_string(convolution_op->type));
        return xnn_status_out_of_memory;
      }

      convolution_op->context.conv2d = (struct conv2d_context) {
        .input_height = input_height,
        .input_width = input_width,
        .input_batch_stride = input_batch_stride,
        .zero = convolution_op->zero_buffer,
        .packed_weights = packed_weights(convolution_op),
        .output_batch_stride = output_batch_stride,
        .input_padding_top = convolution_op->padding_top,
        .output_channels = convolution_op->group_output_channels,
        .output_height_stride = output_width << log2_output_element_size,
        .output_channel_stride = output_height * output_width << log2_output_element_size,
        .hwc2chw_ukernel = convolution_op->ukernel.conv2d.hwc2chw_fn,
      };
      memcpy(&convolution_op->context.conv2d.params, params, sizeof(convolution_op->context.conv2d.params));

      const size_t output_height_tile = convolution_op->ukernel.conv2d.output_height_tile;
      size_t output_height_slice = output_height;
      if (num_threads > 1) {
        const size_t target_tiles_per_thread = 5;
        const size_t max_output_height_slice = divide_round_up(output_height, num_threads * target_tiles_per_thread);
        if (max_output_height_slice < output_height_slice) {
          output_height_slice = min(output_height_slice,
            divide_round_up(output_height_slice, max_output_height_slice * output_height_tile) * output_height_tile);
        }
      }
      convolution_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
      convolution_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_conv2d_hwc2chw;
      convolution_op->compute[0].range[0] = batch_size;
      convolution_op->compute[0].range[1] = output_height;
      convolution_op->compute[0].tile[0] = output_height_slice;
      convolution_op->state = xnn_run_state_needs_setup;

      return xnn_status_success;
    }
    case xnn_microkernel_type_dwconv:
    {
      const size_t zero_size = (input_width << log2_input_element_size) + 2 * XNN_EXTRA_BYTES;

      // Note: zero buffer must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(convolution_op->zero_buffer);
      convolution_op->zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
      if (convolution_op->zero_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator zero padding",
          sizeof(struct xnn_operator), xnn_operator_type_to_string(convolution_op->type));
        return xnn_status_out_of_memory;
      }

      if (convolution_op->ukernel.dwconv2d.update_params != NULL) {
        convolution_op->ukernel.dwconv2d.update_params(chw_params, (uint32_t) input_width);
      }
      convolution_op->context.dwconv2d = (struct dwconv2d_context) {
        .input_height = input_height,
        .input_width = input_width << log2_input_element_size,
        .zero = convolution_op->zero_buffer,
        .input_padding_top = convolution_op->padding_top,
        .input_channel_stride = input_height * input_width << log2_input_element_size,
        .input_batch_stride = input_batch_stride,
        .packed_weights = packed_weights(convolution_op),
        .weights_channel_stride = bias_element_size +
          (convolution_op->kernel_height * convolution_op->kernel_width << log2_filter_element_size),
        .output_channel_stride = output_height * output_width << log2_output_element_size,
        .output_batch_stride = output_batch_stride,
        .chw_ukernel = convolution_op->ukernel.dwconv2d.chw_fn,
      };
      memcpy(&convolution_op->context.dwconv2d.params, chw_params, sizeof(convolution_op->context.dwconv2d.params));

      convolution_op->compute[0].type = xnn_parallelization_type_2d;
      convolution_op->compute[0].task_2d = (pthreadpool_task_2d_t) xnn_compute_dwconv2d_chw;
      convolution_op->compute[0].range[0] = batch_size;
      convolution_op->compute[0].range[1] = convolution_op->groups;
      convolution_op->state = xnn_run_state_needs_setup;

      return xnn_status_success;
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_reshape_convolution2d_nchw_f16(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  return reshape_convolution2d_nchw(
    convolution_op,
    xnn_operator_type_convolution_nchw_f16,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*bias_element_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    &convolution_op->params.f16_minmax,
    &convolution_op->params.f16_chw,
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nchw_f32(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  return reshape_convolution2d_nchw(
    convolution_op,
    xnn_operator_type_convolution_nchw_f32,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*bias_element_size=*/sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &convolution_op->params.f32_minmax,
    &convolution_op->params.f32_chw,
    output_height_out, output_width_out,
    threadpool);
}

static enum xnn_status setup_convolution2d_nchw(
  xnn_operator_t convolution_op,
  enum xnn_operator_type expected_operator_type,
  const void* input,
  void* output)
{
  if (convolution_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }

  if (convolution_op->weights_cache != NULL && !xnn_weights_cache_is_finalized(convolution_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_state;
  }

  switch (convolution_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(convolution_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  switch (convolution_op->ukernel.type) {
    case xnn_microkernel_type_spmm:
    {
      convolution_op->context.spmm.input = (const void*) ((uintptr_t) input + (convolution_op->first_input_channel *
                                                                               convolution_op->context.spmm.scaled_m));
      convolution_op->context.spmm.output = output;
      break;
    }
    case xnn_microkernel_type_conv2d_hwc2chw:
    {
      convolution_op->context.conv2d.input = input;
      convolution_op->context.conv2d.output = output;
      break;
    }
    case xnn_microkernel_type_dwconv:
    {
      convolution_op->context.dwconv2d.input = input;
      convolution_op->context.dwconv2d.output = output;
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_convolution2d_nchw_f16(
    xnn_operator_t convolution_op,
    const void* input,
    void* output)
{
  return setup_convolution2d_nchw(
    convolution_op,
    xnn_operator_type_convolution_nchw_f16,
    input, output);
}

enum xnn_status xnn_setup_convolution2d_nchw_f32(
    xnn_operator_t convolution_op,
    const float* input,
    float* output)
{
  return setup_convolution2d_nchw(
    convolution_op,
    xnn_operator_type_convolution_nchw_f32,
    input, output);
}
