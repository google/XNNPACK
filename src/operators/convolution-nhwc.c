// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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
#include "xnnpack/indirection.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-type.h"
#include "xnnpack/microkernel-utils.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/pack.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

#ifndef XNN_ENABLE_GEMM_M_SPECIALIZATION
#error "XNN_ENABLE_GEMM_M_SPECIALIZATION is not defined"
#endif

static inline size_t compute_output_dimension_with_tf_same_padding(
    size_t input_dimension,
    size_t subsampling_dimension)
{
  return divide_round_up(input_dimension, subsampling_dimension);
}

static inline const struct xnn_dwconv_config* find_dwconv_ukernel(
    size_t kernel_size,
    const struct xnn_dwconv_config* ukernel,
    size_t num_ukernels)
{
  const struct xnn_dwconv_config* best_ukernel = NULL;
  while (num_ukernels-- != 0) {
    // Find the smallest unipass primary_tile that is at least as big as kernel_size.
    if (ukernel->last_tile == 0 && ukernel->primary_tile >= kernel_size) {
      if (best_ukernel == NULL || ukernel->primary_tile < best_ukernel->primary_tile) {
        best_ukernel = ukernel;
      }
    } else if (ukernel->last_tile != 0) {
      // Use multi-pass if it fits the kernel size nicely, or if kernel_size is large.
      if (ukernel->primary_tile + ukernel->middle_tile + ukernel->last_tile == kernel_size || kernel_size >= 25) {
        best_ukernel = ukernel;
      }
    }
    ukernel++;
  }
  if (best_ukernel == NULL) {
    xnn_log_debug("no dwconv ukernel found");
  } else if (best_ukernel->last_tile == 0) {
    xnn_log_debug("dwconv unipass ukernel of primary tile %"PRIu8" found", best_ukernel->primary_tile);
  } else {
    xnn_log_debug("dwconv multipass ukernel of tiles %"PRIu8", %"PRIu8", %"PRIu8" found",
                  best_ukernel->primary_tile,
                  best_ukernel->middle_tile,
                  best_ukernel->last_tile);
  }
  return best_ukernel;
}

static enum xnn_status create_vmulcaddc_path(
    uint32_t groups,
    const void* kernel,
    const void* bias,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w,
    const void* packing_params,
    int packed_weights_padding_byte,
    const void* vmulcaddc_params,
    size_t vmulcaddc_params_size,
    const struct xnn_vmulcaddc_config* vmulcaddc_config,
    enum xnn_operator_type operator_type,
    xnn_operator_t convolution_op)
{
  assert(vmulcaddc_config != NULL);
  assert(vmulcaddc_params != NULL);

  enum xnn_status status = xnn_status_out_of_memory;

  const size_t c_stride = round_up_po2(groups, vmulcaddc_config->channel_tile);
  const size_t packed_weights_size = ((UINT32_C(1) << log2_filter_element_size) + bias_element_size) * c_stride;
  size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size, packed_weights_padding_byte);
  if (weights_ptr == NULL) {
    xnn_log_error("failed to reserve or allocated %zu bytes for %s operator vmulcaddc packed weights",
                  aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  pack_vmulcaddc_w(groups, vmulcaddc_config->channel_tile, kernel, bias, weights_ptr, packing_params);

  if (use_weights_cache(convolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = groups ^ vmulcaddc_config->channel_tile;
    cache_key.kernel = kernel;
    cache_key.bias = bias;
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr, aligned_total_weights_size);
  }

  memcpy(&convolution_op->params, vmulcaddc_params, vmulcaddc_params_size);

  convolution_op->ukernel.vmulcaddc = (struct xnn_ukernel_vmulcaddc) {
    .function = vmulcaddc_config->ukernel,
    .mr = vmulcaddc_config->row_tile,
  };
  return xnn_status_success;

error:
  return status;
}

static enum xnn_status create_dwconv_path(
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t groups,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w,
    xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w,
    const void* packing_params,
    int packed_weights_padding_byte,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params,
    const void* dwconv_params,
    size_t dwconv_params_size,
    const struct xnn_dwconv_config* dwconv_ukernel,
    bool linear_activation,
    enum xnn_operator_type operator_type,
    size_t* zero_size,
    xnn_operator_t convolution_op)
{
  assert(dwconv_ukernel != NULL);
  enum xnn_status status = xnn_status_out_of_memory;
  const uint8_t primary_tile = dwconv_ukernel->primary_tile;
  const bool is_unipass = dwconv_ukernel->last_tile == 0;
  const size_t kernel_size = kernel_height * kernel_width;
  if (is_unipass) {
    assert(primary_tile >= kernel_size);
    xnn_log_debug("using dwconv unipass of primary_tile %u", primary_tile);
  } else {
    assert(kernel_size > primary_tile);
    xnn_log_debug("using dwconv multipass ukernel of tiles %d, %d, %d",
                  primary_tile,
                  dwconv_ukernel->middle_tile,
                  dwconv_ukernel->last_tile);
  }

  const size_t c_stride = round_up_po2(groups, dwconv_ukernel->channel_tile);
  size_t tile_size = 0;
  size_t packed_weights_size = 0;
  if (is_unipass) {
    tile_size = primary_tile;
    packed_weights_size = ((primary_tile << log2_filter_element_size) + bias_element_size + extra_weights_bytes) * c_stride;
  } else {
    tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size, primary_tile, dwconv_ukernel->middle_tile, dwconv_ukernel->last_tile);
    packed_weights_size = xnn_dwconv_multipass_weights_size(
      tile_size, groups, dwconv_ukernel->channel_tile, dwconv_ukernel->channel_subtile,
      dwconv_ukernel->channel_round, bias_element_size, log2_filter_element_size, extra_weights_bytes);
  }
  size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size, packed_weights_padding_byte);
  if (weights_ptr == NULL) {
    xnn_log_error("failed to reserve or allocated %zu bytes for %s operator dwconv packed weights",
                  aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  memcpy(&convolution_op->params, dwconv_params, dwconv_params_size);

  if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
    pack_dwconv_hwg_w(
        primary_tile,
        dwconv_ukernel->middle_tile,
        dwconv_ukernel->last_tile,
        kernel_height, kernel_width,
        groups,
        dwconv_ukernel->channel_tile, dwconv_ukernel->channel_subtile, dwconv_ukernel->channel_round,
        kernel, bias, /*scale=*/NULL, weights_ptr,
        dwconv_ukernel->channel_tile * extra_weights_bytes,
        dwconv_ukernel->channel_subtile * extra_weights_bytes,
        packing_params);
  } else {
    pack_dwconv_ghw_w(
        primary_tile,
        dwconv_ukernel->middle_tile,
        dwconv_ukernel->last_tile,
        kernel_height, kernel_width,
        groups,
        dwconv_ukernel->channel_tile, dwconv_ukernel->channel_subtile, dwconv_ukernel->channel_round,
        kernel, bias, /*scale=*/NULL, weights_ptr,
        dwconv_ukernel->channel_tile * extra_weights_bytes,
        dwconv_ukernel->channel_subtile * extra_weights_bytes,
        packing_params);
  }

  if (scale_params != NULL) {
    assert(init_scale_params != NULL);
    // TODO(zhin): QC8 DWCONV multipass is not implemented for now, fix this when it is supported.
    assert(is_unipass);
    size_t stride = dwconv_ukernel->channel_tile *
                    ((primary_tile << log2_filter_element_size) + bias_element_size + extra_weights_bytes);

    init_scale_params(
      /*channels=*/groups,
      /*channels_tile=*/dwconv_ukernel->channel_tile,
      /*channels_subtile=*/dwconv_ukernel->channel_tile,
      /*stride=*/stride,
      /*substride=*/stride,
      /*stride_offset=*/0,
      /*scale=*/scale_params,
      /*packed_w=*/
      (void*) ((uintptr_t) weights_ptr +
               dwconv_ukernel->channel_tile * ((primary_tile << log2_filter_element_size) + bias_element_size)));
  }

  uint32_t cache_seed = primary_tile ^ dwconv_ukernel->middle_tile ^ dwconv_ukernel->last_tile ^ kernel_height ^ kernel_width
      ^ groups ^ dwconv_ukernel->channel_tile ^ dwconv_ukernel->channel_subtile ^ dwconv_ukernel->channel_round ^ extra_weights_bytes;
  if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
    cache_seed = ~cache_seed;
  }
  if (use_weights_cache(convolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = cache_seed;
    cache_key.kernel = kernel;
    cache_key.bias = bias;
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr, aligned_total_weights_size);
  }

  const union xnn_dwconv_ukernel* ukernels = &dwconv_ukernel->minmax;
  if (linear_activation && dwconv_ukernel->linear.unipass != NULL) {
    ukernels = &dwconv_ukernel->linear;
  }
  convolution_op->ukernel.dwconv = (struct xnn_ukernel_dwconv) {
    .primary_tile = primary_tile,
    .middle_tile = dwconv_ukernel->middle_tile,
    .last_tile = dwconv_ukernel->last_tile,
    .tile_size = tile_size,
  };

  if (is_unipass) {
    convolution_op->ukernel.dwconv.unipass_fn = ukernels->unipass;
  } else {
    convolution_op->ukernel.dwconv.multipass_fn = ukernels->multipass;
  }

  *zero_size = XNN_EXTRA_BYTES + (c_stride << log2_input_element_size);
  return xnn_status_success;
error:
  return status;
}

static enum xnn_status create_gemm_or_igemm(
    enum xnn_microkernel_type ukernel_type,
    uint32_t kernel_size,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w,
    xnn_pack_conv_kgo_w_fn pack_conv_kgo_w,
    xnn_pack_conv_goki_w_fn pack_conv_goki_w,
    const void* packing_params,
    int packed_weights_padding_byte,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params,
    xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params,
    const float* kernel_scale_params,
    const void* gemm_params,
    size_t gemm_params_size,
    const struct xnn_gemm_config* gemm_config,
    bool linear_activation,
    bool relu_activation,
    enum xnn_operator_type operator_type,
    xnn_operator_t convolution_op,
    size_t* zero_size)
{
  enum xnn_status status = xnn_status_out_of_memory;
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t n_stride = round_up(group_output_channels, nr);
  const size_t k_stride = round_up_po2(group_input_channels, kr * sr);

  const uint32_t cache_seed = groups ^ group_input_channels ^ group_output_channels ^ nr ^ kr ^ sr ^ ukernel_type ^ flags;

  if (use_weights_cache(convolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = cache_seed;
    cache_key.kernel = kernel;
    cache_key.bias = bias;
    convolution_op->packed_weights.offset = xnn_weights_cache_look_up(
        convolution_op->weights_cache, &cache_key);
  }

  const bool weights_already_cached = use_weights_cache(convolution_op) &&
      convolution_op->packed_weights.offset != XNN_CACHE_NOT_FOUND;

  const size_t packed_group_weights_size =
      ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes) * n_stride;
  const size_t aligned_total_weights_size = round_up_po2(packed_group_weights_size * groups, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = NULL;

  if (!weights_already_cached) {
    weights_ptr = xnn_get_pointer_to_write_weights(convolution_op, aligned_total_weights_size,
                                                   packed_weights_padding_byte);
    if (!weights_already_cached && weights_ptr == NULL) {
      xnn_log_error( "failed to reserve or allocated %zu bytes for %s operator gemm packed weights",
          aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
      goto error;
    }
    xnn_log_debug("allocated %zu bytes for packed weights in %s operator", aligned_total_weights_size,
                  xnn_operator_type_to_string(operator_type));
  }

  memcpy(&convolution_op->params, gemm_params, gemm_params_size);

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const uint32_t mr = gemm_config->mr;
  if (linear_activation && gemm_config->linear.gemm[mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  } else if (relu_activation && gemm_config->relu.gemm[mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->relu;
  }
  switch (ukernel_type) {
    case xnn_microkernel_type_gemm:
      if(!weights_already_cached) {
        pack_gemm_goi_w(groups, group_output_channels, group_input_channels,
                        nr, kr, sr,
                        kernel, bias, /*scale=*/NULL, weights_ptr, gemm_config->nr * extra_weights_bytes,
                        packing_params);
      }
      convolution_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
        .mr = mr,
        .nr = nr,
        .kr = kr,
        .sr = sr,
      };

      assert(XNN_MAX_MR >= mr);
      for (size_t i = 0; i < mr; i++) {
        convolution_op->ukernel.gemm.gemm_cases[i] = gemm_ukernels->gemm[i];
      }

      break;
    case xnn_microkernel_type_igemm:
      if(!weights_already_cached) {
        if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
          pack_conv_kgo_w(
              groups, group_output_channels, kernel_size,
              nr, kr, sr,
              kernel, bias, /*scale=*/NULL, weights_ptr, gemm_config->nr * extra_weights_bytes, packing_params);
        } else {
          pack_conv_goki_w(
              groups, group_output_channels, kernel_size, group_input_channels,
              nr, kr, sr,
              kernel, bias, /*scale=*/NULL, weights_ptr, gemm_config->nr * extra_weights_bytes, packing_params);
        }
      }
      convolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
        .mr = mr,
            .nr = nr,
            .kr = kr,
            .sr = sr,
      };

      assert(XNN_MAX_MR >= mr);
      for (size_t i = 0; i < mr; i++) {
        convolution_op->ukernel.igemm.igemm_cases[i] = gemm_ukernels->igemm[i];
      }

      break;
    default:
      XNN_UNREACHABLE;
  }

  if (kernel_scale_params != NULL) {
    assert(init_kernel_scale_params != NULL);

    if(!weights_already_cached) {
      void* group_weights =
          (void*)((uintptr_t) weights_ptr +
                  gemm_config->nr * ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size));
      const size_t weights_stride =
          (kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
      for (uint32_t group = 0; group < groups; group++) {
        init_kernel_scale_params(
            group_output_channels, gemm_config->nr, gemm_config->nr,
            gemm_config->nr * weights_stride, gemm_config->nr * weights_stride, 0,
            kernel_scale_params, group_weights);
        kernel_scale_params += group_output_channels;
        group_weights = (void*) ((uintptr_t) group_weights + n_stride * weights_stride);
      }
    }
  }

  if (scale_params != NULL) {
    assert(init_scale_params != NULL);

    if(!weights_already_cached) {
      void* group_weights =
          (void*)((uintptr_t) weights_ptr +
                  gemm_config->nr * ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size));
      if (kernel_scale_params != NULL) {
        group_weights = (void*) ((uintptr_t) group_weights + gemm_config->nr * sizeof(float));
      }
      const size_t weights_stride =
          (kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
      for (uint32_t group = 0; group < groups; group++) {
        init_scale_params(
            group_output_channels, gemm_config->nr, gemm_config->nr,
            gemm_config->nr * weights_stride, gemm_config->nr * weights_stride, 0,
            scale_params, group_weights);
        scale_params += group_output_channels;
        group_weights = (void*) ((uintptr_t) group_weights + n_stride * weights_stride);
      }
    }
  }

  if (use_weights_cache(convolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = cache_seed;
    cache_key.kernel = kernel;
    cache_key.bias = bias;
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr, aligned_total_weights_size);
  }

  *zero_size = XNN_EXTRA_BYTES + (k_stride << log2_input_element_size);
  return xnn_status_success;

error:
  return status;
}

static enum xnn_status create_convolution2d_nhwc(
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
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w,
    xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w,
    xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w,
    xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w,
    xnn_pack_conv_kgo_w_fn pack_conv_kgo_w,
    xnn_pack_conv_goki_w_fn pack_conv_goki_w,
    const void* packing_params,
    int input_padding_byte,
    int packed_weights_padding_byte,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params,
    xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params,
    const float* kernel_scale_params,
    const void* gemm_params,
    size_t gemm_params_size,
    const void* dwconv_params,
    size_t dwconv_params_size,
    const void* vmulcaddc_params,
    size_t vmulcaddc_params_size,
    const struct xnn_gemm_config* gemm_config,
    const struct xnn_dwconv_config* dwconv_ukernel,
    const struct xnn_vmulcaddc_config* vmulcaddc_config,
    bool linear_activation,
    bool relu_activation,
    enum xnn_operator_type operator_type,
    bool dynamic_quantization,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to create %s operator: XNNPACK is not initialized",
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

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create depthwise %s operator with %zu input channels per group: "
      "depthwise convolution must have exactly 1 input channel per group",
      xnn_operator_type_to_string(operator_type), group_input_channels);
    goto error;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to create %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        xnn_operator_type_to_string(operator_type),
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      goto error;
    }
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  convolution_op->weights_cache = weights_cache;

  const size_t kernel_size = kernel_height * kernel_width;

  enum xnn_microkernel_type ukernel_type = xnn_microkernel_type_default;
  const bool unit_subsampling = (subsampling_width | subsampling_height) == 1;
  if (group_input_channels == 1 && group_output_channels == 1 && kernel_size == 1 && unit_subsampling && !any_padding && vmulcaddc_config != NULL) {
    ukernel_type = xnn_microkernel_type_vmulcaddc;
  } else if (group_input_channels == 1 && group_output_channels == 1 && dwconv_ukernel != NULL) {
    ukernel_type = xnn_microkernel_type_dwconv;
  } else if (kernel_size == 1 && unit_subsampling && !any_padding && !dynamic_quantization) {
    ukernel_type = xnn_microkernel_type_gemm;
  } else {
    ukernel_type = xnn_microkernel_type_igemm;
  }
  assert(ukernel_type != xnn_microkernel_type_default);

  size_t zero_size = 0;
  switch (ukernel_type) {
    case xnn_microkernel_type_vmulcaddc:
    {
      status = create_vmulcaddc_path(
          groups, kernel, bias, log2_filter_element_size, bias_element_size,
          pack_vmulcaddc_w, packing_params, packed_weights_padding_byte,
          vmulcaddc_params, vmulcaddc_params_size, vmulcaddc_config,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_dwconv:
    {
      status = create_dwconv_path(
          kernel_height, kernel_width,
          groups, kernel, bias, flags,
          log2_input_element_size, log2_filter_element_size, bias_element_size,
          pack_dwconv_hwg_w, pack_dwconv_ghw_w,
          packing_params, packed_weights_padding_byte, extra_weights_bytes,
          init_scale_params, scale_params,
          dwconv_params, dwconv_params_size, dwconv_ukernel,
          linear_activation, operator_type, &zero_size, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_gemm:
    case xnn_microkernel_type_igemm:
    {
      status = create_gemm_or_igemm(
          ukernel_type, kernel_size,
          groups, group_input_channels, group_output_channels,
          kernel, bias, flags,
          log2_input_element_size, log2_filter_element_size, bias_element_size,
          pack_gemm_goi_w, pack_conv_kgo_w, pack_conv_goki_w, packing_params,
          packed_weights_padding_byte, extra_weights_bytes,
          init_scale_params, scale_params, init_kernel_scale_params, kernel_scale_params,
          gemm_params, gemm_params_size, gemm_config,
          linear_activation, relu_activation,
          operator_type,
          convolution_op,
          &zero_size);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  convolution_op->zero_size = 0;
  const bool tf_same_padding = (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && kernel_size != 1;
  if (any_padding || tf_same_padding) {
    convolution_op->zero_size = zero_size;
    convolution_op->zero_buffer = xnn_allocate_simd_memory(zero_size);
    if (convolution_op->zero_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator zero padding",
        zero_size, xnn_operator_type_to_string(operator_type));
      goto error;
    }
    memset(convolution_op->zero_buffer, input_padding_byte, zero_size);
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
  convolution_op->flags = flags & ~XNN_FLAG_TENSORFLOW_SAME_PADDING;
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

enum xnn_status xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
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
    const float* kernel_scale,
    const int8_t* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qd8_f16_qc8w));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qd8_f16_qc8w));
    return xnn_status_invalid_parameter;
  }
  const uint16_t fp16_output_min = fp16_ieee_from_fp32_value(output_min);
  const uint16_t fp16_output_max = fp16_ieee_from_fp32_value(output_max);
  const float rounded_output_min = fp16_ieee_to_fp32_value(fp16_output_min);
  const float rounded_output_max = fp16_ieee_to_fp32_value(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_f16), rounded_output_min, rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = 1, };

  const struct xnn_gemm_config* gemm_config = xnn_init_qd8_f16_qc8w_gemm_config();
  if (gemm_config == NULL) {
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params gemm_params;
  if XNN_LIKELY(gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&gemm_params, fp16_output_min, fp16_output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, /*bias=*/NULL, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*bias_element_size=*/sizeof(float),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) NULL,
    (xnn_pack_dwconv_ghw_w_fn) NULL,
    (xnn_packw_gemm_goi_ukernel_fn) gemm_config->pack_gemm_goi,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/sizeof(float) * 2,
    xnn_init_qs8_qc8w_scale_fp32_params, bias,
    xnn_init_qs8_qc8w_scale_fp32_params, kernel_scale,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/NULL,
    /*dwconv_params_size=*/0,
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_config=*/gemm_config,
    /*dwconv_ukernel=*/NULL,
    /*vmulcaddc_config=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qd8_f16_qc8w,
    /*dynamic_quantization=*/true,
    /*weights_cache=*/weights_cache,
    convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
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
    const float* kernel_scale,
    const int8_t* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qd8_f32_qc8w));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qd8_f32_qc8w));
    return xnn_status_invalid_parameter;
  }
  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qd8_f32_qc8w), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = 1, };

  const struct xnn_gemm_config* gemm_config = xnn_init_qd8_f32_qc8w_gemm_config();
  assert(gemm_config != NULL);

  union xnn_f32_minmax_params gemm_params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&gemm_params, output_min, output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, /*bias=*/NULL, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*bias_element_size=*/sizeof(float),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) NULL,
    (xnn_pack_dwconv_ghw_w_fn) NULL,
    (xnn_packw_gemm_goi_ukernel_fn) gemm_config->pack_gemm_goi,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/sizeof(float) * 2,
    xnn_init_qs8_qc8w_scale_fp32_params, bias,
    xnn_init_qs8_qc8w_scale_fp32_params, kernel_scale,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/NULL,
    /*dwconv_params_size=*/0,
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_config=*/gemm_config,
    /*dwconv_ukernel=*/NULL,
    /*vmulcaddc_config=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qd8_f32_qc8w,
    /*dynamic_quantization=*/true,
    /*weights_cache=*/weights_cache,
    convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qu8(
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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_qu8_packing_params packing_params = {
    .input_zero_point = input_zero_point,
    .kernel_zero_point = kernel_zero_point,
  };

  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  assert(gemm_config != NULL);

  union xnn_qu8_conv_minmax_params gemm_params;
  if XNN_LIKELY(gemm_config->init.qu8 != NULL) {
    gemm_config->init.qu8(&gemm_params,
      kernel_zero_point, requantization_scale, output_zero_point, output_min, output_max);
  }

  const struct xnn_dwconv_config* dwconv_config = xnn_init_qu8_dwconv_config();
  assert(dwconv_config != NULL);

  union xnn_qu8_conv_minmax_params dwconv_params;
  const struct xnn_dwconv_config* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, dwconv_config, XNN_MAX_QU8_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.qu8(&dwconv_params,
      kernel_zero_point, requantization_scale, output_zero_point, output_min, output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_qu8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_qu8_dwconv_ghw_w,
    (xnn_packw_gemm_goi_ukernel_fn) gemm_config->pack_gemm_goi,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_qu8_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_qu8_conv_goki_w,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/input_zero_point,
    /*packed_weights_padding_byte=*/kernel_zero_point,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_config=*/gemm_config,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_config=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qu8,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qs8(
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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  float* duplicated_requantization_scale = xnn_allocate_simd_memory(groups * group_output_channels * sizeof(float));
  if (duplicated_requantization_scale == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      groups * group_output_channels * sizeof(float),
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8));
    return xnn_status_out_of_memory;
  }
  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    duplicated_requantization_scale[output_channel] = requantization_scale;
  }

  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = input_zero_point, };

  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  assert(gemm_config != NULL);

  union xnn_qs8_qc8w_conv_minmax_params gemm_params;
  if XNN_LIKELY(gemm_config->init.qs8_qc8w != NULL) {
    gemm_config->init.qs8_qc8w(&gemm_params,
      output_zero_point, output_min, output_max);
  }

  const struct xnn_dwconv_config* dwconv_config = xnn_init_qs8_qc8w_dwconv_config();
  assert(dwconv_config != NULL);

  union xnn_qs8_qc8w_conv_minmax_params dwconv_params;
  const struct xnn_dwconv_config* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, dwconv_config, XNN_MAX_QC8_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.qs8_qc8w(&dwconv_params,
      output_zero_point, output_min, output_max);
  }

  enum xnn_status status = create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_qs8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_qs8_dwconv_ghw_w,
    (xnn_packw_gemm_goi_ukernel_fn) gemm_config->pack_gemm_goi,
    (xnn_pack_conv_kgo_w_fn) gemm_config->pack_igemm_kgo,
    (xnn_pack_conv_goki_w_fn) gemm_config->pack_igemm_goki,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/input_zero_point,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/sizeof(float),
    /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
    /*scale_params=*/duplicated_requantization_scale,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_config=*/gemm_config,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_config=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qs8,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    convolution_op_out);

  xnn_release_simd_memory(duplicated_requantization_scale);
  return status;
}

enum xnn_status xnn_create_convolution2d_nhwc_qs8_qc8w(
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
    int8_t input_zero_point,
    float input_scale,
    const float* kernel_scale,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), input_scale);
    return xnn_status_invalid_parameter;
  }

  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    if (kernel_scale[output_channel] <= 0.0f || !isnormal(kernel_scale[output_channel])) {
      xnn_log_error(
        "failed to create %s operator with %.7g kernel scale in output channel #%zu: "
        "scale must be finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), kernel_scale[output_channel],
        output_channel);
      return xnn_status_invalid_parameter;
    }
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  float* requantization_scale = xnn_allocate_simd_memory(groups * group_output_channels * sizeof(float));
  if (requantization_scale == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      groups * group_output_channels * sizeof(float),
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8));
    return xnn_status_out_of_memory;
  }
  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    requantization_scale[output_channel] = input_scale * kernel_scale[output_channel] / output_scale;
    if (requantization_scale[output_channel] >= 256.0f) {
      xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale in output channel #%zu: "
        "requantization scale %.7g is greater or equal to 256.0",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8),
        input_scale, kernel_scale[output_channel], output_scale,
        output_channel, requantization_scale[output_channel]);

      xnn_release_simd_memory(requantization_scale);
      return xnn_status_unsupported_parameter;
    }
  }

  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = input_zero_point, };

  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  assert(gemm_config != NULL);

  union xnn_qs8_qc8w_conv_minmax_params gemm_params;
  if XNN_LIKELY(gemm_config->init.qs8_qc8w != NULL) {
    gemm_config->init.qs8_qc8w(&gemm_params,
      output_zero_point, output_min, output_max);
  }

  const struct xnn_dwconv_config* dwconv_config = xnn_init_qs8_qc8w_dwconv_config();
  assert(dwconv_config != NULL);

  union xnn_qs8_qc8w_conv_minmax_params dwconv_params;
  const struct xnn_dwconv_config* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, dwconv_config, XNN_MAX_QC8_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.qs8_qc8w(&dwconv_params,
      output_zero_point, output_min, output_max);
  }

  enum xnn_status status = create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_qs8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_qs8_dwconv_ghw_w,
    (xnn_packw_gemm_goi_ukernel_fn) gemm_config->pack_gemm_goi,
    (xnn_pack_conv_kgo_w_fn) gemm_config->pack_igemm_kgo,
    (xnn_pack_conv_goki_w_fn) gemm_config->pack_igemm_goki,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/input_zero_point,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/sizeof(float),
    /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
    /*scale_params=*/requantization_scale,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_config=*/gemm_config,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_config=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qc8,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    convolution_op_out);

  xnn_release_simd_memory(requantization_scale);
  return status;
}

enum xnn_status xnn_create_convolution2d_nhwc_f16(
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
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  const uint16_t fp16_output_min = fp16_ieee_from_fp32_value(output_min);
  const uint16_t fp16_output_max = fp16_ieee_from_fp32_value(output_max);
  const float rounded_output_min = fp16_ieee_to_fp32_value(fp16_output_min);
  const float rounded_output_max = fp16_ieee_to_fp32_value(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16), rounded_output_min, rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params gemm_params;
  if XNN_LIKELY(gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&gemm_params, fp16_output_min, fp16_output_max);
  }

  const struct xnn_dwconv_config* dwconv_config = xnn_init_f16_dwconv_config();
  if (dwconv_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params dwconv_params;
  const struct xnn_dwconv_config* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, dwconv_config, XNN_MAX_F16_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.f16(&dwconv_params, fp16_output_min, fp16_output_max);
  }

  const struct xnn_vmulcaddc_config* vmulcaddc_config = xnn_init_f16_vmulcaddc_config();
  if (vmulcaddc_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params vmulcaddc_params;
  if XNN_LIKELY(vmulcaddc_config->init.f16 != NULL) {
    vmulcaddc_config->init.f16(&vmulcaddc_params, fp16_output_min, fp16_output_max);
  }

  xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w = (xnn_pack_vmulcaddc_w_fn) xnn_pack_f16_vmulcaddc_w;
  xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w = (xnn_pack_dwconv_hwg_w_fn) xnn_pack_f16_dwconv_hwg_w;
  xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w = (xnn_pack_dwconv_ghw_w_fn) xnn_pack_f16_dwconv_ghw_w;
  xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w = (xnn_packw_gemm_goi_ukernel_fn) gemm_config->pack_gemm_goi;
  xnn_pack_conv_kgo_w_fn pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn) xnn_pack_f16_conv_kgo_w;
  xnn_pack_conv_goki_w_fn pack_conv_goki_w = (xnn_pack_conv_goki_w_fn) xnn_pack_f16_conv_goki_w;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_vmulcaddc_w = (xnn_pack_vmulcaddc_w_fn) xnn_pack_f32_to_f16_vmulcaddc_w;
    pack_dwconv_hwg_w = (xnn_pack_dwconv_hwg_w_fn) xnn_pack_f32_to_f16_dwconv_hwg_w;
    pack_dwconv_ghw_w = (xnn_pack_dwconv_ghw_w_fn) xnn_pack_f32_to_f16_dwconv_ghw_w;
    pack_gemm_goi_w = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_to_f16_gemm_goi_w;
    pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn) xnn_pack_f32_to_f16_conv_kgo_w;
    pack_conv_goki_w = (xnn_pack_conv_goki_w_fn) xnn_pack_f32_to_f16_conv_goki_w;
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*bias_element_size=*/sizeof(uint16_t),
    pack_vmulcaddc_w,
    pack_dwconv_hwg_w,
    pack_dwconv_ghw_w,
    pack_gemm_goi_w,
    pack_conv_kgo_w,
    pack_conv_goki_w,
    /*packing_params=*/NULL,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/&vmulcaddc_params,
    /*vmulcaddc_params_size=*/sizeof(vmulcaddc_params),
    /*gemm_config=*/gemm_config,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_config=*/vmulcaddc_config,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_f16,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    convolution_op_out);
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
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  const bool relu_activation = (output_max == INFINITY) && (output_min == 0.0f);

  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_gemm_config* gemm_nr2_config = xnn_init_f32_gemm_nr2_config();
  if (gemm_nr2_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  if (gemm_config->nr > group_output_channels) {
    // Default micro-kernel is suboptimal. Try to find a better micro-kernel.

    if (gemm_nr2_config->minmax.igemm[gemm_config->mr].function[XNN_UARCH_DEFAULT] != NULL) {
      gemm_config = gemm_nr2_config;
    }
  }

  union xnn_f32_minmax_params gemm_params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&gemm_params, output_min, output_max);
  }

  const struct xnn_dwconv_config* dwconv_config = xnn_init_f32_dwconv_config();
  if (dwconv_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_minmax_params dwconv_params;
  const struct xnn_dwconv_config* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, dwconv_config, XNN_MAX_F32_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.f32(&dwconv_params, output_min, output_max);
  }

  const struct xnn_vmulcaddc_config* vmulcaddc_config = xnn_init_f32_vmulcaddc_config();
  if (vmulcaddc_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_minmax_params vmulcaddc_params;
  if XNN_LIKELY(vmulcaddc_config->init.f32 != NULL) {
    vmulcaddc_config->init.f32(&vmulcaddc_params, output_min, output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*bias_element_size=*/sizeof(float),
    (xnn_pack_vmulcaddc_w_fn) xnn_pack_f32_vmulcaddc_w,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_f32_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_f32_dwconv_ghw_w,
    (xnn_packw_gemm_goi_ukernel_fn) gemm_config->pack_gemm_goi,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_f32_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_f32_conv_goki_w,
    /*packing_params=*/NULL,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/&vmulcaddc_params,
    /*vmulcaddc_params_size=*/sizeof(vmulcaddc_params),
    /*gemm_config=*/gemm_config,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_config=*/vmulcaddc_config,
    /*linear_activation=*/linear_activation,
    /*relu_activation=*/relu_activation,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_f32,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    convolution_op_out);
}

enum xnn_status xnn_create_fused_convolution2d_nhwc_f32(
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
    size_t num_post_operations,
    struct xnn_post_operation* post_operations,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out)
{
  xnn_log_error(
    "failed to create %s operator: JIT operators are deprecated",
    xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
  return xnn_status_invalid_parameter;
}

static inline bool input_size_changed(xnn_operator_t convolution_op)
{
  return convolution_op->input_height != convolution_op->last_input_height ||
         convolution_op->input_width != convolution_op->last_input_width;
}

static enum xnn_status reshape_gemm(
    xnn_operator_t convolution_op,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t extra_weights_elements_size,
    uint32_t log2_output_element_size,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t num_threads)
{
  // Convolution maps directly to GEMM and doesn't use indirection buffer.
  const size_t batch_size = convolution_op->batch_size;

  const size_t output_height = convolution_op->output_height;
  const size_t output_width = convolution_op->output_width;
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = batch_size * output_size;

  const size_t groups = convolution_op->groups;
  const size_t group_input_channels = convolution_op->group_input_channels;
  const size_t w_stride = extra_weights_elements_size +
    (round_up_po2(group_input_channels, convolution_op->ukernel.gemm.kr * convolution_op->ukernel.gemm.sr) << log2_filter_element_size);
  const size_t group_output_channels = convolution_op->group_output_channels;

  uint32_t mr = convolution_op->ukernel.gemm.mr;
  const uint32_t nr = convolution_op->ukernel.gemm.nr;
  struct xnn_hmp_gemm_ukernel *gemm_cases = convolution_op->ukernel.gemm.gemm_cases;

  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
    mr = xnn_get_heuristic_mr_gemm(batch_output_size, mr, nr, gemm_cases);
  #else
    if (batch_output_size == 1 && gemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
      mr = 1;
    }
  #endif

  struct xnn_hmp_gemm_ukernel gemm_ukernel = gemm_cases[mr - 1];

  convolution_op->context.gemm = (struct gemm_context){
      .k_scaled = group_input_channels << log2_input_element_size,
      .a_stride = convolution_op->input_pixel_stride << log2_input_element_size,
      .ga_stride = group_input_channels << log2_input_element_size,
      .packed_w = packed_weights(convolution_op),
      .w_stride = w_stride,
      .gw_stride = w_stride * round_up(group_output_channels, nr),
      .cm_stride = convolution_op->output_pixel_stride
                   << log2_output_element_size,
      .cn_stride = nr << log2_output_element_size,
      .gc_stride = group_output_channels << log2_output_element_size,
      .log2_csize = log2_output_element_size,
      .num_batch_dims = 1,
      .ukernel = gemm_ukernel,
  };
  convolution_op->context.gemm.batch_dims_a[0] = groups;
  convolution_op->context.gemm.batch_dims_b[0] = groups;
  convolution_op->context.gemm.batch_strides_c[0] = 1;
  memcpy(&convolution_op->context.gemm.params, &convolution_op->params, sizeof(convolution_op->context.gemm.params));
  convolution_op->context.gemm.fused_params = &convolution_op->context.gemm.params;

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
        convolution_op->compute[0].type = xnn_parallelization_type_2d_tile_2d_with_uarch;
        convolution_op->compute[0].task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_gemm;
      } else {
        convolution_op->compute[0].type = xnn_parallelization_type_2d_tile_2d;
        convolution_op->compute[0].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
      }
    #else
      convolution_op->compute[0].type = xnn_parallelization_type_2d_tile_2d;
      convolution_op->compute[0].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
    #endif
    convolution_op->compute[0].range[0] = batch_output_size;
    convolution_op->compute[0].range[1] = group_output_channels;
    convolution_op->compute[0].tile[0] = mr;
    convolution_op->compute[0].tile[1] = nc;
  } else {
    #if XNN_MAX_UARCH_TYPES > 1
      if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
        convolution_op->compute[0].type = xnn_parallelization_type_3d_tile_2d_with_uarch;
        convolution_op->compute[0].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_gemm;
      } else {
        convolution_op->compute[0].type = xnn_parallelization_type_3d_tile_2d;
        convolution_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
      }
    #else
      convolution_op->compute[0].type = xnn_parallelization_type_3d_tile_2d;
      convolution_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
    #endif
    convolution_op->compute[0].range[0] = groups;
    convolution_op->compute[0].range[1] = batch_output_size;
    convolution_op->compute[0].range[2] = group_output_channels;
    convolution_op->compute[0].tile[0] = mr;
    convolution_op->compute[0].tile[1] = nc;
  }
  convolution_op->state = xnn_run_state_needs_setup;

  *workspace_size = 0;
  *workspace_alignment = 1;

  return xnn_status_success;
}

static enum xnn_status reshape_igemm(
    xnn_operator_t convolution_op,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t extra_weights_elements_size,
    uint32_t log2_output_element_size,
    bool dynamic_quantization,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t num_threads)
{
  const size_t batch_size = convolution_op->batch_size;
  const size_t input_height = convolution_op->input_height;
  const size_t input_width = convolution_op->input_width;
  const size_t groups = convolution_op->groups;
  const size_t kernel_height = convolution_op->kernel_height;
  const size_t kernel_width = convolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_height = convolution_op->output_height;
  const size_t output_width = convolution_op->output_width;
  const size_t output_size = output_height * output_width;

  uint32_t mr = convolution_op->ukernel.igemm.mr;
  const uint32_t nr = convolution_op->ukernel.igemm.nr;
  struct xnn_hmp_igemm_ukernel* igemm_cases = convolution_op->ukernel.igemm.igemm_cases;

  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
    mr = xnn_get_heuristic_mr_igemm(output_size, mr, nr, igemm_cases);
  #else
    if (output_size == 1 && igemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
      mr = 1;
    }
  #endif

  struct xnn_hmp_igemm_ukernel igemm_ukernel = igemm_cases[mr - 1];

  const size_t tiled_output_size = round_up(output_size, mr);
  const size_t indirection_buffer_size = sizeof(void*) * kernel_size * tiled_output_size;
  size_t igemm_compute_index;
  if (convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    *workspace_size = indirection_buffer_size;
    *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
    igemm_compute_index = 1;

    convolution_op->context.conv2d_igemm_indirection_init = (struct conv2d_igemm_indirection_init_context) {
      .zero_buffer = convolution_op->zero_buffer,
      .input_pixel_stride = convolution_op->input_pixel_stride << log2_input_element_size,
      .input_height = input_height,
      .input_width = input_width,
      .output_height = output_height,
      .output_width = output_width,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .stride_height = convolution_op->stride_height,
      .stride_width = convolution_op->stride_width,
      .dilation_height = convolution_op->dilation_height,
      .dilation_width = convolution_op->dilation_width,
      .input_padding_top = convolution_op->padding_top,
      .input_padding_left = convolution_op->padding_left,
    };

    convolution_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
    convolution_op->compute[0].context_offset = offsetof(struct xnn_operator, context.conv2d_igemm_indirection_init) - offsetof(struct xnn_operator, context);
    convolution_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_conv2d_igemm_indirection;
    convolution_op->compute[0].range[0] = tiled_output_size;
    convolution_op->compute[0].tile[0] = mr;
  } else {
    *workspace_size = 0;
    *workspace_alignment = 1;
    igemm_compute_index = 0;

    if (input_size_changed(convolution_op)) {
      const void** indirection_buffer =
        (const void**) xnn_reallocate_memory((void*) convolution_op->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error(
            "failed to allocate %zu bytes for %s operator indirection buffer",
            indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));
        return xnn_status_out_of_memory;
      }
      convolution_op->indirection_buffer = indirection_buffer;
      xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
                    indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));

      // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
      // This offset must be aligned properly because inputs and input offsets need to be aligned.
      convolution_op->input = (void*) ((uintptr_t) convolution_op->zero_buffer + XNN_ALLOCATION_ALIGNMENT);
      convolution_op->last_input = convolution_op->input;
      convolution_op->last_input_height = convolution_op->input_height;
      convolution_op->last_input_width = convolution_op->input_width;

      xnn_indirection_init_conv2d(
        /*output_tile_size=*/mr,
        /*output_start=*/0,
        /*output_end=*/tiled_output_size,
        convolution_op->indirection_buffer,
        convolution_op->input,
        convolution_op->zero_buffer,
        convolution_op->input_pixel_stride << log2_input_element_size,
        convolution_op->input_height, convolution_op->input_width,
        convolution_op->output_height, convolution_op->output_width,
        convolution_op->kernel_height, convolution_op->kernel_width,
        convolution_op->stride_height, convolution_op->stride_width,
        convolution_op->dilation_height, convolution_op->dilation_width,
        convolution_op->padding_top, convolution_op->padding_left);
    }
  }


  const size_t group_input_channels = convolution_op->group_input_channels;
  const size_t w_stride = extra_weights_elements_size +
    (round_up_po2(group_input_channels, convolution_op->ukernel.igemm.kr * convolution_op->ukernel.igemm.sr) * kernel_size << log2_filter_element_size);
  const size_t group_output_channels = convolution_op->group_output_channels;
  convolution_op->context.igemm = (struct igemm_context) {
      .ks = kernel_size,
      .ks_scaled = kernel_size * mr * sizeof(void*),
      .kc = group_input_channels << log2_input_element_size,
      .w_stride = w_stride,
      .indirect_a = convolution_op->indirection_buffer,
      .zero = convolution_op->zero_buffer,
      .packed_w = packed_weights(convolution_op),
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
  memcpy(&convolution_op->context.igemm.params, &convolution_op->params, sizeof(convolution_op->context.igemm.params));

  size_t nc = group_output_channels;
  if (num_threads > 1) {
    const size_t num_other_tiles = groups * batch_size * divide_round_up(output_size, mr);
    const size_t target_tiles_per_thread = 5;
    const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
    if (max_nc < nc) {
      nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
    }
  }

  if (dynamic_quantization && convolution_op->zero_size > 0) {
    convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_1d;
    convolution_op->compute[igemm_compute_index].task_1d = (pthreadpool_task_1d_t) xnn_compute_dq_zero_buffer_igemm;
    convolution_op->compute[igemm_compute_index].range[0] = batch_size;
    ++igemm_compute_index;
  }
  if (groups == 1) {
    #if XNN_MAX_UARCH_TYPES > 1
      if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
        if (batch_size > 1) {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d_with_uarch;
            if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_batch_hmp_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_batch_hmp_igemm;
          }
        } else {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_2d_tile_2d_with_uarch;
          if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_igemm;
          }
        }
      } else {
        if (batch_size > 1) {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
          if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_igemm;
          }
        } else {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_2d_tile_2d;
          if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_igemm;
          }
        }
      }
    #else
      if (batch_size > 1) {
        convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
        if (dynamic_quantization) {
          convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_dqigemm;
        } else {
          convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_igemm;
        }
      } else {
        convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_2d_tile_2d;
        if (dynamic_quantization) {
          convolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_dqigemm;
        } else {
          convolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_igemm;
        }
      }
    #endif
    if (batch_size > 1) {
      convolution_op->compute[igemm_compute_index].range[0] = batch_size;
      convolution_op->compute[igemm_compute_index].range[1] = output_size;
      convolution_op->compute[igemm_compute_index].range[2] = group_output_channels;
    } else {
      convolution_op->compute[igemm_compute_index].range[0] = output_size;
      convolution_op->compute[igemm_compute_index].range[1] = group_output_channels;
    }
    convolution_op->compute[igemm_compute_index].tile[0] = mr;
    convolution_op->compute[igemm_compute_index].tile[1] = nc;
  } else {
    #if XNN_MAX_UARCH_TYPES > 1
      if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
        if (batch_size > 1) {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_4d_tile_2d_with_uarch;
           if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_4d_tile_2d_with_id = (pthreadpool_task_4d_tile_2d_with_id_t) xnn_compute_hmp_grouped_batch_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_4d_tile_2d_with_id = (pthreadpool_task_4d_tile_2d_with_id_t) xnn_compute_hmp_grouped_batch_igemm;
          }
        } else {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d_with_uarch;
          if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_igemm;
          }
        }
      } else {
        if (batch_size > 1) {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_4d_tile_2d;
          if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_igemm;
          }
        } else {
          convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
          if (dynamic_quantization) {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_dqigemm;
          } else {
            convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_igemm;
          }
        }
      }
    #else
      if (batch_size > 1) {
        convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_4d_tile_2d;
        if (dynamic_quantization) {
          convolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_dqigemm;
        } else {
          convolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_igemm;
        }
      } else {
        convolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
        if (dynamic_quantization) {
          convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_dqigemm;
        } else {
          convolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_igemm;
        }
      }
    #endif
    if (batch_size > 1) {
      convolution_op->compute[igemm_compute_index].range[0] = batch_size;
      convolution_op->compute[igemm_compute_index].range[1] = groups;
      convolution_op->compute[igemm_compute_index].range[2] = output_size;
      convolution_op->compute[igemm_compute_index].range[3] = group_output_channels;
    } else {
      convolution_op->compute[igemm_compute_index].range[0] = groups;
      convolution_op->compute[igemm_compute_index].range[1] = output_size;
      convolution_op->compute[igemm_compute_index].range[2] = group_output_channels;
    }
    convolution_op->compute[igemm_compute_index].tile[0] = mr;
    convolution_op->compute[igemm_compute_index].tile[1] = nc;
  }
  convolution_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static enum xnn_status reshape_dwconv(
    xnn_operator_t convolution_op,
    uint32_t log2_input_element_size,
    uint32_t log2_accumulator_element_size,
    uint32_t log2_output_element_size,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t num_threads)
{
  const size_t input_height = convolution_op->input_height;
  const size_t input_width = convolution_op->input_width;
  const size_t kernel_height = convolution_op->kernel_height;
  const size_t kernel_width = convolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_height = convolution_op->output_height;
  const size_t output_width = convolution_op->output_width;
  const size_t step_width = convolution_op->dilation_width == 1 ?
      min(convolution_op->stride_width, kernel_width) : kernel_width;
  const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;
  const struct xnn_ukernel_dwconv dwconv_ukernel = convolution_op->ukernel.dwconv;
  const bool is_unipass = dwconv_ukernel.last_tile == 0;
  const size_t tile_size = dwconv_ukernel.tile_size;
  size_t total_workspace_size = 0;

  // Micro-kernel will read (tile_size - kernel_size) elements after the end of indirection buffer.
  const size_t indirection_buffer_size =
    round_up_po2(sizeof(void*) * (tile_size - kernel_size + output_height * step_height), XNN_ALLOCATION_ALIGNMENT);

  size_t dwconv_compute_index;
  const bool is_transient_indirection_buffer = convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
  if (is_transient_indirection_buffer) {
    total_workspace_size += indirection_buffer_size;
    dwconv_compute_index = 1;

    convolution_op->context.dwconv_indirection_init = (struct dwconv_indirection_init_context) {
      .zero_buffer = convolution_op->zero_buffer,
      .input_pixel_stride = convolution_op->input_pixel_stride << log2_input_element_size,
      .input_height = input_height,
      .input_width = input_width,
      .output_height = output_height,
      .output_width = output_width,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .stride_height = convolution_op->stride_height,
      .stride_width = convolution_op->stride_width,
      .dilation_height = convolution_op->dilation_height,
      .dilation_width = convolution_op->dilation_width,
      .input_padding_top = convolution_op->padding_top,
      .input_padding_left = convolution_op->padding_left,
      .step_height = step_height,
      .step_width = step_width,
      .tile_size = tile_size,
    };

    convolution_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
    convolution_op->compute[0].context_offset = offsetof(struct xnn_operator, context.dwconv_indirection_init) - offsetof(struct xnn_operator, context);
    convolution_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_dwconv_indirection;
    convolution_op->compute[0].range[0] = output_height;

    if (num_threads > 1) {
      const size_t target_tiles_per_thread = 5;
      convolution_op->compute[0].tile[0] = divide_round_up(output_height, num_threads * target_tiles_per_thread);
    } else {
      convolution_op->compute[0].tile[0] = output_height;
    }
  } else {
    dwconv_compute_index = 0;

    if (input_size_changed(convolution_op)) {
      const void** indirection_buffer =
        (const void**) xnn_reallocate_memory(convolution_op->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error("failed to allocate %zu bytes for %s operator indirection buffer",
          indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));
        return xnn_status_out_of_memory;
      }
      convolution_op->indirection_buffer = indirection_buffer;
      xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
        indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));

      // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
      // This offset must be aligned properly because inputs and input offsets need to be aligned.
      convolution_op->input = (void* ) ((uintptr_t) convolution_op->zero_buffer + XNN_ALLOCATION_ALIGNMENT);
      convolution_op->last_input = convolution_op->input;
      convolution_op->last_input_height = convolution_op->input_height;
      convolution_op->last_input_width = convolution_op->input_width;

      xnn_indirection_init_dwconv2d(
        /*output_y_start=*/0, /*output_y_end=*/convolution_op->output_height,
        convolution_op->indirection_buffer,
        convolution_op->input,
        convolution_op->input_pixel_stride << log2_input_element_size,
        convolution_op->zero_buffer,
        convolution_op->input_height, convolution_op->input_width,
        convolution_op->output_height, convolution_op->output_width,
        convolution_op->kernel_height, convolution_op->kernel_width,
        convolution_op->stride_height, convolution_op->stride_width,
        convolution_op->dilation_height, convolution_op->dilation_width,
        convolution_op->padding_top, convolution_op->padding_left,
        step_height, step_width, tile_size);
    }
  }

  const size_t groups = convolution_op->groups;
  int32_t extra_input_advanced = is_unipass ? 0 : tile_size - convolution_op->ukernel.dwconv.last_tile;
  convolution_op->context.dwconv = (struct dwconv_context) {
      .kernel_size = kernel_size,
      .indirect_input = convolution_op->indirection_buffer,
      .indirect_input_width_stride = (kernel_height * step_width - extra_input_advanced) * sizeof(void*),
      .indirect_input_height_stride = step_height * sizeof(void*),
      .input_batch_stride = (input_height * input_width * convolution_op->input_pixel_stride) << log2_input_element_size,
      .packed_weights = packed_weights(convolution_op),
      .output_batch_stride = (output_height * output_width * convolution_op->output_pixel_stride) << log2_output_element_size,
      .output_height_stride = (output_width * convolution_op->output_pixel_stride) << log2_output_element_size,
      .output_height = output_height,
      .output_width = output_width,
      .groups = groups,
      .zero = convolution_op->zero_buffer,
      .output_increment = (convolution_op->output_pixel_stride - groups) << log2_output_element_size,
  };
  memcpy(&convolution_op->context.dwconv.params, &convolution_op->params, sizeof(convolution_op->context.dwconv.params));

  const size_t batch_size = convolution_op->batch_size;
  convolution_op->compute[dwconv_compute_index].range[0] = batch_size;
  convolution_op->compute[dwconv_compute_index].range[1] = output_height;
  convolution_op->state = xnn_run_state_needs_setup;

  if (is_unipass) {
    convolution_op->compute[dwconv_compute_index].type = xnn_parallelization_type_2d;
    convolution_op->compute[dwconv_compute_index].task_2d = (pthreadpool_task_2d_t) xnn_compute_dwconv_unipass;
    convolution_op->context.dwconv.unipass_ukernel = convolution_op->ukernel.dwconv.unipass_fn;
  } else {
    const size_t buffer_size =
      round_up_po2(
        (groups + (XNN_MULTIPASS_EXTRA_BYTES >> log2_input_element_size)) << log2_accumulator_element_size,
        XNN_ALLOCATION_ALIGNMENT);
    convolution_op->context.dwconv.buffer_size = buffer_size;
    if (is_transient_indirection_buffer) {
      convolution_op->context.dwconv.multipass_buffer_offset = indirection_buffer_size;
    }
    const bool use_threads_workspace_size = num_threads < batch_size * output_height;
    if (use_threads_workspace_size) {
      convolution_op->compute[dwconv_compute_index].type = xnn_parallelization_type_2d_with_thread;
      convolution_op->compute[dwconv_compute_index].task_2d_with_thread =
        (pthreadpool_task_2d_with_thread_t) xnn_compute_dwconv_multipass_with_thread;
      total_workspace_size += num_threads * buffer_size;
    } else {
      convolution_op->compute[dwconv_compute_index].type = xnn_parallelization_type_2d;
      convolution_op->compute[dwconv_compute_index].task_2d =
        (pthreadpool_task_2d_t) xnn_compute_dwconv_multipass;
      total_workspace_size += batch_size * output_height * buffer_size;
    }

    convolution_op->context.dwconv.multipass_ukernel = convolution_op->ukernel.dwconv.multipass_fn;
  }

  *workspace_size = total_workspace_size;
  *workspace_alignment = total_workspace_size == 0 ? 1 : XNN_ALLOCATION_ALIGNMENT;

  return xnn_status_success;
}

static enum xnn_status reshape_vmulcaddc(
  xnn_operator_t convolution_op,
  uint32_t log2_input_element_size,
  uint32_t log2_output_element_size,
  size_t* workspace_size,
  size_t* workspace_alignment,
  size_t num_threads)
{
  const size_t batch_output_size = convolution_op->batch_size * convolution_op->output_height * convolution_op->output_width;

  convolution_op->context.vmulcaddc = (struct vmulcaddc_context) {
    .n = convolution_op->groups << log2_input_element_size,
    .x_stride = convolution_op->input_pixel_stride << log2_input_element_size,
    .w = packed_weights(convolution_op),
    .y_stride = convolution_op->output_pixel_stride << log2_output_element_size,
    .ukernel = convolution_op->ukernel.vmulcaddc.function,
  };
  memcpy(&convolution_op->context.vmulcaddc.params, &convolution_op->params,
         sizeof(convolution_op->context.vmulcaddc.params));

  size_t mc = batch_output_size;
  if (num_threads > 1) {
    const size_t target_tiles_per_thread = 5;
    const size_t max_mc = divide_round_up(batch_output_size, num_threads * target_tiles_per_thread);
    if (max_mc < mc) {
      const uint32_t mr = convolution_op->ukernel.vmulcaddc.mr;
      mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
    }
  }

  convolution_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
  convolution_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_vmulcaddc;
  convolution_op->compute[0].range[0] = batch_output_size;
  convolution_op->compute[0].tile[0] = mc;
  convolution_op->state = xnn_run_state_needs_setup;

  *workspace_size = 0;
  *workspace_alignment = 1;

  return xnn_status_success;
}

static enum xnn_status reshape_convolution2d_nhwc(
  xnn_operator_t convolution_op,
  enum xnn_operator_type expected_operator_type,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t log2_accumulator_element_size,
  uint32_t extra_weights_elements_size,
  uint32_t log2_output_element_size,
  bool dynamic_quantization,
  size_t* workspace_size,
  size_t* workspace_alignment,
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
    convolution_op->output_height = xnn_compute_convolution_output_dimension(
        convolution_op->padding_top + input_height + convolution_op->padding_bottom,
        convolution_op->kernel_height,
        convolution_op->dilation_height,
        convolution_op->stride_height);
    convolution_op->output_width = xnn_compute_convolution_output_dimension(
        convolution_op->padding_left + input_width + convolution_op->padding_right,
        convolution_op->kernel_width,
        convolution_op->dilation_width,
        convolution_op->stride_width);
  }

  if (output_height_out != NULL) {
    *output_height_out = convolution_op->output_height;
  }
  if (output_width_out != NULL) {
    *output_width_out = convolution_op->output_width;
  }

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  switch (convolution_op->ukernel.type) {
    case xnn_microkernel_type_gemm:
      return reshape_gemm(
          convolution_op,
          log2_input_element_size, log2_filter_element_size, extra_weights_elements_size, log2_output_element_size,
          workspace_size, workspace_alignment, num_threads);
    case xnn_microkernel_type_igemm:
      return reshape_igemm(
          convolution_op,
          log2_input_element_size, log2_filter_element_size, extra_weights_elements_size, log2_output_element_size, dynamic_quantization,
          workspace_size, workspace_alignment, num_threads);
    case xnn_microkernel_type_dwconv:
      return reshape_dwconv(
          convolution_op,
          log2_input_element_size, log2_accumulator_element_size, log2_output_element_size,
          workspace_size, workspace_alignment, num_threads);
    case xnn_microkernel_type_vmulcaddc:
      return reshape_vmulcaddc(
          convolution_op,
          log2_input_element_size, log2_output_element_size,
          workspace_size, workspace_alignment, num_threads);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  convolution_op->last_input_height = convolution_op->input_height;
  convolution_op->last_input_width = convolution_op->input_width;
  convolution_op->input_height = input_height;
  convolution_op->input_width = input_width;
  if (input_size_changed(convolution_op)) {
    if (convolution_op->zero_buffers) {
      for (size_t i = 1; i < batch_size; ++i) {
        xnn_release_simd_memory(convolution_op->zero_buffers[i]);
      }
    }
    convolution_op->zero_buffers = xnn_reallocate_memory(convolution_op->zero_buffers, batch_size * sizeof(void*));
    convolution_op->zero_buffers[0] = convolution_op->zero_buffer;
    for (size_t i = 1; i < batch_size; ++i) {
      convolution_op->zero_buffers[i] = xnn_allocate_simd_memory(convolution_op->zero_size);
    }
  }
  return reshape_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qd8_f16_qc8w,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float) * 2,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*dynamic_quantization=*/true,
    workspace_size, workspace_alignment,
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  convolution_op->last_input_height = convolution_op->input_height;
  convolution_op->last_input_width = convolution_op->input_width;
  convolution_op->input_height = input_height;
  convolution_op->input_width = input_width;
  if (input_size_changed(convolution_op)) {
    if (convolution_op->zero_buffers) {
      for (size_t i = 1; i < batch_size; ++i) {
        xnn_release_simd_memory(convolution_op->zero_buffers[i]);
      }
    }
    convolution_op->zero_buffers = xnn_reallocate_memory(convolution_op->zero_buffers, batch_size * sizeof(void*));
    convolution_op->zero_buffers[0] = convolution_op->zero_buffer;
    for (size_t i = 1; i < batch_size; ++i) {
      convolution_op->zero_buffers[i] = xnn_allocate_simd_memory(convolution_op->zero_size);
    }
  }
  return reshape_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qd8_f32_qc8w,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float) * 2,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*dynamic_quantization=*/true,
    workspace_size, workspace_alignment,
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qu8(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  return reshape_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qu8,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
    /*extra_weights_elements_size=*/sizeof(int32_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*dynamic_quantization=*/false,
    workspace_size, workspace_alignment,
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qs8(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  return reshape_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qs8,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
    /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*dynamic_quantization=*/false,
    workspace_size, workspace_alignment,
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qs8_qc8w(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  return reshape_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qc8,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
    /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*dynamic_quantization=*/false,
    workspace_size, workspace_alignment,
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_f16(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  return reshape_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_f16,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*extra_weights_elements_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*dynamic_quantization=*/false,
    workspace_size, workspace_alignment,
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_f32(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  return reshape_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_f32,
    batch_size, input_height, input_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*extra_weights_elements_size=*/sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*dynamic_quantization=*/false,
    workspace_size, workspace_alignment,
    output_height_out, output_width_out,
    threadpool);
}

static enum xnn_status setup_gemm(xnn_operator_t convolution_op)
{
  convolution_op->context.gemm.a = convolution_op->input;
  convolution_op->context.gemm.c = convolution_op->output;
  convolution_op->context.gemm.quantization_params = convolution_op->quantization_params;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_igemm(
    xnn_operator_t convolution_op,
    void* workspace,
    uint32_t log2_input_element_size)
{
  if (convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    convolution_op->context.igemm.a_offset = (size_t) 0;
    convolution_op->context.igemm.indirect_a = (const void**) workspace;
    convolution_op->context.conv2d_igemm_indirection_init.indirection_buffer = (const void**) workspace;
    convolution_op->context.conv2d_igemm_indirection_init.input = convolution_op->input;
  } else {
    convolution_op->context.igemm.a_offset = (size_t) ((uintptr_t) convolution_op->input - (uintptr_t) convolution_op->last_input);
  }
  convolution_op->context.igemm.zero_size = convolution_op->zero_size;
  convolution_op->context.igemm.zero_buffers = convolution_op->zero_buffers;
  convolution_op->context.igemm.c = convolution_op->output;
  convolution_op->context.igemm.quantization_params = convolution_op->quantization_params;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_dwconv(
    xnn_operator_t convolution_op,
    void* workspace,
    uint32_t log2_input_element_size)
{
  if (convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    convolution_op->context.dwconv.input_offset = (size_t) 0;
    convolution_op->context.dwconv.indirect_input = (const void**) workspace;
    convolution_op->context.dwconv_indirection_init.input = convolution_op->input;
    convolution_op->context.dwconv_indirection_init.indirection_buffer = (const void**) workspace;
  } else {
    convolution_op->context.dwconv.input_offset = (size_t) ((uintptr_t) convolution_op->input - (uintptr_t) convolution_op->last_input);
  }

  if (convolution_op->context.dwconv.buffer_size) {
    assert(workspace != NULL);
    convolution_op->context.dwconv.multipass_buffer =
      (void*) ((uintptr_t) workspace + convolution_op->context.dwconv.multipass_buffer_offset);
  }

  convolution_op->context.dwconv.output = convolution_op->output;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_vmulcaddc(xnn_operator_t convolution_op)
{
  convolution_op->context.vmulcaddc.x = convolution_op->input;
  convolution_op->context.vmulcaddc.y = convolution_op->output;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_convolution2d_nhwc(
  xnn_operator_t convolution_op,
  enum xnn_operator_type expected_operator_type,
  void* workspace,
  const void* input,
  void* output,
  const void* quantization_params,
  uint32_t log2_input_element_size)
{
  if (convolution_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
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

  if (convolution_op->weights_cache != NULL && !xnn_weights_cache_is_finalized(convolution_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_state;
  }

  convolution_op->input = input;
  convolution_op->output = output;
  convolution_op->quantization_params = quantization_params;

  switch (convolution_op->ukernel.type) {
    case xnn_microkernel_type_gemm:
      return setup_gemm(convolution_op);
    case xnn_microkernel_type_igemm:
      return setup_igemm(convolution_op, workspace, log2_input_element_size);
    case xnn_microkernel_type_dwconv:
      return setup_dwconv(convolution_op, workspace, log2_input_element_size);
    case xnn_microkernel_type_vmulcaddc:
      return setup_vmulcaddc(convolution_op);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
    xnn_operator_t convolution_op,
    void* workspace,
    const int8_t* input,
    void* output,
    const struct xnn_dynamic_quantization_params* quantization_params)
{
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qd8_f16_qc8w,
    workspace, input, output, quantization_params,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op,
    void* workspace,
    const int8_t* input,
    float* output,
    const struct xnn_dynamic_quantization_params* quantization_params)
{
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qd8_f32_qc8w,
    workspace, input, output, quantization_params,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qu8(
    xnn_operator_t convolution_op,
    void* workspace,
    const uint8_t* input,
    uint8_t* output)
{
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qu8,
    workspace, input, output, /*quantization_params=*/NULL,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qs8(
    xnn_operator_t convolution_op,
    void* workspace,
    const int8_t* input,
    int8_t* output)
{
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qs8,
    workspace, input, output, /*quantization_params=*/NULL,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qs8_qc8w(
    xnn_operator_t convolution_op,
    void* workspace,
    const int8_t* input,
    int8_t* output)
{
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qc8,
    workspace, input, output, /*quantization_params=*/NULL,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_f16(
    xnn_operator_t convolution_op,
    void* workspace,
    const void* input,
    void* output)
{
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_f16,
    workspace, input, output, /*quantization_params=*/NULL,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF);
}

enum xnn_status xnn_setup_convolution2d_nhwc_f32(
    xnn_operator_t convolution_op,
    void* workspace,
    const float* input,
    float* output)
{
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_f32,
    workspace, input, output, /*quantization_params=*/NULL,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT);
}
