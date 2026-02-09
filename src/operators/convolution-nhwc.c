// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/operators/fingerprint_cache.h"
#include "src/operators/fingerprint_id.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microkernel-type.h"
#include "src/xnnpack/microkernel-utils.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

// Helper state structure that holds:
// - the parameters that are passed to the top-level create function;
// - an internal state that may be used by the case specific initialization
//   helpers;
// - the final parameters that will be passed to the underlying create function.
struct convolution2d_nhwc_context {
  // Parameters to the top-level create function.
  uint32_t input_padding_top;
  uint32_t input_padding_right;
  uint32_t input_padding_bottom;
  uint32_t input_padding_left;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t subsampling_height;
  uint32_t subsampling_width;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t groups;
  size_t group_input_channels;
  size_t group_output_channels;
  size_t input_channel_stride;
  size_t output_channel_stride;
  int8_t input_zero_point;
  float input_scale;
  int8_t kernel_zero_point;
  const void* kernel_scale;
  float kernel_scale_value;
  const void* kernel;
  const void* bias;
  int8_t output_zero_point;
  float output_scale;
  float output_min;
  float output_max;
  uint32_t flags;
  xnn_weights_cache_t weights_cache;
  const struct xnn_gemm_config* gemm_config;
  enum xnn_operator_type operator_type;

  const void* kernel_for_cache_key;
  const void* bias_for_cache_key;

  // State
  union {
    struct xnn_qs8_packing_params qs8;
    struct xnn_qu8_packing_params qu8;
  } packing_params;
  union {
    struct xnn_f16_minmax_params f16;
    struct xnn_f32_minmax_params f32;
    union xnn_qu8_conv_minmax_params qu8;
    union xnn_qs8_qc8w_conv_minmax_params qs8_qc8w;
  } gemm_params;
  union {
    struct xnn_f16_minmax_params f16;
    struct xnn_f32_minmax_params f32;
    union xnn_qu8_conv_minmax_params qu8;
    union xnn_qs8_qc8w_conv_minmax_params qs8_qc8w;
  } dwconv_params;
  union {
    struct xnn_f16_minmax_params f16;
    struct xnn_f32_minmax_params f32;
  } vmuladdc_params;
  xnn_float16 fp16_output_min;
  xnn_float16 fp16_output_max;
  void* requantization_scale;
  float requantization_scale_value;
  enum xnn_microkernel_type microkernel_type;

  // `create_helper` implementation parameters.
  xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w;
  xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w;
  xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w;
  xnn_pack_conv_kgo_w_fn pack_conv_kgo_w;
  xnn_pack_conv_goki_w_fn pack_conv_goki_w;
  void* packing_params_ptr;
  size_t gemm_params_size;
  void* gemm_params_ptr;
  size_t dwconv_params_size;
  void* dwconv_params_ptr;
  const float* scale_params;
  const float* kernel_scale_params;
  size_t input_padding_bytes;
  size_t vmulcaddc_params_size;
  void* vmulcaddc_params_ptr;
  const struct xnn_dwconv_config* dwconv_ukernel;
  const struct xnn_vmulcaddc_config* vmulcaddc_config;
  bool linear_activation;
  bool relu_activation;
  enum xnn_fingerprint_id fingerprint_id;
};

static inline const void* get_kernel_for_cache_key(
    const struct convolution2d_nhwc_context* const context) {
  return context->kernel_for_cache_key ? context->kernel_for_cache_key
                                       : context->kernel;
}

static inline const void* get_bias_for_cache_key(
    const struct convolution2d_nhwc_context* const context) {
  return context->bias_for_cache_key ? context->bias_for_cache_key
                                     : context->bias;
}

static inline size_t compute_output_dimension_with_tf_same_padding(
    size_t input_dimension, size_t subsampling_dimension) {
  return divide_round_up(input_dimension, subsampling_dimension);
}

static inline const struct xnn_dwconv_config* find_dwconv_ukernel(
    size_t kernel_size, const struct xnn_dwconv_config* ukernel,
    size_t num_ukernels) {
  const struct xnn_dwconv_config* best_ukernel = NULL;
  while (num_ukernels-- != 0) {
    // Find the smallest unipass primary_tile that is at least as big as
    // kernel_size.
    if (ukernel->primary_tile >= kernel_size) {
      if (best_ukernel == NULL ||
          ukernel->primary_tile < best_ukernel->primary_tile) {
        best_ukernel = ukernel;
      }
    }
    ukernel++;
  }
  if (best_ukernel == NULL) {
    xnn_log_debug("no dwconv ukernel found");
  }
  return best_ukernel;
}

static enum xnn_status create_vmulcaddc_path(
    uint32_t groups, const void* kernel, const void* bias,
    uint32_t log2_filter_element_size, uint32_t bias_element_size,
    xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w, const void* packing_params,
    const void* vmulcaddc_params, size_t vmulcaddc_params_size,
    const struct xnn_vmulcaddc_config* vmulcaddc_config,
    enum xnn_fingerprint_id fingerprint_id,
    enum xnn_operator_type operator_type, xnn_operator_t convolution_op,
    const struct convolution2d_nhwc_context* context) {
  assert(vmulcaddc_config != NULL);
  assert(vmulcaddc_params != NULL);

  enum xnn_status status = xnn_status_out_of_memory;

  const size_t c_stride = round_up_po2(groups, vmulcaddc_config->channel_tile);
  const size_t packed_weights_size =
      ((UINT32_C(1) << log2_filter_element_size) + bias_element_size) *
      c_stride;
  size_t aligned_total_weights_size =
      round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size);
  if (weights_ptr == NULL) {
    xnn_log_error(
        "failed to reserve or allocated %zu bytes for %s operator vmulcaddc "
        "packed weights",
        aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size,
                xnn_operator_type_to_string(operator_type));

  pack_vmulcaddc_w(groups, vmulcaddc_config->channel_tile, kernel, bias,
                   weights_ptr, packing_params);

  const struct xnn_weights_cache_look_up_key cache_key = {
    .seed = groups ^ vmulcaddc_config->channel_tile,
    .kernel = get_kernel_for_cache_key(context),
    .bias = get_bias_for_cache_key(context),
    .fingerprint_id = fingerprint_id,
  };
  if (use_weights_cache(convolution_op)) {
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr,
        aligned_total_weights_size);
  }

  memcpy(&convolution_op->params, vmulcaddc_params, vmulcaddc_params_size);

  convolution_op->ukernel.vmulcaddc = (struct xnn_ukernel_vmulcaddc){
      .function = vmulcaddc_config->ukernel,
      .mr = vmulcaddc_config->row_tile,
      .channel_tile = vmulcaddc_config->channel_tile,
  };
  return xnn_status_success;

error:
  return status;
}

static enum xnn_status create_dwconv_path(
    uint32_t kernel_height, uint32_t kernel_width, uint32_t groups,
    const void* kernel, const void* bias, uint32_t flags,
    uint32_t log2_input_element_size, uint32_t log2_filter_element_size,
    uint32_t bias_element_size, xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w,
    xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w, const void* packing_params,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params, const void* dwconv_params,
    size_t dwconv_params_size, const struct xnn_dwconv_config* dwconv_ukernel,
    bool linear_activation, enum xnn_fingerprint_id fingerprint_id,
    enum xnn_operator_type operator_type, size_t* zero_size,
    xnn_operator_t convolution_op,
    const struct convolution2d_nhwc_context* context) {
  assert(dwconv_ukernel != NULL);
  enum xnn_status status = xnn_status_out_of_memory;
  const uint8_t primary_tile = dwconv_ukernel->primary_tile;
  assert(primary_tile >= kernel_height * kernel_width);
  xnn_log_debug("using dwconv unipass of primary_tile %u", primary_tile);

  const size_t c_stride = round_up_po2(groups, dwconv_ukernel->channel_tile);
  size_t packed_weights_size = 0;
  packed_weights_size = ((primary_tile << log2_filter_element_size) +
                         bias_element_size + extra_weights_bytes) *
                        c_stride;

  size_t aligned_total_weights_size =
      round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size);
  if (weights_ptr == NULL) {
    xnn_log_error(
        "failed to reserve or allocated %zu bytes for %s operator dwconv "
        "packed weights",
        aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size,
                xnn_operator_type_to_string(operator_type));
  if (extra_weights_bytes > 0) {
    // TODO(b/402602597): We shouldn't need this initialization.
    memset(weights_ptr, 0, aligned_total_weights_size);
  }
  memcpy(&convolution_op->params, dwconv_params, dwconv_params_size);

  if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
    pack_dwconv_hwg_w(
        primary_tile, kernel_height, kernel_width, groups,
        dwconv_ukernel->channel_tile, kernel, bias, /*scale=*/NULL, weights_ptr,
        dwconv_ukernel->channel_tile * extra_weights_bytes, packing_params);
  } else {
    pack_dwconv_ghw_w(
        primary_tile, kernel_height, kernel_width, groups,
        dwconv_ukernel->channel_tile, kernel, bias, /*scale=*/NULL, weights_ptr,
        dwconv_ukernel->channel_tile * extra_weights_bytes, packing_params);
  }

  if (scale_params != NULL) {
    assert(init_scale_params != NULL);
    size_t stride = dwconv_ukernel->channel_tile *
                    ((primary_tile << log2_filter_element_size) +
                     bias_element_size + extra_weights_bytes);

    init_scale_params(
        /*channels=*/groups,
        /*channels_tile=*/dwconv_ukernel->channel_tile,
        /*stride=*/stride,
        /*scale=*/scale_params,
        /*packed_w=*/
        (void*)((uintptr_t)weights_ptr +
                dwconv_ukernel->channel_tile *
                    ((primary_tile << log2_filter_element_size) +
                     bias_element_size)));
  }

  uint32_t cache_seed = primary_tile ^ kernel_height ^ kernel_width ^ groups ^
                        dwconv_ukernel->channel_tile ^ extra_weights_bytes;
  if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
    cache_seed = ~cache_seed;
  }
  if (use_weights_cache(convolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key = {
        .seed = cache_seed,
        .kernel = get_kernel_for_cache_key(context),
        .bias = get_bias_for_cache_key(context),
        .fingerprint_id = fingerprint_id,
    };
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr,
        aligned_total_weights_size);
  }

  xnn_dwconv_ukernel_fn ukernel = dwconv_ukernel->minmax;
  if (linear_activation && dwconv_ukernel->linear != NULL) {
    ukernel = dwconv_ukernel->linear;
  }
  convolution_op->ukernel.dwconv = (struct xnn_ukernel_dwconv){
      .channel_tile = dwconv_ukernel->channel_tile,
      .primary_tile = primary_tile,
  };

  convolution_op->ukernel.dwconv.ukernel = ukernel;

  *zero_size = XNN_EXTRA_BYTES + (c_stride << log2_input_element_size);
  return xnn_status_success;
error:
  return status;
}

static enum xnn_status create_igemm(
    enum xnn_microkernel_type ukernel_type, uint32_t kernel_size,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    const void* kernel, const void* bias, uint32_t flags,
    uint32_t log2_input_element_size, uint32_t log2_filter_element_size,
    uint32_t bias_element_size, xnn_pack_conv_kgo_w_fn pack_conv_kgo_w,
    xnn_pack_conv_goki_w_fn pack_conv_goki_w, const void* packing_params,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params,
    xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params,
    const float* kernel_scale_params, const void* gemm_params,
    size_t gemm_params_size, const struct xnn_gemm_config* gemm_config,
    bool linear_activation, bool relu_activation,
    enum xnn_fingerprint_id fingerprint_id,
    enum xnn_operator_type operator_type, xnn_operator_t convolution_op,
    size_t* zero_size, const struct convolution2d_nhwc_context* context) {
  enum xnn_status status = xnn_status_out_of_memory;
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t n_stride = round_up(group_output_channels, nr);
  const size_t k_stride = round_up_po2(group_input_channels, kr * sr);

  const uint32_t cache_seed = groups ^ group_input_channels ^
                              group_output_channels ^ nr ^ kr ^ sr ^
                              ukernel_type ^ flags;

  const struct xnn_weights_cache_look_up_key cache_key = {
    .seed = cache_seed,
    .kernel = get_kernel_for_cache_key(context),
    .bias = get_bias_for_cache_key(context),
    .fingerprint_id = fingerprint_id,
  };
  if (use_weights_cache(convolution_op)) {
    convolution_op->packed_weights.offset =
        xnn_weights_cache_look_up(convolution_op->weights_cache, &cache_key);
  }

  bool weights_already_cached =
      use_weights_cache(convolution_op) &&
      convolution_op->packed_weights.offset != XNN_CACHE_NOT_FOUND;

  const size_t packed_group_weights_size =
      ((kernel_size * k_stride << log2_filter_element_size) +
       bias_element_size + extra_weights_bytes) *
      n_stride;
  const size_t aligned_total_weights_size = round_up_po2(
      packed_group_weights_size * groups, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = NULL;

  if (!weights_already_cached) {
    weights_ptr = xnn_get_pointer_to_write_weights(convolution_op,
                                                   aligned_total_weights_size);
    if (!weights_already_cached && weights_ptr == NULL) {
      xnn_log_error(
          "failed to reserve or allocated %zu bytes for %s operator gemm "
          "packed weights",
          aligned_total_weights_size,
          xnn_operator_type_to_string(operator_type));
      goto error;
    }
    xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                  aligned_total_weights_size,
                  xnn_operator_type_to_string(operator_type));
    if (extra_weights_bytes > 0) {
      // TODO(b/402602597): We shouldn't need this initialization.
      memset(weights_ptr, 0, aligned_total_weights_size);
    }
  }

  memcpy(&convolution_op->params, gemm_params, gemm_params_size);

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const uint32_t mr = gemm_config->mr;
  if (linear_activation &&
      gemm_config->linear.gemm[mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  } else if (relu_activation &&
             gemm_config->relu.gemm[mr - 1].function[XNN_UARCH_DEFAULT] !=
                 NULL) {
    gemm_ukernels = &gemm_config->relu;
  }
  switch (ukernel_type) {
    case xnn_microkernel_type_igemm:
      if (!weights_already_cached) {
        if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
          pack_conv_kgo_w(groups, group_output_channels, kernel_size, nr, kr,
                          sr, kernel, bias, /*scale=*/scale_params, weights_ptr,
                          gemm_config->nr * extra_weights_bytes,
                          packing_params);
        } else {
          pack_conv_goki_w(
              groups, group_output_channels, kernel_size, group_input_channels,
              nr, kr, sr, kernel, bias, /*scale=*/scale_params, weights_ptr,
              gemm_config->nr * extra_weights_bytes, packing_params);
        }
      }
      convolution_op->ukernel.igemm->mr = mr;
      convolution_op->ukernel.igemm->nr = nr;
      convolution_op->ukernel.igemm->kr = kr;
      convolution_op->ukernel.igemm->sr = sr;
      convolution_op->ukernel.igemm->mr_packed =
          gemm_config->mr_packed ? gemm_config->mr_packed : mr;

      assert(XNN_MAX_MR >= mr);
      for (size_t i = 0; i < mr; i++) {
        convolution_op->ukernel.igemm->igemm_cases[i] = gemm_ukernels->igemm[i];
      }

      break;
    default:
      XNN_UNREACHABLE;
  }

  if (kernel_scale_params != NULL) {
    assert(init_kernel_scale_params != NULL);

    if (!weights_already_cached) {
      void* group_weights =
          (void*)((uintptr_t)weights_ptr +
                  gemm_config->nr *
                      ((kernel_size * k_stride << log2_filter_element_size) +
                       bias_element_size));
      const size_t weights_stride =
          (kernel_size * k_stride << log2_filter_element_size) +
          bias_element_size + extra_weights_bytes;
      for (uint32_t group = 0; group < groups; group++) {
        init_kernel_scale_params(group_output_channels, gemm_config->nr,
                                 gemm_config->nr * weights_stride,
                                 kernel_scale_params, group_weights);
        kernel_scale_params += group_output_channels;
        group_weights =
            (void*)((uintptr_t)group_weights + n_stride * weights_stride);
      }
    }
  }

  if (scale_params != NULL) {
    assert(init_scale_params != NULL);

    if (!weights_already_cached) {
      void* group_weights =
          (void*)((uintptr_t)weights_ptr +
                  gemm_config->nr *
                      ((kernel_size * k_stride << log2_filter_element_size) +
                       bias_element_size));
      if (kernel_scale_params != NULL) {
        group_weights =
            (void*)((uintptr_t)group_weights + gemm_config->nr * sizeof(float));
      }
      const size_t weights_stride =
          (kernel_size * k_stride << log2_filter_element_size) +
          bias_element_size + extra_weights_bytes;
      for (uint32_t group = 0; group < groups; group++) {
        init_scale_params(group_output_channels, gemm_config->nr,
                          gemm_config->nr * weights_stride, scale_params,
                          group_weights);
        scale_params += group_output_channels;
        group_weights =
            (void*)((uintptr_t)group_weights + n_stride * weights_stride);
      }
    }
  }

  if (use_weights_cache(convolution_op)) {
    convolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        convolution_op->weights_cache, &cache_key, weights_ptr,
        aligned_total_weights_size);
  }

  *zero_size = XNN_EXTRA_BYTES + (k_stride << log2_input_element_size);
  return xnn_status_success;

error:
  return status;
}

struct convolution2d_nhwc_variant;

typedef enum xnn_status (*variant_setup_fn)(
    const struct convolution2d_nhwc_variant*,
    struct convolution2d_nhwc_context*);

// Holds parameters that are reused by multiple implementations.
struct convolution2d_nhwc_variant {
  variant_setup_fn check_input_scale;
  variant_setup_fn check_kernel_scale;
  variant_setup_fn check_output_scale;
  variant_setup_fn check_output_range;
  variant_setup_fn init_linear_activation;
  variant_setup_fn init_requantization_scale;
  variant_setup_fn init_packing_params;
  variant_setup_fn init_gemm_params;
  variant_setup_fn init_dwconv_params;
  variant_setup_fn init_vmuladdc_params;
  variant_setup_fn set_context_params;
  variant_setup_fn cleanup;

  size_t extra_weights_bytes;
  xnn_init_qs8_qc8w_scale_params_fn init_scale_params;
  xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params;
  bool dynamic_quantization;
};


static enum xnn_status select_microkernel_type(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const size_t kernel_size = context->kernel_height * context->kernel_width;
  const bool unit_subsampling = (context->subsampling_width | context->subsampling_height) == 1;
  const bool any_padding = (context->input_padding_left | context->input_padding_top |
                            context->input_padding_right | context->input_padding_bottom) != 0;
  if (context->group_input_channels == 1 && context->group_output_channels == 1 &&
      kernel_size == 1 && unit_subsampling && !any_padding &&
      context->vmulcaddc_config != NULL) {
    context->microkernel_type = xnn_microkernel_type_vmulcaddc;
  } else if (context->group_input_channels == 1 && context->group_output_channels == 1 &&
             context->dwconv_ukernel != NULL) {
    context->microkernel_type = xnn_microkernel_type_dwconv;

  } else {
    context->microkernel_type = xnn_microkernel_type_igemm;
  }
  return xnn_status_success;
}

static enum xnn_status create_convolution2d_nhwc(
    const struct convolution2d_nhwc_variant* variant,
    const struct convolution2d_nhwc_context* context,
    xnn_operator_t* convolution_op_out) {
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string(context->operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (context->kernel_width == 0 || context->kernel_height == 0) {
    xnn_log_error("failed to create %s operator with %" PRIu32 "x%" PRIu32
                  " kernel: kernel dimensions must be non-zero",
                  xnn_operator_type_to_string(context->operator_type),
                  context->kernel_width, context->kernel_height);
    goto error;
  }

  if (context->subsampling_width == 0 || context->subsampling_height == 0) {
    xnn_log_error("failed to create %s operator with %" PRIu32 "x%" PRIu32
                  " subsampling: subsampling dimensions must be non-zero",
                  xnn_operator_type_to_string(context->operator_type),
                  context->subsampling_width, context->subsampling_height);
    goto error;
  }

  if (context->dilation_width == 0 || context->dilation_height == 0) {
    xnn_log_error("failed to create %s operator with %" PRIu32 "x%" PRIu32
                  " dilation: dilation dimensions must be non-zero",
                  xnn_operator_type_to_string(context->operator_type),
                  context->dilation_width, context->dilation_height);
    goto error;
  }

  if (context->groups == 0) {
    xnn_log_error("failed to create %s operator with %" PRIu32
                  " groups: number of groups must be non-zero",
                  xnn_operator_type_to_string(context->operator_type),
                  context->groups);
    goto error;
  }

  if (context->group_input_channels == 0) {
    xnn_log_error(
        "failed to create %s operator with %zu input channels per group: "
        "number of channels must be non-zero",
        xnn_operator_type_to_string(context->operator_type),
        context->group_input_channels);
    goto error;
  }

  if (context->group_output_channels == 0) {
    xnn_log_error(
        "failed to create %s operator with %zu output channels per group: "
        "number of channels must be non-zero",
        xnn_operator_type_to_string(context->operator_type),
        context->group_output_channels);
    goto error;
  }

  const size_t input_channels = context->groups * context->group_input_channels;
  if (context->input_channel_stride < input_channels) {
    xnn_log_error(
        "failed to create %s operator with input channel stride of %zu: stride "
        "must be at least as large as the number of input channels (%" PRIu32
        "x%zu)",
        xnn_operator_type_to_string(context->operator_type),
        context->input_channel_stride, context->groups,
        context->group_input_channels);
    goto error;
  }

  const size_t output_channels = context->groups * context->group_output_channels;
  if (context->output_channel_stride < output_channels) {
    xnn_log_error(
        "failed to create %s operator with output channel stride of %zu: "
        "stride must be at least as large as the number of output channels "
        "(%" PRIu32 "x%zu)",
        xnn_operator_type_to_string(context->operator_type),
        context->output_channel_stride, context->groups,
        context->group_output_channels);
    goto error;
  }

  if ((context->flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 &&
      context->group_input_channels != 1) {
    xnn_log_error(
        "failed to create depthwise %s operator with %zu input channels per "
        "group: depthwise convolution must have exactly 1 input channel per "
        "group",
        xnn_operator_type_to_string(context->operator_type),
        context->group_input_channels);
    goto error;
  }

  const bool any_padding = (context->input_padding_left | context->input_padding_top |
                            context->input_padding_right | context->input_padding_bottom) != 0;
  if ((context->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error("failed to create %s operator with %" PRIu32 "+%" PRIu32
                    "x%" PRIu32 "+%" PRIu32
                    " padding: TensorFlow SAME padding can't be combined with "
                    "explicit padding specification",
                    xnn_operator_type_to_string(context->operator_type),
                    context->input_padding_top, context->input_padding_left,
                    context->input_padding_bottom,
                    context->input_padding_right);
      goto error;
    }
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_operator),
                  xnn_operator_type_to_string(context->operator_type));
    goto error;
  }
  const int num_compute_invocations = 3;
  convolution_op->compute = xnn_allocate_zero_memory(
      num_compute_invocations * sizeof(struct compute_parameters));
  if (convolution_op->compute == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct compute_parameters),
                  xnn_operator_type_to_string(context->operator_type));
    goto error;
  }
  convolution_op->num_compute_invocations = num_compute_invocations;

  convolution_op->convolution_op =
      xnn_allocate_zero_memory(sizeof(struct xnn_convolution_operator));
  if (convolution_op->convolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_convolution_operator),
                  xnn_operator_type_to_string(context->operator_type));
    goto error;
  }

  convolution_op->ukernel.igemm =
      xnn_allocate_zero_simd_memory(sizeof(struct xnn_ukernel_igemm));
  if (convolution_op->ukernel.igemm == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_ukernel_igemm),
                  xnn_operator_type_to_string(context->operator_type));
    goto error;
  }

  convolution_op->weights_cache = context->weights_cache;

  const size_t kernel_size = context->kernel_height * context->kernel_width;

  size_t zero_size = 0;
  switch (context->microkernel_type) {
    case xnn_microkernel_type_vmulcaddc: {
      status = create_vmulcaddc_path(
          context->groups, context->kernel, context->bias,
          context->gemm_config->log2_filter_element_size,
          context->gemm_config->bias_element_size, context->pack_vmulcaddc_w,
          context->packing_params_ptr, context->vmulcaddc_params_ptr,
          context->vmulcaddc_params_size, context->vmulcaddc_config,
          context->fingerprint_id, context->operator_type, convolution_op, context);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_dwconv: {
      convolution_op->dynamic_context.dwconv =
          xnn_allocate_zero_simd_memory(sizeof(struct dwconv_op_context));
      if (convolution_op->dynamic_context.dwconv == NULL) {
        xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                      sizeof(struct dwconv_op_context),
                      xnn_operator_type_to_string(context->operator_type));
        goto error;
      }
      status = create_dwconv_path(
          context->kernel_height, context->kernel_width, context->groups,
          context->kernel, context->bias, context->flags,
          context->gemm_config->log2_input_element_size,
          context->gemm_config->log2_filter_element_size,
          context->gemm_config->bias_element_size, context->pack_dwconv_hwg_w,
          context->pack_dwconv_ghw_w, context->packing_params_ptr,
          variant->extra_weights_bytes, variant->init_scale_params,
          context->scale_params, context->dwconv_params_ptr,
          context->dwconv_params_size, context->dwconv_ukernel,
          context->linear_activation, context->fingerprint_id,
          context->operator_type, &zero_size, convolution_op, context);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_igemm: {
      convolution_op->dynamic_context.igemm =
          xnn_allocate_zero_simd_memory(sizeof(struct igemm_op_context));
      if (convolution_op->dynamic_context.igemm == NULL) {
        xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                      sizeof(struct igemm_op_context),
                      xnn_operator_type_to_string(context->operator_type));
        goto error;
      }
      status = create_igemm(
          context->microkernel_type, kernel_size, context->groups,
          context->group_input_channels, context->group_output_channels,
          context->kernel, context->bias, context->flags,
          context->gemm_config->log2_input_element_size,
          context->gemm_config->log2_filter_element_size,
          context->gemm_config->bias_element_size, context->pack_conv_kgo_w,
          context->pack_conv_goki_w, context->packing_params_ptr,
          variant->extra_weights_bytes, variant->init_scale_params,
          context->scale_params, variant->init_kernel_scale_params,
          context->kernel_scale_params, context->gemm_params_ptr,
          context->gemm_params_size, context->gemm_config,
          context->linear_activation, context->relu_activation,
          context->fingerprint_id, context->operator_type, convolution_op,
          &zero_size, context);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  convolution_op->convolution_op->zero_size = 0;
  const bool tf_same_padding =
      (context->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && kernel_size != 1;
  if (any_padding || tf_same_padding) {
    convolution_op->convolution_op->zero_size = zero_size;
    convolution_op->zero_buffer = xnn_allocate_simd_memory(zero_size);
    if (convolution_op->zero_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for %s operator zero padding",
                    zero_size, xnn_operator_type_to_string(context->operator_type));
      goto error;
    }
    memset(convolution_op->zero_buffer, context->input_padding_bytes, zero_size);
  }

  convolution_op->convolution_op->padding_top = context->input_padding_top;
  convolution_op->convolution_op->padding_right = context->input_padding_right;
  convolution_op->convolution_op->padding_bottom = context->input_padding_bottom;
  convolution_op->convolution_op->padding_left = context->input_padding_left;

  convolution_op->convolution_op->kernel_height = context->kernel_height;
  convolution_op->convolution_op->kernel_width = context->kernel_width;
  convolution_op->convolution_op->stride_height = context->subsampling_height;
  convolution_op->convolution_op->stride_width = context->subsampling_width;
  convolution_op->convolution_op->dilation_height = context->dilation_height;
  convolution_op->convolution_op->dilation_width = context->dilation_width;
  convolution_op->convolution_op->groups = context->groups;
  convolution_op->convolution_op->group_input_channels = context->group_input_channels;
  convolution_op->convolution_op->group_output_channels = context->group_output_channels;
  convolution_op->input_pixel_stride = context->input_channel_stride;
  convolution_op->output_pixel_stride = context->output_channel_stride;

  convolution_op->type = context->operator_type;
  convolution_op->ukernel.type = context->microkernel_type;
  convolution_op->flags = context->flags & ~XNN_FLAG_TENSORFLOW_SAME_PADDING;
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

static enum xnn_status UNUSED_FUNCTION(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  return xnn_status_success;
}

static enum xnn_status check_scale_qx8(enum xnn_operator_type operator_type,
                                       float value, const char* name) {
  if (value <= 0.0f || !isnormal(value)) {
    xnn_log_error(
        "failed to create %s operator with %.7g %s scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(operator_type), value, name);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_input_scale_qx8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  return check_scale_qx8(context->operator_type, context->input_scale, "input");
}

static enum xnn_status check_kernel_scale_qx8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  return check_scale_qx8(context->operator_type, context->kernel_scale_value,
                         "kernel");
}

static enum xnn_status check_kernel_scale_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const float* kernel_scale = context->kernel_scale;
  for (size_t output_channel = 0;
       output_channel < context->groups * context->group_output_channels;
       output_channel++) {
    if (kernel_scale[output_channel] <= 0.0f ||
        !isnormal(kernel_scale[output_channel])) {
      xnn_log_error(
          "failed to create %s operator with %.7g kernel scale in output "
          "channel #%zu: scale must be finite, normalized, and positive",
          xnn_operator_type_to_string(context->operator_type),
          kernel_scale[output_channel], output_channel);
      return xnn_status_invalid_parameter;
    }
  }
  return xnn_status_success;
}

static enum xnn_status check_output_scale_qx8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  return check_scale_qx8(context->operator_type, context->output_scale,
                         "output");
}

static enum xnn_status check_output_range_f32(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  if (isnan(context->output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(context->output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_invalid_parameter;
  }

  if (context->output_min > context->output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(context->operator_type),
        context->output_min, context->output_max);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_output_range_f16(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  enum xnn_status status = check_output_range_f32(variant, context);
  if (status != xnn_status_success) {
    return status;
  }

  context->fp16_output_min = xnn_float16_from_float(context->output_min);
  context->fp16_output_max = xnn_float16_from_float(context->output_max);
  const float rounded_output_min =
      xnn_float16_to_float(context->fp16_output_min);
  const float rounded_output_max =
      xnn_float16_to_float(context->fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be below upper bound",
        xnn_operator_type_to_string(context->operator_type), rounded_output_min,
        rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  return xnn_status_success;
}

static enum xnn_status init_linear_activation_f32(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->linear_activation = (context->output_max == INFINITY) &&
                               (context->output_min == -context->output_max);
  context->relu_activation =
      (context->output_max == INFINITY) && (context->output_min == 0.0f);
  return xnn_status_success;
}

static enum xnn_status init_requantization_scale_qu8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->requantization_scale_value = context->input_scale *
                                        context->kernel_scale_value /
                                        context->output_scale;
  if (context->requantization_scale_value >= 256.0f) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel "
        "scale, and %.7g output scale: requantization scale %.7g is greater or "
        "equal to 256.0",
        xnn_operator_type_to_string(context->operator_type),
        context->input_scale, context->kernel_scale_value,
        context->output_scale, context->requantization_scale_value);
    return xnn_status_unsupported_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status init_requantization_scale_qs8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  enum xnn_status status = init_requantization_scale_qu8(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  float* scale_params = xnn_allocate_simd_memory(
      context->groups * context->group_output_channels * sizeof(float));
  if (scale_params == NULL) {
    xnn_log_error(
        "failed to allocate %zu bytes for %s operator packed weights",
        context->groups * context->group_output_channels * sizeof(float),
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_out_of_memory;
  }
  for (size_t output_channel = 0;
       output_channel < context->groups * context->group_output_channels;
       output_channel++) {
    scale_params[output_channel] = context->requantization_scale_value;
  }
  context->requantization_scale = scale_params;
  context->scale_params = scale_params;
  return xnn_status_success;
}

static enum xnn_status init_requantization_scale_qx8_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  float* requantization_scale = xnn_allocate_simd_memory(
      context->groups * context->group_output_channels * sizeof(float));
  if (requantization_scale == NULL) {
    xnn_log_error(
        "failed to allocate %zu bytes for %s operator packed weights",
        context->groups * context->group_output_channels * sizeof(float),
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_out_of_memory;
  }
  const float* kernel_scale = context->kernel_scale;
  for (size_t output_channel = 0;
       output_channel < context->groups * context->group_output_channels;
       output_channel++) {
    requantization_scale[output_channel] = context->input_scale *
                                           kernel_scale[output_channel] /
                                           context->output_scale;
    if (requantization_scale[output_channel] >= 256.0f) {
      xnn_log_error(
          "failed to create %s operator with %.7g input scale, %.7g kernel "
          "scale, and %.7g output scale in output channel #%zu: requantization "
          "scale %.7g is greater or equal to 256.0",
          xnn_operator_type_to_string(context->operator_type),
          context->input_scale, kernel_scale[output_channel],
          context->output_scale, output_channel,
          requantization_scale[output_channel]);
      xnn_release_simd_memory(requantization_scale);
      return xnn_status_unsupported_parameter;
    }
  }
  context->requantization_scale = requantization_scale;
  context->scale_params = requantization_scale;
  return xnn_status_success;
}

static enum xnn_status init_packing_params_qu8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->packing_params.qu8.input_zero_point = context->input_zero_point;
  context->packing_params.qu8.kernel_zero_point = context->kernel_zero_point;
  context->packing_params_ptr = &(context->packing_params.qu8);
  return xnn_status_success;
}

static enum xnn_status init_packing_params_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->packing_params.qs8.input_zero_point = context->input_zero_point;
  context->packing_params_ptr = &(context->packing_params.qs8);
  return xnn_status_success;
}

static enum xnn_status init_packing_params_fxx_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->packing_params.qs8.input_zero_point = 1;
  context->packing_params_ptr = &(context->packing_params.qs8);
  return xnn_status_success;
}

static enum xnn_status init_gemm_params_qu8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  if (context->output_max > UINT8_MAX) {
    context->output_max = UINT8_MAX;
  }
  if (context->output_min < 0) {
    context->output_min = 0;
  }
  if XNN_LIKELY (context->gemm_config->init.qu8 != NULL) {
    context->gemm_config->init.qu8(
        &context->gemm_params.qu8, context->kernel_zero_point,
        context->requantization_scale_value, context->output_zero_point,
        context->output_min, context->output_max);
    context->gemm_params_size = sizeof(context->gemm_params.qu8);
    context->gemm_params_ptr = &context->gemm_params.qu8;
  }
  return xnn_status_success;
}

static enum xnn_status init_gemm_params_qs8_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  if (context->output_max > INT8_MAX) {
    context->output_max = INT8_MAX;
  }
  if (context->output_min < INT8_MIN) {
    context->output_min = INT8_MIN;
  }
  if XNN_LIKELY (context->gemm_config->init.qs8_qc8w != NULL) {
    context->gemm_config->init.qs8_qc8w(
        &context->gemm_params.qs8_qc8w, context->output_zero_point,
        context->output_min, context->output_max);
    context->gemm_params_size = sizeof(context->gemm_params.qs8_qc8w);
    context->gemm_params_ptr = &context->gemm_params.qs8_qc8w;
  }
  return xnn_status_success;
}

static enum xnn_status init_gemm_params_f16(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  if XNN_LIKELY (context->gemm_config->init.f16 != NULL) {
    context->gemm_config->init.f16(&context->gemm_params.f16,
                                   context->fp16_output_min,
                                   context->fp16_output_max);
    context->gemm_params_size = sizeof(context->gemm_params.f16);
    context->gemm_params_ptr = &context->gemm_params.f16;
  }
  return xnn_status_success;
}

static enum xnn_status init_gemm_params_f32(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  if XNN_LIKELY (context->gemm_config->init.f32 != NULL) {
    context->gemm_config->init.f32(&context->gemm_params.f32,
                                   context->output_min, context->output_max);
    context->gemm_params_size = sizeof(context->gemm_params.f32);
    context->gemm_params_ptr = &context->gemm_params.f32;
  }
  return xnn_status_success;
}

static enum xnn_status init_dwconv_params_qu8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const struct xnn_dwconv_config* dwconv_config = xnn_init_qu8_dwconv_config();
  if (dwconv_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }

  context->dwconv_ukernel =
      find_dwconv_ukernel(context->kernel_height * context->kernel_width,
                          dwconv_config, XNN_MAX_QU8_DWCONV_UKERNELS);
  if XNN_LIKELY (context->dwconv_ukernel != NULL) {
    context->dwconv_ukernel->init.qu8(
        &context->dwconv_params.qu8, context->kernel_zero_point,
        context->requantization_scale_value, context->output_zero_point,
        context->output_min, context->output_max);
    context->dwconv_params_size = sizeof(context->dwconv_params.qu8);
    context->dwconv_params_ptr = &context->dwconv_params.qu8;
  }
  return xnn_status_success;
}

static enum xnn_status init_dwconv_params_qs8_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const struct xnn_dwconv_config* dwconv_config =
      xnn_init_qs8_qc8w_dwconv_config();
  if (dwconv_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }

  context->dwconv_ukernel =
      find_dwconv_ukernel(context->kernel_height * context->kernel_width,
                          dwconv_config, XNN_MAX_QC8_DWCONV_UKERNELS);
  if XNN_LIKELY (context->dwconv_ukernel != NULL) {
    context->dwconv_ukernel->init.qs8_qc8w(
        &context->dwconv_params.qs8_qc8w, context->output_zero_point,
        context->output_min, context->output_max);
    context->dwconv_params_size = sizeof(context->dwconv_params.qs8_qc8w);
    context->dwconv_params_ptr = &context->dwconv_params.qs8_qc8w;
  }
  return xnn_status_success;
}

static enum xnn_status init_dwconv_params_f16(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const struct xnn_dwconv_config* dwconv_config = xnn_init_f16_dwconv_config();
  if (dwconv_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }
  context->dwconv_ukernel =
      find_dwconv_ukernel(context->kernel_height * context->kernel_width,
                          dwconv_config, XNN_MAX_F16_DWCONV_UKERNELS);
  if XNN_LIKELY (context->dwconv_ukernel != NULL) {
    context->dwconv_ukernel->init.f16(&context->dwconv_params.f16,
                                      context->fp16_output_min,
                                      context->fp16_output_max);
    context->dwconv_params_size = sizeof(context->dwconv_params.f16);
    context->dwconv_params_ptr = &context->dwconv_params.f16;
  }
  return xnn_status_success;
}

static enum xnn_status init_dwconv_params_f32(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const struct xnn_dwconv_config* dwconv_config = xnn_init_f32_dwconv_config();
  if (dwconv_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }
  context->dwconv_ukernel =
      find_dwconv_ukernel(context->kernel_height * context->kernel_width,
                          dwconv_config, XNN_MAX_F32_DWCONV_UKERNELS);
  if XNN_LIKELY (context->dwconv_ukernel != NULL) {
    context->dwconv_ukernel->init.f32(&context->dwconv_params.f32,
                                      context->output_min, context->output_max);
    context->dwconv_params_size = sizeof(context->dwconv_params.f32);
    context->dwconv_params_ptr = &context->dwconv_params.f32;
  }
  return xnn_status_success;
}

static enum xnn_status init_vmuladdc_params_f16(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const struct xnn_vmulcaddc_config* vmulcaddc_config =
      xnn_init_f16_vmulcaddc_config();
  if (vmulcaddc_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }

  if XNN_LIKELY (vmulcaddc_config->init.f16 != NULL) {
    vmulcaddc_config->init.f16(&context->vmuladdc_params.f16,
                               context->fp16_output_min,
                               context->fp16_output_max);
    context->vmulcaddc_params_size = sizeof(context->vmuladdc_params.f16);
    context->vmulcaddc_params_ptr = &context->vmuladdc_params.f16;
  }
  return xnn_status_success;
}

static enum xnn_status init_vmuladdc_params_f32(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const struct xnn_vmulcaddc_config* vmulcaddc_config =
      xnn_init_f32_vmulcaddc_config();
  if (vmulcaddc_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }

  if XNN_LIKELY (vmulcaddc_config->init.f32 != NULL) {
    vmulcaddc_config->init.f32(&context->vmuladdc_params.f32,
                               context->output_min, context->output_max);
    context->vmulcaddc_params_size = sizeof(context->vmuladdc_params.f32);
    context->vmulcaddc_params_ptr = &context->vmuladdc_params.f32;
  }
  return xnn_status_success;
}

static enum xnn_status set_context_params_qu8(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->pack_dwconv_hwg_w =
      (xnn_pack_dwconv_hwg_w_fn)xnn_pack_qu8_dwconv_hwg_w;
  context->pack_dwconv_ghw_w =
      (xnn_pack_dwconv_ghw_w_fn)xnn_pack_qu8_dwconv_ghw_w;
  context->pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn)xnn_pack_qu8_conv_kgo_w;
  context->pack_conv_goki_w = (xnn_pack_conv_goki_w_fn)xnn_pack_qu8_conv_goki_w;
  context->input_padding_bytes = context->input_zero_point;
  return xnn_status_success;
}

static enum xnn_status set_context_params_qx8_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->pack_dwconv_hwg_w =
      (xnn_pack_dwconv_hwg_w_fn)xnn_pack_qs8_dwconv_hwg_w;
  context->pack_dwconv_ghw_w =
      (xnn_pack_dwconv_ghw_w_fn)xnn_pack_qs8_dwconv_ghw_w;
  context->pack_conv_kgo_w =
      (xnn_pack_conv_kgo_w_fn)context->gemm_config->pack_igemm_kgo;
  context->pack_conv_goki_w =
      (xnn_pack_conv_goki_w_fn)context->gemm_config->pack_igemm_goki;
  context->input_padding_bytes = context->input_zero_point;
  return xnn_status_success;
}

static enum xnn_status set_context_params_fxx_qc8w(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  const float* const scale_params = context->bias;
  context->bias = NULL;  // The bias is put in the scale params.
  context->pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn)xnn_pack_qs8_conv_kgo_w,
  context->pack_conv_goki_w = (xnn_pack_conv_goki_w_fn)xnn_pack_qs8_conv_goki_w,
  context->packing_params_ptr = &context->packing_params.qs8;
  context->scale_params = scale_params;
  context->kernel_scale_params = context->kernel_scale;
  return xnn_status_success;
}

static enum xnn_status set_context_params_f16(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  if (context->flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    context->pack_vmulcaddc_w =
        (xnn_pack_vmulcaddc_w_fn)xnn_pack_f32_to_f16_vmulcaddc_w;
    context->pack_dwconv_hwg_w =
        (xnn_pack_dwconv_hwg_w_fn)xnn_pack_f32_to_f16_dwconv_hwg_w;
    context->pack_dwconv_ghw_w =
        (xnn_pack_dwconv_ghw_w_fn)xnn_pack_f32_to_f16_dwconv_ghw_w;
    context->pack_conv_kgo_w =
        (xnn_pack_conv_kgo_w_fn)xnn_pack_f32_to_f16_conv_kgo_w;
    context->pack_conv_goki_w =
        (xnn_pack_conv_goki_w_fn)xnn_pack_f32_to_f16_conv_goki_w;
  } else {
    context->pack_vmulcaddc_w =
        (xnn_pack_vmulcaddc_w_fn)xnn_pack_f16_vmulcaddc_w;
    context->pack_dwconv_hwg_w =
        (xnn_pack_dwconv_hwg_w_fn)xnn_pack_f16_dwconv_hwg_w;
    context->pack_dwconv_ghw_w =
        (xnn_pack_dwconv_ghw_w_fn)xnn_pack_f16_dwconv_ghw_w;
    context->pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn)xnn_pack_f16_conv_kgo_w;
    context->pack_conv_goki_w =
        (xnn_pack_conv_goki_w_fn)xnn_pack_f16_conv_goki_w;
  }
  return xnn_status_success;
}

static enum xnn_status set_context_params_f32(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  context->pack_vmulcaddc_w = (xnn_pack_vmulcaddc_w_fn)xnn_pack_f32_vmulcaddc_w;
  context->pack_dwconv_hwg_w =
      (xnn_pack_dwconv_hwg_w_fn)xnn_pack_f32_dwconv_hwg_w;
  context->pack_dwconv_ghw_w =
      (xnn_pack_dwconv_ghw_w_fn)xnn_pack_f32_dwconv_ghw_w;
  context->pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn)xnn_pack_f32_conv_kgo_w;
  context->pack_conv_goki_w = (xnn_pack_conv_goki_w_fn)xnn_pack_f32_conv_goki_w;
  return xnn_status_success;
}

static enum xnn_status cleanup_requantization_scale(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  xnn_release_simd_memory(context->requantization_scale);
  return xnn_status_success;
}

static struct convolution2d_nhwc_variant qs8_variant = {
    .check_input_scale = check_input_scale_qx8,
    .check_kernel_scale = check_kernel_scale_qx8,
    .check_output_scale = check_output_scale_qx8,
    .check_output_range = check_output_range_f32,
    .init_linear_activation = UNUSED_FUNCTION,
    .init_requantization_scale = init_requantization_scale_qs8,
    .init_packing_params = init_packing_params_qc8w,
    .init_gemm_params = init_gemm_params_qs8_qc8w,
    .init_dwconv_params = init_dwconv_params_qs8_qc8w,
    .init_vmuladdc_params = UNUSED_FUNCTION,
    .set_context_params = set_context_params_qx8_qc8w,
    .cleanup = cleanup_requantization_scale,
    .init_scale_params = xnn_init_qs8_qc8w_scale_fp32_params,
    .extra_weights_bytes = sizeof(float),
};

static struct convolution2d_nhwc_variant qu8_variant = {
    .check_input_scale = check_input_scale_qx8,
    .check_kernel_scale = check_kernel_scale_qx8,
    .check_output_scale = check_output_scale_qx8,
    .check_output_range = check_output_range_f32,
    .init_linear_activation = UNUSED_FUNCTION,
    .init_requantization_scale = init_requantization_scale_qu8,
    .init_packing_params = init_packing_params_qu8,
    .init_gemm_params = init_gemm_params_qu8,
    .init_dwconv_params = init_dwconv_params_qu8,
    .init_vmuladdc_params = UNUSED_FUNCTION,
    .set_context_params = set_context_params_qu8,
    .cleanup = UNUSED_FUNCTION,
};

static struct convolution2d_nhwc_variant qx8_qc8w_variant = {
    .check_input_scale = check_input_scale_qx8,
    .check_kernel_scale = check_kernel_scale_qc8w,
    .check_output_scale = check_output_scale_qx8,
    .check_output_range = check_output_range_f32,
    .init_linear_activation = UNUSED_FUNCTION,
    .init_requantization_scale = init_requantization_scale_qx8_qc8w,
    .init_packing_params = init_packing_params_qc8w,
    .init_gemm_params = init_gemm_params_qs8_qc8w,
    .init_dwconv_params = init_dwconv_params_qs8_qc8w,
    .init_vmuladdc_params = UNUSED_FUNCTION,
    .set_context_params = set_context_params_qx8_qc8w,
    .cleanup = cleanup_requantization_scale,
    .extra_weights_bytes = sizeof(float),
    .init_scale_params = xnn_init_qs8_qc8w_scale_fp32_params,
};

static struct convolution2d_nhwc_variant qx8_f16_qc8w_variant = {
    .check_input_scale = UNUSED_FUNCTION,
    .check_kernel_scale = UNUSED_FUNCTION,
    .check_output_scale = UNUSED_FUNCTION,
    .check_output_range = check_output_range_f16,
    .init_linear_activation = UNUSED_FUNCTION,
    .init_requantization_scale = UNUSED_FUNCTION,
    .init_packing_params = init_packing_params_fxx_qc8w,
    .init_gemm_params = init_gemm_params_f16,
    .init_dwconv_params = UNUSED_FUNCTION,
    .init_vmuladdc_params = UNUSED_FUNCTION,
    .set_context_params = set_context_params_fxx_qc8w,
    .cleanup = UNUSED_FUNCTION,
    .extra_weights_bytes = sizeof(float) * 2,
    .init_scale_params = xnn_init_qs8_qc8w_scale_fp32_params,
    .init_kernel_scale_params = xnn_init_qs8_qc8w_scale_fp32_params,
    .dynamic_quantization = true,
};

static struct convolution2d_nhwc_variant qx8_f32_qc8w_variant = {
    .check_input_scale = UNUSED_FUNCTION,
    .check_kernel_scale = UNUSED_FUNCTION,
    .check_output_scale = UNUSED_FUNCTION,
    .check_output_range = check_output_range_f32,
    .init_linear_activation = UNUSED_FUNCTION,
    .init_requantization_scale = UNUSED_FUNCTION,
    .init_packing_params = init_packing_params_fxx_qc8w,
    .init_gemm_params = init_gemm_params_f32,
    .init_dwconv_params = UNUSED_FUNCTION,
    .init_vmuladdc_params = UNUSED_FUNCTION,
    .set_context_params = set_context_params_fxx_qc8w,
    .cleanup = UNUSED_FUNCTION,
    .extra_weights_bytes = sizeof(float) * 2,
    .init_scale_params = xnn_init_qs8_qc8w_scale_fp32_params,
    .init_kernel_scale_params = xnn_init_qs8_qc8w_scale_fp32_params,
    .dynamic_quantization = true,
};


static struct convolution2d_nhwc_variant f16_variant = {
    .check_input_scale = UNUSED_FUNCTION,
    .check_kernel_scale = UNUSED_FUNCTION,
    .check_output_scale = UNUSED_FUNCTION,
    .check_output_range = check_output_range_f16,
    .init_linear_activation = UNUSED_FUNCTION,
    .init_requantization_scale = UNUSED_FUNCTION,
    .init_packing_params = UNUSED_FUNCTION,
    .init_gemm_params = init_gemm_params_f16,
    .init_dwconv_params = init_dwconv_params_f16,
    .init_vmuladdc_params = init_vmuladdc_params_f16,
    .set_context_params = set_context_params_f16,
    .cleanup = UNUSED_FUNCTION,
};

static struct convolution2d_nhwc_variant f32_variant = {
    .check_input_scale = UNUSED_FUNCTION,
    .check_kernel_scale = UNUSED_FUNCTION,
    .check_output_scale = UNUSED_FUNCTION,
    .check_output_range = check_output_range_f32,
    .init_linear_activation = init_linear_activation_f32,
    .init_requantization_scale = UNUSED_FUNCTION,
    .init_packing_params = UNUSED_FUNCTION,
    .init_gemm_params = init_gemm_params_f32,
    .init_dwconv_params = init_dwconv_params_f32,
    .init_vmuladdc_params = init_vmuladdc_params_f32,
    .set_context_params = set_context_params_f32,
    .cleanup = UNUSED_FUNCTION,
};

static enum xnn_status compute_fingerprint_id(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  enum xnn_fingerprint_id_helper in, out, weights, flags = 0;

#define OPERATOR_TYPE_CASE(...)                                           \
  XNN_EXPAND(XNN_GLUE(                                                    \
      XNN_OVERLOAD(XNN_OPERATOR_TYPE_NAME_, XNN_COUNT_ARGS(__VA_ARGS__)), \
      (__VA_ARGS__)))

#define XNN_OPERATOR_TYPE_NAME_1(IN_OUT_WEIGHTS)            \
  case xnn_operator_type_convolution_nhwc_##IN_OUT_WEIGHTS: \
    in = xnn_fingerprint_id_helper_##IN_OUT_WEIGHTS;        \
    out = xnn_fingerprint_id_helper_##IN_OUT_WEIGHTS;       \
    weights = xnn_fingerprint_id_helper_##IN_OUT_WEIGHTS;   \
    break

#define XNN_OPERATOR_TYPE_NAME_2(IN_OUT, WEIGHTS)               \
  case xnn_operator_type_convolution_nhwc_##IN_OUT##_##WEIGHTS: \
    in = xnn_fingerprint_id_helper_##IN_OUT;                    \
    out = xnn_fingerprint_id_helper_##IN_OUT;                   \
    weights = xnn_fingerprint_id_helper_##WEIGHTS;              \
    break

#define XNN_OPERATOR_TYPE_NAME_3(IN, OUT, WEIGHTS)                  \
  case xnn_operator_type_convolution_nhwc_##IN##_##OUT##_##WEIGHTS: \
    in = xnn_fingerprint_id_helper_##IN;                            \
    out = xnn_fingerprint_id_helper_##OUT;                          \
    weights = xnn_fingerprint_id_helper_##WEIGHTS;                  \
    break

  switch (context->operator_type) {
    OPERATOR_TYPE_CASE(f16);
    OPERATOR_TYPE_CASE(pf16);
    OPERATOR_TYPE_CASE(pf32);
    OPERATOR_TYPE_CASE(f32);
    OPERATOR_TYPE_CASE(qd8, f16, qc8w);
    OPERATOR_TYPE_CASE(qdu8, f16, qc8w);
    OPERATOR_TYPE_CASE(qd8, f32, qc8w);
    OPERATOR_TYPE_CASE(qdu8, f32, qc8w);
    OPERATOR_TYPE_CASE(qu8);
    OPERATOR_TYPE_CASE(qs8);
    OPERATOR_TYPE_CASE(qc8);
    OPERATOR_TYPE_CASE(pqs8, qs8, qc8w);
    default:
      xnn_log_error(
          "Unsupported operator type when computing the fingerprint id for "
          "operator %s",
          xnn_operator_type_to_string(context->operator_type));
      return xnn_status_unsupported_parameter;
  }

  if (context->flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    flags |= xnn_fingerprint_id_helper_fp32_static_weights;
  }
  switch (context->microkernel_type) {
    case xnn_microkernel_type_dwconv:
      flags |= xnn_fingerprint_id_helper_dwconv;
      break;
    case xnn_microkernel_type_vmulcaddc:
      flags |= xnn_fingerprint_id_helper_vmulcaddc;
      break;
    case xnn_microkernel_type_igemm:
      break;
    default:
      xnn_log_error(
          "Unsupported microkernel type when computing the fingerprint id of "
          "%s",
          xnn_operator_type_to_string(context->operator_type));
      return xnn_status_unsupported_parameter;
  }

#undef OPERATOR_TYPE_CASE
#undef OPERATOR_TYPE_NAME_1
#undef OPERATOR_TYPE_NAME_2
#undef OPERATOR_TYPE_NAME_3

  context->fingerprint_id = xnn_compute_fingerprint_id_value(
      xnn_fingerprint_id_helper_convolution2d_nhwc, in, out, weights, flags, 0);
  return xnn_status_success;
}

static enum xnn_status select_gemm_config(
    struct convolution2d_nhwc_context* context) {
  switch (context->operator_type) {
    case xnn_operator_type_convolution_nhwc_qd8_f16_qc8w:
      context->gemm_config = xnn_init_qd8_f16_qc8w_igemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_qdu8_f16_qc8w:
      context->gemm_config = xnn_init_qdu8_f16_qc8w_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_qd8_f32_qc8w:
      context->gemm_config = xnn_init_qd8_f32_qc8w_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_qdu8_f32_qc8w:
      context->gemm_config = xnn_init_qdu8_f32_qc8w_igemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_qu8:
      context->gemm_config = xnn_init_qu8_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_qs8:
      context->gemm_config = xnn_init_qs8_qc8w_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_qc8:
      context->gemm_config = xnn_init_qs8_qc8w_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_pqs8_qs8_qc8w:
      context->gemm_config = xnn_init_pqs8_qc8w_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_f16:
      context->gemm_config = xnn_init_f16_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_pf16:
      context->gemm_config = xnn_init_pf16_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_pf32:
      context->gemm_config = xnn_init_pf32_gemm_config();
      break;
    case xnn_operator_type_convolution_nhwc_f32:
      context->gemm_config = xnn_init_f32_igemm_config();
      if (context->gemm_config &&
          context->gemm_config->nr > context->group_output_channels) {
        const struct xnn_gemm_config* gemm_nr2_config =
            xnn_init_f32_gemm_nr2_config(context->flags);
        // Default micro-kernel is suboptimal. Try to find a better
        // micro-kernel.
        if (gemm_nr2_config->minmax.igemm[gemm_nr2_config->mr - 1]
                .function[XNN_UARCH_DEFAULT] != NULL) {
          context->gemm_config = gemm_nr2_config;
        }
      }
      break;
    default:
      xnn_log_error(
          "Unsupported operator type (%s) when initializing gemm_config.",
          xnn_operator_type_to_string(context->operator_type));
      return xnn_status_unsupported_parameter;
  }
  if (context->gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }
  return xnn_status_success;
}


static enum xnn_status init_convolution2d_nhwc_context(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context) {
  enum xnn_status status = xnn_status_invalid_parameter;
  status = select_gemm_config(context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->check_input_scale(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->check_kernel_scale(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->check_output_scale(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->check_output_range(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  if (context->gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }
  status = variant->init_linear_activation(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->init_requantization_scale(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->init_packing_params(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->init_gemm_params(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->init_dwconv_params(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->init_vmuladdc_params(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = variant->set_context_params(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  status = select_microkernel_type(variant, context);
  if (status != xnn_status_success) {
    xnn_log_error("failed to select a microkernel type for %s operator",
                  xnn_operator_type_to_string(context->operator_type));
    return status;
  }
  status = compute_fingerprint_id(variant, context);
  if (status != xnn_status_success) {
    return status;
  }
  return status;
}

static const struct convolution2d_nhwc_context fingerprint_context_base = {
  .input_padding_top = 0,
  .input_padding_right = 0,
  .input_padding_bottom = 0,
  .input_padding_left = 0,
  .kernel_height = 1,
  .kernel_width = 1,
  .subsampling_height = 1,
  .subsampling_width = 1,
  .dilation_height = 1,
  .dilation_width = 1,
  .groups = 1,
  .group_input_channels = 1,
  .group_output_channels = 1,
  .input_channel_stride = 1,
  .output_channel_stride = 1,
  .input_zero_point = 0,
  .input_scale = 1,
  .kernel_zero_point = 0,
  .kernel_scale_value = 1,
  .output_zero_point = 0,
  .output_scale = 1,
  .output_min = FLT_MIN,
  .output_max = FLT_MAX,
};

struct fingerprint_buffers {
  void* data;
  void* kernel;
  void* bias;
  float* kernel_scale;
};

// Returns the given `buffer` pointer, then advances it by `bytes` bytes,
// rounded up to `XNN_ALLOCATION_ALIGNMENT`.
static void* get_and_advance_simd_buffer(uint8_t** buffer, size_t bytes) {
  uint8_t* const res = *buffer;
  *buffer += bytes + (XNN_ALLOCATION_ALIGNMENT - (bytes % XNN_ALLOCATION_ALIGNMENT));
  return res;
};

static struct fingerprint_buffers generate_fingerprint_data(
    const struct convolution2d_nhwc_variant* variant,
    const struct convolution2d_nhwc_context* context) {
  const size_t kernel_size = context->kernel_width * context->kernel_height *
                             context->group_input_channels *
                             context->group_output_channels * context->groups;
  const size_t kernel_element_size = 1 << context->gemm_config->log2_filter_element_size;
  const size_t bias_size = context->group_output_channels * context->groups;
  const size_t kernel_scale_size = context->group_output_channels * context->groups;
  const size_t bytes =
      kernel_size * kernel_element_size +
      bias_size * context->gemm_config->bias_element_size +
      kernel_scale_size * sizeof(float) +
      XNN_ALLOCATION_ALIGNMENT * 3;
  uint8_t* buffer = xnn_allocate_simd_memory(bytes);
  fill_fingerprint_buffer(buffer, bytes);
  struct fingerprint_buffers data = {
    .data = buffer,
    .kernel = get_and_advance_simd_buffer(&buffer, kernel_size * kernel_element_size),
    .bias = get_and_advance_simd_buffer(&buffer, bias_size * context->gemm_config->bias_element_size),
    .kernel_scale = get_and_advance_simd_buffer(&buffer, kernel_scale_size * sizeof(float)),
  };
  for (size_t i = 0; i < kernel_scale_size; ++i) {
    data.kernel_scale[i] = 1;
  }
  return data;
}

enum xnn_status xnn_fingerprint_convolution2d_nhwc(
    const enum xnn_fingerprint_id fingerprint_id) {
  const struct convolution2d_nhwc_variant* variant;
  struct convolution2d_nhwc_context context = fingerprint_context_base;
  context.fingerprint_id = fingerprint_id;
  switch (fingerprint_id & ~xnn_fingerprint_id_helper_flag_mask) {
    case xnn_fingerprint_id_convolution2d_nhwc_qd8_f16_qc8w:
      variant = &qx8_f16_qc8w_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_qd8_f16_qc8w;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_qdu8_f16_qc8w:
      variant = &qx8_f16_qc8w_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_qdu8_f16_qc8w;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_qd8_f32_qc8w:
      variant = &qx8_f32_qc8w_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_qd8_f32_qc8w;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_qdu8_f32_qc8w:
      variant = &qx8_f32_qc8w_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_qdu8_f32_qc8w;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_qu8_qu8_qu8:
      variant = &qu8_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_qu8;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_qs8_qs8_qs8:
      variant = &qs8_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_qs8;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_qc8_qc8_qc8:
      variant = &qx8_qc8w_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_qc8;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_pqs8_qs8_qc8w:
      variant = &qx8_qc8w_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_pqs8_qs8_qc8w;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_pf16_pf16_pf16:
      variant = &f16_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_pf16;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_f16_f16_f16:
      variant = &f16_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_f16;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_f32_f32_f32:
      variant = &f32_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_f32;
      break;
    case xnn_fingerprint_id_convolution2d_nhwc_pf32_pf32_pf32:
      variant = &f32_variant;
      context.operator_type = xnn_operator_type_convolution_nhwc_pf32;
      break;
    default:
      xnn_log_error(
          "failed to fingerprint: unsupported operator stored in fingerprint: "
          "%ud",
          (uint32_t)(fingerprint_id & ~xnn_fingerprint_id_helper_flag_mask) >>
              XNN_FINGERPRINT_ID_OP_OFFSET);
      return xnn_status_unsupported_parameter;
  }

  if (fingerprint_id & xnn_fingerprint_id_helper_fp32_static_weights) {
    context.flags |= XNN_FLAG_FP32_STATIC_WEIGHTS;
  }

  if (fingerprint_id & xnn_fingerprint_id_helper_vmulcaddc) {
    // The base context is initialized to vmulcaddc path conditions.
  } else if (fingerprint_id & xnn_fingerprint_id_helper_dwconv) {
      context.input_padding_top = 1;
      context.input_padding_left = 1;
      context.input_padding_bottom = 1;
      context.input_padding_right = 1;
  } else { // igemm
      context.input_padding_top = 2;
      context.input_padding_left = 2;
      context.input_padding_bottom = 2;
      context.input_padding_right = 2;
  }

  enum xnn_status status = select_gemm_config(&context);
  if (status != xnn_status_success) {
    return status;
  }

  if (fingerprint_id & xnn_fingerprint_id_helper_nr2) {
    context.group_output_channels = context.gemm_config->nr;
    status = select_gemm_config(&context);
    if (status != xnn_status_success) {
      return status;
    }
  }

  struct fingerprint_context f_context = create_fingerprint_context(fingerprint_id);
  if (f_context.status != xnn_status_uninitialized) {
    if (f_context.status != xnn_status_success) {
      xnn_log_error("Error creating fingerprint context for convolution 2D NHWC: %d.", fingerprint_id);
    }
    return f_context.status;
  }
  struct fingerprint_buffers data = generate_fingerprint_data(variant, &context);
  context.kernel = data.kernel;
  context.bias = data.bias;
  context.kernel_scale = data.kernel_scale;
  context.weights_cache = &f_context.cache;

  // Create a dummy operation.
  status = init_convolution2d_nhwc_context(variant, &context);
  if (status == xnn_status_success) {
    status = create_convolution2d_nhwc(variant, &context, &f_context.op);
    variant->cleanup(variant, &context);
  }
  finalize_fingerprint_context(&f_context);
  xnn_release_simd_memory(data.data);
  return status;
}

enum xnn_status create_convolution2d_nhwc_helper(
    const struct convolution2d_nhwc_variant* variant,
    struct convolution2d_nhwc_context* context,
    xnn_operator_t* convolution_op_out) {
  enum xnn_status status = init_convolution2d_nhwc_context(variant, context);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_fingerprint_convolution2d_nhwc(context->fingerprint_id);
  if (status != xnn_status_success) {
    xnn_log_error("failed fingerprinting %s",
                  xnn_operator_type_to_string(context->operator_type));
    return status;
  }

  status = create_convolution2d_nhwc(variant, context,
                                     convolution_op_out);
  variant->cleanup(variant, context);
  return status;
}

#define CONV2D_NHWC_QX8_FXX_QC8W_PARAMS                                       \
  .input_padding_top = input_padding_top,                                     \
  .input_padding_right = input_padding_right,                                 \
  .input_padding_bottom = input_padding_bottom,                               \
  .input_padding_left = input_padding_left, .kernel_height = kernel_height,   \
  .kernel_width = kernel_width, .subsampling_height = subsampling_height,     \
  .subsampling_width = subsampling_width, .dilation_height = dilation_height, \
  .dilation_width = dilation_width, .groups = groups,                         \
  .group_input_channels = group_input_channels,                               \
  .group_output_channels = group_output_channels,                             \
  .input_channel_stride = input_channel_stride,                               \
  .output_channel_stride = output_channel_stride,                             \
  .kernel_scale = kernel_scale, .kernel = kernel, .bias = bias,               \
  .output_min = output_min, .output_max = output_max, .flags = flags,

enum xnn_status xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {CONV2D_NHWC_QX8_FXX_QC8W_PARAMS};
  context.operator_type = xnn_operator_type_convolution_nhwc_qd8_f16_qc8w;
  context.weights_cache = weights_cache;
  return create_convolution2d_nhwc_helper(&qx8_f16_qc8w_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qdu8_f16_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {CONV2D_NHWC_QX8_FXX_QC8W_PARAMS};
  context.operator_type = xnn_operator_type_convolution_nhwc_qdu8_f16_qc8w;
  context.weights_cache = weights_cache;
  return create_convolution2d_nhwc_helper(&qx8_f16_qc8w_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {CONV2D_NHWC_QX8_FXX_QC8W_PARAMS};
  context.operator_type = xnn_operator_type_convolution_nhwc_qd8_f32_qc8w;
  context.weights_cache = weights_cache;
  return create_convolution2d_nhwc_helper(&qx8_f32_qc8w_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qdu8_f32_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {CONV2D_NHWC_QX8_FXX_QC8W_PARAMS};
  context.operator_type = xnn_operator_type_convolution_nhwc_qdu8_f32_qc8w;
  context.weights_cache = weights_cache;
  return create_convolution2d_nhwc_helper(&qx8_f32_qc8w_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qu8(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, uint8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, float kernel_scale, const uint8_t* kernel,
    const int32_t* bias, uint8_t output_zero_point, float output_scale,
    uint8_t output_min, uint8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale_value = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_qu8,
  };
  return create_convolution2d_nhwc_helper(&qu8_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qs8(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, int8_t input_zero_point, float input_scale,
    float kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_scale_value = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_qs8,
  };
  return create_convolution2d_nhwc_helper(&qs8_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qs8_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_scale = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_qc8,
  };
  return create_convolution2d_nhwc_helper(&qx8_qc8w_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_pqs8_qs8_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_scale = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_pqs8_qs8_qc8w,
  };
  return create_convolution2d_nhwc_helper(&qx8_qc8w_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_pqs8_qs8_qs8(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, int8_t input_zero_point, float input_scale,
    float kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  float* broadcast_kernel_scale =
      xnn_allocate_simd_memory(groups * group_output_channels * sizeof(float));
  if (broadcast_kernel_scale == NULL) {
    xnn_log_error(
        "failed to allocate %zu bytes for %s operator packed weights",
        groups * group_output_channels * sizeof(float),
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8));
    return xnn_status_out_of_memory;
  }
  for (size_t output_channel = 0;
       output_channel < groups * group_output_channels; output_channel++) {
    broadcast_kernel_scale[output_channel] = kernel_scale;
  }
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_scale = broadcast_kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_pqs8_qs8_qc8w,
  };
  enum xnn_status status = create_convolution2d_nhwc_helper(
      &qx8_qc8w_variant, &context, convolution_op_out);
  xnn_release_simd_memory(broadcast_kernel_scale);
  return status;
}

enum xnn_status xnn_create_convolution2d_nhwc_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_f16,
  };
  return create_convolution2d_nhwc_helper(&f16_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_f32,
  };
  return create_convolution2d_nhwc_helper(&f32_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_pf16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_pf16,
  };
  return create_convolution2d_nhwc_helper(&f16_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_pf32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_pf32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .gemm_config = gemm_config,
      .operator_type = xnn_operator_type_convolution_nhwc_pf32,
  };
  return create_convolution2d_nhwc_helper(&f32_variant, &context,
                                          convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_f32_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  // Convert the `f16` kernel and bias to `f32` in temporary buffers.
  const size_t num_kernel_entries = groups * group_input_channels *
                                    group_output_channels * kernel_width *
                                    kernel_height;
  float* fp32_kernel_buffer =
      (float*)xnn_allocate_memory(num_kernel_entries * sizeof(float));
  const void* bias_buffer = NULL;
  float* fp32_bias_buffer = NULL;
  const xnn_float16* f16_kernel = (const xnn_float16*)kernel;
  const xnn_float16* f16_bias = (const xnn_float16*)bias;
  for (size_t i = 0; i < num_kernel_entries; ++i) {
    fp32_kernel_buffer[i] = xnn_float16_to_float(f16_kernel[i]);
  }
  if (bias && !(flags & XNN_FLAG_FP32_STATIC_BIASES)) {
    fp32_bias_buffer = (float*)xnn_allocate_memory(
        groups * group_output_channels * sizeof(float));
    bias_buffer = fp32_bias_buffer;
    for (size_t i = 0; i < groups * group_output_channels; ++i) {
      fp32_bias_buffer[i] = xnn_float16_to_float(f16_bias[i]);
    }
  } else {
    bias_buffer = bias;
  }

  // Delegate creation to the `f32` operator.
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .kernel = fp32_kernel_buffer,
      .bias = bias_buffer,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_f32,
      .kernel_for_cache_key = kernel,
      .bias_for_cache_key = bias,
  };
  enum xnn_status status = create_convolution2d_nhwc_helper(
      &f32_variant, &context, convolution_op_out);

  // Release temporary `f32` buffers.
  xnn_release_memory(fp32_kernel_buffer);
  xnn_release_memory(fp32_bias_buffer);

  return status;
}

enum xnn_status xnn_create_convolution2d_nhwc_pf32_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  // Convert the `f16` kernel and bias to `f32` in temporary buffers.
  const size_t num_kernel_entries = groups * group_input_channels *
                                    group_output_channels * kernel_width *
                                    kernel_height;
  float* fp32_kernel_buffer =
      (float*)xnn_allocate_memory(num_kernel_entries * sizeof(float));
  const void* bias_buffer = NULL;
  float* fp32_bias_buffer = NULL;
  const xnn_float16* f16_kernel = (const xnn_float16*)kernel;
  const xnn_float16* f16_bias = (const xnn_float16*)bias;
  for (size_t i = 0; i < num_kernel_entries; ++i) {
    fp32_kernel_buffer[i] = xnn_float16_to_float(f16_kernel[i]);
  }
  if (bias && !(flags & XNN_FLAG_FP32_STATIC_BIASES)) {
    fp32_bias_buffer = (float*)xnn_allocate_memory(
        groups * group_output_channels * sizeof(float));
    bias_buffer = fp32_bias_buffer;
    for (size_t i = 0; i < groups * group_output_channels; ++i) {
      fp32_bias_buffer[i] = xnn_float16_to_float(f16_bias[i]);
    }
  } else {
    bias_buffer = bias;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_pf32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_pf32));
    return xnn_status_unsupported_hardware;
  }

  // Delegate creation to the `pf32` operator.
  struct convolution2d_nhwc_context context = {
      .input_padding_top = input_padding_top,
      .input_padding_right = input_padding_right,
      .input_padding_bottom = input_padding_bottom,
      .input_padding_left = input_padding_left,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .subsampling_height = subsampling_height,
      .subsampling_width = subsampling_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .gemm_config = gemm_config,
      .groups = groups,
      .group_input_channels = group_input_channels,
      .group_output_channels = group_output_channels,
      .input_channel_stride = input_channel_stride,
      .output_channel_stride = output_channel_stride,
      .kernel = fp32_kernel_buffer,
      .bias = bias_buffer,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_convolution_nhwc_pf32,
      .kernel_for_cache_key = kernel,
      .bias_for_cache_key = bias,
  };
  enum xnn_status status = create_convolution2d_nhwc_helper(
      &f32_variant, &context, convolution_op_out);

  // Release temporary `f32` buffers.
  xnn_release_memory(fp32_kernel_buffer);
  xnn_release_memory(fp32_bias_buffer);

  return status;
}

enum xnn_status xnn_create_fused_convolution2d_nhwc_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    size_t num_post_operations, struct xnn_post_operation* post_operations,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  xnn_log_error(
      "failed to create %s operator: JIT operators are deprecated",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
  return xnn_status_invalid_parameter;
}

static inline bool input_size_changed(xnn_operator_t convolution_op) {
  return convolution_op->convolution_op->input_height !=
             convolution_op->convolution_op->last_input_height ||
         convolution_op->convolution_op->input_width !=
             convolution_op->convolution_op->last_input_width;
}

static enum xnn_status reshape_igemm(
    xnn_operator_t convolution_op, uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size, uint32_t extra_weights_elements_size,
    uint32_t log2_output_element_size, bool dynamic_quantization,
    size_t* workspace_size, size_t num_threads) {
  const size_t batch_size = convolution_op->batch_size;
  const size_t input_height = convolution_op->convolution_op->input_height;
  const size_t input_width = convolution_op->convolution_op->input_width;
  const size_t groups = convolution_op->convolution_op->groups;
  const size_t kernel_height = convolution_op->convolution_op->kernel_height;
  const size_t kernel_width = convolution_op->convolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_height = convolution_op->convolution_op->output_height;
  const size_t output_width = convolution_op->convolution_op->output_width;
  const size_t output_size = output_height * output_width;

  const uint32_t nr = convolution_op->ukernel.igemm->nr;
  struct xnn_hmp_igemm_ukernel* igemm_cases =
      convolution_op->ukernel.igemm->igemm_cases;
  const uint32_t mr = xnn_get_heuristic_mr_igemm(
      output_size, convolution_op->ukernel.igemm->mr, nr, igemm_cases);
  const uint32_t mr_packed = convolution_op->ukernel.igemm->mr_packed;
  const uint32_t kr = convolution_op->ukernel.igemm->kr;
  const uint32_t sr = convolution_op->ukernel.igemm->sr;

  struct xnn_hmp_igemm_ukernel igemm_ukernel = igemm_cases[mr - 1];

  const size_t tiled_output_size = round_up(output_size, mr);
  const size_t indirection_buffer_size =
      sizeof(void*) * kernel_size * tiled_output_size;
  struct compute_parameters* igemm_compute = &convolution_op->compute[0];

  if (convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    *workspace_size = indirection_buffer_size;
    struct compute_parameters* indirection_compute = igemm_compute++;

    convolution_op->dynamic_context.igemm->conv2d_igemm_indirection_init =
        (struct conv2d_igemm_indirection_init_context){
            .zero_buffer = convolution_op->zero_buffer,
            .input_pixel_stride = convolution_op->input_pixel_stride
                                  << log2_input_element_size,
            .input_height = input_height,
            .input_width = input_width,
            .output_height = output_height,
            .output_width = output_width,
            .kernel_height = kernel_height,
            .kernel_width = kernel_width,
            .stride_height = convolution_op->convolution_op->stride_height,
            .stride_width = convolution_op->convolution_op->stride_width,
            .dilation_height = convolution_op->convolution_op->dilation_height,
            .dilation_width = convolution_op->convolution_op->dilation_width,
            .input_padding_top = convolution_op->convolution_op->padding_top,
            .input_padding_left = convolution_op->convolution_op->padding_left,
            .mr = mr,
        };

    indirection_compute->type = xnn_parallelization_type_1d_tile_1d_dynamic;
    indirection_compute->context_offset =
        offsetof(struct igemm_op_context, conv2d_igemm_indirection_init);
    indirection_compute->task_1d_tile_1d_dynamic =
        (pthreadpool_task_1d_tile_1d_dynamic_t)xnn_compute_conv2d_igemm_indirection;
    indirection_compute->range[0] = tiled_output_size;
    indirection_compute->tile[0] = mr;
  } else {
    *workspace_size = 0;

    if (input_size_changed(convolution_op)) {
      const void** indirection_buffer = (const void**)xnn_reallocate_memory(
          (void*)convolution_op->convolution_op->indirection_buffer,
          indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error(
            "failed to allocate %zu bytes for %s operator indirection buffer",
            indirection_buffer_size,
            xnn_operator_type_to_string_v2(convolution_op));
        return xnn_status_out_of_memory;
      }
      convolution_op->convolution_op->indirection_buffer = indirection_buffer;
      xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
                    indirection_buffer_size,
                    xnn_operator_type_to_string_v2(convolution_op));

      // Set a dummy input first, the actual input offset is calculated in setup
      // when we have the input pointer. This offset must be aligned properly
      // because inputs and input offsets need to be aligned.
      convolution_op->convolution_op->input =
          (void*)((uintptr_t)convolution_op->zero_buffer +
                  XNN_ALLOCATION_ALIGNMENT);
      convolution_op->convolution_op->last_input =
          convolution_op->convolution_op->input;
      convolution_op->convolution_op->last_input_height =
          convolution_op->convolution_op->input_height;
      convolution_op->convolution_op->last_input_width =
          convolution_op->convolution_op->input_width;

      xnn_indirection_init_conv2d(
          /*output_tile_size=*/mr,
          /*output_start=*/0,
          /*output_end=*/tiled_output_size,
          convolution_op->convolution_op->indirection_buffer,
          convolution_op->convolution_op->input, convolution_op->zero_buffer,
          convolution_op->input_pixel_stride << log2_input_element_size,
          convolution_op->convolution_op->input_height,
          convolution_op->convolution_op->input_width,
          convolution_op->convolution_op->output_height,
          convolution_op->convolution_op->output_width,
          convolution_op->convolution_op->kernel_height,
          convolution_op->convolution_op->kernel_width,
          convolution_op->convolution_op->stride_height,
          convolution_op->convolution_op->stride_width,
          convolution_op->convolution_op->dilation_height,
          convolution_op->convolution_op->dilation_width,
          convolution_op->convolution_op->padding_top,
          convolution_op->convolution_op->padding_left);
    }
  }

  const size_t group_input_channels =
      convolution_op->convolution_op->group_input_channels;
  const size_t w_stride = extra_weights_elements_size +
                          (round_up_po2(group_input_channels,
                                        convolution_op->ukernel.igemm->kr *
                                            convolution_op->ukernel.igemm->sr) *
                               kernel_size
                           << log2_filter_element_size);
  const size_t group_output_channels =
      convolution_op->convolution_op->group_output_channels;

  const struct xnn_pack_lh_config* packed_lh_config = NULL;
  bool inline_lhs_packing =
      (convolution_op->flags & XNN_FLAG_INLINE_LHS_PACKING) ||
      (igemm_ukernel.packed_lhs_function[XNN_UARCH_DEFAULT] != NULL);
  switch (convolution_op->type) {
    case xnn_operator_type_convolution_nhwc_pqs8_qs8_qc8w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_x8_igemm_pack_lh_config();
      }
      break;
    case xnn_operator_type_convolution_nhwc_pf16:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_x16_igemm_pack_lh_config();
      }
      break;
    case xnn_operator_type_convolution_nhwc_pf32:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_x32_igemm_pack_lh_config();
      }
      break;  
    default:
      break;
  }

  // If we are packing the LHS, provide a per-thread workspace to do so inline.
  size_t workspace_offset = 0;
  size_t per_thread_workspace_size = 0;
  if (packed_lh_config) {
    if (inline_lhs_packing) {
      assert(workspace_size);
      per_thread_workspace_size = packed_lh_config->size_for_igemm_fn(
          mr, /*kc=*/group_input_channels,
          /*ks=*/kernel_size, mr_packed, kr, sr);
      xnn_log_debug("Inlining LHS packing for %s.",
                    xnn_operator_type_to_string(convolution_op->type));
      // We need a buffer for `mr` packed rows for each thread for inlined
      // LHS packing.
      workspace_offset =
          round_up_po2(*workspace_size, XNN_ALLOCATION_ALIGNMENT);
      *workspace_size =
          workspace_offset + num_threads * per_thread_workspace_size;
      log2_input_element_size = packed_lh_config->log2_input_element_size;
      xnn_log_debug(
          "Requesting workspace of size %zu x %zu bytes for LHS packing.",
          num_threads, *workspace_size);
    } else {
      log2_input_element_size = packed_lh_config->log2_packed_element_size;
    }
  }

  convolution_op->dynamic_context.igemm->igemm = (struct igemm_context){
      .ks = kernel_size,
      .ks_scaled = kernel_size * mr * sizeof(void*),
      .kc = group_input_channels << log2_input_element_size,
      .w_stride = w_stride,
      .indirect_a = convolution_op->convolution_op->indirection_buffer,
      .zero = convolution_op->zero_buffer,
      .packed_w = packed_weights(convolution_op),
      .cm_stride = convolution_op->output_pixel_stride
                   << log2_output_element_size,
      .cn_stride = nr << log2_output_element_size,
      .ga_stride = group_input_channels << log2_input_element_size,
      .gw_stride = w_stride * round_up(group_output_channels, nr),
      .gc_stride = group_output_channels << log2_output_element_size,
      .ba_stride =
          input_height * input_width * convolution_op->input_pixel_stride
          << log2_input_element_size,
      .bc_stride = output_size * convolution_op->output_pixel_stride
                   << log2_output_element_size,
      .log2_csize = log2_output_element_size,
      .ukernel = igemm_ukernel,
      .mr = mr,
      .nc = group_output_channels,
      .kr = kr,
      .sr = sr,
      .packed_lh_config = packed_lh_config,
      .workspace_offset = workspace_offset,
      .per_thread_workspace_size = per_thread_workspace_size,
      .mr_packed = mr_packed,
  };
  memcpy(&convolution_op->dynamic_context.igemm->igemm.params,
         &convolution_op->params,
         sizeof(convolution_op->dynamic_context.igemm->igemm.params));

  // Compute the optimal tile size for this iGEMM.
  const size_t nc =
      (packed_lh_config && inline_lhs_packing)
          ? group_output_channels
          : xnn_gemm_best_tile_size(
                groups * batch_size, /*m=*/output_size,
                /*n=*/group_output_channels,
                /*m_stride=*/kernel_size * sizeof(void*) +
                    (input_width * convolution_op->input_pixel_stride
                     << log2_input_element_size),
                /*n_stride=*/
                convolution_op->dynamic_context.igemm->igemm.w_stride,
                /*cn_stride=*/1 << log2_output_element_size, mr, nr,
                num_threads);

  if (dynamic_quantization && convolution_op->convolution_op->zero_size > 0) {
    struct compute_parameters* dq_zero_buffer_compute = igemm_compute++;
    dq_zero_buffer_compute->type = xnn_parallelization_type_1d;
    dq_zero_buffer_compute->task_1d =
        (pthreadpool_task_1d_t)xnn_compute_dq_zero_buffer_igemm;
    dq_zero_buffer_compute->range[0] = batch_size;
  }

  if (groups == 1) {
#if XNN_MAX_UARCH_TYPES > 1
    if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
      if (packed_lh_config && inline_lhs_packing) {
        igemm_compute->type =
            xnn_parallelization_type_2d_tile_1d_dynamic_with_uarch_with_thread;
        igemm_compute->task_2d_tile_1d_dynamic_with_id_with_thread =
            (pthreadpool_task_2d_tile_1d_dynamic_with_id_with_thread_t)
                xnn_compute_batch_hmp_inline_packed_igemm;
      } else if (batch_size > 1) {
        igemm_compute->type =
            xnn_parallelization_type_3d_tile_2d_dynamic_with_uarch;
        if (dynamic_quantization) {
          igemm_compute->task_3d_tile_2d_dynamic_with_id =
              (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
                  xnn_compute_batch_hmp_dqigemm;
        } else {
          igemm_compute->task_3d_tile_2d_dynamic_with_id =
              (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
                  xnn_compute_batch_hmp_igemm;
        }
      } else {
        igemm_compute->type =
            xnn_parallelization_type_2d_tile_2d_dynamic_with_uarch;
        if (dynamic_quantization) {
          igemm_compute->task_2d_tile_2d_dynamic_with_id =
              (pthreadpool_task_2d_tile_2d_dynamic_with_id_t)
                  xnn_compute_hmp_dqigemm;
        } else {
          igemm_compute->task_2d_tile_2d_dynamic_with_id =
              (pthreadpool_task_2d_tile_2d_dynamic_with_id_t)
                  xnn_compute_hmp_igemm;
        }
      }
    } else
#endif  // XNN_MAX_UARCH_TYPES > 1
      if (packed_lh_config && inline_lhs_packing) {
        igemm_compute->type =
            xnn_parallelization_type_2d_tile_1d_dynamic_with_thread;
        igemm_compute->task_2d_tile_1d_dynamic_with_id =
            (pthreadpool_task_2d_tile_1d_dynamic_with_id_t)
                xnn_compute_batch_inline_packed_igemm;
      } else if (batch_size > 1) {
        igemm_compute->type = xnn_parallelization_type_3d_tile_2d_dynamic;
        if (dynamic_quantization) {
          igemm_compute->task_3d_tile_2d_dynamic =
              (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_batch_dqigemm;
        } else {
          igemm_compute->task_3d_tile_2d_dynamic =
              (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_batch_igemm;
        }
      } else {
        igemm_compute->type = xnn_parallelization_type_2d_tile_2d_dynamic;
        if (dynamic_quantization) {
          igemm_compute->task_2d_tile_2d_dynamic =
              (pthreadpool_task_2d_tile_2d_dynamic_t)xnn_compute_dqigemm;
        } else {
          igemm_compute->task_2d_tile_2d_dynamic =
              (pthreadpool_task_2d_tile_2d_dynamic_t)xnn_compute_igemm;
        }
      }
    if (packed_lh_config && inline_lhs_packing) {
      igemm_compute->range[0] = batch_size;
      igemm_compute->range[1] = output_size;
      igemm_compute->tile[0] = mr;
    } else if (batch_size > 1) {
      igemm_compute->range[0] = batch_size;
      igemm_compute->range[1] = group_output_channels;
      igemm_compute->range[2] = output_size;
      igemm_compute->tile[0] = nc;
      igemm_compute->tile[1] = mr;
    } else {
      igemm_compute->range[0] = group_output_channels;
      igemm_compute->range[1] = output_size;
      igemm_compute->tile[0] = nc;
      igemm_compute->tile[1] = mr;
    }
  } else {
#if XNN_MAX_UARCH_TYPES > 1
    if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
      if (packed_lh_config && inline_lhs_packing) {
        igemm_compute->type =
            xnn_parallelization_type_3d_tile_1d_dynamic_with_uarch_with_thread;
        igemm_compute->task_3d_tile_1d_dynamic_with_id_with_thread =
            (pthreadpool_task_3d_tile_1d_dynamic_with_id_with_thread_t)
                xnn_compute_grouped_batch_hmp_inline_packed_igemm;
      } else if (batch_size > 1) {
        igemm_compute->type =
            xnn_parallelization_type_4d_tile_2d_dynamic_with_uarch;
        if (dynamic_quantization) {
          igemm_compute->task_4d_tile_2d_dynamic_with_id =
              (pthreadpool_task_4d_tile_2d_dynamic_with_id_t)
                  xnn_compute_hmp_grouped_batch_dqigemm;
        } else {
          igemm_compute->task_4d_tile_2d_dynamic_with_id =
              (pthreadpool_task_4d_tile_2d_dynamic_with_id_t)
                  xnn_compute_hmp_grouped_batch_igemm;
        }
      } else {
        igemm_compute->type =
            xnn_parallelization_type_3d_tile_2d_dynamic_with_uarch;
        if (dynamic_quantization) {
          igemm_compute->task_3d_tile_2d_dynamic_with_id =
              (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
                  xnn_compute_hmp_grouped_dqigemm;
        } else {
          igemm_compute->task_3d_tile_2d_dynamic_with_id =
              (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
                  xnn_compute_hmp_grouped_igemm;
        }
      }
    } else
#endif  // XNN_MAX_UARCH_TYPES > 1
      if (packed_lh_config && inline_lhs_packing) {
        igemm_compute->type =
            xnn_parallelization_type_3d_tile_1d_dynamic_with_thread;
        igemm_compute->task_3d_tile_1d_dynamic_with_id =
            (pthreadpool_task_3d_tile_1d_dynamic_with_id_t)
                xnn_compute_grouped_batch_inline_packed_igemm;
      } else if (batch_size > 1) {
        igemm_compute->type = xnn_parallelization_type_4d_tile_2d_dynamic;
        if (dynamic_quantization) {
          igemm_compute->task_4d_tile_2d_dynamic =
              (pthreadpool_task_4d_tile_2d_dynamic_t)
                  xnn_compute_grouped_batch_dqigemm;
        } else {
          igemm_compute->task_4d_tile_2d =
              (pthreadpool_task_4d_tile_2d_dynamic_t)
                  xnn_compute_grouped_batch_igemm;
        }
      } else {
        igemm_compute->type = xnn_parallelization_type_3d_tile_2d_dynamic;
        if (dynamic_quantization) {
          igemm_compute->task_3d_tile_2d_dynamic =
              (pthreadpool_task_3d_tile_2d_dynamic_t)
                  xnn_compute_grouped_dqigemm;
        } else {
          igemm_compute->task_3d_tile_2d_dynamic =
              (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_grouped_igemm;
        }
      }
    if (packed_lh_config && inline_lhs_packing) {
      igemm_compute->range[0] = batch_size;
      igemm_compute->range[1] = groups;
      igemm_compute->range[2] = output_size;
      igemm_compute->tile[0] = mr;
    } else if (batch_size > 1) {
      igemm_compute->range[0] = batch_size;
      igemm_compute->range[1] = groups;
      igemm_compute->range[2] = group_output_channels;
      igemm_compute->range[3] = output_size;
      igemm_compute->tile[0] = nc;
      igemm_compute->tile[1] = mr;
    } else {
      igemm_compute->range[0] = groups;
      igemm_compute->range[1] = group_output_channels;
      igemm_compute->range[2] = output_size;
      igemm_compute->tile[0] = nc;
      igemm_compute->tile[1] = mr;
    }
  }
  convolution_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static enum xnn_status reshape_dwconv(
    xnn_operator_t convolution_op, uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size, uint32_t extra_weights_elements_size,
    uint32_t log2_accumulator_element_size, uint32_t log2_output_element_size,
    size_t* workspace_size, size_t num_threads) {
  const size_t input_height = convolution_op->convolution_op->input_height;
  const size_t input_width = convolution_op->convolution_op->input_width;
  const size_t kernel_height = convolution_op->convolution_op->kernel_height;
  const size_t kernel_width = convolution_op->convolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_height = convolution_op->convolution_op->output_height;
  const size_t output_width = convolution_op->convolution_op->output_width;
  const size_t step_width =
      convolution_op->convolution_op->dilation_width == 1
          ? min(convolution_op->convolution_op->stride_width, kernel_width)
          : kernel_width;
  const size_t step_height =
      kernel_size + (output_width - 1) * step_width * kernel_height;
  const struct xnn_ukernel_dwconv dwconv_ukernel =
      convolution_op->ukernel.dwconv;
  const size_t primary_tile = dwconv_ukernel.primary_tile;
  size_t total_workspace_size = 0;

  // Micro-kernel will read (tile_size - kernel_size) elements after the end of
  // indirection buffer.
  const size_t indirection_buffer_size =
      round_up_po2(sizeof(void*) * (primary_tile - kernel_size +
                                    output_height * step_height),
                   XNN_ALLOCATION_ALIGNMENT);

  size_t dwconv_compute_index;
  const bool is_transient_indirection_buffer =
      convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
  if (is_transient_indirection_buffer) {
    total_workspace_size += indirection_buffer_size;
    dwconv_compute_index = 1;

    convolution_op->dynamic_context.dwconv->dwconv_indirection_init =
        (struct dwconv_indirection_init_context){
            .zero_buffer = convolution_op->zero_buffer,
            .input_pixel_stride = convolution_op->input_pixel_stride
                                  << log2_input_element_size,
            .input_height = input_height,
            .input_width = input_width,
            .output_height = output_height,
            .output_width = output_width,
            .kernel_height = kernel_height,
            .kernel_width = kernel_width,
            .stride_height = convolution_op->convolution_op->stride_height,
            .stride_width = convolution_op->convolution_op->stride_width,
            .dilation_height = convolution_op->convolution_op->dilation_height,
            .dilation_width = convolution_op->convolution_op->dilation_width,
            .input_padding_top = convolution_op->convolution_op->padding_top,
            .input_padding_left = convolution_op->convolution_op->padding_left,
            .step_height = step_height,
            .step_width = step_width,
            .tile_size = primary_tile,
        };

    convolution_op->compute[0].type = xnn_parallelization_type_1d_tile_1d_dynamic;
    convolution_op->compute[0].context_offset =
        offsetof(struct dwconv_op_context, dwconv_indirection_init);
    convolution_op->compute[0].task_1d_tile_1d_dynamic =
        (pthreadpool_task_1d_tile_1d_dynamic_t)xnn_compute_dwconv_indirection;
    convolution_op->compute[0].range[0] = output_height;
    convolution_op->compute[0].tile[0] = 1;

  } else {
    dwconv_compute_index = 0;

    if (input_size_changed(convolution_op)) {
      const void** indirection_buffer = (const void**)xnn_reallocate_memory(
          convolution_op->convolution_op->indirection_buffer,
          indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error(
            "failed to allocate %zu bytes for %s operator indirection buffer",
            indirection_buffer_size,
            xnn_operator_type_to_string_v2(convolution_op));
        return xnn_status_out_of_memory;
      }
      convolution_op->convolution_op->indirection_buffer = indirection_buffer;
      xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
                    indirection_buffer_size,
                    xnn_operator_type_to_string_v2(convolution_op));

      // Set a dummy input first, the actual input offset is calculated in setup
      // when we have the input pointer. This offset must be aligned properly
      // because inputs and input offsets need to be aligned.
      convolution_op->convolution_op->input =
          (void*)((uintptr_t)convolution_op->zero_buffer +
                  XNN_ALLOCATION_ALIGNMENT);
      convolution_op->convolution_op->last_input =
          convolution_op->convolution_op->input;
      convolution_op->convolution_op->last_input_height =
          convolution_op->convolution_op->input_height;
      convolution_op->convolution_op->last_input_width =
          convolution_op->convolution_op->input_width;

      xnn_indirection_init_dwconv2d(
          /*output_y_start=*/0,
          /*output_y_end=*/convolution_op->convolution_op->output_height,
          convolution_op->convolution_op->indirection_buffer,
          convolution_op->convolution_op->input,
          convolution_op->input_pixel_stride << log2_input_element_size,
          convolution_op->zero_buffer,
          convolution_op->convolution_op->input_height,
          convolution_op->convolution_op->input_width,
          convolution_op->convolution_op->output_height,
          convolution_op->convolution_op->output_width,
          convolution_op->convolution_op->kernel_height,
          convolution_op->convolution_op->kernel_width,
          convolution_op->convolution_op->stride_height,
          convolution_op->convolution_op->stride_width,
          convolution_op->convolution_op->dilation_height,
          convolution_op->convolution_op->dilation_width,
          convolution_op->convolution_op->padding_top,
          convolution_op->convolution_op->padding_left, step_height, step_width,
          primary_tile);
    }
  }

  const size_t groups = convolution_op->convolution_op->groups;
  convolution_op->dynamic_context.dwconv->dwconv = (struct dwconv_context){
      .kernel_size = kernel_size,
      .indirect_input = convolution_op->convolution_op->indirection_buffer,
      .indirect_input_width_stride =
          (kernel_height * step_width) * sizeof(void*),
      .indirect_input_height_stride = step_height * sizeof(void*),
      .input_batch_stride =
          (input_height * input_width * convolution_op->input_pixel_stride)
          << log2_input_element_size,
      .input_channel_stride = 1 << log2_input_element_size,
      .packed_weights = packed_weights(convolution_op),
      .weights_channel_stride = (primary_tile << log2_filter_element_size) +
                                extra_weights_elements_size,
      .output_batch_stride =
          (output_height * output_width * convolution_op->output_pixel_stride)
          << log2_output_element_size,
      .output_height_stride =
          (output_width * convolution_op->output_pixel_stride)
          << log2_output_element_size,
      .output_pixel_stride = convolution_op->output_pixel_stride
                             << log2_output_element_size,
      .output_channel_stride = 1 << log2_output_element_size,
      .output_height = output_height,
      .output_width = output_width,
      .groups = groups,
      .zero = convolution_op->zero_buffer,
  };
  memcpy(&convolution_op->dynamic_context.dwconv->dwconv.params,
         &convolution_op->params,
         sizeof(convolution_op->dynamic_context.dwconv->dwconv.params));

  const size_t batch_size = convolution_op->batch_size;
  convolution_op->compute[dwconv_compute_index].range[0] = batch_size;
  convolution_op->compute[dwconv_compute_index].range[1] = output_height;
  convolution_op->state = xnn_run_state_needs_setup;

  const size_t channel_tile = convolution_op->ukernel.dwconv.channel_tile;
  // Be defensive against bogus hardware_config cache size info, assume the L1
  // cache is at least 32KB.
  const size_t cache_size =
      max(32768, xnn_init_hardware_config()->l1_data_cache_bytes / 2);
  const size_t output_working_set_per_channel =
      (primary_tile << log2_input_element_size) +
      (primary_tile << log2_filter_element_size) + extra_weights_elements_size +
      (1 << log2_output_element_size);
  const size_t tile_size =
      divide_round_up(cache_size / output_working_set_per_channel,
                      channel_tile) *
      channel_tile;

  convolution_op->compute[dwconv_compute_index].range[2] = groups;
  convolution_op->compute[dwconv_compute_index].tile[0] =
      max(tile_size, channel_tile);
  convolution_op->compute[dwconv_compute_index].type =
      xnn_parallelization_type_3d_tile_1d_dynamic;
  convolution_op->compute[dwconv_compute_index].task_3d_tile_1d_dynamic =
      (pthreadpool_task_3d_tile_1d_dynamic_t)xnn_compute_dwconv_unipass;
  convolution_op->dynamic_context.dwconv->dwconv.ukernel =
      convolution_op->ukernel.dwconv.ukernel;

  *workspace_size = total_workspace_size;

  return xnn_status_success;
}

static enum xnn_status reshape_vmulcaddc(xnn_operator_t convolution_op,
                                         uint32_t log2_input_element_size,
                                         uint32_t log2_output_element_size,
                                         size_t* workspace_size,
                                         size_t num_threads) {
  const size_t batch_output_size =
      convolution_op->batch_size *
      convolution_op->convolution_op->output_height *
      convolution_op->convolution_op->output_width;

  convolution_op->context.vmulcaddc = (struct vmulcaddc_context){
      .n = convolution_op->convolution_op->groups << log2_input_element_size,
      .x_stride = convolution_op->input_pixel_stride << log2_input_element_size,
      .w = packed_weights(convolution_op),
      .y_stride = convolution_op->output_pixel_stride
                  << log2_output_element_size,
      .ukernel = convolution_op->ukernel.vmulcaddc.function,
  };
  memcpy(&convolution_op->context.vmulcaddc.params, &convolution_op->params,
         sizeof(convolution_op->context.vmulcaddc.params));

  size_t mc = batch_output_size;
  if (num_threads > 1) {
    const size_t target_tiles_per_thread = 5;
    const size_t max_mc = divide_round_up(
        batch_output_size, num_threads * target_tiles_per_thread);
    if (max_mc < mc) {
      const uint32_t mr = convolution_op->ukernel.vmulcaddc.mr;
      mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
    }
  }

  convolution_op->compute[0].type = xnn_parallelization_type_1d_tile_1d_dynamic;
  convolution_op->compute[0].task_1d_tile_1d_dynamic =
      (pthreadpool_task_1d_tile_1d_dynamic_t)xnn_compute_vmulcaddc;
  convolution_op->compute[0].range[0] = batch_output_size;
  convolution_op->compute[0].tile[0] = mc;
  convolution_op->state = xnn_run_state_needs_setup;

  *workspace_size = 0;

  return xnn_status_success;
}

static enum xnn_status reshape_convolution2d_nhwc(
    xnn_operator_t convolution_op,
    enum xnn_operator_type expected_operator_type, size_t batch_size,
    size_t input_height, size_t input_width, uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size, uint32_t log2_accumulator_element_size,
    uint32_t extra_weights_elements_size, uint32_t log2_output_element_size,
    bool dynamic_quantization, size_t* workspace_size,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  if (convolution_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(convolution_op));
    return xnn_status_invalid_parameter;
  }
  convolution_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string_v2(convolution_op));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zux%zu input: input dimensions "
        "must be non-zero",
        xnn_operator_type_to_string_v2(convolution_op), input_width,
        input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    convolution_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  convolution_op->batch_size = batch_size;
  convolution_op->convolution_op->input_height = input_height;
  convolution_op->convolution_op->input_width = input_width;

  if (convolution_op->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    convolution_op->convolution_op->output_height =
        compute_output_dimension_with_tf_same_padding(
            input_height, convolution_op->convolution_op->stride_height);
    convolution_op->convolution_op->output_width =
        compute_output_dimension_with_tf_same_padding(
            input_width, convolution_op->convolution_op->stride_width);

    const uint32_t effective_kernel_height =
        (convolution_op->convolution_op->kernel_height - 1) *
            convolution_op->convolution_op->dilation_height +
        1;
    const uint32_t effective_kernel_width =
        (convolution_op->convolution_op->kernel_width - 1) *
            convolution_op->convolution_op->dilation_width +
        1;
    const size_t total_padding_height =
        (convolution_op->convolution_op->output_height - 1) *
            convolution_op->convolution_op->stride_height +
        effective_kernel_height - input_height;
    const size_t total_padding_width =
        (convolution_op->convolution_op->output_width - 1) *
            convolution_op->convolution_op->stride_width +
        effective_kernel_width - input_width;
    convolution_op->convolution_op->padding_top = total_padding_height / 2;
    convolution_op->convolution_op->padding_left = total_padding_width / 2;
    convolution_op->convolution_op->padding_bottom =
        total_padding_height - convolution_op->convolution_op->padding_top;
    convolution_op->convolution_op->padding_right =
        total_padding_width - convolution_op->convolution_op->padding_left;
  } else {
    convolution_op->convolution_op->output_height =
        xnn_compute_convolution_output_dimension(
            convolution_op->convolution_op->padding_top + input_height +
                convolution_op->convolution_op->padding_bottom,
            convolution_op->convolution_op->kernel_height,
            convolution_op->convolution_op->dilation_height,
            convolution_op->convolution_op->stride_height);
    convolution_op->convolution_op->output_width =
        xnn_compute_convolution_output_dimension(
            convolution_op->convolution_op->padding_left + input_width +
                convolution_op->convolution_op->padding_right,
            convolution_op->convolution_op->kernel_width,
            convolution_op->convolution_op->dilation_width,
            convolution_op->convolution_op->stride_width);
  }

  if (output_height_out != NULL) {
    *output_height_out = convolution_op->convolution_op->output_height;
  }
  if (output_width_out != NULL) {
    *output_width_out = convolution_op->convolution_op->output_width;
  }

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  switch (convolution_op->ukernel.type) {
    case xnn_microkernel_type_igemm:
      return reshape_igemm(
          convolution_op, log2_input_element_size, log2_filter_element_size,
          extra_weights_elements_size, log2_output_element_size,
          dynamic_quantization, workspace_size, num_threads);
    case xnn_microkernel_type_dwconv:
      return reshape_dwconv(
          convolution_op, log2_input_element_size, log2_filter_element_size,
          extra_weights_elements_size, log2_accumulator_element_size,
          log2_output_element_size, workspace_size, num_threads);
    case xnn_microkernel_type_vmulcaddc:
      return reshape_vmulcaddc(convolution_op, log2_input_element_size,
                               log2_output_element_size, workspace_size,
                               num_threads);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status reshape_convolution2d_nhwc_qx8_f16_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, enum xnn_operator_type expected_operator_type,
    pthreadpool_t threadpool) {
  convolution_op->convolution_op->last_input_height =
      convolution_op->convolution_op->input_height;
  convolution_op->convolution_op->last_input_width =
      convolution_op->convolution_op->input_width;
  convolution_op->convolution_op->input_height = input_height;
  convolution_op->convolution_op->input_width = input_width;
  if (convolution_op->convolution_op->valid_batch_size != batch_size) {
    if (convolution_op->convolution_op->zero_buffers) {
      for (size_t i = 1; i < convolution_op->convolution_op->valid_batch_size;
           ++i) {
        xnn_release_simd_memory(
            convolution_op->convolution_op->zero_buffers[i]);
      }
    }
    convolution_op->convolution_op->zero_buffers =
        xnn_reallocate_memory(convolution_op->convolution_op->zero_buffers,
                              batch_size * sizeof(void*));
    convolution_op->convolution_op->zero_buffers[0] =
        convolution_op->zero_buffer;
    for (size_t i = 1; i < batch_size; ++i) {
      convolution_op->convolution_op->zero_buffers[i] =
          xnn_allocate_simd_memory(convolution_op->convolution_op->zero_size);
    }
    convolution_op->convolution_op->valid_batch_size = batch_size;
  }
  return reshape_convolution2d_nhwc(
      convolution_op, expected_operator_type, batch_size, input_height,
      input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float) * 2,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*dynamic_quantization=*/true, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc_qx8_f16_qc8w(
      convolution_op, batch_size, input_height, input_width, workspace_size,
      output_height_out, output_width_out,
      xnn_operator_type_convolution_nhwc_qd8_f16_qc8w, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qdu8_f16_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc_qx8_f16_qc8w(
      convolution_op, batch_size, input_height, input_width, workspace_size,
      output_height_out, output_width_out,
      xnn_operator_type_convolution_nhwc_qdu8_f16_qc8w, threadpool);
}

enum xnn_status reshape_convolution2d_nhwc_qx8_f32_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, enum xnn_operator_type expected_operator_type,
    pthreadpool_t threadpool) {
  convolution_op->convolution_op->last_input_height =
      convolution_op->convolution_op->input_height;
  convolution_op->convolution_op->last_input_width =
      convolution_op->convolution_op->input_width;
  convolution_op->convolution_op->input_height = input_height;
  convolution_op->convolution_op->input_width = input_width;
  if (convolution_op->convolution_op->valid_batch_size != batch_size) {
    if (convolution_op->convolution_op->zero_buffers) {
      for (size_t i = 1; i < convolution_op->convolution_op->valid_batch_size;
           ++i) {
        xnn_release_simd_memory(
            convolution_op->convolution_op->zero_buffers[i]);
      }
    }
    convolution_op->convolution_op->zero_buffers =
        xnn_reallocate_memory(convolution_op->convolution_op->zero_buffers,
                              batch_size * sizeof(void*));
    convolution_op->convolution_op->zero_buffers[0] =
        convolution_op->zero_buffer;
    for (size_t i = 1; i < batch_size; ++i) {
      convolution_op->convolution_op->zero_buffers[i] =
          xnn_allocate_simd_memory(convolution_op->convolution_op->zero_size);
    }
    convolution_op->convolution_op->valid_batch_size = batch_size;
  }
  return reshape_convolution2d_nhwc(
      convolution_op, expected_operator_type, batch_size, input_height,
      input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float) * 2,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*dynamic_quantization=*/true, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc_qx8_f32_qc8w(
      convolution_op, batch_size, input_height, input_width, workspace_size,
      output_height_out, output_width_out,
      xnn_operator_type_convolution_nhwc_qd8_f32_qc8w, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qdu8_f32_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc_qx8_f32_qc8w(
      convolution_op, batch_size, input_height, input_width, workspace_size,
      output_height_out, output_width_out,
      xnn_operator_type_convolution_nhwc_qdu8_f32_qc8w, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qu8(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qu8, batch_size,
      input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
      /*extra_weights_elements_size=*/sizeof(int32_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qs8(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qs8, batch_size,
      input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
      /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_qs8_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qc8, batch_size,
      input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
      /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_pqs8_qs8_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_pqs8_qs8_qc8w,
      batch_size, input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
      /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_f16(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_f16, batch_size,
      input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*extra_weights_elements_size=*/sizeof(uint16_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_f32(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_f32, batch_size,
      input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*extra_weights_elements_size=*/sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_pf16(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_pf16, batch_size,
      input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*extra_weights_elements_size=*/sizeof(uint16_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

enum xnn_status xnn_reshape_convolution2d_nhwc_pf32(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  return reshape_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_pf32, batch_size,
      input_height, input_width,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*extra_weights_elements_size=*/sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*dynamic_quantization=*/false, workspace_size, output_height_out,
      output_width_out, threadpool);
}

static enum xnn_status setup_igemm(xnn_operator_t convolution_op,
                                   void* workspace,
                                   uint32_t log2_input_element_size) {
  struct igemm_op_context* igemm_context =
      convolution_op->dynamic_context.igemm;
  if (convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    igemm_context->igemm.a_offset = (size_t)0;
    igemm_context->igemm.indirect_a = (const void**)workspace;
    igemm_context->conv2d_igemm_indirection_init.indirection_buffer =
        (const void**)workspace;
    igemm_context->conv2d_igemm_indirection_init.input =
        convolution_op->convolution_op->input;
  } else {
    igemm_context->igemm.a_offset =
        (size_t)((uintptr_t)convolution_op->convolution_op->input -
                 (uintptr_t)convolution_op->convolution_op->last_input);
  }
  igemm_context->igemm.zero_size = convolution_op->convolution_op->zero_size;
  igemm_context->igemm.zero_buffers =
      convolution_op->convolution_op->zero_buffers;
  igemm_context->igemm.c = convolution_op->convolution_op->output;
  igemm_context->igemm.quantization_params =
      convolution_op->quantization_params;
  igemm_context->igemm.workspace = workspace;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_dwconv(xnn_operator_t convolution_op,
                                    void* workspace,
                                    uint32_t log2_input_element_size) {
  if (convolution_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    convolution_op->dynamic_context.dwconv->dwconv.input_offset = (size_t)0;
    convolution_op->dynamic_context.dwconv->dwconv.indirect_input =
        (const void**)workspace;
    convolution_op->dynamic_context.dwconv->dwconv_indirection_init.input =
        convolution_op->convolution_op->input;
    convolution_op->dynamic_context.dwconv->dwconv_indirection_init
        .indirection_buffer = (const void**)workspace;
  } else {
    convolution_op->dynamic_context.dwconv->dwconv.input_offset =
        (size_t)((uintptr_t)convolution_op->convolution_op->input -
                 (uintptr_t)convolution_op->convolution_op->last_input);
  }

  convolution_op->dynamic_context.dwconv->dwconv.output =
      convolution_op->convolution_op->output;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_vmulcaddc(xnn_operator_t convolution_op) {
  convolution_op->context.vmulcaddc.x = convolution_op->convolution_op->input;
  convolution_op->context.vmulcaddc.y = convolution_op->convolution_op->output;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_convolution2d_nhwc(
    xnn_operator_t convolution_op,
    enum xnn_operator_type expected_operator_type, void* workspace,
    const void* input, void* output, const void* quantization_params,
    uint32_t log2_input_element_size) {
  if (convolution_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(convolution_op));
    return xnn_status_invalid_parameter;
  }

  switch (convolution_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(convolution_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different
      // pointers.
      break;
  }

  if (convolution_op->weights_cache != NULL &&
      !xnn_weights_cache_is_finalized(convolution_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
                  xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_state;
  }

  convolution_op->convolution_op->input = input;
  convolution_op->convolution_op->output = output;
  convolution_op->quantization_params = quantization_params;

  switch (convolution_op->ukernel.type) {
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
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    void* output, const struct xnn_quantization_params* quantization_params) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qd8_f16_qc8w,
      workspace, input, output, quantization_params,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qdu8_f16_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    void* output, const struct xnn_quantization_params* quantization_params) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qdu8_f16_qc8w,
      workspace, input, output, quantization_params,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    float* output, const struct xnn_quantization_params* quantization_params) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qd8_f32_qc8w,
      workspace, input, output, quantization_params,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qdu8_f32_qc8w(
    xnn_operator_t convolution_op, void* workspace, const uint8_t* input,
    float* output, const struct xnn_quantization_params* quantization_params) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qdu8_f32_qc8w,
      workspace, input, output, quantization_params,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qu8(xnn_operator_t convolution_op,
                                                 void* workspace,
                                                 const uint8_t* input,
                                                 uint8_t* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qu8, workspace, input,
      output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qs8(xnn_operator_t convolution_op,
                                                 void* workspace,
                                                 const int8_t* input,
                                                 int8_t* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qs8, workspace, input,
      output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_qs8_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    int8_t* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_qc8, workspace, input,
      output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_pqs8_qs8_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    int8_t* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_pqs8_qs8_qc8w,
      workspace, input, output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T);
}

enum xnn_status xnn_setup_convolution2d_nhwc_f16(xnn_operator_t convolution_op,
                                                 void* workspace,
                                                 const void* input,
                                                 void* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_f16, workspace, input,
      output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT16);
}

enum xnn_status xnn_setup_convolution2d_nhwc_pf16(xnn_operator_t convolution_op,
                                                  void* workspace,
                                                  const void* input,
                                                  void* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_pf16, workspace, input,
      output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT16);
}

enum xnn_status xnn_setup_convolution2d_nhwc_f32(xnn_operator_t convolution_op,
                                                 void* workspace,
                                                 const float* input,
                                                 float* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_f32, workspace, input,
      output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT);
}

enum xnn_status xnn_setup_convolution2d_nhwc_pf32(xnn_operator_t convolution_op,
                                                 void* workspace,
                                                 const float* input,
                                                 float* output) {
  return setup_convolution2d_nhwc(
      convolution_op, xnn_operator_type_convolution_nhwc_pf32, workspace, input,
      output, /*quantization_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT);
}
