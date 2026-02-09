// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/operators/fingerprint_cache.h"
#include "src/operators/fingerprint_id.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/internal.h"
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
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

static float clamp(const float value, const float minimum, const float maximum) {
  const float a = minimum <= value ? value : minimum;
  return maximum >= a ? a : maximum;
}

static enum xnn_operator_type get_operator_type(
    const enum xnn_fingerprint_id fingerprint_id) {
  switch (fingerprint_id) {
#define XNNPACK_FINGERPRINT_TO_OP_TYPE(...)                                 \
  case XNN_EXPAND_TYPES(xnn_fingerprint_id_fully_connected_nc, __VA_ARGS__): \
    return XNN_CONCAT_TYPES(xnn_operator_type_fully_connected_nc, __VA_ARGS__);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(f16);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(pf16);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qd8, f32, qc2w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qd8, f16, qc4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qdu8, f16, qc4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qd8, f16, qb4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qd8, f32, qc4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qdu8, f32, qc4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qp8, f32, qc4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qp8, f32, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qp8, f32, qb4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qd8, f32, qb4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qdu8, f32, qb4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qd8, f32, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qdu8, f32, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qd8, f16, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qdu8, f16, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(bf16, f32);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(f32);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(pf32);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(f32, qc4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(f32, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qs8);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qs8, qc2w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qs8, qc4w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qs8, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(pqs8, qc8w);
    XNNPACK_FINGERPRINT_TO_OP_TYPE(qu8);
  case xnn_fingerprint_id_fully_connected_nc_f32_f32_f32_nr2:
    return xnn_operator_type_fully_connected_nc_f32;
    default:
      return xnn_operator_type_invalid;
  }
}

#undef XNNPACK_OP_TYPE_TO_FINGERPRINT

static enum xnn_status create_fully_connected_nc(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias, uint32_t flags,
    size_t block_size, const uint16_t* blockwise_kernel_scale_params,
    xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio_w,
    xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w, const void* packing_params,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params,
    xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params,
    const float* kernel_scale_params, const void* params, size_t params_size,
    const struct xnn_gemm_config* gemm_config,
    const struct gemm_fused_ukernels* gemm_ukernels,
    const enum xnn_operator_type operator_type,
    xnn_weights_cache_t weights_cache, enum xnn_fingerprint_id fingerprint_id,
    xnn_operator_t* fully_connected_op_out,
    const void* original_kernel_for_cache_key,
    const void* original_bias_for_cache_key) {
  xnn_operator_t fully_connected_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;
  assert(gemm_config);
  const uint32_t log2_filter_element_size = gemm_config->log2_filter_element_size;
  const uint32_t bias_element_size = gemm_config->bias_element_size;
  const bool filter_is_nibble = gemm_config->log2_filter_element_bit_size == XNN_LOG2_BIT_SIZEOF_INT4;
  const bool filter_is_crumb = gemm_config->log2_filter_element_bit_size == XNN_LOG2_BIT_SIZEOF_INT2;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (input_channels == 0) {
    xnn_log_error(
        "failed to create %s operator with %zu input channels: number of "
        "channels must be non-zero",
        xnn_operator_type_to_string(operator_type), input_channels);
    goto error;
  }

  if (output_channels == 0) {
    xnn_log_error(
        "failed to create %s operator with %zu output channels: number of "
        "channels must be non-zero",
        xnn_operator_type_to_string(operator_type), output_channels);
    goto error;
  }

  if (input_stride < input_channels) {
    xnn_log_error(
        "failed to create %s operator with input element stride of %zu: stride "
        "must be at least as large as the number of input channels (%zu)",
        xnn_operator_type_to_string(operator_type), input_stride,
        input_channels);
    goto error;
  }

  if (output_stride < output_channels) {
    xnn_log_error(
        "failed to create %s operator with output element stride of %zu: "
        "stride must be at least as large as the number of output channels "
        "(%zu)",
        xnn_operator_type_to_string(operator_type), output_stride,
        output_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  fully_connected_op =
      xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (fully_connected_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_operator),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }
  fully_connected_op->compute =
      xnn_allocate_zero_memory(2 * sizeof(struct compute_parameters));
  if (fully_connected_op->compute == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct compute_parameters),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }
  fully_connected_op->num_compute_invocations = 1;
  fully_connected_op->convolution_op =
      xnn_allocate_zero_memory(sizeof(struct xnn_convolution_operator));
  if (fully_connected_op->convolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_convolution_operator),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  fully_connected_op->ukernel.gemm_ukernels =
      xnn_allocate_zero_simd_memory(sizeof(struct gemm_types));
  if (fully_connected_op->ukernel.gemm_ukernels == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct gemm_types),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  fully_connected_op->dynamic_context.gemm =
      xnn_allocate_zero_simd_memory(sizeof(struct gemm_op_context));
  if (fully_connected_op->dynamic_context.gemm == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct gemm_op_context),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  fully_connected_op->weights_cache = weights_cache;

  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const uint32_t planes = gemm_config->planes;

  const size_t n_stride = round_up(output_channels, nr);

  size_t k_stride = round_up_po2(input_channels, kr * sr);

  if (filter_is_crumb) {
    if (planes != 4) {
      xnn_log_error(
        "planes is %u but expected to be 4 for 2 bit", planes);
      goto error;
    }
    k_stride = round_up_po2(input_channels, kr * sr * planes);

    // If filter is 2-bit, quarter k_stride (since we will scale k_stride by
    // log2_filter_element_size, and we pass 0 for qc2).
    k_stride = round_up_po2(k_stride, 4) >> 2;
  } else if (filter_is_nibble) {
    input_channels = round_up_po2(input_channels, planes);

    if (planes < 1 || planes > 2) {
      xnn_log_error("planes is %u but expected to be 1 or 2 for 4 bit", planes);
      goto error;
    }
    k_stride = round_up_po2(input_channels, kr * sr * planes);

    // If filter is 4-bit, half k_stride (since we will scale k_stride by
    // log2_filter_element_size, and we pass 0 for qc4).
    k_stride = round_up_po2(k_stride, 2) >> 1;
  }

  size_t block_scale_bytes = 0;
  size_t num_blocks = 0;
  const bool block_wise = (block_size != 0);
  if (block_wise) {
    num_blocks = input_channels / block_size;
    block_scale_bytes += num_blocks * sizeof(uint16_t);
  }

  const size_t weights_stride =
      gemm_config->packed_stride_weights_and_biases
          ? gemm_config->packed_stride_weights_and_biases(
                gemm_config, input_channels, block_size, k_stride,
                extra_weights_bytes)
          : (k_stride << log2_filter_element_size) + bias_element_size +
                extra_weights_bytes + block_scale_bytes;
  const size_t packed_weights_size = n_stride * weights_stride;
  fully_connected_op->weights_stride = weights_stride;
  size_t aligned_total_weights_size =
      round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);

  uint32_t cache_seed = output_channels ^ input_channels ^ nr ^ kr ^ sr ^
                        extra_weights_bytes ^ operator_type;
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    cache_seed = ~cache_seed;
  }
  size_t cache_offset = XNN_CACHE_NOT_FOUND;
  struct xnn_weights_cache_look_up_key cache_key;
  cache_key.seed = cache_seed;
  cache_key.kernel =
      original_kernel_for_cache_key ? original_kernel_for_cache_key : kernel;
  cache_key.bias =
      original_bias_for_cache_key ? original_bias_for_cache_key : bias;
  cache_key.fingerprint_id = fingerprint_id;
  if (use_weights_cache(fully_connected_op)) {
    cache_offset = xnn_weights_cache_look_up(fully_connected_op->weights_cache,
                                             &cache_key);
  }

  if (cache_offset == XNN_CACHE_NOT_FOUND) {
    void* weights_ptr = xnn_get_pointer_to_write_weights(
        fully_connected_op, aligned_total_weights_size);
    if (weights_ptr == NULL) {
      xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_weights_size, xnn_operator_type_to_string(operator_type));
      goto error;
    }
    xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                  aligned_total_weights_size,
                  xnn_operator_type_to_string(operator_type));
    if (extra_weights_bytes > 0) {
      // TODO(b/402602597): We shouldn't need this initialization.
      memset(weights_ptr, 0, aligned_total_weights_size);
    }

    if (gemm_config->pack_weights_and_biases) {
      gemm_config->pack_weights_and_biases(
          flags, gemm_config, input_channels, output_channels,
          /*groups=*/1,
          /*block_wise=*/block_size,
          /*k_stride=*/
          (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) ? output_channels
                                               : input_channels,
          /*accumulator_init=*/bias,
          /*weights=*/kernel,
          /*int_extra_data0_fn=*/(xnn_init_scale_params_fn)init_scale_params,
          /*extra_data0=*/scale_params,
          /*extra_data0_size=*/init_scale_params != NULL ? sizeof(float) : 0,
          /*init_extra_data1_fn=*/
          (xnn_init_scale_params_fn)init_kernel_scale_params,
          /*extra_data1=*/
          block_wise ? (const void*)blockwise_kernel_scale_params
                     : (const void*)kernel_scale_params,
          /*extra_data1_size=*/init_kernel_scale_params != NULL ? sizeof(float)
                                                                : 0,
          /*packed_weights_ptr=*/weights_ptr, packing_params);
    } else {
      if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
        pack_gemm_gio_w(
            /*groups=*/1, output_channels, input_channels, nr, kr, sr,
            output_channels, kernel, bias, /*scale=*/NULL, weights_ptr,
            nr * extra_weights_bytes, packing_params);
      } else {
        pack_gemm_goi_w(
            /*groups=*/1, output_channels, input_channels, nr, kr, sr, kernel,
            bias, /*scale=*/NULL, weights_ptr, nr * extra_weights_bytes,
            packing_params);
      }
      if (kernel_scale_params != NULL) {
        assert(init_kernel_scale_params != NULL);

        void* weights =
            (void*)((uintptr_t)weights_ptr +
                    nr * ((k_stride << log2_filter_element_size) +
                        bias_element_size));
        init_kernel_scale_params(output_channels, nr, nr * weights_stride,
                                 kernel_scale_params, weights);
      }

      if (scale_params != NULL) {
        assert(init_scale_params != NULL);
        void* weights =
            (void*)((uintptr_t)weights_ptr +
                    nr * ((k_stride << log2_filter_element_size) +
                        bias_element_size));
        if (kernel_scale_params != NULL) {
          weights = (void*)((uintptr_t)weights + nr * sizeof(float));
        }
        init_scale_params(output_channels, nr, nr * weights_stride,
                          scale_params, weights);
      }
    }

    if (use_weights_cache(fully_connected_op)) {
      fully_connected_op->packed_weights.offset =
          xnn_look_up_or_insert_weights_cache(fully_connected_op->weights_cache,
                                              &cache_key, weights_ptr,
                                              aligned_total_weights_size);
    }
  } else {
    fully_connected_op->packed_weights.offset = cache_offset;
  }

  fully_connected_op->convolution_op->group_input_channels = input_channels;
  fully_connected_op->convolution_op->group_output_channels = output_channels;
  fully_connected_op->input_pixel_stride = input_stride;
  fully_connected_op->output_pixel_stride = output_stride;

  memcpy(&fully_connected_op->params, params, params_size);
  fully_connected_op->type = operator_type;
  fully_connected_op->flags = flags;

  const size_t mr = gemm_config->mr;
  const uint32_t mr_packed =
      gemm_config->mr_packed ? gemm_config->mr_packed : gemm_config->mr;
  fully_connected_op->ukernel.type = xnn_microkernel_type_gemm;
  fully_connected_op->ukernel.gemm_ukernels->gemm = (struct xnn_ukernel_gemm){
      .mr = mr,
      .nr = nr,
      .kr = kr,
      .sr = sr,
      .kp = planes,
      .mr_packed = mr_packed,
  };
  assert(XNN_MAX_MR >= mr);
  for (size_t i = 0; i < mr; i++) {
    fully_connected_op->ukernel.gemm_ukernels->gemm.gemm_cases[i] =
        gemm_ukernels->gemm[i];
  }
  fully_connected_op->gemm_config = gemm_config;

  fully_connected_op->state = xnn_run_state_invalid;

  *fully_connected_op_out = fully_connected_op;
  return xnn_status_success;

error:
  xnn_delete_operator(fully_connected_op);
  return status;
}

struct fc_context {
  // Parameters.
  size_t input_channels;
  size_t output_channels;
  size_t input_stride;
  size_t output_stride;
  int8_t input_zero_point;
  float input_scale;
  uint8_t kernel_zero_point;
  union {
    const float* f32;
    const uint16_t* bf16;
  } kernel_scale;
  float kernel_scale_value;
  const void* kernel;
  const void* bias;
  int8_t output_zero_point;
  float output_scale;
  float output_min;
  float output_max;
  uint32_t flags;
  bool should_fingerprint;

  // Create function parameters.
  size_t block_size;
  const uint16_t* blockwise_kernel_scale_params;
  xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio_w;
  xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w;
  const void* packing_params;
  xnn_init_qs8_qc8w_scale_params_fn init_scale_params;
  const float* scale_params;
  xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params;
  const float* kernel_scale_params;
  union {
    struct xnn_f16_minmax_params f16;
    struct xnn_f16_qc4w_minmax_params f16_qc4w;
    struct xnn_f16_qb4w_minmax_params f16_qb4w;
    struct xnn_f32_minmax_params f32;
    struct xnn_f32_qc4w_minmax_params f32_qc4w;
    struct xnn_f32_qb4w_minmax_params f32_qb4w;
    union xnn_qs8_qc8w_conv_minmax_params qs8_qc8w;
    union xnn_qu8_conv_minmax_params qu8;
  } params;
  size_t params_size;
  const struct xnn_gemm_config* gemm_config;
  const struct gemm_fused_ukernels* gemm_ukernels;
  const enum xnn_operator_type operator_type;
  xnn_weights_cache_t weights_cache;
  xnn_operator_t* fully_connected_op_out;

  // State
  enum xnn_fingerprint_id fingerprint_id;
  const xnn_float16 fp16_output_min;
  const xnn_float16 fp16_output_max;
  void* kernel_scales;
  const void* kernel_zero_points;
  union {
    struct xnn_qd8_qc2w_packing_params qd8_qc2w;
    struct xnn_qs8_qc2w_packing_params qs8_qc2w;
    struct xnn_qs8_qc4w_packing_params qs8_qc4w;
    struct xnn_qs8_qc8w_packing_params qs8_qc8w;
    struct xnn_qs8_packing_params qs8;
    struct xnn_qu8_packing_params qu8;
  } packing_params_data;
  union {
    float f32_value;
    float* f32;
  } requantization_scale;

  // These pointers are used to keep track of the original data address when
  // doing on-the-fly conversions (for instance to fp16) within the create
  // functions.
  //
  // They are used to compute the cache_key to recover the link of the original
  // buffer in the cache.
  const void* original_kernel;
  const void* original_bias;

  void* fingerprint_data_to_release;
  void* requantization_scale_to_release;
};


struct fc_variant;

static enum xnn_status check_input_scale_value(const struct fc_variant* variant,
                                               struct fc_context* context) {
  if (context->input_scale <= 0.0f || !isnormal(context->input_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(context->operator_type),
        context->input_scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_output_scale_value(
    const struct fc_variant* variant, struct fc_context* context) {
  if (context->output_scale <= 0.0f || !isnormal(context->output_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(context->operator_type),
        context->output_scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_kernel_scale_value(
    const struct fc_variant* variant, struct fc_context* context) {
  if (context->kernel_scale_value <= 0.0f ||
      !isnormal(context->kernel_scale_value)) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(context->operator_type),
        context->kernel_scale_value);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_output_bounds_f32(const struct fc_variant* variant,
                                               struct fc_context* context) {
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
        "bound must be below upper bound",
        xnn_operator_type_to_string(context->operator_type),
        context->output_min, context->output_max);
    return xnn_status_invalid_parameter;
  }

  return xnn_status_success;
}

static enum xnn_status check_kernel_zero_point_is_0_or_8_qu8(
    const struct fc_variant* variant, struct fc_context* context) {
  if (context->kernel_zero_point != 8 && context->kernel_zero_point != 0) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equals to 8 "
                  "(unsigned weights) or 0 (signed weights)",
                  xnn_operator_type_to_string(context->operator_type),
                  context->kernel_zero_point);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_kernel_zero_point_is_8_qu8(
    const struct fc_variant* variant, struct fc_context* context) {
  if (context->kernel_zero_point != 8) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equals to 8",
                  xnn_operator_type_to_string(context->operator_type),
                  context->kernel_zero_point);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_block_size(const struct fc_variant* variant,
                                        struct fc_context* context) {
  if (context->block_size < XNN_MIN_BLOCKSIZE ||
      context->block_size % XNN_MIN_BLOCKSIZE != 0) {
    xnn_log_error(
        "failed to create %s operator with block_size: %zu: expecting "
        "block_size to be a multiple of %d.",
        xnn_operator_type_to_string(context->operator_type),
        context->block_size, XNN_MIN_BLOCKSIZE);
    return xnn_status_invalid_parameter;
  }

  if (context->input_channels % context->block_size != 0) {
    xnn_log_error(
        "failed to create %s operator with input_channels: %zu, and "
        "block_size: %zu: expecting input_channels %% block_size == 0.",
        xnn_operator_type_to_string(context->operator_type),
        context->input_channels, context->block_size);
    return xnn_status_invalid_parameter;
  }

  // Assuming kernel_scale.size() is output_channels * num_blocks.
  size_t num_blocks = context->input_channels / context->block_size;
  for (size_t output_channel = 0; output_channel < context->output_channels;
       output_channel++) {
    for (size_t block_index = 0; block_index < num_blocks; block_index++) {
      const size_t scale_index = output_channel * num_blocks + block_index;
      const float fp32_scale = math_cvt_fp32_bf16(context->kernel_scale.bf16[scale_index]);
      if (fp32_scale <= 0.0f || !isnormal(fp32_scale)) {
        xnn_log_error(
            "failed to create %s operator with %.7g kernel scale in output "
            "channel #%zu, block #%zu: scale must be finite and positive",
            xnn_operator_type_to_string(context->operator_type), fp32_scale,
            output_channel, block_index);
        return xnn_status_invalid_parameter;
      }
    }
  }
  context->blockwise_kernel_scale_params = context->kernel_scale.bf16;
  return xnn_status_success;
}

static enum xnn_status check_no_transpose_flag(const struct fc_variant* variant,
                                               struct fc_context* context) {
  if (context->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    xnn_log_error(
        "failed to create %s operator with XNN_FLAG_TRANSPOSE_WEIGHTS: not "
        "supported",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status check_kernel_scale_f32(const struct fc_variant* variant,
                                               struct fc_context* context) {
  for (size_t output_channel = 0; output_channel < context->output_channels;
       output_channel++) {
    if (context->kernel_scale.f32[output_channel] <= 0.0f ||
        !isnormal(context->kernel_scale.f32[output_channel])) {
      xnn_log_error(
          "failed to create %s operator with %.7g kernel scale in output "
          "channel #%zu: scale must be finite and positive",
          xnn_operator_type_to_string(context->operator_type),
          context->kernel_scale.f32[output_channel], output_channel);
      return xnn_status_invalid_parameter;
    }
  }
  return xnn_status_success;
}

static enum xnn_status setup_gemm_ukernels(const struct fc_variant* variant,
                                           struct fc_context* context) {
  const bool linear_activation = (context->output_max == INFINITY) &&
                                 (context->output_min == -context->output_max);
  if (linear_activation &&
      context->gemm_config->linear.gemm[context->gemm_config->mr - 1]
              .function[XNN_UARCH_DEFAULT] != NULL) {
    context->gemm_ukernels = &context->gemm_config->linear;
  }
  return xnn_status_success;
}

static enum xnn_status setup_params_f16(const struct fc_variant* variant,
                                        struct fc_context* context) {
  const xnn_float16 fp16_output_min =
      xnn_float16_from_float(context->output_min);
  const xnn_float16 fp16_output_max =
      xnn_float16_from_float(context->output_max);
  if XNN_LIKELY (context->gemm_config->init.f16 != NULL) {
    context->gemm_config->init.f16(&context->params.f16, fp16_output_min,
                                   fp16_output_max);
  }
  context->params_size = sizeof(context->params.f16);
  return xnn_status_success;
}

static enum xnn_status setup_params_f16_qc4w(const struct fc_variant* variant,
                                             struct fc_context* context) {
  const xnn_float16 fp16_output_min =
      xnn_float16_from_float(context->output_min);
  const xnn_float16 fp16_output_max =
      xnn_float16_from_float(context->output_max);
  if XNN_LIKELY (context->gemm_config->init.f16_qc4w != NULL) {
    context->gemm_config->init.f16_qc4w(&context->params.f16_qc4w,
                                        fp16_output_min, fp16_output_max,
                                        context->kernel_zero_point);
  }
  context->params_size = sizeof(context->params.f16_qc4w);
  return xnn_status_success;
}

static enum xnn_status setup_params_f16_qb4w(const struct fc_variant* variant,
                                             struct fc_context* context) {
  const xnn_float16 fp16_output_min =
      xnn_float16_from_float(context->output_min);
  const xnn_float16 fp16_output_max =
      xnn_float16_from_float(context->output_max);
  if XNN_LIKELY (context->gemm_config->init.f16_qb4w != NULL) {
    context->gemm_config->init.f16_qb4w(
        &context->params.f16_qb4w, fp16_output_min, fp16_output_max,
        context->kernel_zero_point, context->block_size);
  }
  context->params_size = sizeof(context->params.f16_qb4w);
  return xnn_status_success;
}

static enum xnn_status setup_params_f32(const struct fc_variant* variant,
                                        struct fc_context* context) {
  if XNN_LIKELY (context->gemm_config->init.f32 != NULL) {
    context->gemm_config->init.f32(&context->params.f32, context->output_min,
                                   context->output_max);
  }
  context->params_size = sizeof(context->params.f32);
  return xnn_status_success;
}

static enum xnn_status setup_params_f32_qc4w(const struct fc_variant* variant,
                                             struct fc_context* context) {
  if XNN_LIKELY (context->gemm_config->init.f32_qc4w != NULL) {
    context->gemm_config->init.f32_qc4w(
        &context->params.f32_qc4w, context->output_min, context->output_max,
        context->kernel_zero_point);
  }
  context->params_size = sizeof(context->params.f32_qc4w);
  return xnn_status_success;
}

static enum xnn_status setup_params_f32_qb4w(const struct fc_variant* variant,
                                             struct fc_context* context) {
  if XNN_LIKELY (context->gemm_config->init.f32_qb4w != NULL) {
    context->gemm_config->init.f32_qb4w(
        &context->params.f32_qb4w, context->output_min, context->output_max,
        context->kernel_zero_point, context->block_size);
  }
  context->params_size = sizeof(context->params.f32_qb4w);
  return xnn_status_success;
}

static enum xnn_status setup_params_qs8_qc8w(const struct fc_variant* variant,
                                             struct fc_context* context) {
  context->output_min = clamp(context->output_min, INT8_MIN, INT8_MAX);
  context->output_max = clamp(context->output_max, INT8_MIN, INT8_MAX);
  if XNN_LIKELY (context->gemm_config->init.qs8_qc8w != NULL) {
    context->gemm_config->init.qs8_qc8w(
        &context->params.qs8_qc8w, context->output_zero_point,
        context->output_min, context->output_max);
  }
  context->params_size = sizeof(context->params.qs8_qc8w);
  return xnn_status_success;
}

static enum xnn_status setup_params_qu8(const struct fc_variant* variant,
                                        struct fc_context* context) {
  const float requantization_scale = context->input_scale *
                                     context->kernel_scale_value /
                                     context->output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel "
        "scale, and %.7g output scale: requantization scale %.7g is greater or "
        "equal to 256.0",
        xnn_operator_type_to_string(context->operator_type),
        context->input_scale, context->kernel_scale_value,
        context->output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  context->output_min = clamp(context->output_min, 0, UINT8_MAX);
  context->output_max = clamp(context->output_max, 0, UINT8_MAX);
  if XNN_LIKELY (context->gemm_config->init.qu8 != NULL) {
    context->gemm_config->init.qu8(&context->params.qu8, context->kernel_zero_point,
                          requantization_scale, context->output_zero_point,
                          context->output_min, context->output_max);
  }
  context->params_size = sizeof(context->params.qu8);
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qs8_qc2w(
    const struct fc_variant* variant, struct fc_context* context) {
  context->packing_params_data.qs8_qc2w = (struct xnn_qs8_qc2w_packing_params){
      .input_zero_point = context->input_zero_point, .kernel_zero_point = 0};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qd8_qc2w(
    const struct fc_variant* variant, struct fc_context* context) {
  context->packing_params_data.qd8_qc2w = (struct xnn_qd8_qc2w_packing_params){
      .input_zero_point = 1, .kernel_zero_point = context->kernel_zero_points};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qs8_qc4w(
    const struct fc_variant* variant, struct fc_context* context) {
  context->packing_params_data.qs8_qc4w = (struct xnn_qs8_qc4w_packing_params){
      .input_zero_point = context->input_zero_point,
      .kernel_zero_point = context->kernel_zero_point};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qs8_qc4w_izp_1(
    const struct fc_variant* variant, struct fc_context* context) {
  context->packing_params_data.qs8_qc4w = (struct xnn_qs8_qc4w_packing_params){
      .input_zero_point = 1, context->kernel_zero_point};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qs8(
    const struct fc_variant* variant, struct fc_context* context) {
  context->packing_params_data.qs8 = (struct xnn_qs8_packing_params){
      .input_zero_point = context->input_zero_point};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qs8_izp1(
    const struct fc_variant* variant, struct fc_context* context) {
  context->packing_params_data.qs8 =
      (struct xnn_qs8_packing_params){.input_zero_point = 1};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qs8_qc8w(
    const struct fc_variant* variant, struct fc_context* context) {
  // For qp__f32_qc8w, we don't know input zero point until runtime, row sum is
  // multiplied by it during packing, so set it to 1.
  context->packing_params_data.qs8_qc8w = (struct xnn_qs8_qc8w_packing_params){
      .input_zero_point = 1, .scale_multiplier = 1.0f};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_params_qu8(
    const struct fc_variant* variant, struct fc_context* context) {
  context->packing_params_data.qu8 = (struct xnn_qu8_packing_params){
      .input_zero_point = context->input_zero_point,
      .kernel_zero_point = context->kernel_zero_point};
  context->packing_params = &context->packing_params_data;
  return xnn_status_success;
}

static enum xnn_status setup_packing_functions_f16(const struct fc_variant* variant,
                                        struct fc_context* context) {
   context->pack_gemm_gio_w =
      (xnn_packw_gemm_gio_ukernel_fn)context->gemm_config->pack_gemm_gio;
  context->pack_gemm_goi_w =
      (xnn_packw_gemm_goi_ukernel_fn)context->gemm_config->pack_gemm_goi;
  if (context->flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    context->pack_gemm_gio_w =
        (xnn_packw_gemm_gio_ukernel_fn)xnn_pack_f32_to_f16_gemm_gio_w;
    context->pack_gemm_goi_w =
        (xnn_packw_gemm_goi_ukernel_fn)xnn_pack_f32_to_f16_gemm_goi_w;
  }
  return xnn_status_success;
}

static enum xnn_status setup_packing_functions_from_gemm_config(
    const struct fc_variant* variant, struct fc_context* context) {
  context->pack_gemm_gio_w =
      (xnn_packw_gemm_gio_ukernel_fn)context->gemm_config->pack_gemm_gio;
  context->pack_gemm_goi_w =
      (xnn_packw_gemm_goi_ukernel_fn)context->gemm_config->pack_gemm_goi;
  return xnn_status_success;
}

static enum xnn_status setup_scale_params_qs8_qc8w(
    const struct fc_variant* variant, struct fc_context* context) {
  context->scale_params = context->bias;
  context->bias = NULL;
  context->init_scale_params = xnn_init_qs8_qc8w_scale_fp32_params;
  context->init_kernel_scale_params = xnn_init_qs8_qc8w_scale_fp32_params;
  context->kernel_scale_params = context->kernel_scale.f32;
  return xnn_status_success;
}

static enum xnn_status setup_scale_params_qs8_qc2w(
    const struct fc_variant* variant, struct fc_context* context) {
  context->scale_params = context->bias;
  context->bias = NULL;
  context->kernel_scale_params = context->kernel_scale.f32;
  return xnn_status_success;
}

static enum xnn_status setup_scale_params_f32_qcxw(
    const struct fc_variant* variant, struct fc_context* context) {
  context->scale_params = context->kernel_scale.f32;
  context->init_scale_params = xnn_init_qs8_qc8w_scale_fp32_params;
  return xnn_status_success;
}

static enum xnn_status setup_scale_params_qs8(const struct fc_variant* variant,
                                              struct fc_context* context) {
  context->requantization_scale.f32_value = context->input_scale *
                                  context->kernel_scale_value /
                                  context->output_scale;
  if (context->requantization_scale.f32_value >= 256.0f) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel "
        "scale, and %.7g output scale: requantization scale %.7g is greater or "
        "equal to 256.0",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8),
        context->input_scale, context->kernel_scale_value,
        context->output_scale, context->requantization_scale.f32_value);
    return xnn_status_unsupported_parameter;
  }

  context->scale_params = &context->requantization_scale.f32_value;
  context->init_scale_params = xnn_init_qs8_to_qs8_qc8w_scale_fp32_params;
  return xnn_status_success;
}

static enum xnn_status setup_scale_params_qx8_qcyw(const struct fc_variant* variant,
                                              struct fc_context* context) {
  context->requantization_scale.f32 = xnn_allocate_simd_memory(context->output_channels * sizeof(float));
  context->requantization_scale_to_release = context->requantization_scale.f32;
  if (context->requantization_scale.f32 == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator packed weights",
                  context->output_channels * sizeof(float),
                  xnn_operator_type_to_string(context->operator_type));
    return xnn_status_out_of_memory;
  }

  for (size_t output_channel = 0; output_channel < context->output_channels;
       output_channel++) {
    context->requantization_scale.f32[output_channel] =
        context->input_scale * context->kernel_scale.f32[output_channel] / context->output_scale;
    if (context->requantization_scale.f32[output_channel] >= 256.0f) {
      xnn_log_error(
          "failed to create %s operator with %.7g input scale, %.7g kernel "
          "scale, and %.7g output scale in output channel #%zu: requantization "
          "scale %.7g is greater or equal to 256.0",
          xnn_operator_type_to_string(context->operator_type), context->input_scale,
          context->kernel_scale.f32[output_channel], context->output_scale, output_channel,
          context->requantization_scale.f32[output_channel]);
      return xnn_status_unsupported_parameter;
    }
  }

  context->scale_params = context->requantization_scale.f32;
  context->init_scale_params = xnn_init_qs8_qc8w_scale_fp32_params;
  return xnn_status_success;
}

static enum xnn_status UNUSED(const struct fc_variant* variant,
                              struct fc_context* context) {
  return xnn_status_success;
}

static void cleanup_context(const struct fc_variant* variant, struct fc_context* context) {
  xnn_release_simd_memory(context->fingerprint_data_to_release);
  xnn_release_simd_memory(context->requantization_scale_to_release);
}

static enum xnn_status force_coherent_kernel_scale_values_bf16(const struct fc_variant* variant,
                              struct fc_context* context) {
  // We cast the `const` away because we know that the data was created for the
  // fingerprinting and that its safe to modify.
  uint16_t* kernel_scale = (uint16_t*)(uintptr_t)context->kernel_scale.bf16;
  for (size_t i = 0; i < context->output_channels; ++i) {
    kernel_scale[i] =
        math_cvt_bf16_fp32(0.5 + ((float)i) / context->output_channels / 3);
  }
  return xnn_status_success;
}

static enum xnn_status force_coherent_kernel_scale_values_f32(const struct fc_variant* variant,
                              struct fc_context* context) {
  // We cast the `const` away because we know that the data was created for the
  // fingerprinting and that its safe to modify.
  float* kernel_scale = (float*)(uintptr_t)context->kernel_scale.f32;
  for (size_t i = 0; i < context->output_channels; ++i) {
    kernel_scale[i] = 0.5 + ((float)i) / context->output_channels / 3;
  }
  return xnn_status_success;
}

static enum xnn_status force_coherent_bias_values_i32(
    const struct fc_variant* variant, struct fc_context* context) {
  // We cast the `const` away because we know that the data was created for the
  // fingerprinting and that its safe to modify.
  int32_t* bias = (int32_t*)(uintptr_t)context->bias;
  for (size_t i = 0; i < context->output_channels; ++i) {
    // Set the topmost 5 bits to 0.
    bias[i] = bias[i] & 0x07ffffff;
  }
  return xnn_status_success;
}

static enum xnn_status set_min_block_size(const struct fc_variant* variant,
                              struct fc_context* context) {
  context->block_size = XNN_MIN_BLOCKSIZE;
  return xnn_status_success;
}

static enum xnn_status set_kernel_zero_point_to_8(const struct fc_variant* variant,
                              struct fc_context* context) {
  context->kernel_zero_point = 8;
  return xnn_status_success;
}

typedef enum xnn_status (*fc_setup_function)(const struct fc_variant*,
                                             struct fc_context*);

#define XNN_FC_VARIANT_FINGERPRINT_CONSTRAINT_MAX_COUNT 8

struct fc_variant {
  // Functions.
  fc_setup_function check_output_bounds;
  fc_setup_function check_kernel_zero_point;
  fc_setup_function check_block_size;
  fc_setup_function check_flags;
  fc_setup_function check_input_scale;
  fc_setup_function check_kernel_scale;
  fc_setup_function check_output_scale;
  fc_setup_function setup_gemm_ukernels;
  fc_setup_function setup_params;
  fc_setup_function setup_packing_params;
  fc_setup_function setup_packing_functions;
  fc_setup_function setup_scale_params;
  // These functions will enforce constraints on the data that is generated to
  // create a fingerprint.
  fc_setup_function
      fingerprint_constraints[XNN_FC_VARIANT_FINGERPRINT_CONSTRAINT_MAX_COUNT];
  // Constants.
  size_t extra_weights_bytes;
  size_t kernel_scale_element_size;
};

static const struct fc_variant f16_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = UNUSED,
    .setup_params = setup_params_f16,
    .setup_packing_params = UNUSED,
    .setup_packing_functions = setup_packing_functions_f16,
    .setup_scale_params = UNUSED,
    .fingerprint_constraints = {},
    .extra_weights_bytes = 0,
    .kernel_scale_element_size = 0,
};

static const struct fc_variant qx8_f16_qc4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = check_kernel_zero_point_is_0_or_8_qu8,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f16_qc4w,
    .setup_packing_params = setup_packing_params_qs8_qc4w_izp_1,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8_qc8w,
    .fingerprint_constraints = {},
    .extra_weights_bytes = sizeof(float) * 2,
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant qx8_f32_qc4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = check_kernel_zero_point_is_0_or_8_qu8,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32_qc4w,
    .setup_packing_params = setup_packing_params_qs8_qc4w_izp_1,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8_qc8w,
    .fingerprint_constraints = {},
    .extra_weights_bytes = sizeof(float) * 2,
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant qp8_f32_qc4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = check_kernel_zero_point_is_0_or_8_qu8,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32,
    .setup_packing_params = setup_packing_params_qs8_qc4w_izp_1,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8_qc8w,
    .fingerprint_constraints = {},
    .extra_weights_bytes = sizeof(float) * 2,
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant qp8_f32_qc8w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32,
    .setup_packing_params = setup_packing_params_qs8_qc8w,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8_qc8w,
    .fingerprint_constraints = {},
    .extra_weights_bytes = sizeof(float) * 2,
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant qd8_f16_qb4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = check_kernel_zero_point_is_0_or_8_qu8,
    .check_block_size = check_block_size,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f16_qb4w,
    .setup_packing_params = setup_packing_params_qs8_qc4w_izp_1,
    .setup_packing_functions = UNUSED,
    .setup_scale_params = UNUSED,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_bf16,
                                set_min_block_size},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = sizeof(uint16_t),
};

static const struct fc_variant qx8_f32_qb4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = check_kernel_zero_point_is_0_or_8_qu8,
    .check_block_size = check_block_size,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32_qb4w,
    .setup_packing_params = setup_packing_params_qs8_qc4w_izp_1,
    .setup_packing_functions = UNUSED,
    .setup_scale_params = UNUSED,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_bf16,
                                set_min_block_size},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = sizeof(uint16_t),
};

static const struct fc_variant qp8_f32_qb4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = check_kernel_zero_point_is_8_qu8,
    .check_block_size = check_block_size,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32_qb4w,
    .setup_packing_params = setup_packing_params_qs8_qc4w_izp_1,
    .setup_packing_functions = UNUSED,
    .setup_scale_params = UNUSED,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_bf16,
                                set_min_block_size, set_kernel_zero_point_to_8},
    .extra_weights_bytes = 0,
    .kernel_scale_element_size = sizeof(uint16_t),
};

static const struct fc_variant qd8_f32_qc2w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32,
    .setup_packing_params = setup_packing_params_qd8_qc2w,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8_qc2w,
    .fingerprint_constraints = {},
    .extra_weights_bytes = sizeof(float) * 2,
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant qdx8_f32_qc8w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32,
    .setup_packing_params = setup_packing_params_qs8_izp1,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8_qc8w,
    .fingerprint_constraints = {},
    .extra_weights_bytes = sizeof(float) * 2,
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant qdx8_f16_qc8w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f16,
    .setup_packing_params = setup_packing_params_qs8_izp1,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8_qc8w,
    .fingerprint_constraints = {},
    .extra_weights_bytes = sizeof(float) * 2,
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant f32_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32,
    .setup_packing_params = UNUSED,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = UNUSED,
    .fingerprint_constraints = {},
    .extra_weights_bytes = 0,
    .kernel_scale_element_size = 0,
};

static const struct fc_variant f32_qc4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = check_no_transpose_flag,
    .check_kernel_scale = check_kernel_scale_f32,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32_qc4w,
    .setup_packing_params = UNUSED,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_f32_qcxw,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_bf16},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant f32_qc8w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .check_kernel_scale = check_kernel_scale_f32,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_f32,
    .setup_packing_params = UNUSED,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_f32_qcxw,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_bf16},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = sizeof(float),
};

static const struct fc_variant qs8_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .check_input_scale = check_input_scale_value,
    .check_kernel_scale = check_kernel_scale_value,
    .check_output_scale = check_output_scale_value,
    .setup_gemm_ukernels = setup_gemm_ukernels,
    .setup_params = setup_params_qs8_qc8w,
    .setup_packing_params = setup_packing_params_qs8,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qs8,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_bf16,
                                force_coherent_bias_values_i32},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = 0,
};

static const struct fc_variant qs8_qc2w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .check_input_scale = check_input_scale_value,
    .check_kernel_scale = check_kernel_scale_value,
    .check_output_scale = check_output_scale_value,
    .setup_gemm_ukernels = UNUSED,
    .setup_params = setup_params_qs8_qc8w,
    .setup_packing_params = setup_packing_params_qs8_qc2w,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qx8_qcyw,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_f32,
                                force_coherent_bias_values_i32},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = 0,
};

static const struct fc_variant qs8_qc4w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .check_input_scale = check_input_scale_value,
    .check_kernel_scale = check_kernel_scale_value,
    .check_output_scale = check_output_scale_value,
    .setup_gemm_ukernels = UNUSED,
    .setup_params = setup_params_qs8_qc8w,
    .setup_packing_params = setup_packing_params_qs8_qc4w,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qx8_qcyw,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_f32,
                                force_coherent_bias_values_i32},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = 0,
};

static const struct fc_variant qs8_qc8w_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .check_input_scale = check_input_scale_value,
    .check_kernel_scale = check_kernel_scale_value,
    .check_output_scale = check_output_scale_value,
    .setup_gemm_ukernels = UNUSED,
    .setup_params = setup_params_qs8_qc8w,
    .setup_packing_params = setup_packing_params_qs8,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = setup_scale_params_qx8_qcyw,
    .fingerprint_constraints = {force_coherent_kernel_scale_values_f32,
                                force_coherent_bias_values_i32},
    .extra_weights_bytes = sizeof(float),
    .kernel_scale_element_size = 0,
};

static const struct fc_variant qu8_variant = {
    .check_output_bounds = check_output_bounds_f32,
    .check_kernel_zero_point = UNUSED,
    .check_block_size = UNUSED,
    .check_flags = UNUSED,
    .check_input_scale = check_input_scale_value,
    .check_kernel_scale = check_kernel_scale_value,
    .check_output_scale = check_output_scale_value,
    .setup_gemm_ukernels = UNUSED,
    .setup_params = setup_params_qu8,
    .setup_packing_params = setup_packing_params_qu8,
    .setup_packing_functions = setup_packing_functions_from_gemm_config,
    .setup_scale_params = UNUSED,
    .fingerprint_constraints = {force_coherent_bias_values_i32},
    .extra_weights_bytes = 0,
    .kernel_scale_element_size = 0,
};

static enum xnn_status setup_variant_and_gemm_config(
    const struct fc_variant** variant, struct fc_context* context) {
  switch (context->operator_type) {
    case xnn_operator_type_fully_connected_nc_f16:
      *variant = &f16_variant;
      context->gemm_config = xnn_init_f16_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_f16_f16_f16;
      break;
    case xnn_operator_type_fully_connected_nc_pf16:
      *variant = &f16_variant;
      context->gemm_config = xnn_init_pf16_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_pf16_pf16_pf16;
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc4w:
      *variant = &qx8_f16_qc4w_variant;
      context->gemm_config = xnn_init_qd8_f16_qc4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qd8_f16_qc4w;
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w:
      *variant = &qx8_f16_qc4w_variant;
      context->gemm_config = xnn_init_qdu8_f16_qc4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qdu8_f16_qc4w;
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qb4w:
      *variant = &qd8_f16_qb4w_variant;
      context->gemm_config = xnn_init_qd8_f16_qb4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qd8_f16_qb4w;
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc2w:
      *variant = &qd8_f32_qc2w_variant;
      context->gemm_config = xnn_init_qd8_f32_qc2w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qd8_f32_qc2w;
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc4w:
      *variant = &qx8_f32_qc4w_variant;
      context->gemm_config = xnn_init_qd8_f32_qc4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qd8_f32_qc4w;
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w:
      *variant = &qx8_f32_qc4w_variant;
      context->gemm_config = xnn_init_qdu8_f32_qc4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qdu8_f32_qc4w;
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc4w:
      *variant = &qp8_f32_qc4w_variant;
      context->gemm_config = xnn_init_qp8_f32_qc4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qp8_f32_qc4w;
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc8w:
      *variant = &qp8_f32_qc8w_variant;
      context->gemm_config = xnn_init_qp8_f32_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qp8_f32_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qb4w:
      *variant = &qp8_f32_qb4w_variant;
      context->gemm_config = xnn_init_qp8_f32_qb4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qp8_f32_qb4w;
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qb4w:
      *variant = &qx8_f32_qb4w_variant;
      context->gemm_config = xnn_init_qd8_f32_qb4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qd8_f32_qb4w;
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w:
      *variant = &qx8_f32_qb4w_variant;
      context->gemm_config = xnn_init_qdu8_f32_qb4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qdu8_f32_qb4w;
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc8w:
      *variant = &qdx8_f32_qc8w_variant;
      context->gemm_config = xnn_init_qd8_f32_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qd8_f32_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w:
      *variant = &qdx8_f32_qc8w_variant;
      context->gemm_config = xnn_init_qdu8_f32_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qdu8_f32_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc8w:
      *variant = &qdx8_f16_qc8w_variant;
      context->gemm_config = xnn_init_qd8_f16_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qd8_f16_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w:
      *variant = &qdx8_f16_qc8w_variant;
      context->gemm_config = xnn_init_qdu8_f16_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qdu8_f16_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_bf16_f32:
      *variant = &f32_variant;
      context->gemm_config = xnn_init_bf16_f32_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_bf16_bf16_f32;
      break;
    case xnn_operator_type_fully_connected_nc_f32: {
      *variant = &f32_variant;
      context->gemm_config = xnn_init_f32_gemm_config(context->flags);
      const struct xnn_gemm_config* gemm_nr2_config = xnn_init_f32_gemm_nr2_config(context->flags);
      // When we are directly computing a fingerprint id, we don't have the data
      // needed to choose which config we want but we have the fingerprint id
      // available.
      if (context->fingerprint_id ==
          xnn_fingerprint_id_fully_connected_nc_f32_f32_f32_nr2) {
        context->gemm_config = gemm_nr2_config;
      } else if (gemm_nr2_config != NULL &&
                 gemm_nr2_config->minmax.gemm[gemm_nr2_config->mr - 1]
                         .function[XNN_UARCH_DEFAULT] != NULL &&
                 xnn_use_nr2(context->gemm_config->nr, gemm_nr2_config->nr,
                             context->output_channels)) {
        // Select microkernel configuration based on output channels
        xnn_log_debug("Using `nr2` GEMM config for %s op.",
                      xnn_operator_type_to_string(
                          context->operator_type));
        context->gemm_config = gemm_nr2_config;
        context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_f32_f32_f32_nr2;
      } else {
        context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_f32_f32_f32;
      }
      break;
    }
    case xnn_operator_type_fully_connected_nc_pf32:
      *variant = &f32_variant;
      context->gemm_config = xnn_init_pf32_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_pf32_pf32_pf32;
      break;
    case xnn_operator_type_fully_connected_nc_f32_qc4w:
      *variant = &f32_qc4w_variant;
      context->gemm_config = xnn_init_f32_qc4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_f32_f32_qc4w;
      break;
    case xnn_operator_type_fully_connected_nc_f32_qc8w:
      *variant = &f32_qc8w_variant;
      context->gemm_config = xnn_init_f32_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_f32_f32_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_qs8:
      *variant = &qs8_variant;
      context->gemm_config = xnn_init_qs8_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qs8_qs8_qs8;
      break;
    case xnn_operator_type_fully_connected_nc_qs8_qc2w:
      *variant = &qs8_qc2w_variant;
      context->gemm_config = xnn_init_qs8_qc2w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qs8_qs8_qc2w;
      break;
    case xnn_operator_type_fully_connected_nc_qs8_qc4w:
      *variant = &qs8_qc4w_variant;
      context->gemm_config = xnn_init_qs8_qc4w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qs8_qs8_qc4w;
      break;
    case xnn_operator_type_fully_connected_nc_qs8_qc8w:
      *variant = &qs8_qc8w_variant;
      context->gemm_config = xnn_init_qs8_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qs8_qs8_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_pqs8_qc8w:
      *variant = &qs8_qc8w_variant;
      context->gemm_config = xnn_init_pqs8_qc8w_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_pqs8_pqs8_qc8w;
      break;
    case xnn_operator_type_fully_connected_nc_qu8:
      *variant = &qu8_variant;
      context->gemm_config = xnn_init_qu8_gemm_config();
      context->fingerprint_id = xnn_fingerprint_id_fully_connected_nc_qu8_qu8_qu8;
      break;
    default:
      xnn_log_error(
          "Could not select gemm config for operator %s: unhandled operator "
          "type",
          xnn_operator_type_to_string(context->operator_type));
      return xnn_status_invalid_state;
  }

  if (context->gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(context->operator_type));
    return xnn_status_unsupported_hardware;
  }

  return xnn_status_success;
}

// Returns the given `buffer` pointer, then advances it by `bytes` bytes,
// rounded up to `XNN_ALLOCATION_ALIGNMENT`.
static void* get_and_advance_simd_buffer(uint8_t** buffer, size_t bytes) {
  uint8_t* const res = *buffer;
  *buffer += bytes + (XNN_ALLOCATION_ALIGNMENT - (bytes % XNN_ALLOCATION_ALIGNMENT));
  return res;
};

// Generates a buffer that holds fak weights and bias.
//
// - The fake weights have input_channels * output_channels elements.
// - The bias has output_channels elements.
// - The kernel scale has output_channels elements.
static enum xnn_status generate_fingerprint_data(const struct fc_variant* variant, struct fc_context* context) {
  const int32_t input_channels = max(1 << (context->gemm_config->log2_kr + context->gemm_config->log2_sr), XNN_MIN_BLOCKSIZE);
  const int32_t output_channels = context->gemm_config->nr;
  const uint32_t bias_element_size = context->gemm_config->bias_element_size;
  const uint32_t kernel_element_size = 1 << context->gemm_config->log2_filter_element_size;
  const size_t weights_bytes = input_channels * output_channels * kernel_element_size;
  const size_t bias_bytes = output_channels * bias_element_size;
  const size_t kernel_scale_bytes = output_channels * variant->kernel_scale_element_size;
  const size_t kernel_zero_points_bytes = output_channels * sizeof(float);
  const size_t bytes = weights_bytes + bias_bytes + kernel_scale_bytes +
                       kernel_zero_points_bytes + 4 * XNN_ALLOCATION_ALIGNMENT;
  uint8_t* buffer = xnn_allocate_simd_memory(bytes);
  fill_fingerprint_buffer(buffer, bytes);
  context->fingerprint_data_to_release = buffer;
  context->input_channels = input_channels;
  context->input_stride = input_channels;
  context->output_channels = output_channels;
  context->output_stride = output_channels;
  context->kernel = get_and_advance_simd_buffer(&buffer, weights_bytes);
  context->bias = get_and_advance_simd_buffer(&buffer, bias_bytes);
  context->kernel_scale.f32 = get_and_advance_simd_buffer(&buffer, kernel_scale_bytes);
  context->kernel_zero_points = get_and_advance_simd_buffer(&buffer, kernel_zero_points_bytes);
  return xnn_status_success;
}

static enum xnn_status create_fully_connected_nc_helper(
    struct fc_context* context) {
  const struct fc_variant* variant;
  enum xnn_status status = xnn_status_uninitialized;
  XNN_IF_ERROR_GOTO(error, setup_variant_and_gemm_config(&variant, context));
  if (context->should_fingerprint) {
    XNN_IF_ERROR_GOTO(error, xnn_fingerprint_fully_connected_nc(context->fingerprint_id));
  }

  XNN_IF_ERROR_GOTO(error, variant->check_output_bounds(variant, context));
  XNN_IF_ERROR_GOTO(error, variant->check_kernel_zero_point(variant, context));
  XNN_IF_ERROR_GOTO(error, variant->check_block_size(variant, context));
  XNN_IF_ERROR_GOTO(error, variant->setup_gemm_ukernels(variant, context));
  XNN_IF_ERROR_GOTO(error, variant->setup_params(variant, context));
  XNN_IF_ERROR_GOTO(error, variant->setup_packing_params(variant, context));
  XNN_IF_ERROR_GOTO(error, variant->setup_packing_functions(variant, context));
  XNN_IF_ERROR_GOTO(error, variant->setup_scale_params(variant, context));

  XNN_IF_ERROR_GOTO(
      error,
      create_fully_connected_nc(
          context->input_channels, context->output_channels,
          context->input_stride, context->output_stride, context->kernel,
          context->bias, context->flags, context->block_size,
          context->blockwise_kernel_scale_params, context->pack_gemm_gio_w,
          context->pack_gemm_goi_w, context->packing_params,
          variant->extra_weights_bytes, context->init_scale_params,
          context->scale_params, context->init_kernel_scale_params,
          context->kernel_scale_params, &context->params, context->params_size,
          context->gemm_config, &context->gemm_config->minmax,
          context->operator_type, context->weights_cache,
          context->fingerprint_id, context->fully_connected_op_out,
          context->original_kernel, context->original_bias));
error:
  cleanup_context(variant, context);
  return status;
}

enum xnn_status xnn_fingerprint_fully_connected_nc(
    const enum xnn_fingerprint_id fingerprint_id) {
  enum xnn_status status = xnn_status_uninitialized;
  struct fingerprint_context fingerprint_context =
      create_fingerprint_context(fingerprint_id);
  struct fc_context context = {
      .weights_cache = &fingerprint_context.cache,
      .fully_connected_op_out = &fingerprint_context.op,
      .output_min = FLT_MIN,
      .output_max = FLT_MAX,
      .input_scale = 1,
      .output_scale = 1,
      .kernel_scale_value = 1,
      .fingerprint_id = fingerprint_id,
      .operator_type = get_operator_type(fingerprint_id),
  };
  if (fingerprint_context.status == xnn_status_uninitialized) {
    const struct fc_variant* variant;
    XNN_IF_ERROR_GOTO(error, setup_variant_and_gemm_config(&variant, &context));
    XNN_IF_ERROR_GOTO(error, generate_fingerprint_data(variant, &context));
    for (int i = 0; i < XNN_FC_VARIANT_FINGERPRINT_CONSTRAINT_MAX_COUNT &&
                    variant->fingerprint_constraints[i];
         ++i) {
      XNN_IF_ERROR_GOTO(error,
                        variant->fingerprint_constraints[i](variant, &context));
    }
    status = create_fully_connected_nc_helper(&context);
  } else {
    status = fingerprint_context.status;
  }
error:
  finalize_fingerprint_context(&fingerprint_context);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_f16,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_pf16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_pf16,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qd8_f16_qc4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f16_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .block_size = block_size,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.bf16 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qd8_f16_qb4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qb4w_f16_scales(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  enum xnn_status status = xnn_status_success;
  const size_t num_blocks =
      (input_channels + block_size - 1) / block_size * output_channels;
  xnn_bfloat16* bf16_scale_buffer =
      (xnn_bfloat16*)xnn_allocate_memory(num_blocks * sizeof(xnn_bfloat16));
  for (size_t i = 0; i < num_blocks; ++i) {
    bf16_scale_buffer[i] = xnn_bfloat16_from_float(
        xnn_float16_to_float(((const xnn_float16*)kernel_scale)[i]));
  }
  status = xnn_create_fully_connected_nc_qd8_f16_qb4w(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, (const uint16_t*)bf16_scale_buffer, kernel, bias,
      output_min, output_max, flags, weights_cache, fully_connected_op_out);
  xnn_release_memory(bf16_scale_buffer);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qc2w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_zero_point,
    const float* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_zero_points = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qd8_f32_qc2w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qd8_f32_qc4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qp8_f32_qc4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const void* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qp8_f32_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .block_size = block_size,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.bf16 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qp8_f32_qb4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qb4w_f16_scales(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  enum xnn_status status = xnn_status_success;
  const size_t num_blocks =
      (input_channels + block_size - 1) / block_size * output_channels;
  xnn_bfloat16* bf16_scale_buffer =
      (xnn_bfloat16*)xnn_allocate_memory(num_blocks * sizeof(xnn_bfloat16));
  for (size_t i = 0; i < num_blocks; ++i) {
    bf16_scale_buffer[i] = xnn_bfloat16_from_float(
        xnn_float16_to_float(((const xnn_float16*)kernel_scale)[i]));
  }
  // Fingerprinting is done by xnn_create_fully_connected_nc_qp8_f32_qb4w.
  status = xnn_create_fully_connected_nc_qp8_f32_qb4w(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, (const uint16_t*)bf16_scale_buffer, kernel, bias,
      output_min, output_max, flags, weights_cache, fully_connected_op_out);
  xnn_release_memory(bf16_scale_buffer);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .block_size = block_size,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.bf16 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qb4w_f16_scales(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const xnn_float16* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const size_t num_blocks =
      (input_channels + block_size - 1) / block_size * output_channels;
  xnn_bfloat16* bf16_scale_buffer =
      (xnn_bfloat16*)xnn_allocate_memory(num_blocks * sizeof(xnn_bfloat16));
  for (size_t i = 0; i < num_blocks; ++i) {
    bf16_scale_buffer[i] =
        xnn_bfloat16_from_float(xnn_float16_to_float(kernel_scale[i]));
  }
  enum xnn_status status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, (const uint16_t*)bf16_scale_buffer, kernel, bias, output_min, output_max,
      flags, weights_cache, fully_connected_op_out);
  xnn_release_memory(bf16_scale_buffer);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .block_size = block_size,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.bf16 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qb4w_f16_scales(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const xnn_float16* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const size_t num_blocks =
      (input_channels + block_size - 1) / block_size * output_channels;
  xnn_bfloat16* bf16_scale_buffer =
      (xnn_bfloat16*)xnn_allocate_memory(num_blocks * sizeof(xnn_bfloat16));
  for (size_t i = 0; i < num_blocks; ++i) {
    bf16_scale_buffer[i] =
        xnn_bfloat16_from_float(xnn_float16_to_float(kernel_scale[i]));
  }
  return xnn_create_fully_connected_nc_qdu8_f32_qb4w(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, (const uint16_t*)bf16_scale_buffer, kernel, bias, output_min, output_max,
      flags, weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qd8_f32_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qd8_f16_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f16_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_f32_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  float* fp32_kernel_buffer = (float*)xnn_allocate_memory(
      input_channels * output_channels * sizeof(float));
  float* fp32_bias_buffer = NULL;
  float* fp32_bias_buffer_to_release = NULL;
  const xnn_float16* f16_kernel = (const xnn_float16*)kernel;
  const xnn_float16* f16_bias = (const xnn_float16*)bias;
  for (size_t i = 0; i < input_channels * output_channels; ++i) {
    fp32_kernel_buffer[i] = xnn_float16_to_float(f16_kernel[i]);
  }
  if (bias && !(flags & XNN_FLAG_FP32_STATIC_BIASES)) {
    fp32_bias_buffer_to_release =
        (float*)xnn_allocate_memory(output_channels * sizeof(float));
    fp32_bias_buffer = fp32_bias_buffer_to_release;
    for (size_t i = 0; i < output_channels; ++i) {
      fp32_bias_buffer[i] = xnn_float16_to_float(f16_bias[i]);
    }
  } else {
    fp32_bias_buffer = (float*)(uintptr_t)bias;
  }
  // Fingerprinting is done by xnn_create_fully_connected_nc_f32.
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel = fp32_kernel_buffer,
      .bias = fp32_bias_buffer,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_f32,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
      .original_kernel = kernel,
      .original_bias = bias,
  };
  enum xnn_status status = create_fully_connected_nc_helper(&context);
  xnn_release_memory(fp32_kernel_buffer);
  xnn_release_memory(fp32_bias_buffer_to_release);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_pf32_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  float* fp32_kernel_buffer = (float*)xnn_allocate_memory(
      input_channels * output_channels * sizeof(float));
  float* fp32_bias_buffer = NULL;
  float* fp32_bias_buffer_to_release = NULL;
  const xnn_float16* f16_kernel = (const xnn_float16*)kernel;
  const xnn_float16* f16_bias = (const xnn_float16*)bias;
  for (size_t i = 0; i < input_channels * output_channels; ++i) {
    fp32_kernel_buffer[i] = xnn_float16_to_float(f16_kernel[i]);
  }
  if (bias && !(flags & XNN_FLAG_FP32_STATIC_BIASES)) {
    fp32_bias_buffer_to_release =
        (float*)xnn_allocate_memory(output_channels * sizeof(float));
    fp32_bias_buffer = fp32_bias_buffer_to_release;
    for (size_t i = 0; i < output_channels; ++i) {
      fp32_bias_buffer[i] = xnn_float16_to_float(f16_bias[i]);
    }
  } else {
    fp32_bias_buffer = (float*)(uintptr_t)bias;
  }
  // Fingerprinting is done by xnn_create_fully_connected_nc_pf32.
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel = fp32_kernel_buffer,
      .bias = fp32_bias_buffer,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_pf32,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
      .original_kernel = kernel,
      .original_bias = bias,
  };
  enum xnn_status status = create_fully_connected_nc_helper(&context);
  xnn_release_memory(fp32_kernel_buffer);
  xnn_release_memory(fp32_bias_buffer_to_release);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_bf16_f32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_bf16_f32,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}


enum xnn_status xnn_create_fully_connected_nc_f32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_f32,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_pf32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_pf32,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const uint8_t* kernel, const float* bias, float output_min,
    float output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_f32_qc4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_f32_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qs8(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    float kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
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
      .operator_type = xnn_operator_type_fully_connected_nc_qs8,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qs8_qc2w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const void* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .input_zero_point = input_zero_point,
      .output_zero_point = output_zero_point,
      .input_scale = input_scale,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qs8_qc2w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qs8_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, const float* kernel_scale, const void* kernel,
    const int32_t* bias, int8_t output_zero_point, float output_scale,
    int8_t output_min, int8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_zero_point = kernel_zero_point,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qs8_qc4w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_qs8_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_qs8_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

enum xnn_status xnn_create_fully_connected_nc_pqs8_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
      .input_zero_point = input_zero_point,
      .input_scale = input_scale,
      .kernel_scale.f32 = kernel_scale,
      .kernel = kernel,
      .bias = bias,
      .output_zero_point = output_zero_point,
      .output_scale = output_scale,
      .output_min = output_min,
      .output_max = output_max,
      .flags = flags,
      .weights_cache = weights_cache,
      .operator_type = xnn_operator_type_fully_connected_nc_pqs8_qc8w,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}


enum xnn_status xnn_create_fully_connected_nc_qu8(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, float kernel_scale, const uint8_t* kernel,
    const int32_t* bias, uint8_t output_zero_point, float output_scale,
    uint8_t output_min, uint8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  struct fc_context context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_stride = input_stride,
      .output_stride = output_stride,
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
      .operator_type = xnn_operator_type_fully_connected_nc_qu8,
      .fully_connected_op_out = fully_connected_op_out,
      .should_fingerprint = true,
  };
  return create_fully_connected_nc_helper(&context);
}

static enum xnn_status reshape_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    enum xnn_operator_type expected_operator_type, size_t batch_size,
    bool dynamic_quantization, uint32_t log2_output_element_size,
    const void* params, size_t params_size, size_t* workspace_size,
    pthreadpool_t threadpool) {
  uint32_t log2_input_element_size = fully_connected_op->gemm_config->log2_input_element_size;
  const bool filter_is_nibble =
      fully_connected_op->gemm_config->log2_filter_element_bit_size == XNN_LOG2_BIT_SIZEOF_INT4;
  const bool filter_is_crumb =
      fully_connected_op->gemm_config->log2_filter_element_bit_size == XNN_LOG2_BIT_SIZEOF_INT2;
  if (fully_connected_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(fully_connected_op));
    return xnn_status_invalid_parameter;
  }
  fully_connected_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string_v2(fully_connected_op));
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    fully_connected_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  size_t input_channels =
      fully_connected_op->convolution_op->group_input_channels;
  const size_t output_channels =
      fully_connected_op->convolution_op->group_output_channels;

  uint32_t mr = fully_connected_op->ukernel.gemm_ukernels->gemm.mr;
  uint32_t mr_packed =
      fully_connected_op->ukernel.gemm_ukernels->gemm.mr_packed;
  const uint32_t nr = fully_connected_op->ukernel.gemm_ukernels->gemm.nr;
  const uint32_t kr = fully_connected_op->ukernel.gemm_ukernels->gemm.kr;
  const uint32_t sr = fully_connected_op->ukernel.gemm_ukernels->gemm.sr;
  struct xnn_hmp_gemm_ukernel* gemm_cases =
      fully_connected_op->ukernel.gemm_ukernels->gemm.gemm_cases;
  const size_t num_threads = pthreadpool_get_threads_count(threadpool);

  if (batch_size == 1 &&
      fully_connected_op->ukernel.gemm_ukernels->gemm.gemm_cases[0]
              .function[XNN_UARCH_DEFAULT] != NULL) {
    mr = 1;
    mr_packed = 1;
  }

  assert(mr != 0 && mr <= XNN_MAX_MR);
  struct xnn_hmp_gemm_ukernel gemm_ukernel = gemm_cases[mr - 1];
  if (filter_is_nibble || filter_is_crumb) {
    const uint32_t planes = fully_connected_op->ukernel.gemm_ukernels->gemm.kp;
    input_channels = round_up_po2(input_channels, planes);
  }

  const struct xnn_pack_lh_config* packed_lh_config = NULL;
  bool inline_lhs_packing =
      fully_connected_op->flags & XNN_FLAG_INLINE_LHS_PACKING;
  switch (fully_connected_op->type) {
    case xnn_operator_type_fully_connected_nc_qd8_f16_qb4w:
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc4w:
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc8w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_f16_qdint8_pack_lh_config();
      }
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w:
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_f16_qduint8_pack_lh_config();
      }
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc2w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_f32_qdint8_row_sums_pack_lh_config();
      }
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qb4w:
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc4w:
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc8w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_f32_qdint8_pack_lh_config();
      }
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w:
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w:
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_f32_qduint8_pack_lh_config();
      }
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qb4w:
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc4w:
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc8w:
      packed_lh_config = xnn_init_qp8_pack_lh_config();
      break;
    case xnn_operator_type_fully_connected_nc_pf16:
      packed_lh_config = xnn_init_x16_pack_lh_config();
      break;
    case xnn_operator_type_fully_connected_nc_pf32:
      packed_lh_config = xnn_init_x32_pack_lh_config();
      break;
    case xnn_operator_type_fully_connected_nc_pqs8_qc8w:
      packed_lh_config = xnn_init_x8_pack_lh_config();
      break;
    default:
      break;
  }

  // Clear the operator's compute data to avoid accidentally reusing values from
  // a previous reshape (this was an interesting bug to track down).
  memset(fully_connected_op->compute, 0, 2 * sizeof(struct compute_parameters));
  struct compute_parameters* gemm_compute = &fully_connected_op->compute[0];
  fully_connected_op->num_compute_invocations = 1;
  struct gemm_op_context* gemm_context =
      fully_connected_op->dynamic_context.gemm;

  // Compute the optimal tile size for this GEMM.
  const size_t nc = xnn_gemm_best_tile_size(
      /*num_groups=*/1, /*m=*/batch_size, /*n=*/output_channels,
      /*m_stride=*/
      fully_connected_op->input_pixel_stride
          << (packed_lh_config ? packed_lh_config->log2_packed_element_size
                               : log2_input_element_size),
      /*n_stride=*/
      fully_connected_op->weights_stride,
      /*cn_stride=*/1 << log2_output_element_size, mr, nr,
      /*num_threads=*/num_threads);

  // If we are packing the LHS, provide a per-thread workspace to do so inline.
  if (packed_lh_config) {
    if (inline_lhs_packing) {
      assert(workspace_size);
      const size_t per_thread_workspace_size = packed_lh_config->size_fn(
          mr, /*k=*/input_channels, mr_packed, kr, sr);

      // If `xnn_gemm_best_tile_size` suggests an `nc` that is smaller than `n`,
      // i.e. it suggests splitting along `output_channels`, then it's probably
      // not a good idea to inline the packing, which requires using `nc == n`.
      //
      // Similarly, inlining the packing also doesn't make sense if the number
      // of threads exceeds the number of tiles that we can parallelize over.
      //
      // In either case, we pack the entire LHS into the workspace in a separate
      // `compute`, just as if it were a separate op.
      const bool should_inline_lhs_packing = xnn_should_inline_lhs_packing(
          fully_connected_op->gemm_config,
          /*m_packed_stride=*/divide_round_up(per_thread_workspace_size, mr),
          /*n_stride=*/fully_connected_op->weights_stride,
          /*cn_stride=*/1 << log2_output_element_size, /*mc=*/batch_size,
          /*nc=*/output_channels);

      if (packed_lh_config->gemv_noop && mr == 1) {
        xnn_log_debug(
            "Skipping inline packing for %s with m=%zu, n=%zu, and k=%zu since "
            "it is a no-op for GEMV.",
            xnn_operator_type_to_string(fully_connected_op->type), batch_size,
            output_channels, input_channels);
      } else if (!should_inline_lhs_packing ||
                 num_threads * mr > round_up(batch_size, mr)) {
        xnn_log_debug(
            "Pre-packing LHS of %s with m=%zu, n=%zu, and k=%zu despite "
            "request to inline because %s.",
            xnn_operator_type_to_string(fully_connected_op->type), batch_size,
            output_channels, input_channels,
            !should_inline_lhs_packing
                ? "packed lhs will likely not stay in cache"
                : "batch size does not parallelize well over the number of "
                  "threads");

        // Allocate a workspace for the entire LHS.
        *workspace_size = packed_lh_config->size_fn(
            batch_size, /*k=*/input_channels, mr_packed, kr, sr);

        // Set up the LHS packing as a separate compute.
        gemm_context->pack_lh = (struct pack_lh_context){
            .m = batch_size,
            .k = input_channels,
            .mr = mr_packed,
            .kr = kr,
            .sr = sr,
            .lhs_stride = input_channels
                          << packed_lh_config->log2_input_element_size,
            .packed_offset_fn = packed_lh_config->offset_fn,
            .pack_lh_ukernel = packed_lh_config->pack_lh_fn,
        };
        fully_connected_op->compute[0].context_offset =
            offsetof(struct gemm_op_context, pack_lh);
        fully_connected_op->compute[0].type =
            xnn_parallelization_type_2d_tile_1d_dynamic;
        fully_connected_op->compute[0].task_2d_tile_1d_dynamic =
            (pthreadpool_task_2d_tile_1d_dynamic_t)xnn_compute_pack_lh;
        fully_connected_op->compute[0].range[0] = 1;
        fully_connected_op->compute[0].range[1] = batch_size;
        fully_connected_op->compute[0].tile[0] = mr_packed;

        fully_connected_op->num_compute_invocations = 2;
        gemm_compute = &fully_connected_op->compute[1];
        log2_input_element_size = packed_lh_config->log2_packed_element_size;
        inline_lhs_packing = false;
        xnn_log_debug("Requesting workspace of size %zu bytes for LHS packing.",
                      *workspace_size);
      } else {
        xnn_log_debug(
            "Inlining LHS packing for %s with m=%zu, n=%zu, and k=%zu.",
            xnn_operator_type_to_string(fully_connected_op->type), batch_size,
            output_channels, input_channels);
        // We need a buffer for `mr` packed rows for each thread for inlined
        // LHS packing.
        *workspace_size = num_threads * per_thread_workspace_size;
        log2_input_element_size = packed_lh_config->log2_input_element_size;
        xnn_log_debug(
            "Requesting workspace of size %zu x %zu bytes for LHS packing.",
            num_threads, *workspace_size);
      }
    } else {
      log2_input_element_size = packed_lh_config->log2_packed_element_size;
    }
  }

  gemm_context->gemm = (struct gemm_context){
      .k_scaled = input_channels << log2_input_element_size,
      .w_stride = fully_connected_op->weights_stride,
      .a_stride = fully_connected_op->input_pixel_stride
                  << log2_input_element_size,
      .packed_w = packed_weights(fully_connected_op),
      .cm_stride = fully_connected_op->output_pixel_stride
                   << log2_output_element_size,
      .cn_stride = nr << log2_output_element_size,
      .log2_csize = log2_output_element_size,
      .ukernel = gemm_ukernel,
      .mr = mr,
      .nc = output_channels,
      .kr = kr,
      .sr = sr,
      .kc = input_channels,
      .packed_lh_config = packed_lh_config,
      .workspace_offset = 0,
      .mr_packed = mr_packed,
      .dynamic_quantization = dynamic_quantization,
      .with_row_sum = fully_connected_op->type == xnn_operator_type_fully_connected_nc_qd8_f32_qc2w,
  };

  memcpy(&gemm_context->gemm.params, params, params_size);
  gemm_context->gemm.fused_params = &gemm_context->gemm.params;

#if XNN_MAX_UARCH_TYPES > 1
  if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
    if (packed_lh_config) {
      if (inline_lhs_packing) {
        gemm_compute->type =
            xnn_parallelization_type_1d_tile_1d_dynamic_with_uarch_with_thread;
        fully_connected_op->compute[0]
            .task_1d_tile_1d_dynamic_with_id_with_thread =
            (pthreadpool_task_1d_tile_1d_dynamic_with_id_with_thread_t)
                xnn_compute_hmp_inline_packed_qp8gemm;
      } else {
        gemm_compute->type =
            xnn_parallelization_type_2d_tile_2d_dynamic_with_uarch;
        gemm_compute->task_2d_tile_2d_dynamic_with_id =
            (pthreadpool_task_2d_tile_2d_dynamic_with_id_t)
                xnn_compute_hmp_qp8gemm;
      }
    } else {
      gemm_compute->type =
          xnn_parallelization_type_2d_tile_2d_dynamic_with_uarch;
      if (dynamic_quantization) {
        gemm_compute->task_2d_tile_2d_dynamic_with_id =
            (pthreadpool_task_2d_tile_2d_dynamic_with_id_t)
                xnn_compute_hmp_dqgemm;
      } else {
        gemm_compute->task_2d_tile_2d_dynamic_with_id =
            (pthreadpool_task_2d_tile_2d_dynamic_with_id_t)xnn_compute_hmp_gemm;
      }
    }
  } else
#endif  // XNN_MAX_UARCH_TYPES > 1
    if (packed_lh_config) {
      if (inline_lhs_packing) {
        gemm_compute->type =
            xnn_parallelization_type_1d_tile_1d_dynamic_with_thread;
        gemm_compute->task_1d_tile_1d_dynamic_with_id =
            (pthreadpool_task_1d_tile_1d_dynamic_with_id_t)
                xnn_compute_inline_packed_qp8gemm;
      } else {
        gemm_compute->type = xnn_parallelization_type_2d_tile_2d_dynamic;
        gemm_compute->task_2d_tile_2d_dynamic =
            (pthreadpool_task_2d_tile_2d_dynamic_t)xnn_compute_qp8gemm;
      }
    } else {
      gemm_compute->type = xnn_parallelization_type_2d_tile_2d_dynamic;
      if (dynamic_quantization) {
        gemm_compute->task_2d_tile_2d_dynamic =
            (pthreadpool_task_2d_tile_2d_dynamic_t)xnn_compute_dqgemm;
      } else {
        gemm_compute->type = xnn_parallelization_type_2d_tile_2d_dynamic;
        gemm_compute->task_2d_tile_2d_dynamic =
            (pthreadpool_task_2d_tile_2d_dynamic_t)xnn_compute_gemm;
      }
    }

  if (packed_lh_config && inline_lhs_packing) {
    gemm_compute->range[0] = batch_size;
    gemm_compute->tile[0] = mr;
  } else {
    gemm_compute->range[0] = output_channels;
    gemm_compute->range[1] = batch_size;
    gemm_compute->tile[0] = nc;
    gemm_compute->tile[1] = mr;
  }
  fully_connected_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_fully_connected_nc_f16(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f16, batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      &fully_connected_op->params.f16_minmax,
      sizeof(fully_connected_op->params.f16_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_pf16(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pf16, batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      &fully_connected_op->params.f16_minmax,
      sizeof(fully_connected_op->params.f16_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_f32_f16(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return xnn_reshape_fully_connected_nc_f32(fully_connected_op, batch_size,
                                            threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_bf16_f32(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_bf16_f32,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_f32(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32, batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_pf32(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pf32, batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32_qc4w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_qc4w_minmax,
      sizeof(fully_connected_op->params.f32_qc4w_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_f32_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32_qc8w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qc4w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      &fully_connected_op->params.f32_qc4w_minmax,
      sizeof(fully_connected_op->params.f32_qc4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f16_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      &fully_connected_op->params.f32_qc4w_minmax,
      sizeof(fully_connected_op->params.f32_qc4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qb4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qb4w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      &fully_connected_op->params.f32_qb4w_minmax,
      sizeof(fully_connected_op->params.f32_qb4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc2w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
    fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc2w,
    batch_size,
    /*dynamic_quantization=*/true,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &fully_connected_op->params.f32_minmax,
    sizeof(fully_connected_op->params.f32_minmax), workspace_size,
    threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc4w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_qc4w_minmax,
      sizeof(fully_connected_op->params.f32_qc4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_qc4w_minmax,
      sizeof(fully_connected_op->params.f32_qc4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_qb4w_minmax,
      sizeof(fully_connected_op->params.f32_qb4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f32_qb4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_qb4w_minmax,
      sizeof(fully_connected_op->params.f32_qb4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qc8w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      &fully_connected_op->params.f16_minmax,
      sizeof(fully_connected_op->params.f16_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f16_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT16,
      &fully_connected_op->params.f16_minmax,
      sizeof(fully_connected_op->params.f16_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc8w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f32_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w,
      batch_size,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qc4w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qc8w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_minmax,
      sizeof(fully_connected_op->params.f32_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qb4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qb4w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &fully_connected_op->params.f32_qb4w_minmax,
      sizeof(fully_connected_op->params.f32_qb4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qs8(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8, batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      &fully_connected_op->params.qs8_qc8w_conv_minmax,
      sizeof(fully_connected_op->params.qs8_qc8w_conv_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qs8_qc2w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc2w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      &fully_connected_op->params.qs8_qc8w_conv_minmax,
      sizeof(fully_connected_op->params.qs8_qc8w_conv_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qs8_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc4w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      &fully_connected_op->params.qs8_qc8w_conv_minmax,
      sizeof(fully_connected_op->params.qs8_qc8w_conv_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qs8_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc8w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      &fully_connected_op->params.qs8_qc8w_conv_minmax,
      sizeof(fully_connected_op->params.qs8_qc8w_conv_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_pqs8_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pqs8_qc8w,
      batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      &fully_connected_op->params.qs8_qc8w_conv_minmax,
      sizeof(fully_connected_op->params.qs8_qc8w_conv_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qu8(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qu8, batch_size,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      &fully_connected_op->params.qu8_conv_minmax,
      sizeof(fully_connected_op->params.qu8_conv_minmax),
      /*workspace_size=*/NULL, threadpool);
}

static enum xnn_status setup_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    enum xnn_operator_type expected_operator_type, const void* input,
    void* output, void* workspace, const void* row_sum,
    const void* quantization_params) {
  if (fully_connected_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(fully_connected_op));
    return xnn_status_invalid_parameter;
  }

  if (fully_connected_op->weights_cache != NULL &&
      !xnn_weights_cache_is_finalized(fully_connected_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
                  xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_state;
  }

  switch (fully_connected_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(fully_connected_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different
      // pointers.
      break;
  }

  struct gemm_op_context* gemm_context =
      fully_connected_op->dynamic_context.gemm;

  if (fully_connected_op->num_compute_invocations == 2) {
    gemm_context->pack_lh.lhs = input;
    gemm_context->pack_lh.lhs_packed = workspace;
    gemm_context->gemm.a = workspace;
  } else {
    gemm_context->gemm.a = input;
    gemm_context->gemm.workspace = workspace;
  }
  gemm_context->gemm.c = output;
  gemm_context->gemm.quantization_params = quantization_params;
  gemm_context->gemm.row_sum = row_sum;

  fully_connected_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_fully_connected_nc_f16(
    xnn_operator_t fully_connected_op, const void* input, void* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f16, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_pf16(
    xnn_operator_t fully_connected_op, const void* input, void* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pf16, input,
      output, workspace, /*row_sum=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32_f16(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return xnn_setup_fully_connected_nc_f32(fully_connected_op, input, output);
}

enum xnn_status xnn_setup_fully_connected_nc_bf16_f32(
    xnn_operator_t fully_connected_op, const void* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_bf16_f32, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_pf32(
    xnn_operator_t fully_connected_op, const float* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pf32, input,
      output, workspace, /*row_sum=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32_qc4w(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32_qc4w, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32_qc8w(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32_qc8w, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qc4w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f16_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qb4w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc2w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace, const float* row_sum,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
    fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc2w,
    input, output, workspace, row_sum, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc4w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qc4w(
    xnn_operator_t fully_connected_op, const uint8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qc8w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f16_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qc4w,
      input, output, workspace, /*row_sum=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qc8w,
      input, output, workspace, /*row_sum=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qb4w,
      input, output, workspace, /*row_sum=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc8w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w,
      input, output, workspace, /*row_sum=*/NULL, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qs8(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qs8_qc2w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc2w, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qs8_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc4w, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qs8_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc8w, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_pqs8_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pqs8_qc8w, input,
      output, workspace, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qu8(
    xnn_operator_t fully_connected_op, const uint8_t* input, uint8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qu8, input,
      output, /*workspace=*/NULL, /*row_sum=*/NULL,
      /*quantization_params=*/NULL);
}
