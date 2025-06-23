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

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
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

static enum xnn_status create_fully_connected_nc(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias, uint32_t flags,
    size_t block_size, const uint16_t* blockwise_kernel_scale_params,
    uint32_t log2_input_element_size, uint32_t log2_filter_element_size,
    bool filter_is_nibble, uint32_t bias_element_size,
    xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio_w,
    xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w, const void* packing_params,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params,
    xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params,
    const float* kernel_scale_params, const void* params, size_t params_size,
    const struct xnn_gemm_config* gemm_config,
    const struct gemm_fused_ukernels* gemm_ukernels,
    enum xnn_operator_type operator_type, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  xnn_operator_t fully_connected_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

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
        "failed to create %s operator with input element stride of %zu: "
        "stride must be at least as large as the number of input channels "
        "(%zu)",
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

  if (filter_is_nibble) {
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
  cache_key.kernel = kernel;
  cache_key.bias = bias;
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
            gemm_config->nr * extra_weights_bytes, packing_params);
      } else {
        pack_gemm_goi_w(
            /*groups=*/1, output_channels, input_channels, nr, kr, sr, kernel,
            bias, /*scale=*/NULL, weights_ptr,
            gemm_config->nr * extra_weights_bytes, packing_params);
      }
      if (kernel_scale_params != NULL) {
        assert(init_kernel_scale_params != NULL);

        void* weights =
            (void*)((uintptr_t)weights_ptr +
                    gemm_config->nr * ((k_stride << log2_filter_element_size) +
                                       bias_element_size));
        init_kernel_scale_params(output_channels, gemm_config->nr,
                                 gemm_config->nr * weights_stride,
                                 kernel_scale_params, weights);
      }

      if (scale_params != NULL) {
        assert(init_scale_params != NULL);
        void* weights =
            (void*)((uintptr_t)weights_ptr +
                    gemm_config->nr * ((k_stride << log2_filter_element_size) +
                                       bias_element_size));
        if (kernel_scale_params != NULL) {
          weights =
              (void*)((uintptr_t)weights + gemm_config->nr * sizeof(float));
        }
        init_scale_params(output_channels, gemm_config->nr,
                          gemm_config->nr * weights_stride, scale_params,
                          weights);
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

enum xnn_status create_fully_connected_nc_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  const xnn_float16 fp16_output_min = xnn_float16_from_float(output_min);
  const xnn_float16 fp16_output_max = xnn_float16_from_float(output_max);
  const float rounded_output_min = xnn_float16_to_float(fp16_output_min);
  const float rounded_output_max = xnn_float16_to_float(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be below upper bound",
        xnn_operator_type_to_string(expected_operator_type), rounded_output_min,
        rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  struct xnn_f16_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, fp16_output_min, fp16_output_max);
  }
  xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio_w =
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio;
  xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w =
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_gemm_gio_w =
        (xnn_packw_gemm_gio_ukernel_fn)xnn_pack_f32_to_f16_gemm_gio_w;
    pack_gemm_goi_w =
        (xnn_packw_gemm_goi_ukernel_fn)xnn_pack_f32_to_f16_gemm_goi_w;
  }
  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*filter_is_nibble=*/false,
      /*bias_element_size=*/sizeof(uint16_t), pack_gemm_gio_w, pack_gemm_goi_w,
      /*packing_params=*/NULL,
      /*extra_weights_bytes=*/0,
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, &gemm_config->minmax, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  return create_fully_connected_nc_f16(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, output_min, output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_f16, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_pf16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_pf16_gemm_config();
  return create_fully_connected_nc_f16(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, output_min, output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_pf16, fully_connected_op_out);
}

enum xnn_status create_fully_connected_nc_qx8_f16_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  const xnn_float16 fp16_output_min = xnn_float16_from_float(output_min);
  const xnn_float16 fp16_output_max = xnn_float16_from_float(output_max);
  const float rounded_output_min = xnn_float16_to_float(fp16_output_min);
  const float rounded_output_max = xnn_float16_to_float(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be below upper bound",
        xnn_operator_type_to_string(expected_operator_type), rounded_output_min,
        rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  if (kernel_zero_point != 8 && kernel_zero_point != 0) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equals to 8 "
                  "(unsigned weights) or 0 (signed weights)",
                  xnn_operator_type_to_string(expected_operator_type),
                  kernel_zero_point);
    return xnn_status_invalid_parameter;
  }

  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f16_qc4w_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f16_qc4w != NULL) {
    gemm_config->init.f16_qc4w(&params, fp16_output_min, fp16_output_max,
                               kernel_zero_point);
  }

  // We don't know input zero point until runtime, row sum is multiplied by it
  // during packing, so set it to 1.
  const struct xnn_qs8_qc4w_packing_params packing_params = {
      /*input_zero_point=*/1, kernel_zero_point};

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      /*bias=*/NULL, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*bias_element_size=*/sizeof(float),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      &packing_params,
      /*extra_weights_bytes=*/sizeof(float) * 2,
      /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
      /*scale_params=*/bias,
      /*init_kernel_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
      /*kernel_scale_params=*/kernel_scale, &params, sizeof(params),
      gemm_config, gemm_ukernels, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f16_qc4w_gemm_config();
  return create_fully_connected_nc_qx8_f16_qc4w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qd8_f16_qc4w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f16_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qdu8_f16_qc4w_gemm_config();
  return create_fully_connected_nc_qx8_f16_qc4w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qd8_f16_qb4w));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qd8_f16_qb4w));
    return xnn_status_invalid_parameter;
  }

  const xnn_float16 fp16_output_min = xnn_float16_from_float(output_min);
  const xnn_float16 fp16_output_max = xnn_float16_from_float(output_max);
  const float rounded_output_min = xnn_float16_to_float(fp16_output_min);
  const float rounded_output_max = xnn_float16_to_float(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be below upper bound",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qd8_f16_qb4w),
        rounded_output_min, rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  if (block_size < XNN_MIN_BLOCKSIZE || block_size % XNN_MIN_BLOCKSIZE != 0) {
    xnn_log_error(
        "failed to create %s operator with block_size: %zu: expecting "
        "block_size to be a multiple of %d.",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qd8_f16_qb4w),
        block_size, XNN_MIN_BLOCKSIZE);
    return xnn_status_invalid_parameter;
  }

  if (input_channels % block_size != 0) {
    xnn_log_error(
        "failed to create %s operator with input_channels: %zu, and "
        "block_size: %zu: expecting input_channels %% block_size == 0.",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qd8_f16_qb4w),
        input_channels, block_size);
    return xnn_status_invalid_parameter;
  }

  // Assuming kernel_scale.size() is output_channels * num_blocks.
  size_t num_blocks = input_channels / block_size;
  for (size_t output_channel = 0; output_channel < output_channels;
       output_channel++) {
    for (size_t block_index = 0; block_index < num_blocks; block_index++) {
      size_t scale_index = output_channel * num_blocks + block_index;
      float fp32_scale = math_cvt_fp32_bf16(kernel_scale[scale_index]);
      if (fp32_scale <= 0.0f || !isnormal(fp32_scale)) {
        xnn_log_error(
            "failed to create %s operator with %.7g kernel scale in output "
            "channel #%zu, block #%zu: scale must be finite and positive",
            xnn_operator_type_to_string(
                xnn_operator_type_fully_connected_nc_qd8_f16_qb4w),
            fp32_scale, output_channel, block_index);
        return xnn_status_invalid_parameter;
      }
    }
  }

  if (kernel_zero_point != 8) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equal to 8 "
                  "(unsigned weights) or 0 (signed weights)",
                  xnn_operator_type_to_string(
                      xnn_operator_type_fully_connected_nc_qd8_f16_qb4w),
                  kernel_zero_point);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f16_qb4w_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qd8_f16_qb4w));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f16_qb4w_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f16_qb4w != NULL) {
    gemm_config->init.f16_qb4w(&params, fp16_output_min, fp16_output_max,
                               kernel_zero_point, block_size);
  }

  // We don't know input zero point until runtime, row sum is multiplied by it
  // during packing, so set it to 1.
  const struct xnn_qs8_qc4w_packing_params packing_params = {
      /*input_zero_point=*/1, kernel_zero_point};

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/block_size,
      /*blockwise_kernel_scale_params=*/kernel_scale,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*bias_element_size=*/sizeof(float),
      /*pack_gemm_gio_w,=*/NULL,
      /*pack_gemm_goi_w=*/NULL,
      /*packing_params=*/&packing_params,
      /*extra_weights_bytes=*/sizeof(float),
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_fully_connected_nc_qd8_f16_qb4w,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status create_fully_connected_nc_qx8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(expected_operator_type), output_min,
        output_max);
    return xnn_status_invalid_parameter;
  }

  if (kernel_zero_point != 8 && kernel_zero_point != 0) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equal to 8 "
                  "(unsigned weights) or 0 (signed weights)",
                  xnn_operator_type_to_string(expected_operator_type),
                  kernel_zero_point);
    return xnn_status_invalid_parameter;
  }

  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_qc4w_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32_qc4w != NULL) {
    gemm_config->init.f32_qc4w(&params, output_min, output_max,
                               kernel_zero_point);
  }

  // We don't know input zero point until runtime, row sum is multiplied by it
  // during packing, so set it to 1.
  const struct xnn_qs8_qc4w_packing_params packing_params = {
      /*input_zero_point=*/1, kernel_zero_point};

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      /*bias=*/NULL, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*bias_element_size=*/sizeof(float),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      &packing_params,
      /*extra_weights_bytes=*/sizeof(float) * 2,
      /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
      /*scale_params=*/bias,
      /*init_kernel_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
      /*kernel_scale_params=*/kernel_scale, &params, sizeof(params),
      gemm_config, gemm_ukernels, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f32_qc4w_gemm_config();
  return create_fully_connected_nc_qx8_f32_qc4w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qd8_f32_qc4w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qdu8_f32_qc4w_gemm_config();
  return create_fully_connected_nc_qx8_f32_qc4w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w,
      fully_connected_op_out);
}

static enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qcxw(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const void* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, enum xnn_operator_type operator_type,
    const struct xnn_gemm_config* gemm_config, bool filter_is_nibble,
    const void* packing_params, xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(operator_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(operator_type), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      /*bias=*/NULL, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/filter_is_nibble,
      /*bias_element_size=*/sizeof(float),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi, packing_params,
      /*extra_weights_bytes=*/0,
      /*init_scale_params=*/NULL,
      /*scale_params=*/bias,
      /*init_kernel_scale_params=*/NULL,
      /*kernel_scale_params=*/kernel_scale, &params, sizeof(params),
      gemm_config, gemm_ukernels, operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  if (kernel_zero_point != 8 && kernel_zero_point != 0) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equals to 8 "
                  "(unsigned weights) or 0 (signed weights)",
                  xnn_operator_type_to_string(
                      xnn_operator_type_fully_connected_nc_qp8_f32_qc4w),
                  kernel_zero_point);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config =
      xnn_init_qp8_f32_qc4w_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qc4w));
    return xnn_status_unsupported_hardware;
  }

  // We don't know input zero point until runtime, row sum is multiplied by it
  // during packing, so set it to 1.
  const struct xnn_qs8_qc4w_packing_params packing_params = {
      /*input_zero_point=*/1, kernel_zero_point};

  return xnn_create_fully_connected_nc_qp8_f32_qcxw(
      input_channels, output_channels, input_stride, output_stride,
      kernel_scale, kernel, bias, output_min, output_max, flags, weights_cache,
      /*operator_type=*/xnn_operator_type_fully_connected_nc_qp8_f32_qc4w,
      gemm_config, /*filter_is_nibble=*/true, &packing_params,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const void* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qp8_f32_qc8w_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qc8w));
    return xnn_status_unsupported_hardware;
  }

  // We don't know input zero point until runtime, row sum is multiplied by it
  // during packing, so set it to 1.
  const struct xnn_qs8_qc8w_packing_params packing_params = {
      /*input_zero_point=*/1, 1.0f};

  return xnn_create_fully_connected_nc_qp8_f32_qcxw(
      input_channels, output_channels, input_stride, output_stride,
      kernel_scale, kernel, bias, output_min, output_max, flags, weights_cache,
      /*operator_type=*/xnn_operator_type_fully_connected_nc_qp8_f32_qc8w,
      gemm_config, /*filter_is_nibble=*/false, &packing_params,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qb4w));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qb4w));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qb4w),
        output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config =
      xnn_init_qp8_f32_qb4w_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qb4w));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  if (block_size < XNN_MIN_BLOCKSIZE || block_size % XNN_MIN_BLOCKSIZE != 0) {
    xnn_log_error(
        "failed to create %s operator with block_size: %zu: expecting "
        "block_size to be a multiple of %d.",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qb4w),
        block_size, XNN_MIN_BLOCKSIZE);
    return xnn_status_invalid_parameter;
  }

  if (input_channels % block_size != 0) {
    xnn_log_error(
        "failed to create %s operator with input_channels: %zu, and "
        "block_size: %zu: expecting input_channels %% block_size == 0.",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_qp8_f32_qb4w),
        input_channels, block_size);
    return xnn_status_invalid_parameter;
  }

  if (kernel_zero_point != 8) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equal to 8 "
                  "(unsigned weights) or 0 (signed weights)",
                  xnn_operator_type_to_string(
                      xnn_operator_type_fully_connected_nc_qp8_f32_qb4w),
                  kernel_zero_point);
    return xnn_status_invalid_parameter;
  }
  // Assuming kernel_scale.size() is output_channels * num_blocks.
  size_t num_blocks = input_channels / block_size;
  for (size_t output_channel = 0; output_channel < output_channels;
       output_channel++) {
    for (size_t block_index = 0; block_index < num_blocks; block_index++) {
      size_t scale_index = output_channel * num_blocks + block_index;
      float fp32_scale = math_cvt_fp32_bf16(kernel_scale[scale_index]);
      if (fp32_scale <= 0.0f || !isnormal(fp32_scale)) {
        xnn_log_error(
            "failed to create %s operator with %.7g kernel scale in output "
            "channel #%zu, block #%zu: scale must be finite and positive",
            xnn_operator_type_to_string(
                xnn_operator_type_fully_connected_nc_qp8_f32_qb4w),
            fp32_scale, output_channel, block_index);
        return xnn_status_invalid_parameter;
      }
    }
  }

  struct xnn_f32_qb4w_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32_qb4w != NULL) {
    gemm_config->init.f32_qb4w(&params, output_min, output_max,
                               kernel_zero_point, block_size);
  }

  // We don't know input zero point until runtime, row sum is multiplied by it
  // during packing, so set it to 1.
  const struct xnn_qs8_qc4w_packing_params packing_params = {
      /*input_zero_point=*/1, kernel_zero_point};

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/block_size,
      /*blockwise_kernel_scale_params=*/kernel_scale,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*bias_element_size=*/sizeof(float),
      /*pack_gemm_gio_w,=*/NULL,
      /*pack_gemm_goi_w=*/NULL, &packing_params,
      /*extra_weights_bytes=*/0,
      /*init_scale_params=*/NULL,
      /*scale_params=*/NULL,
      /*init_kernel_scale_params=*/NULL,
      /*kernel_scale_params=*/NULL, &params, sizeof(params), gemm_config,
      gemm_ukernels, xnn_operator_type_fully_connected_nc_qp8_f32_qb4w,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status create_fully_connected_nc_qx8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(expected_operator_type), output_min,
        output_max);
    return xnn_status_invalid_parameter;
  }

  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  if (block_size < XNN_MIN_BLOCKSIZE || block_size % XNN_MIN_BLOCKSIZE != 0) {
    xnn_log_error(
        "failed to create %s operator with block_size: %zu: expecting "
        "block_size to be a multiple of %d.",
        xnn_operator_type_to_string(expected_operator_type), block_size,
        XNN_MIN_BLOCKSIZE);
    return xnn_status_invalid_parameter;
  }

  if (input_channels % block_size != 0) {
    xnn_log_error(
        "failed to create %s operator with input_channels: %zu, and "
        "block_size: %zu: expecting input_channels %% block_size == 0.",
        xnn_operator_type_to_string(expected_operator_type), input_channels,
        block_size);
    return xnn_status_invalid_parameter;
  }

  if (kernel_zero_point != 8 && kernel_zero_point != 0) {
    xnn_log_error("failed to create %s operator with %" PRIu8
                  " kernel zero point: kernel zero point must be equal to 8 "
                  "(unsigned weights) or 0 (signed weights)",
                  xnn_operator_type_to_string(expected_operator_type),
                  kernel_zero_point);
    return xnn_status_invalid_parameter;
  }
  // Assuming kernel_scale.size() is output_channels * num_blocks.
  size_t num_blocks = input_channels / block_size;
  for (size_t output_channel = 0; output_channel < output_channels;
       output_channel++) {
    for (size_t block_index = 0; block_index < num_blocks; block_index++) {
      size_t scale_index = output_channel * num_blocks + block_index;
      float fp32_scale = math_cvt_fp32_bf16(kernel_scale[scale_index]);
      if (fp32_scale <= 0.0f || !isnormal(fp32_scale)) {
        xnn_log_error(
            "failed to create %s operator with %.7g kernel scale in output "
            "channel #%zu, block #%zu: scale must be finite and positive",
            xnn_operator_type_to_string(expected_operator_type), fp32_scale,
            output_channel, block_index);
        return xnn_status_invalid_parameter;
      }
    }
  }

  struct xnn_f32_qb4w_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32_qb4w != NULL) {
    gemm_config->init.f32_qb4w(&params, output_min, output_max,
                               kernel_zero_point, block_size);
  }

  // We don't know input zero point until runtime, row sum is multiplied by it
  // during packing, so set it to 1.
  const struct xnn_qs8_qc4w_packing_params packing_params = {
      /*input_zero_point=*/1, kernel_zero_point};

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/block_size,
      /*blockwise_kernel_scale_params=*/kernel_scale,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*bias_element_size=*/sizeof(float),
      /*pack_gemm_gio_w,=*/NULL,
      /*pack_gemm_goi_w=*/NULL, &packing_params,
      /*extra_weights_bytes=*/sizeof(float),
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, gemm_ukernels, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f32_qb4w_gemm_config();
  return create_fully_connected_nc_qx8_f32_qb4w(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      fully_connected_op_out);
}

enum xnn_status create_fully_connected_nc_qd8_f32_qb4w_f16_scales(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const xnn_float16* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  const size_t num_blocks =
      (input_channels + block_size - 1) / block_size * output_channels;
  xnn_bfloat16* bf16_scale_buffer =
      (xnn_bfloat16*)xnn_allocate_memory(num_blocks * sizeof(xnn_bfloat16));
  for (size_t i = 0; i < num_blocks; ++i) {
    bf16_scale_buffer[i] =
        xnn_bfloat16_from_float(xnn_float16_to_float(kernel_scale[i]));
  }
  enum xnn_status status = create_fully_connected_nc_qx8_f32_qb4w(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, (const uint16_t*)bf16_scale_buffer, kernel, bias,
      output_min, output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      fully_connected_op_out);
  xnn_release_memory(bf16_scale_buffer);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qb4w_f16_scales(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const xnn_float16* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f32_qb4w_gemm_config();
  return create_fully_connected_nc_qd8_f32_qb4w_f16_scales(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qb4w_f16_scales(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const xnn_float16* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qdu8_f32_qb4w_gemm_config();
  return create_fully_connected_nc_qd8_f32_qb4w_f16_scales(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qdu8_f32_qb4w_gemm_config();
  return create_fully_connected_nc_qx8_f32_qb4w(
      input_channels, output_channels, input_stride, output_stride, block_size,
      kernel_zero_point, kernel_scale, kernel, bias, output_min, output_max,
      flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w,
      fully_connected_op_out);
}

enum xnn_status create_fully_connected_nc_qdx8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(expected_operator_type), output_min,
        output_max);
    return xnn_status_invalid_parameter;
  }

  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  const struct xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      NULL, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
      /*bias_element_size=*/sizeof(float),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      &packing_params,
      /*extra_weights_bytes=*/sizeof(float) * 2,
      xnn_init_qs8_qc8w_scale_fp32_params, bias,
      xnn_init_qs8_qc8w_scale_fp32_params, kernel_scale, &params,
      sizeof(params), gemm_config, gemm_ukernels, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f32_qc8w_gemm_config();
  return create_fully_connected_nc_qdx8_f32_qc8w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_scale, kernel, bias, output_min, output_max, flags, weights_cache,
      gemm_config, xnn_operator_type_fully_connected_nc_qd8_f32_qc8w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qdu8_f32_qc8w_gemm_config();
  return create_fully_connected_nc_qdx8_f32_qc8w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_scale, kernel, bias, output_min, output_max, flags, weights_cache,
      gemm_config, xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w,
      fully_connected_op_out);
}

enum xnn_status create_fully_connected_nc_qx8_f16_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  const xnn_float16 fp16_output_min = xnn_float16_from_float(output_min);
  const xnn_float16 fp16_output_max = xnn_float16_from_float(output_max);
  const float rounded_output_min = xnn_float16_to_float(fp16_output_min);
  const float rounded_output_max = xnn_float16_to_float(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be below upper bound",
        xnn_operator_type_to_string(expected_operator_type), rounded_output_min,
        rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f16_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, fp16_output_min, fp16_output_max);
  }

  const struct xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      NULL, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
      /*bias_element_size=*/sizeof(float),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      &packing_params,
      /*extra_weights_bytes=*/sizeof(float) * 2,
      xnn_init_qs8_qc8w_scale_fp32_params, bias,
      xnn_init_qs8_qc8w_scale_fp32_params, kernel_scale, &params,
      sizeof(params), gemm_config, gemm_ukernels, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f16_qc8w_gemm_config();
  return create_fully_connected_nc_qx8_f16_qc8w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_scale, kernel, bias, output_min, output_max, flags, weights_cache,
      gemm_config, xnn_operator_type_fully_connected_nc_qd8_f16_qc8w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qdu8_f16_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qdu8_f16_qc8w_gemm_config();
  return create_fully_connected_nc_qx8_f16_qc8w(
      input_channels, output_channels, input_stride, output_stride,
      kernel_scale, kernel, bias, output_min, output_max, flags, weights_cache,
      gemm_config, xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w,
      fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_f32_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  float* fp32_kernel_buffer = (float*)xnn_allocate_memory(
      input_channels * output_channels * sizeof(float));
  float* fp32_bias_buffer = NULL;
  const xnn_float16* f16_kernel = (const xnn_float16*)kernel;
  const xnn_float16* f16_bias = (const xnn_float16*)bias;
  for (size_t i = 0; i < input_channels * output_channels; ++i) {
    fp32_kernel_buffer[i] = xnn_float16_to_float(f16_kernel[i]);
  }
  if (bias && !(flags & XNN_FLAG_FP32_STATIC_BIASES)) {
    fp32_bias_buffer =
        (float*)xnn_allocate_memory(output_channels * sizeof(float));
    for (size_t i = 0; i < output_channels; ++i) {
      fp32_bias_buffer[i] = xnn_float16_to_float(f16_bias[i]);
    }
    bias = fp32_bias_buffer;
  }
  enum xnn_status status = xnn_create_fully_connected_nc_f32(
      input_channels, output_channels, input_stride, output_stride,
      fp32_kernel_buffer, bias, output_min, output_max, flags, weights_cache,
      fully_connected_op_out);
  xnn_release_memory(fp32_kernel_buffer);
  xnn_release_memory(fp32_bias_buffer);
  return status;
}

enum xnn_status create_fully_connected_nc_f32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(expected_operator_type), output_min,
        output_max);
    return xnn_status_invalid_parameter;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*filter_is_nibble=*/false,
      /*bias_element_size=*/sizeof(float),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      /*packing_params=*/NULL,
      /*extra_weights_bytes=*/0,
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, gemm_ukernels, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_bf16_f32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_bf16_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_bf16_f32));
    return xnn_status_unsupported_hardware;
  }

  return create_fully_connected_nc_f32(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, output_min, output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_bf16_f32, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_f32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config(flags);
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_gemm_config* gemm_nr2_config =
      xnn_init_f32_gemm_nr2_config(flags);

  // Select microkernel configuration based on output channels
  if (gemm_nr2_config != NULL) {
    const size_t nr = gemm_config->nr;
    const size_t nr2 = gemm_nr2_config->nr;
    size_t nr_overcompute = (nr - output_channels % nr) % nr;
    size_t nr2_overcompute = (nr2 - output_channels % nr2) % nr2;
    // Switch to alternative microkernel when:
    // 1. Alternative microkernel better supports less output channels, or
    // 2. Alternative microkernel has less overcompute and default wastes >1% of
    // output channels
    if (nr > output_channels || (nr2_overcompute < nr_overcompute &&
                                 nr_overcompute * 100 > output_channels)) {
      // Default microkernel is suboptimal, use a microkernel that better
      // supports less output channels.
      if (gemm_nr2_config->minmax.gemm[gemm_nr2_config->mr - 1]
              .function[XNN_UARCH_DEFAULT] != NULL) {
        gemm_config = gemm_nr2_config;
      }
    }
  }

  return create_fully_connected_nc_f32(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, output_min, output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_f32, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_pf32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_pf32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_pf32));
    return xnn_status_unsupported_hardware;
  }

  return create_fully_connected_nc_f32(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, output_min, output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_pf32, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const uint8_t* kernel, const float* bias, float output_min,
    float output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc4w));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc4w));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc4w),
        output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    xnn_log_error(
        "failed to create %s operator with XNN_FLAG_TRANSPOSE_WEIGHTS: not "
        "supported",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc4w));
    return xnn_status_unsupported_parameter;
  }

  for (size_t output_channel = 0; output_channel < output_channels;
       output_channel++) {
    if (kernel_scale[output_channel] <= 0.0f ||
        !isnormal(kernel_scale[output_channel])) {
      xnn_log_error(
          "failed to create %s operator with %.7g kernel scale in output "
          "channel #%zu: scale must be finite and positive",
          xnn_operator_type_to_string(
              xnn_operator_type_fully_connected_nc_f32_qc4w),
          kernel_scale[output_channel], output_channel);
      return xnn_status_invalid_parameter;
    }
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_f32_qc4w_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc4w));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_qc4w_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32_qc4w != NULL) {
    gemm_config->init.f32_qc4w(&params, output_min, output_max,
                               kernel_zero_point);
  }

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*bias_element_size=*/sizeof(float), (xnn_packw_gemm_gio_ukernel_fn)NULL,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      /*packing_params=*/NULL,
      /*extra_weights_bytes=*/sizeof(float),
      /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
      /*scale_params=*/kernel_scale,
      /*init_kernel_scale_params=*/NULL,
      /*kernel_scale_params=*/NULL, &params, sizeof(params), gemm_config,
      gemm_ukernels, xnn_operator_type_fully_connected_nc_f32_qc4w,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
        "failed to create %s operator with NaN output lower bound: lower bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc8w));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
        "failed to create %s operator with NaN output upper bound: upper bound "
        "must be non-NaN",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc8w));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%.7g, %.7g] output range: lower "
        "bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc8w),
        output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  for (size_t output_channel = 0; output_channel < output_channels;
       output_channel++) {
    if (kernel_scale[output_channel] <= 0.0f ||
        !isnormal(kernel_scale[output_channel])) {
      xnn_log_error(
          "failed to create %s operator with %.7g kernel scale in output "
          "channel #%zu: scale must be finite and positive",
          xnn_operator_type_to_string(
              xnn_operator_type_fully_connected_nc_f32_qc8w),
          kernel_scale[output_channel], output_channel);
      return xnn_status_invalid_parameter;
    }
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_f32_qc8w_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_fully_connected_nc_f32_qc8w));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation =
      (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1]
                                   .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
      /*bias_element_size=*/sizeof(float),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      /*packing_params=*/NULL,
      /*extra_weights_bytes=*/sizeof(float),
      /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
      /*scale_params=*/kernel_scale,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_fully_connected_nc_f32_qc8w,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qs8(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    float kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8),
        input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g kernel scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8),
        kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g output scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8),
        output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%" PRId8 ", %" PRId8
        "] output range: lower bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8),
        output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string(
                      xnn_operator_type_fully_connected_nc_qs8));
    return xnn_status_uninitialized;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel "
        "scale, and %.7g output scale: "
        "requantization scale %.7g is greater or equal to 256.0",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8),
        input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  assert(gemm_config != NULL);

  union xnn_qs8_qc8w_conv_minmax_params params;
  if XNN_LIKELY (gemm_config->init.qs8_qc8w != NULL) {
    gemm_config->init.qs8_qc8w(&params, output_zero_point, output_min,
                               output_max);
  }
  const struct xnn_qs8_packing_params packing_params = {
      .input_zero_point = input_zero_point,
  };
  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
      /*bias_element_size=*/sizeof(int32_t),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      &packing_params,
      /*extra_weights_bytes=*/sizeof(float),
      /*init_scale_params=*/xnn_init_qs8_to_qs8_qc8w_scale_fp32_params,
      /*scale_params=*/&requantization_scale,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, &gemm_config->minmax,
      xnn_operator_type_fully_connected_nc_qs8,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

enum xnn_status create_fully_connected_nc_qx8_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, float input_scale, const float* kernel_scale,
    const int8_t* kernel, const int32_t* bias, int8_t output_zero_point,
    float output_scale, int8_t output_min, int8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type, bool filter_is_nibble,
    const void* packing_params, xnn_operator_t* fully_connected_op_out) {
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(expected_operator_type), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g output scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(expected_operator_type), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%" PRId8 ", %" PRId8
        "] output range: lower bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(expected_operator_type), output_min,
        output_max);
    return xnn_status_invalid_parameter;
  }

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  float* requantization_scale =
      xnn_allocate_simd_memory(output_channels * sizeof(float));
  if (requantization_scale == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator packed weights",
                  output_channels * sizeof(float),
                  xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_out_of_memory;
  }

  for (size_t output_channel = 0; output_channel < output_channels;
       output_channel++) {
    requantization_scale[output_channel] =
        input_scale * kernel_scale[output_channel] / output_scale;
    if (requantization_scale[output_channel] >= 256.0f) {
      xnn_log_error(
          "failed to create %s operator with %.7g input scale, %.7g kernel "
          "scale, and %.7g output scale in output channel #%zu: "
          "requantization scale %.7g is greater or equal to 256.0",
          xnn_operator_type_to_string(expected_operator_type), input_scale,
          kernel_scale[output_channel], output_scale, output_channel,
          requantization_scale[output_channel]);

      xnn_release_simd_memory(requantization_scale);
      return xnn_status_unsupported_parameter;
    }
  }

  assert(gemm_config != NULL);

  union xnn_qs8_qc8w_conv_minmax_params params;
  if XNN_LIKELY (gemm_config->init.qs8_qc8w != NULL) {
    gemm_config->init.qs8_qc8w(&params, output_zero_point, output_min,
                               output_max);
  }

  enum xnn_status status = create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T, filter_is_nibble,
      /*bias_element_size=*/sizeof(int32_t),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi, packing_params,
      /*extra_weights_bytes=*/sizeof(float),
      /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
      /*scale_params=*/requantization_scale,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, &gemm_config->minmax, expected_operator_type,
      /*weights_cache=*/weights_cache, fully_connected_op_out);

  xnn_release_simd_memory(requantization_scale);
  return status;
}

enum xnn_status xnn_create_fully_connected_nc_qs8_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, const float* kernel_scale, const void* kernel,
    const int32_t* bias, int8_t output_zero_point, float output_scale,
    int8_t output_min, int8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc4w_gemm_config();
  const struct xnn_qs8_qc4w_packing_params packing_params = {
      .input_zero_point = input_zero_point,
      .kernel_zero_point = kernel_zero_point};
  return create_fully_connected_nc_qx8_qc8w(
      input_channels, output_channels, input_stride, output_stride, input_scale,
      kernel_scale, kernel, bias, output_zero_point, output_scale, output_min,
      output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qs8_qc4w, /*filter_is_nibble=*/true,
      &packing_params, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qs8_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  const struct xnn_qs8_packing_params packing_params = {.input_zero_point =
                                                            input_zero_point};
  return create_fully_connected_nc_qx8_qc8w(
      input_channels, output_channels, input_stride, output_stride, input_scale,
      kernel_scale, kernel, bias, output_zero_point, output_scale, output_min,
      output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_qs8_qc8w, /*filter_is_nibble=*/false,
      &packing_params, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_pqs8_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_pqs8_qc8w_gemm_config();
  const struct xnn_qs8_packing_params packing_params = {.input_zero_point =
                                                            input_zero_point};
  return create_fully_connected_nc_qx8_qc8w(
      input_channels, output_channels, input_stride, output_stride, input_scale,
      kernel_scale, kernel, bias, output_zero_point, output_scale, output_min,
      output_max, flags, weights_cache, gemm_config,
      xnn_operator_type_fully_connected_nc_pqs8_qc8w,
      /*filter_is_nibble=*/false, &packing_params, fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qu8(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, float kernel_scale, const uint8_t* kernel,
    const int32_t* bias, uint8_t output_zero_point, float output_scale,
    uint8_t output_min, uint8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8),
        input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g kernel scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8),
        kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
        "failed to create %s operator with %.7g output scale: scale must be "
        "finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8),
        output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
        "failed to create %s operator with [%" PRIu8 ", %" PRIu8
        "] output range: lower bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8),
        output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel "
        "scale, and %.7g output scale: "
        "requantization scale %.7g is greater or equal to 256.0",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8),
        input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  assert(gemm_config != NULL);

  union xnn_qu8_conv_minmax_params params;
  if XNN_LIKELY (gemm_config->init.qu8 != NULL) {
    gemm_config->init.qu8(&params, kernel_zero_point, requantization_scale,
                          output_zero_point, output_min, output_max);
  }
  const struct xnn_qu8_packing_params packing_params = {
      .input_zero_point = input_zero_point,
      .kernel_zero_point = kernel_zero_point,
  };
  return create_fully_connected_nc(
      input_channels, output_channels, input_stride, output_stride, kernel,
      bias, flags,
      /*block_size=*/0,
      /*blockwise_kernel_scale_params=*/NULL,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/false,
      /*bias_element_size=*/sizeof(int32_t),
      (xnn_packw_gemm_gio_ukernel_fn)gemm_config->pack_gemm_gio,
      (xnn_packw_gemm_goi_ukernel_fn)gemm_config->pack_gemm_goi,
      &packing_params,
      /*extra_weights_bytes=*/0,
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*init_kernel_scale_params=*/NULL, /*kernel_scale_params=*/NULL, &params,
      sizeof(params), gemm_config, &gemm_config->minmax,
      xnn_operator_type_fully_connected_nc_qu8,
      /*weights_cache=*/weights_cache, fully_connected_op_out);
}

static enum xnn_status reshape_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    enum xnn_operator_type expected_operator_type, size_t batch_size,
    uint32_t log2_input_element_size, uint32_t log2_filter_element_size,
    bool filter_is_nibble, bool dynamic_quantization,
    uint32_t log2_output_element_size, const void* params, size_t params_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
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
  if (filter_is_nibble) {
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*filter_is_nibble=*/false,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
      &fully_connected_op->params.f16_minmax,
      sizeof(fully_connected_op->params.f16_minmax),
      /*workspace_size=*/NULL, threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_pf16(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pf16, batch_size,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*filter_is_nibble=*/false,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
      &fully_connected_op->params.f32_qb4w_minmax,
      sizeof(fully_connected_op->params.f32_qb4w_minmax), workspace_size,
      threadpool);
}

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc4w,
      batch_size,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
      /*dynamic_quantization=*/true,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/0,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/0,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/0,
      // Pass 1 byte even though it is half byte, we handle the division via
      // filter_is_nibble == true.
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/true,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*filter_is_nibble=*/false,
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
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      /*filter_is_nibble=*/false,
      /*dynamic_quantization=*/false,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
      &fully_connected_op->params.qu8_conv_minmax,
      sizeof(fully_connected_op->params.qu8_conv_minmax),
      /*workspace_size=*/NULL, threadpool);
}

static enum xnn_status setup_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    enum xnn_operator_type expected_operator_type, const void* input,
    void* output, void* workspace, const void* quantization_params) {
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

  fully_connected_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_fully_connected_nc_f16(
    xnn_operator_t fully_connected_op, const void* input, void* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f16, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_pf16(
    xnn_operator_t fully_connected_op, const void* input, void* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pf16, input,
      output, workspace, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32_f16(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return xnn_setup_fully_connected_nc_f32(fully_connected_op, input, output);
}

enum xnn_status xnn_setup_fully_connected_nc_bf16_f32(
    xnn_operator_t fully_connected_op, const void* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_bf16_f32, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_pf32(
    xnn_operator_t fully_connected_op, const float* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pf32, input,
      output, workspace, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32_qc4w(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32_qc4w, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_f32_qc8w(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_f32_qc8w, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qc4w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f16_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qb4w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc4w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qc4w(
    xnn_operator_t fully_connected_op, const uint8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qb4w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f16_qc8w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f16_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op, const float* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qc4w,
      input, output, workspace, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qc8w(
    xnn_operator_t fully_connected_op, const float* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qc8w,
      input, output, workspace, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qb4w(
    xnn_operator_t fully_connected_op, const float* input, float* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qp8_f32_qb4w,
      input, output, workspace, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qd8_f32_qc8w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace,
    const struct xnn_quantization_params* quantization_params) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w,
      input, output, workspace, quantization_params);
}

enum xnn_status xnn_setup_fully_connected_nc_qs8(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qs8_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc4w, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qs8_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qs8_qc8w, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_pqs8_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output,
    void* workspace) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_pqs8_qc8w, input,
      output, workspace, /*quantization_params=*/NULL);
}

enum xnn_status xnn_setup_fully_connected_nc_qu8(
    xnn_operator_t fully_connected_op, const uint8_t* input, uint8_t* output) {
  return setup_fully_connected_nc(
      fully_connected_op, xnn_operator_type_fully_connected_nc_qu8, input,
      output, /*workspace=*/NULL, /*quantization_params=*/NULL);
}
