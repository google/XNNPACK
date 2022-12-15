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
#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/pack.h>
#include <xnnpack/params.h>


static enum xnn_status create_fully_connected_nc(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_gemm_io_w_fn pack_gemm_io_w,
    xnn_pack_gemm_goi_w_fn pack_gemm_goi_w,
    const void* packing_params,
    int packed_weights_padding_byte,
    const void* params,
    size_t params_size,
    const struct gemm_parameters* gemm_parameters,
    const struct gemm_fused_ukernels* gemm_ukernels,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    xnn_caches_t caches,
    xnn_operator_t* fully_connected_op_out)
{
  xnn_operator_t fully_connected_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error(
      "failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (input_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu input channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), input_channels);
    goto error;
  }

  if (output_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu output channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), output_channels);
    goto error;
  }

  if (input_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of input channels (%zu)",
      xnn_operator_type_to_string(operator_type), input_stride, input_channels);
    goto error;
  }

  if (output_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu)",
      xnn_operator_type_to_string(operator_type), output_stride, output_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  fully_connected_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (fully_connected_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  if (caches != NULL) {
    fully_connected_op->weights_cache = caches->weights_cache;
  }

  const uint32_t nr = gemm_parameters->nr;
  const uint32_t kr = UINT32_C(1) << gemm_parameters->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_parameters->log2_sr;

  const size_t n_stride = round_up(output_channels, nr);
  const size_t k_stride = round_up_po2(input_channels, kr * sr);

  const size_t packed_weights_size = n_stride * (bias_element_size + (k_stride << log2_filter_element_size));
  size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      fully_connected_op, aligned_total_weights_size, packed_weights_padding_byte);
  if (weights_ptr == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      packed_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
    aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    pack_gemm_io_w(
      output_channels, input_channels,
      nr, kr, sr,
      kernel, bias,
      weights_ptr,
      packing_params);
  } else {
    pack_gemm_goi_w(
      1, output_channels, input_channels,
      nr, kr, sr,
      kernel, bias,
      weights_ptr,
      0 /* extra bytes */,
      packing_params);
  }

  if (use_weights_cache(fully_connected_op)) {
    fully_connected_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
        fully_connected_op->weights_cache, weights_ptr, aligned_total_weights_size);
  }

  fully_connected_op->group_input_channels = input_channels;
  fully_connected_op->group_output_channels = output_channels;
  fully_connected_op->input_pixel_stride = input_stride;
  fully_connected_op->output_pixel_stride = output_stride;

  memcpy(&fully_connected_op->params, params, params_size);
  fully_connected_op->type = operator_type;
  fully_connected_op->flags = flags;

  const size_t mr = gemm_parameters->mr;
  fully_connected_op->ukernel.type = xnn_microkernel_type_gemm;
  fully_connected_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
    .mr = mr,
    .nr = nr,
    .kr = kr,
    .sr = sr,
  };

  assert(XNN_MAX_MR >= mr);
  fully_connected_op->ukernel.gemm.gemm_cases[0] = gemm_ukernels->gemm[0];
  for (size_t i = 1; i < mr; i++) {
    fully_connected_op->ukernel.gemm.gemm_cases[i] = gemm_ukernels->gemm[mr-1];
  }

  fully_connected_op->state = xnn_run_state_invalid;

  *fully_connected_op_out = fully_connected_op;
  return xnn_status_success;

error:
  xnn_delete_operator(fully_connected_op);
  return status;
}

static enum xnn_status setup_fully_connected_nc(
  xnn_operator_t fully_connected_op,
  enum xnn_operator_type expected_operator_type,
  size_t batch_size,
  const void* input,
  void* output,
  uint32_t datatype_init_flags,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  size_t params_size,
  size_t num_threads)
{
  if (fully_connected_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(fully_connected_op->type));
    return xnn_status_invalid_parameter;
  }
  fully_connected_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(fully_connected_op->type));
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    fully_connected_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  if (fully_connected_op->weights_cache != NULL &&
      !xnn_weights_cache_is_finalized(fully_connected_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
      xnn_operator_type_to_string(fully_connected_op->type));
    return xnn_status_invalid_state;
  }

  fully_connected_op->batch_size = 1;
  fully_connected_op->input_height = batch_size;
  fully_connected_op->input_width = 1;
  fully_connected_op->input = input;

  fully_connected_op->output_height = batch_size;
  fully_connected_op->output_width = 1;
  fully_connected_op->output = output;

  const size_t input_channels = fully_connected_op->group_input_channels;
  const size_t output_channels = fully_connected_op->group_output_channels;

  uint32_t mr = fully_connected_op->ukernel.gemm.mr;
  const uint32_t nr = fully_connected_op->ukernel.gemm.nr;

  struct xnn_hmp_gemm_ukernel gemm_ukernel = fully_connected_op->ukernel.gemm.gemm_cases[mr-1];
  if (batch_size == 1 && fully_connected_op->ukernel.gemm.gemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernel = fully_connected_op->ukernel.gemm.gemm_cases[0];
    mr = 1;
  }

  fully_connected_op->context.gemm = (struct gemm_context) {
    .k_scaled = input_channels << log2_input_element_size,
    .w_stride = bias_element_size +
        (round_up_po2(input_channels, fully_connected_op->ukernel.gemm.kr * fully_connected_op->ukernel.gemm.sr) << log2_input_element_size),
    .a = input,
    .a_stride = fully_connected_op->input_pixel_stride << log2_input_element_size,
    .packed_w = packed_weights(fully_connected_op),
    .c = output,
    .cm_stride = fully_connected_op->output_pixel_stride << log2_output_element_size,
    .cn_stride = nr << log2_output_element_size,
    .log2_csize = log2_output_element_size,
    .ukernel = gemm_ukernel,
  };
  memcpy(&fully_connected_op->context.gemm.params, params, params_size);
  fully_connected_op->context.gemm.fused_params = &fully_connected_op->context.gemm.params;

  #if XNN_TEST_MODE
    const size_t nc = nr;
  #else
    size_t nc = output_channels;
    if (num_threads > 1) {
      const size_t num_other_tiles = divide_round_up(batch_size, mr);
      const size_t target_tiles_per_thread = 5;
      const size_t max_nc = divide_round_up(output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
      if (max_nc < nc) {
        nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
      }
    }
  #endif
  #if XNN_MAX_UARCH_TYPES > 1
    if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
      fully_connected_op->compute.type = xnn_parallelization_type_2d_tile_2d_with_uarch;
      fully_connected_op->compute.task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_gemm;
    } else {
      fully_connected_op->compute.type = xnn_parallelization_type_2d_tile_2d;
      fully_connected_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
    }
  #else
    fully_connected_op->compute.type = xnn_parallelization_type_2d_tile_2d;
    fully_connected_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
  #endif
  fully_connected_op->compute.range[0] = batch_size;
  fully_connected_op->compute.range[1] = output_channels;
  fully_connected_op->compute.tile[0] = mr;
  fully_connected_op->compute.tile[1] = nc;
  fully_connected_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_create_fully_connected_nc_f16(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const void* kernel,
    const void* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_caches_t caches,
    xnn_operator_t* fully_connected_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_f16));
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

  union xnn_f16_minmax_params params;
  if XNN_LIKELY(xnn_params.f16.gemm.init.f16 != NULL) {
    xnn_params.f16.gemm.init.f16(&params, fp16_output_min, fp16_output_max);
  }
  xnn_pack_gemm_io_w_fn pack_gemm_io_w = (xnn_pack_gemm_io_w_fn) xnn_pack_f16_gemm_io_w;
  xnn_pack_gemm_goi_w_fn pack_gemm_goi_w = (xnn_pack_gemm_goi_w_fn) xnn_pack_f16_gemm_goi_w;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_gemm_io_w = (xnn_pack_gemm_io_w_fn) xnn_pack_f32_to_f16_gemm_io_w;
    pack_gemm_goi_w = (xnn_pack_gemm_goi_w_fn) xnn_pack_f32_to_f16_gemm_goi_w;
  }
  return create_fully_connected_nc(
    input_channels, output_channels,
    input_stride, output_stride,
    kernel, bias, flags,
    1 /* log2(sizeof(filter element)) = log2(sizeof(uint16_t)) */,
    sizeof(uint16_t) /* sizeof(bias element) */,
    pack_gemm_io_w,
    pack_gemm_goi_w,
    NULL /* packing params */, 0 /* packed weights padding byte */,
    &params, sizeof(params),
    &xnn_params.f16.gemm, &xnn_params.f16.gemm.minmax,
    XNN_INIT_FLAG_F16,
    xnn_operator_type_fully_connected_nc_f16,
    caches,
    fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_f32(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_caches_t caches,
    xnn_operator_t* fully_connected_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &xnn_params.f32.gemm.minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && xnn_params.f32.gemm.linear.gemm[xnn_params.f32.gemm.mr-1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &xnn_params.f32.gemm.linear;
  }

  union xnn_f32_minmax_params params;
  if XNN_LIKELY(xnn_params.f32.gemm.init.f32 != NULL) {
    xnn_params.f32.gemm.init.f32(&params, output_min, output_max);
  }
  return create_fully_connected_nc(
    input_channels, output_channels,
    input_stride, output_stride,
    kernel, bias, flags,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    (xnn_pack_gemm_io_w_fn) xnn_pack_f32_gemm_io_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_f32_gemm_goi_w,
    NULL /* packing params */, 0 /* packed weights padding byte */,
    &params, sizeof(params),
    &xnn_params.f32.gemm, gemm_ukernels,
    XNN_INIT_FLAG_F32,
    xnn_operator_type_fully_connected_nc_f32,
    caches,
    fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qs8(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
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
    xnn_caches_t caches,
    xnn_operator_t* fully_connected_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  union xnn_qs8_conv_minmax_params params;
  if XNN_LIKELY(xnn_params.qs8.gemm.init.qs8 != NULL) {
    xnn_params.qs8.gemm.init.qs8(&params, requantization_scale, output_zero_point, output_min, output_max);
  }
  const struct xnn_qs8_packing_params packing_params = {
    .input_zero_point = input_zero_point,
  };
  return create_fully_connected_nc(
    input_channels, output_channels,
    input_stride, output_stride,
    kernel, bias, flags,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    (xnn_pack_gemm_io_w_fn) xnn_pack_qs8_gemm_io_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_qs8_gemm_goi_w,
    &packing_params, 0 /* packed weights padding byte */,
    &params, sizeof(params),
    &xnn_params.qs8.gemm, &xnn_params.qs8.gemm.minmax,
    XNN_INIT_FLAG_QS8,
    xnn_operator_type_fully_connected_nc_qs8,
    caches,
    fully_connected_op_out);
}

enum xnn_status xnn_create_fully_connected_nc_qu8(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
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
    xnn_caches_t caches,
    xnn_operator_t* fully_connected_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qu8),
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
  return create_fully_connected_nc(
    input_channels, output_channels,
    input_stride, output_stride,
    kernel, bias, flags,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    (xnn_pack_gemm_io_w_fn) xnn_pack_qu8_gemm_io_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_qu8_gemm_goi_w,
    &packing_params, kernel_zero_point /* packed weights padding byte */,
    &params, sizeof(params),
    &xnn_params.qu8.gemm, &xnn_params.qu8.gemm.minmax,
    XNN_INIT_FLAG_QU8,
    xnn_operator_type_fully_connected_nc_qu8,
    caches,
    fully_connected_op_out);
}

enum xnn_status xnn_setup_fully_connected_nc_f16(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_fully_connected_nc(
    fully_connected_op, xnn_operator_type_fully_connected_nc_f16,
    batch_size,
    input, output,
    XNN_INIT_FLAG_F32,
    1 /* log2(sizeof(input element)) = log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(filter element)) = log2(sizeof(uint16_t)) */,
    sizeof(uint16_t) /* sizeof(bias element) */,
    1 /* log2(sizeof(output element)) = log2(sizeof(uint16_t)) */,
    &fully_connected_op->params.f16_minmax,
    sizeof(fully_connected_op->params.f16_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_fully_connected_nc_f32(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_fully_connected_nc(
    fully_connected_op, xnn_operator_type_fully_connected_nc_f32,
    batch_size,
    input, output,
    XNN_INIT_FLAG_F32,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    &fully_connected_op->params.f32_minmax,
    sizeof(fully_connected_op->params.f32_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_fully_connected_nc_qs8(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    const int8_t* input,
    int8_t* output,
    pthreadpool_t threadpool)
{
  return setup_fully_connected_nc(
    fully_connected_op, xnn_operator_type_fully_connected_nc_qs8,
    batch_size,
    input, output,
    XNN_INIT_FLAG_QS8,
    0 /* log2(sizeof(input element)) = log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(int8_t)) */,
    &fully_connected_op->params.qs8_conv_minmax,
    sizeof(fully_connected_op->params.qs8_conv_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_fully_connected_nc_qu8(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  return setup_fully_connected_nc(
    fully_connected_op, xnn_operator_type_fully_connected_nc_qu8,
    batch_size,
    input, output,
    XNN_INIT_FLAG_QU8,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(uint8_t)) */,
    &fully_connected_op->params.qu8_conv_minmax,
    sizeof(fully_connected_op->params.qu8_conv_minmax),
    pthreadpool_get_threads_count(threadpool));
}
