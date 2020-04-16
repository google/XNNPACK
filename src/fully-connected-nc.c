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
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


enum xnn_status xnn_create_fully_connected_nc_q8(
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
    xnn_operator_t* fully_connected_op_out)
{
  xnn_operator_t fully_connected_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Fully Connected operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (input_channels == 0) {
    xnn_log_error(
      "failed to create Fully Connected operator with %zu input channels: number of channels must be non-zero",
      input_channels);
    goto error;
  }

  if (output_channels == 0) {
    xnn_log_error(
      "failed to create Fully Connected operator with %zu output channels: number of channels must be non-zero",
      output_channels);
    goto error;
  }

  if (input_stride < input_channels) {
    xnn_log_error(
      "failed to create Fully Connected operator with input element stride of %zu: "
      "stride must be at least as large as the number of input channels (%zu)",
      input_stride, input_channels);
    goto error;
  }

  if (output_stride < output_channels) {
    xnn_log_error(
      "failed to create Fully Connected operator with output element stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu)",
      output_stride, output_channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create Fully Connected operator with %.7g input scale: scale must be finite, normalized, and positive",
      input_scale);
    goto error;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create Fully Connected operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      kernel_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create Fully Connected operator with %.7g output scale: scale must be finite, normalized, and positive",
      output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Fully Connected operator with [%" PRIu8 ", %" PRIu8 "] output range: "
      "range min must be below range max",
      output_min, output_max);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 1.0f) {
    xnn_log_error(
      "failed to create Fully Connected operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 1.0",
      input_scale, kernel_scale, output_scale, requantization_scale);
    goto error;
  }

  status = xnn_status_out_of_memory;

  fully_connected_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (fully_connected_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Fully Connected operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  const uint32_t nr = xnn_params.q8.gemm.nr;
  const uint32_t kr = UINT32_C(1) << xnn_params.q8.gemm.log2_kr;

  const size_t n_stride = round_up(output_channels, nr);
  const size_t k_stride = round_up_po2(input_channels, kr);

  fully_connected_op->packed_weights = xnn_allocate_simd_memory(n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
  if (fully_connected_op->packed_weights == NULL) {
    xnn_log_error("failed to allocate %zu bytes for packed weights",
      n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
    goto error;
  }
  memset(fully_connected_op->packed_weights, kernel_zero_point, n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    xnn_pack_q8_gemm_io_w(
      output_channels, input_channels,
      nr, kr,
      input_zero_point, kernel_zero_point,
      kernel, bias,
      fully_connected_op->packed_weights);
  } else {
    xnn_pack_q8_gemm_goi_w(
      1, output_channels, input_channels,
      nr, kr,
      input_zero_point, kernel_zero_point,
      kernel, bias,
      fully_connected_op->packed_weights);
  }

  fully_connected_op->group_input_channels = input_channels;
  fully_connected_op->group_output_channels = output_channels;
  fully_connected_op->input_pixel_stride = input_stride;
  fully_connected_op->output_pixel_stride = output_stride;

  fully_connected_op->kernel_zero_point = kernel_zero_point;

  fully_connected_op->q8_gemm_params =
    xnn_init_q8_gemm_params(
      input_zero_point, kernel_zero_point,
      requantization_scale, output_zero_point, output_min, output_max);

  fully_connected_op->type = xnn_operator_type_fully_connected_nc_q8;

  fully_connected_op->ukernel.type = xnn_ukernel_type_gemm;
  fully_connected_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
    .general_case = xnn_params.q8.gemm.minmax.gemm,
    .mr = xnn_params.q8.gemm.mr,
    .nr = nr,
    .kr = kr,
  };

  fully_connected_op->state = xnn_run_state_invalid;

  *fully_connected_op_out = fully_connected_op;
  return xnn_status_success;

error:
  xnn_delete_operator(fully_connected_op);
  return status;
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
    xnn_operator_t* fully_connected_op_out)
{
  xnn_operator_t fully_connected_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Fully Connected operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (input_channels == 0) {
    xnn_log_error(
      "failed to create Fully Connected operator with %zu input channels: number of channels must be non-zero",
      input_channels);
    goto error;
  }

  if (output_channels == 0) {
    xnn_log_error(
      "failed to create Fully Connected operator with %zu output channels: number of channels must be non-zero",
      output_channels);
    goto error;
  }

  if (input_stride < input_channels) {
    xnn_log_error(
      "failed to create Fully Connected operator with input element stride of %zu: "
      "stride must be at least as large as the number of input channels (%zu)",
      input_stride, input_channels);
    goto error;
  }

  if (output_stride < output_channels) {
    xnn_log_error(
      "failed to create Fully Connected operator with output element stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu)",
      output_stride, output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create Fully Connected operator with NaN output lower bound: lower bound must be non-NaN");
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create Fully Connected operator with NaN output upper bound: upper bound must be non-NaN");
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Fully Connected operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      output_min, output_max);
    goto error;
  }

  status = xnn_status_out_of_memory;

  fully_connected_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (fully_connected_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Fully Connected operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  const uint32_t nr = xnn_params.f32.gemm.nr;
  const uint32_t kr = UINT32_C(1) << xnn_params.f32.gemm.log2_kr;
  const uint32_t sr = UINT32_C(1) << xnn_params.f32.gemm.log2_sr;

  const size_t n_stride = round_up(output_channels, nr);
  const size_t k_stride = round_up_po2(input_channels, kr);

  fully_connected_op->packed_weights = xnn_allocate_simd_memory(n_stride * (k_stride * sizeof(float) + sizeof(float)));
  if (fully_connected_op->packed_weights == NULL) {
    xnn_log_error("failed to allocate %zu bytes for packed weights",
      n_stride * (k_stride * sizeof(float) + sizeof(float)));
    goto error;
  }
  memset(fully_connected_op->packed_weights, 0, n_stride * (k_stride * sizeof(float) + sizeof(float)));

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    xnn_pack_f32_gemm_io_w(
      output_channels, input_channels,
      nr, kr, sr,
      kernel, bias,
      fully_connected_op->packed_weights);
  } else {
    xnn_pack_f32_gemm_goi_w(
      1, output_channels, input_channels,
      nr, kr, sr,
      kernel, bias,
      fully_connected_op->packed_weights);
  }

  fully_connected_op->group_input_channels = input_channels;
  fully_connected_op->group_output_channels = output_channels;
  fully_connected_op->input_pixel_stride = input_stride;
  fully_connected_op->output_pixel_stride = output_stride;

  fully_connected_op->f32_minmax_params = xnn_init_f32_minmax_params(output_min, output_max);

  fully_connected_op->type = xnn_operator_type_fully_connected_nc_f32;

  const struct gemm_fused_ukernels* ukernels = &xnn_params.f32.gemm.minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && xnn_params.f32.gemm.linear.gemm.function[XNN_UARCH_DEFAULT] != NULL) {
    ukernels = &xnn_params.f32.gemm.linear;
  }

  fully_connected_op->ukernel.type = xnn_ukernel_type_gemm;
  fully_connected_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
    .general_case = ukernels->gemm,
    .mr1_case = ukernels->gemm1,
    .mr = xnn_params.f32.gemm.mr,
    .nr = nr,
    .kr = kr,
  };

  fully_connected_op->state = xnn_run_state_invalid;

  *fully_connected_op_out = fully_connected_op;
  return xnn_status_success;

error:
  xnn_delete_operator(fully_connected_op);
  return status;
}

static enum xnn_status setup_fully_connected_nc(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  const void* input,
  void* output,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  size_t num_threads)
{
  fully_connected_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Fully Connected operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    fully_connected_op->state = xnn_run_state_skip;
    return xnn_status_success;
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

  struct xnn_hmp_gemm_ukernel gemm_ukernel = fully_connected_op->ukernel.gemm.general_case;
  if (batch_size == 1 && fully_connected_op->ukernel.gemm.mr1_case.function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernel = fully_connected_op->ukernel.gemm.mr1_case;
    mr = 1;
  }

  fully_connected_op->context.gemm = (struct gemm_context) {
    .k_scaled = input_channels << log2_input_element_size,
    .w_stride = (round_up_po2(input_channels, fully_connected_op->ukernel.gemm.kr) << log2_input_element_size) + bias_element_size,
    .a = input,
    .a_stride = fully_connected_op->input_pixel_stride << log2_input_element_size,
    .packed_w = fully_connected_op->packed_weights,
    .c = output,
    .cm_stride = fully_connected_op->output_pixel_stride << log2_output_element_size,
    .cn_stride = nr << log2_output_element_size,
    .log2_csize = log2_output_element_size,
    .ukernel = gemm_ukernel,
  };
  memcpy(&fully_connected_op->context.gemm.params, params, sizeof(fully_connected_op->context.gemm.params));

  size_t nc = output_channels;
  if (num_threads > 1) {
    const size_t num_other_tiles = divide_round_up(batch_size, mr);
    const size_t target_tiles_per_thread = 5;
    const size_t max_nc = divide_round_up(output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
    if (max_nc < nc) {
      nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
    }
  }
  fully_connected_op->compute.type = xnn_parallelization_type_2d_tile_2d;
  fully_connected_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
  fully_connected_op->compute.range[0] = batch_size;
  fully_connected_op->compute.range[1] = output_channels;
  fully_connected_op->compute.tile[0] = mr;
  fully_connected_op->compute.tile[1] = nc;
  fully_connected_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_fully_connected_nc_q8(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (fully_connected_op->type != xnn_operator_type_fully_connected_nc_q8) {
    xnn_log_error("failed to setup Fully Connected (NC, Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_fully_connected_nc(
    fully_connected_op,
    batch_size,
    input, output,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(uint8_t)) */,
    &fully_connected_op->q8_gemm_params,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_fully_connected_nc_f32(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (fully_connected_op->type != xnn_operator_type_fully_connected_nc_f32) {
    xnn_log_error("failed to setup Fully Connected (NC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_fully_connected_nc(
    fully_connected_op,
    batch_size,
    input, output,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    &fully_connected_op->f32_minmax_params,
    pthreadpool_get_threads_count(threadpool));
}
