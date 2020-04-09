// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


enum xnn_status xnn_create_clamp_nc_u8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* clamp_op_out)
{
  xnn_operator_t clamp_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Clamp operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Clamp operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create Clamp operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create Clamp operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Clamp operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      output_min, output_max);
    goto error;
  }

  status = xnn_status_out_of_memory;

  clamp_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (clamp_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Clamp operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  clamp_op->channels = channels;
  clamp_op->input_pixel_stride = input_stride;
  clamp_op->output_pixel_stride = output_stride;
  clamp_op->u8_minmax_params = xnn_init_u8_minmax_params(output_min, output_max);

  clamp_op->type = xnn_operator_type_clamp_nc_u8;
  clamp_op->ukernel.type = xnn_ukernel_type_clamp;

  clamp_op->state = xnn_run_state_invalid;

  *clamp_op_out = clamp_op;
  return xnn_status_success;

error:
  xnn_delete_operator(clamp_op);
  return status;
}

enum xnn_status xnn_create_clamp_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* clamp_op_out)
{
  xnn_operator_t clamp_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Clamp operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Clamp operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create Clamp operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create Clamp operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create Clamp operator with NaN output lower bound: lower bound must be non-NaN");
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create Clamp operator with NaN output upper bound: upper bound must be non-NaN");
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Clamp operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      output_min, output_max);
    goto error;
  }

  status = xnn_status_out_of_memory;

  clamp_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (clamp_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Clamp operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  clamp_op->channels = channels;
  clamp_op->input_pixel_stride = input_stride;
  clamp_op->output_pixel_stride = output_stride;
  clamp_op->f32_minmax_params = xnn_init_f32_minmax_params(output_min, output_max);

  clamp_op->type = xnn_operator_type_clamp_nc_f32;
  clamp_op->ukernel.type = xnn_ukernel_type_clamp;

  clamp_op->state = xnn_run_state_invalid;

  *clamp_op_out = clamp_op;
  return xnn_status_success;

error:
  xnn_delete_operator(clamp_op);
  return status;
}

enum xnn_status xnn_setup_clamp_nc_u8(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (clamp_op->type != xnn_operator_type_clamp_nc_u8) {
    xnn_log_error("failed to setup Clamp (NC, U8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  clamp_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Clamp operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    clamp_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = clamp_op->channels;
  const size_t input_stride = clamp_op->input_pixel_stride;
  const size_t output_stride = clamp_op->output_pixel_stride;
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    clamp_op->context.univector_contiguous = (struct univector_contiguous_context) {
      .x = input,
      .x_stride = input_stride * sizeof(uint8_t),
      .y = output,
      .y_stride = output_stride * sizeof(uint8_t),
      .ukernel = xnn_params.u8.clamp,
      .params.u8_output = clamp_op->u8_minmax_params,
    };
    clamp_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    clamp_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
    clamp_op->compute.range[0] = batch_size * channels * sizeof(uint8_t);
    clamp_op->compute.tile[0] = block_size;
  } else {
    clamp_op->context.univector_strided = (struct univector_strided_context) {
      .n = channels * sizeof(uint8_t),
      .x = input,
      .x_stride = input_stride * sizeof(uint8_t),
      .y = output,
      .y_stride = output_stride * sizeof(uint8_t),
      .ukernel = xnn_params.u8.clamp,
      .params.u8_output = clamp_op->u8_minmax_params,
    };
    clamp_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    clamp_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
    clamp_op->compute.range[0] = batch_size;
    clamp_op->compute.tile[0] = 1;
  }
  clamp_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_clamp_nc_f32(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (clamp_op->type != xnn_operator_type_clamp_nc_f32) {
    xnn_log_error("failed to setup Clamp (NC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  clamp_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Clamp operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    clamp_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = clamp_op->channels;
  const size_t input_stride = clamp_op->input_pixel_stride;
  const size_t output_stride = clamp_op->output_pixel_stride;
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    clamp_op->context.univector_contiguous = (struct univector_contiguous_context) {
      .x = input,
      .x_stride = input_stride * sizeof(float),
      .y = output,
      .y_stride = output_stride * sizeof(float),
      .ukernel = xnn_params.f32.clamp,
      .params.f32_output = clamp_op->f32_minmax_params,
    };
    clamp_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    clamp_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
    clamp_op->compute.range[0] = batch_size * channels * sizeof(float);
    clamp_op->compute.tile[0] = block_size;
  } else {
    clamp_op->context.univector_strided = (struct univector_strided_context) {
      .n = channels * sizeof(float),
      .x = input,
      .x_stride = input_stride * sizeof(float),
      .y = output,
      .y_stride = output_stride * sizeof(float),
      .ukernel = xnn_params.f32.clamp,
      .params.f32_output = clamp_op->f32_minmax_params,
    };
    clamp_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    clamp_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
    clamp_op->compute.range[0] = batch_size;
    clamp_op->compute.tile[0] = 1;
  }
  clamp_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
