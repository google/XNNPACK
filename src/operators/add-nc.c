// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
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


enum xnn_status xnn_create_add_nc_q8(
    size_t channels,
    size_t a_stride,
    size_t b_stride,
    size_t sum_stride,
    uint8_t a_zero_point,
    float a_scale,
    uint8_t b_zero_point,
    float b_scale,
    uint8_t sum_zero_point,
    float sum_scale,
    uint8_t sum_min,
    uint8_t sum_max,
    uint32_t flags,
    xnn_operator_t* add_op_out)
{
  xnn_operator_t add_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Add operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Add operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (a_stride < channels) {
    xnn_log_error(
      "failed to create Add operator with A element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      a_stride, channels);
    goto error;
  }

  if (b_stride < channels) {
    xnn_log_error(
      "failed to create Add operator with B element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      b_stride, channels);
    goto error;
  }

  if (sum_stride < channels) {
    xnn_log_error(
      "failed to create Add operator with Sum element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      sum_stride, channels);
    goto error;
  }

  if (a_scale <= 0.0f || !isnormal(a_scale)) {
    xnn_log_error(
      "failed to create Add operator with %.7g A scale: scale must be finite, normalized, and positive", a_scale);
    goto error;
  }

  if (b_scale <= 0.0f || !isnormal(b_scale)) {
    xnn_log_error(
      "failed to create Add operator with %.7g B scale: scale must be finite, normalized, and positive", b_scale);
    goto error;
  }

  if (sum_scale <= 0.0f || !isnormal(sum_scale)) {
    xnn_log_error(
      "failed to create Add operator with %.7g output scale: scale must be finite, normalized, and positive",
      sum_scale);
    goto error;
  }

  if (sum_min >= sum_max) {
    xnn_log_error(
      "failed to create Add operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      sum_min, sum_max);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  const float a_output_scale = a_scale / sum_scale;
  if (a_output_scale < 0x1.0p-14f || a_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create Add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
      a_output_scale);
    goto error;
  }

  const float b_output_scale = b_scale / sum_scale;
  if (b_output_scale < 0x1.0p-14f || b_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create Add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
      b_output_scale);
    goto error;
  }

  status = xnn_status_out_of_memory;

  add_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (add_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Add operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  add_op->channels = channels;
  add_op->input_pixel_stride = a_stride;
  add_op->input2_pixel_stride = b_stride;
  add_op->output_pixel_stride = sum_stride;
  add_op->q8_add_params =
    xnn_init_q8_add_params(
      a_zero_point, b_zero_point, sum_zero_point,
      a_scale / sum_scale, b_scale / sum_scale,
      sum_min, sum_max);

  add_op->type = xnn_operator_type_add_nc_q8;
  add_op->ukernel.type = xnn_ukernel_type_add;

  add_op->state = xnn_run_state_invalid;

  *add_op_out = add_op;
  return xnn_status_success;

error:
  xnn_delete_operator(add_op);
  return status;
}

enum xnn_status xnn_create_add_nc_f32(
    size_t channels,
    size_t a_stride,
    size_t b_stride,
    size_t sum_stride,
    float sum_min,
    float sum_max,
    uint32_t flags,
    xnn_operator_t* add_op_out)
{
  xnn_operator_t add_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Add operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create add operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (a_stride < channels) {
    xnn_log_error(
      "failed to create Add operator with A element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      a_stride, channels);
    goto error;
  }

  if (b_stride < channels) {
    xnn_log_error(
      "failed to create Add operator with B element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      b_stride, channels);
    goto error;
  }

  if (sum_stride < channels) {
    xnn_log_error(
      "failed to create Add operator with Sum element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      sum_stride, channels);
    goto error;
  }

  if (isnan(sum_min)) {
    xnn_log_error(
      "failed to create Add operator with NaN output lower bound: lower bound must be non-NaN");
    goto error;
  }

  if (isnan(sum_max)) {
    xnn_log_error(
      "failed to create Add operator with NaN output upper bound: upper bound must be non-NaN");
    goto error;
  }

  if (sum_min >= sum_max) {
    xnn_log_error(
      "failed to create Add operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      sum_min, sum_max);
    goto error;
  }

  status = xnn_status_out_of_memory;

  add_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (add_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Add operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  add_op->channels = channels;
  add_op->input_pixel_stride = a_stride;
  add_op->input2_pixel_stride = b_stride;
  add_op->output_pixel_stride = sum_stride;
  add_op->f32_minmax_params = xnn_init_f32_minmax_params(sum_min, sum_max);

  add_op->type = xnn_operator_type_add_nc_f32;
  add_op->ukernel.type = xnn_ukernel_type_add;

  add_op->state = xnn_run_state_invalid;

  *add_op_out = add_op;
  return xnn_status_success;

error:
  xnn_delete_operator(add_op);
  return status;
}

enum xnn_status xnn_setup_add_nc_q8(
    xnn_operator_t add_op,
    size_t batch_size,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* sum,
    pthreadpool_t threadpool)
{
  if (add_op->type != xnn_operator_type_add_nc_q8) {
    xnn_log_error("failed to setup Add (NC, Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  add_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Add operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    add_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = add_op->channels;
  const size_t a_stride = add_op->input_pixel_stride;
  const size_t b_stride = add_op->input2_pixel_stride;
  const size_t sum_stride = add_op->output_pixel_stride;
  if ((((a_stride ^ channels) | (b_stride ^ channels) | (sum_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    add_op->context.add_contiguous = (struct add_contiguous_context) {
      .a = a,
      .b = b,
      .y = sum,
      .params.q8 = add_op->q8_add_params,
      .ukernel = xnn_params.q8.vadd,
    };
    add_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    add_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_add_contiguous;
    add_op->compute.range[0] = batch_size * channels * sizeof(uint8_t);
    add_op->compute.tile[0] = block_size;
  } else {
    add_op->context.add_strided = (struct add_strided_context) {
      .a = a,
      .a_stride = a_stride * sizeof(uint8_t),
      .b = b,
      .b_stride = b_stride * sizeof(uint8_t),
      .y = sum,
      .y_stride = sum_stride * sizeof(uint8_t),
      .n = channels,
      .params.q8 = add_op->q8_add_params,
      .ukernel = xnn_params.q8.vadd,
    };
    add_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    add_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_add_strided;
    add_op->compute.range[0] = batch_size;
    add_op->compute.tile[0] = 1;
  }
  add_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_add_nc_f32(
    xnn_operator_t add_op,
    size_t batch_size,
    const float* a,
    const float* b,
    float* sum,
    pthreadpool_t threadpool)
{
  if (add_op->type != xnn_operator_type_add_nc_f32) {
    xnn_log_error("failed to setup Add (NC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  add_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Add operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    add_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = add_op->channels;
  const size_t a_stride = add_op->input_pixel_stride;
  const size_t b_stride = add_op->input2_pixel_stride;
  const size_t sum_stride = add_op->output_pixel_stride;
  if ((((a_stride ^ channels) | (b_stride ^ channels) | (sum_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    add_op->context.add_contiguous = (struct add_contiguous_context) {
      .a = a,
      .b = b,
      .y = sum,
      .params.f32 = add_op->f32_minmax_params,
      .ukernel = xnn_params.f32.vadd.op_ukernel,
    };
    add_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    add_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_add_contiguous;
    add_op->compute.range[0] = batch_size * channels * sizeof(float);
    add_op->compute.tile[0] = block_size;
  } else {
    add_op->context.add_strided = (struct add_strided_context) {
      .a = a,
      .a_stride = a_stride * sizeof(float),
      .b = b,
      .b_stride = b_stride * sizeof(float),
      .y = sum,
      .y_stride = sum_stride * sizeof(float),
      .n = channels * sizeof(float),
      .params.f32 = add_op->f32_minmax_params,
      .ukernel = xnn_params.f32.vadd.op_ukernel,
    };
    add_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    add_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_add_strided;
    add_op->compute.range[0] = batch_size;
    add_op->compute.tile[0] = 1;
  }
  add_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
