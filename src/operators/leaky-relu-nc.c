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
#include <xnnpack/operator.h>
#include <xnnpack/log.h>


enum xnn_status xnn_create_leaky_relu_nc_q8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float negative_slope,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* leaky_relu_op_out)
{
  xnn_operator_t leaky_relu_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Leaky ReLU operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  if (negative_slope <= 0.0f || !isnormal(negative_slope)) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with %.7g negative slope: slope must be finite, normalized, and positive",
      negative_slope);
    goto error;
  }

  if (negative_slope > 1.0f) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with %.7g negative slope: slope must not exceed 1.0", negative_slope);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with %.7g input scale: scale must be finite, normalized, and positive",
      input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with %.7g output scale: scale must be finite, normalized, and positive",
      output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with [%" PRIu8 ", %" PRIu8 "] output range: "
      "range min must be below range max",
      output_min, output_max);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create Leaky ReLU operator with %.7g input-to-output scale ratio: "
      "scale ratio must be in [2**-8, 2**8) range",
      input_output_scale);
    goto error;
  }

  status = xnn_status_out_of_memory;

  leaky_relu_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (leaky_relu_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Leaky ReLU operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  leaky_relu_op->lookup_table = xnn_allocate_simd_memory(256 * sizeof(uint8_t));
  if (leaky_relu_op->lookup_table == NULL) {
    xnn_log_error("failed to allocate 256 bytes for Leaky ReLU lookup table");
    goto error;
  }

  uint8_t* lookup_table = leaky_relu_op->lookup_table;
  const float scaled_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float scaled_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (int32_t i = 0; i < 256; i++) {
    const float x = input_output_scale * (float) (i - (int32_t) (uint32_t) input_zero_point);
    float y = x < 0.0f ? x * negative_slope : x;
    if (y < scaled_min_less_zero_point) {
      y = scaled_min_less_zero_point;
    }
    if (y > scaled_max_less_zero_point) {
      y = scaled_max_less_zero_point;
    }
    lookup_table[(uint32_t) i] = (uint8_t) (lrintf(y) + (long) output_zero_point);
  }

  leaky_relu_op->channels = channels;
  leaky_relu_op->input_pixel_stride = input_stride;
  leaky_relu_op->output_pixel_stride = output_stride;

  leaky_relu_op->type = xnn_operator_type_leaky_relu_nc_q8;
  leaky_relu_op->ukernel.type = xnn_ukernel_type_lut;

  leaky_relu_op->state = xnn_run_state_invalid;

  *leaky_relu_op_out = leaky_relu_op;
  return xnn_status_success;

error:
  xnn_delete_operator(leaky_relu_op);
  return status;
}

enum xnn_status xnn_setup_leaky_relu_nc_q8(
    xnn_operator_t leaky_relu_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (leaky_relu_op->type != xnn_operator_type_leaky_relu_nc_q8) {
    xnn_log_error("failed to setup Leaky ReLU (NC, Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  leaky_relu_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Leaky ReLU operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    leaky_relu_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = leaky_relu_op->channels;
  const size_t input_stride = leaky_relu_op->input_pixel_stride;
  const size_t output_stride = leaky_relu_op->output_pixel_stride;
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 1024;
    leaky_relu_op->context.lut_contiguous = (struct lut_contiguous_context) {
      .x = input,
      .x_stride = input_stride * sizeof(uint8_t),
      .t = leaky_relu_op->lookup_table,
      .y = output,
      .y_stride = output_stride * sizeof(uint8_t),
      .ukernel = xnn_params.x8.lut,
    };
    leaky_relu_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    leaky_relu_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_lut_contiguous;
    leaky_relu_op->compute.range[0] = batch_size * channels * sizeof(uint8_t);
    leaky_relu_op->compute.tile[0] = block_size;
  } else {
    leaky_relu_op->context.lut_strided = (struct lut_strided_context) {
      .n = channels,
      .x = input,
      .x_stride = input_stride * sizeof(uint8_t),
      .t = leaky_relu_op->lookup_table,
      .y = output,
      .y_stride = output_stride * sizeof(uint8_t),
      .ukernel = xnn_params.x8.lut,
    };
    leaky_relu_op->compute.type = xnn_parallelization_type_1d;
    leaky_relu_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_lut_strided;
    leaky_relu_op->compute.range[0] = batch_size;
    leaky_relu_op->compute.tile[0] = 0;
  }
  leaky_relu_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
