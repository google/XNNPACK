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


enum xnn_status xnn_create_sigmoid_nc_q8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* sigmoid_op_out)
{
  xnn_operator_t sigmoid_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Sigmoid operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Sigmoid operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create Sigmoid operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create Sigmoid operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create Sigmoid operator with %.7g input scale: scale must be finite, normalized, and positive",
      input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create Sigmoid operator with %.7g output scale: scale must be finite, normalized, and positive",
      output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Sigmoid operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      output_min, output_max);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  if (output_scale != 0x1.0p-8f) {
    xnn_log_error(
      "failed to create Sigmoid operator with %.7g output scale: only output scale of 1/256 is supported",
      output_scale);
    goto error;
  }

  if (output_zero_point != 0) {
    xnn_log_error(
      "failed to create Sigmoid operator with %" PRIu8 " output zero point: only output zero point of 0 is supported",
      output_zero_point);
    goto error;
  }

  status = xnn_status_out_of_memory;

  sigmoid_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (sigmoid_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Sigmoid operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  sigmoid_op->lookup_table = xnn_allocate_simd_memory(256 * sizeof(uint8_t));
  if (sigmoid_op->lookup_table == NULL) {
    xnn_log_error("failed to allocate 256 bytes for Sigmoid lookup table");
    goto error;
  }

  uint8_t* lookup_table = sigmoid_op->lookup_table;
  const float scaled_min = (float) (int32_t) output_min;
  const float scaled_max = (float) (int32_t) output_max;
  for (int32_t i = 0; i < 256; i++) {
    const float x = input_scale * (float) (i - (int32_t) (uint32_t) input_zero_point);
    // Scale sigmoid(x) by 1 / output scale = 256.0
    float scaled_sigmoid_x = 256.0f / (1.0f + expf(-x));
    if (scaled_sigmoid_x < scaled_min) {
      scaled_sigmoid_x = scaled_min;
    }
    if (scaled_sigmoid_x > scaled_max) {
      scaled_sigmoid_x = scaled_max;
    }
    lookup_table[(uint32_t) i] = (uint8_t) lrintf(scaled_sigmoid_x);
  }

  sigmoid_op->channels = channels;
  sigmoid_op->input_pixel_stride = input_stride;
  sigmoid_op->output_pixel_stride = output_stride;

  sigmoid_op->type = xnn_operator_type_sigmoid_nc_q8;
  sigmoid_op->ukernel.type = xnn_ukernel_type_lut;

  sigmoid_op->state = xnn_run_state_invalid;

  *sigmoid_op_out = sigmoid_op;
  return xnn_status_success;

error:
  xnn_delete_operator(sigmoid_op);
  return status;
}

enum xnn_status xnn_create_sigmoid_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* sigmoid_op_out)
{
  xnn_operator_t sigmoid_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Sigmoid operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Sigmoid operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create Sigmoid operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create Sigmoid operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if (xnn_params.f32.sigmoid == NULL) {
    xnn_log_error(
      "failed to create Sigmoid operator: "
      "only selected hardware configurations are supported");
    goto error;
  }

  status = xnn_status_out_of_memory;

  sigmoid_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (sigmoid_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for xnn_operator structure", sizeof(struct xnn_operator));
    goto error;
  }

  sigmoid_op->channels = channels;
  sigmoid_op->input_pixel_stride = input_stride;
  sigmoid_op->output_pixel_stride = output_stride;

  sigmoid_op->type = xnn_operator_type_sigmoid_nc_f32;
  sigmoid_op->ukernel.type = xnn_ukernel_type_sigmoid;

  sigmoid_op->state = xnn_run_state_invalid;

  *sigmoid_op_out = sigmoid_op;
  return xnn_status_success;

error:
  xnn_delete_operator(sigmoid_op);
  return status;
}

enum xnn_status xnn_setup_sigmoid_nc_q8(
    xnn_operator_t sigmoid_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (sigmoid_op->type != xnn_operator_type_sigmoid_nc_q8) {
    xnn_log_error("failed to setup Sigmoid (Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  sigmoid_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Sigmoid operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    sigmoid_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  sigmoid_op->batch_size = batch_size;
  sigmoid_op->input = input;
  sigmoid_op->output = output;

  const size_t channels = sigmoid_op->channels;
  const size_t input_stride = sigmoid_op->input_pixel_stride;
  const size_t output_stride = sigmoid_op->output_pixel_stride;
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 1024;
    sigmoid_op->context.lut_contiguous = (struct lut_contiguous_context) {
      .x = input,
      .x_stride = input_stride * sizeof(uint8_t),
      .t = sigmoid_op->lookup_table,
      .y = output,
      .y_stride = output_stride * sizeof(uint8_t),
      .ukernel = xnn_params.x8.lut,
    };
    sigmoid_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    sigmoid_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_lut_contiguous;
    sigmoid_op->compute.range[0] = batch_size * channels * sizeof(uint8_t);
    sigmoid_op->compute.tile[0] = block_size;
  } else {
    sigmoid_op->context.lut_strided = (struct lut_strided_context) {
      .n = channels,
      .x = input,
      .x_stride = input_stride * sizeof(uint8_t),
      .t = sigmoid_op->lookup_table,
      .y = output,
      .y_stride = output_stride * sizeof(uint8_t),
      .ukernel = xnn_params.x8.lut,
    };
    sigmoid_op->compute.type = xnn_parallelization_type_1d;
    sigmoid_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_lut_strided;
    sigmoid_op->compute.range[0] = batch_size;
    sigmoid_op->compute.tile[0] = 0;
  }
  sigmoid_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_sigmoid_nc_f32(
    xnn_operator_t sigmoid_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (sigmoid_op->type != xnn_operator_type_sigmoid_nc_f32) {
    xnn_log_error("failed to setup Sigmoid (F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  sigmoid_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Sigmoid operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    sigmoid_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = sigmoid_op->channels;
  const size_t input_stride = sigmoid_op->input_pixel_stride;
  const size_t output_stride = sigmoid_op->output_pixel_stride;
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    sigmoid_op->context.univector_contiguous = (struct univector_contiguous_context) {
      .x = input,
      .x_stride = input_stride * sizeof(float),
      .y = output,
      .y_stride = output_stride * sizeof(float),
      .ukernel = xnn_params.f32.sigmoid,
    };
    sigmoid_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    sigmoid_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
    sigmoid_op->compute.range[0] = batch_size * channels * sizeof(float);
    sigmoid_op->compute.tile[0] = block_size;
  } else {
    sigmoid_op->context.univector_strided = (struct univector_strided_context) {
      .n = channels * sizeof(float),
      .x = input,
      .x_stride = input_stride * sizeof(float),
      .y = output,
      .y_stride = output_stride * sizeof(float),
      .ukernel = xnn_params.f32.sigmoid,
    };
    sigmoid_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    sigmoid_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
    sigmoid_op->compute.range[0] = batch_size;
    sigmoid_op->compute.tile[0] = 1;
  }
  sigmoid_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
