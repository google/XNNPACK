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

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


enum xnn_status xnn_create_global_average_pooling_nwc_qu8(
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
    xnn_operator_t* global_average_pooling_op_out)
{
  xnn_operator_t global_average_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), output_stride, channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), output_min, output_max);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input-to-output scale ratio: scale ratio must be in [2**-8, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), input_output_scale);
    goto error;
  }

  status = xnn_status_out_of_memory;

  global_average_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (global_average_pooling_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8));
    goto error;
  }

  const size_t zero_size = channels * sizeof(uint8_t) + XNN_EXTRA_BYTES;
  void* zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
  if (zero_buffer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator zero padding",
      zero_size, xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8));
    goto error;
  }
  global_average_pooling_op->zero_buffer = zero_buffer;

  global_average_pooling_op->channels = channels;
  global_average_pooling_op->input_pixel_stride = input_stride;
  global_average_pooling_op->output_pixel_stride = output_stride;
  global_average_pooling_op->input_zero_point = input_zero_point;
  global_average_pooling_op->output_zero_point = output_zero_point;
  global_average_pooling_op->input_scale = input_scale;
  global_average_pooling_op->output_scale = output_scale;
  global_average_pooling_op->output_min = output_min;
  global_average_pooling_op->output_max = output_max;

  global_average_pooling_op->type = xnn_operator_type_global_average_pooling_nwc_qu8;
  global_average_pooling_op->ukernel.type = xnn_ukernel_type_global_average_pooling;

  global_average_pooling_op->state = xnn_run_state_invalid;

  *global_average_pooling_op_out = global_average_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(global_average_pooling_op);
  return status;
}

enum xnn_status xnn_create_global_average_pooling_nwc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* global_average_pooling_op_out)
{
  xnn_operator_t global_average_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_F16) == 0) {
    xnn_log_error("failed to create %s operator: FP16 operations are not supported",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16), channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16), input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16), output_stride, channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    goto error;
  }

  if (fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)) >= fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max))) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max)));
    goto error;
  }

  status = xnn_status_out_of_memory;

  global_average_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (global_average_pooling_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    goto error;
  }

  const size_t zero_size = channels * sizeof(uint16_t) + XNN_EXTRA_BYTES;
  void* zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
  if (zero_buffer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator zero padding",
      zero_size, xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    goto error;
  }
  global_average_pooling_op->zero_buffer = zero_buffer;

  global_average_pooling_op->channels = channels;
  global_average_pooling_op->input_pixel_stride = input_stride;
  global_average_pooling_op->output_pixel_stride = output_stride;
  global_average_pooling_op->params.f16_scaleminmax = xnn_init_f16_scaleminmax_params(
    UINT16_C(0x7E00) /* NaN */,
    fp16_ieee_from_fp32_value(output_min),
    fp16_ieee_from_fp32_value(output_max));

  global_average_pooling_op->type = xnn_operator_type_global_average_pooling_nwc_f16;
  global_average_pooling_op->ukernel.type = xnn_ukernel_type_global_average_pooling;

  global_average_pooling_op->state = xnn_run_state_invalid;

  *global_average_pooling_op_out = global_average_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(global_average_pooling_op);
  return status;
}

enum xnn_status xnn_create_global_average_pooling_nwc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* global_average_pooling_op_out)
{
  xnn_operator_t global_average_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32), channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32), input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32), output_stride, channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32), output_min, output_max);
    goto error;
  }

  status = xnn_status_out_of_memory;

  global_average_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (global_average_pooling_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    goto error;
  }

  const size_t zero_size = channels * sizeof(float) + XNN_EXTRA_BYTES;
  void* zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
  if (zero_buffer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator zero padding",
      zero_size, xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    goto error;
  }
  global_average_pooling_op->zero_buffer = zero_buffer;

  global_average_pooling_op->channels = channels;
  global_average_pooling_op->input_pixel_stride = input_stride;
  global_average_pooling_op->output_pixel_stride = output_stride;
  global_average_pooling_op->params.f32_scaleminmax = xnn_init_f32_scaleminmax_params(nanf(""), output_min, output_max);

  global_average_pooling_op->type = xnn_operator_type_global_average_pooling_nwc_f32;
  global_average_pooling_op->ukernel.type = xnn_ukernel_type_global_average_pooling;

  global_average_pooling_op->state = xnn_run_state_invalid;

  *global_average_pooling_op_out = global_average_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(global_average_pooling_op);
  return status;
}

enum xnn_status xnn_setup_global_average_pooling_nwc_qu8(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != xnn_operator_type_global_average_pooling_nwc_qu8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8));
    return xnn_status_uninitialized;
  }

  assert(xnn_params.qu8.gavgpool.mr != 0);

  if (width == 0) {
    xnn_log_error("failed to setup %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  global_average_pooling_op->batch_size = batch_size;
  global_average_pooling_op->input_width = width;
  global_average_pooling_op->input = input;
  global_average_pooling_op->output = output;

  global_average_pooling_op->params.qu8_avgpool =
    xnn_init_qu8_avgpool_params(
      -(int32_t) width * (int32_t) (uint32_t) global_average_pooling_op->input_zero_point,
      global_average_pooling_op->input_scale / (global_average_pooling_op->output_scale * (float) width),
      global_average_pooling_op->output_zero_point,
      global_average_pooling_op->output_min,
      global_average_pooling_op->output_max);

  const size_t input_stride_in_bytes = global_average_pooling_op->input_pixel_stride * sizeof(uint8_t);
  const size_t channels = global_average_pooling_op->channels;
  global_average_pooling_op->context.global_average_pooling_nwc = (struct global_average_pooling_nwc_context) {
      .input = input,
      .zero = global_average_pooling_op->zero_buffer,
      .input_pixel_stride = input_stride_in_bytes,
      .input_batch_stride = input_stride_in_bytes * width,
      .input_elements = width,
      .channels = channels,
      .output = output,
      .output_batch_stride = global_average_pooling_op->output_pixel_stride * sizeof(uint8_t),
      .params.qu8 = global_average_pooling_op->params.qu8_avgpool,
  };
  global_average_pooling_op->compute.type = xnn_parallelization_type_1d;
  global_average_pooling_op->compute.range[0] = batch_size;

  if (width <= xnn_params.qu8.gavgpool.mr) {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_unipass;
    global_average_pooling_op->context.global_average_pooling_nwc.unipass_ukernel = xnn_params.qu8.gavgpool.up;
  } else {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_multipass;
    global_average_pooling_op->context.global_average_pooling_nwc.multipass_ukernel = xnn_params.qu8.gavgpool.mp;
  }
  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_global_average_pooling_nwc_f16(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != xnn_operator_type_global_average_pooling_nwc_f16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    return xnn_status_uninitialized;
  }

  assert(xnn_params.f16.gavgpool.mr != 0);

  if ((xnn_params.init_flags & XNN_INIT_FLAG_F16) == 0) {
    xnn_log_error("failed to setup %s operator: FP16 operations are not supported",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    return xnn_status_unsupported_hardware;
  }

  if (width == 0) {
    xnn_log_error("failed to setup %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16), width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  global_average_pooling_op->batch_size = batch_size;
  global_average_pooling_op->input_width = width;
  global_average_pooling_op->input = input;
  global_average_pooling_op->output = output;

  xnn_update_f16_scaleminmax_params(&global_average_pooling_op->params.f16_scaleminmax, fp16_ieee_from_fp32_value(1.0f / (float) width));

  const size_t input_stride_in_bytes = global_average_pooling_op->input_pixel_stride * sizeof(uint16_t);
  const size_t channels = global_average_pooling_op->channels;
  global_average_pooling_op->context.global_average_pooling_nwc = (struct global_average_pooling_nwc_context) {
      .input = input,
      .zero = global_average_pooling_op->zero_buffer,
      .input_pixel_stride = input_stride_in_bytes,
      .input_batch_stride = input_stride_in_bytes * width,
      .input_elements = width,
      .channels = channels,
      .output = output,
      .output_batch_stride = global_average_pooling_op->output_pixel_stride * sizeof(uint16_t),
      .params.f16 = global_average_pooling_op->params.f16_scaleminmax,
  };
  global_average_pooling_op->compute.type = xnn_parallelization_type_1d;
  global_average_pooling_op->compute.range[0] = batch_size;

  if (width <= xnn_params.f16.gavgpool.mr) {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_unipass;
    global_average_pooling_op->context.global_average_pooling_nwc.unipass_ukernel = xnn_params.f16.gavgpool.up;
  } else {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_multipass;
    global_average_pooling_op->context.global_average_pooling_nwc.multipass_ukernel = xnn_params.f16.gavgpool.mp;
  }
  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_global_average_pooling_nwc_f32(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != xnn_operator_type_global_average_pooling_nwc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    return xnn_status_uninitialized;
  }

  assert(xnn_params.f32.gavgpool.mr != 0);

  if (width == 0) {
    xnn_log_error("failed to setup %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32), width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  global_average_pooling_op->batch_size = batch_size;
  global_average_pooling_op->input_width = width;
  global_average_pooling_op->input = input;
  global_average_pooling_op->output = output;

  xnn_update_f32_scaleminmax_params(&global_average_pooling_op->params.f32_scaleminmax, 1.0f / (float) width);

  const size_t input_stride_in_bytes = global_average_pooling_op->input_pixel_stride * sizeof(float);
  const size_t channels = global_average_pooling_op->channels;
  global_average_pooling_op->context.global_average_pooling_nwc = (struct global_average_pooling_nwc_context) {
      .input = input,
      .zero = global_average_pooling_op->zero_buffer,
      .input_pixel_stride = input_stride_in_bytes,
      .input_batch_stride = input_stride_in_bytes * width,
      .input_elements = width,
      .channels = channels,
      .output = output,
      .output_batch_stride = global_average_pooling_op->output_pixel_stride * sizeof(float),
      .params.f32 = global_average_pooling_op->params.f32_scaleminmax,
  };
  global_average_pooling_op->compute.type = xnn_parallelization_type_1d;
  global_average_pooling_op->compute.range[0] = batch_size;

  if (width <= xnn_params.f32.gavgpool.mr) {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_unipass;
    global_average_pooling_op->context.global_average_pooling_nwc.unipass_ukernel = xnn_params.f32.gavgpool.up;
  } else {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_multipass;
    global_average_pooling_op->context.global_average_pooling_nwc.multipass_ukernel = xnn_params.f32.gavgpool.mp;
  }
  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
