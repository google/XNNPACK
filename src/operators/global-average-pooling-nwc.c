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


static enum xnn_status create_global_average_pooling_nwc(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    uint32_t log2_element_size,
    size_t params_offset,
    const void* params,
    size_t params_size,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* global_average_pooling_op_out)
{
  xnn_operator_t global_average_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & datatype_init_flags) == 0) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), output_stride, channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  global_average_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (global_average_pooling_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const size_t zero_size = (channels << log2_element_size) + XNN_EXTRA_BYTES;
  void* zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
  if (zero_buffer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator zero padding",
      zero_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  global_average_pooling_op->zero_buffer = zero_buffer;

  global_average_pooling_op->channels = channels;
  global_average_pooling_op->input_pixel_stride = input_stride;
  global_average_pooling_op->output_pixel_stride = output_stride;
  memcpy((void*) ((uintptr_t) global_average_pooling_op + params_offset), params, params_size);

  global_average_pooling_op->type = operator_type;
  global_average_pooling_op->flags = flags;

  global_average_pooling_op->state = xnn_run_state_invalid;

  *global_average_pooling_op_out = global_average_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(global_average_pooling_op);
  return status;
}

static enum xnn_status setup_global_average_pooling_nwc(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const void* input,
    void* output,
    size_t log2_element_size,
    const struct gavgpool_parameters gavgpool[restrict XNN_MIN_ELEMENTS(1)],
    uint32_t datatype_init_flags,
    enum xnn_operator_type expected_operator_type,
    const void* params,
    size_t params_size,
    void (*update_params)(xnn_operator_t, size_t),
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & datatype_init_flags) == 0) {
    xnn_log_error("failed to setup %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_unsupported_hardware;
  }

  if (width == 0) {
    xnn_log_error("failed to setup %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(global_average_pooling_op->type), width);
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

  update_params(global_average_pooling_op, width);

  assert(gavgpool->row_tile != 0);

  const size_t input_stride_in_bytes = global_average_pooling_op->input_pixel_stride << log2_element_size;
  const size_t channels = global_average_pooling_op->channels;
  global_average_pooling_op->context.global_average_pooling_nwc = (struct global_average_pooling_nwc_context) {
      .input = input,
      .zero = global_average_pooling_op->zero_buffer,
      .input_pixel_stride = input_stride_in_bytes,
      .input_batch_stride = input_stride_in_bytes * width,
      .input_elements = width,
      .channels = channels,
      .output = output,
      .output_batch_stride = (global_average_pooling_op->output_pixel_stride << log2_element_size),
  };
  memcpy(&global_average_pooling_op->context.global_average_pooling_nwc.params, params, params_size);
  global_average_pooling_op->compute.type = xnn_parallelization_type_1d;
  global_average_pooling_op->compute.range[0] = batch_size;

  if (width <= gavgpool->row_tile) {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_unipass;
    global_average_pooling_op->context.global_average_pooling_nwc.unipass_ukernel = gavgpool->unipass;
  } else {
    global_average_pooling_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_multipass;
    global_average_pooling_op->context.global_average_pooling_nwc.multipass_ukernel = gavgpool->multipass;
  }
  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

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
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input-to-output scale ratio: scale ratio must be in [2**-8, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qu8), input_output_scale);
    return xnn_status_unsupported_parameter;
  }

  union xnn_qu8_avgpool_minmax_params params;
  if (xnn_params.qu8.gavgpool.init.qu8 != NULL) {
    xnn_params.qu8.gavgpool.init.qu8(&params,
      0 /* bias */, 1.0f /* scale */, output_zero_point, output_min, output_max);
  }
  const enum xnn_status status = create_global_average_pooling_nwc(
    channels, input_stride, output_stride, flags,
    0 /* log2(sizeof(uint8_t)) */,
    offsetof(struct xnn_operator, params.qu8_gavgpool),
    &params, sizeof(params),
    XNN_INIT_FLAG_QU8,
    xnn_operator_type_global_average_pooling_nwc_qu8,
    global_average_pooling_op_out);
  if (status == xnn_status_success) {
    xnn_operator_t global_average_pooling_op = *global_average_pooling_op_out;
    global_average_pooling_op->input_zero_point = (int32_t) (uint32_t) input_zero_point;
    global_average_pooling_op->input_scale = input_scale;
    global_average_pooling_op->output_scale = output_scale;
  }
  return status;
}

enum xnn_status xnn_create_global_average_pooling_nwc_qs8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    int8_t input_zero_point,
    float input_scale,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* global_average_pooling_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input-to-output scale ratio: scale ratio must be in [2**-8, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_qs8), input_output_scale);
    return xnn_status_unsupported_parameter;
  }

  union xnn_qs8_avgpool_minmax_params params;
  if (xnn_params.qs8.gavgpool.init.qs8 != NULL) {
    xnn_params.qs8.gavgpool.init.qs8(&params,
      0 /* bias */, 1.0f /* scale */, output_zero_point, output_min, output_max);
  }
  const enum xnn_status status = create_global_average_pooling_nwc(
    channels, input_stride, output_stride, flags,
    0 /* log2(sizeof(int8_t)) */,
    offsetof(struct xnn_operator, params.qs8_gavgpool),
    &params, sizeof(params),
    XNN_INIT_FLAG_QS8,
    xnn_operator_type_global_average_pooling_nwc_qs8,
    global_average_pooling_op_out);
  if (status == xnn_status_success) {
    xnn_operator_t global_average_pooling_op = *global_average_pooling_op_out;
    global_average_pooling_op->input_zero_point = (int32_t) input_zero_point;
    global_average_pooling_op->input_scale = input_scale;
    global_average_pooling_op->output_scale = output_scale;
  }
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
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)) >= fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max))) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max)));
    return xnn_status_invalid_parameter;
  }

  union xnn_f16_scaleminmax_params params;
  if (xnn_params.f16.gavgpool.init.f16 != NULL) {
    xnn_params.f16.gavgpool.init.f16(&params,
      0 /* scale */, fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max));
  }
  return create_global_average_pooling_nwc(
    channels, input_stride, output_stride, flags,
    1 /* log2(sizeof(uint16_t)) */,
    offsetof(struct xnn_operator, params.f16_scaleminmax),
    &params, sizeof(params),
    XNN_INIT_FLAG_F16,
    xnn_operator_type_global_average_pooling_nwc_f16,
    global_average_pooling_op_out);
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
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  union xnn_f32_scaleminmax_params params;
  if (xnn_params.f32.gavgpool.init.f32 != NULL) {
    xnn_params.f32.gavgpool.init.f32(&params,
      0.0f /* scale */, output_min, output_max);
  }
  return create_global_average_pooling_nwc(
    channels, input_stride, output_stride, flags,
    2 /* log2(sizeof(float)) */,
    offsetof(struct xnn_operator, params.f32_scaleminmax),
    &params, sizeof(params),
    XNN_INIT_FLAG_F32,
    xnn_operator_type_global_average_pooling_nwc_f32,
    global_average_pooling_op_out);
}

static void update_params_qu8(
  xnn_operator_t global_average_pooling_op,
  size_t width)
{
  const int32_t bias = -((int32_t) width * global_average_pooling_op->input_zero_point);
  const float scale = global_average_pooling_op->input_scale / (global_average_pooling_op->output_scale * (float) width);
  xnn_params.qu8.gavgpool.update.qu8(&global_average_pooling_op->params.qu8_gavgpool, bias, scale);
}

enum xnn_status xnn_setup_global_average_pooling_nwc_qu8(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width,
    input, output,
    0 /* log2(sizeof(uint8_t)) */,
    &xnn_params.qu8.gavgpool,
    XNN_INIT_FLAG_QU8,
    xnn_operator_type_global_average_pooling_nwc_qu8,
    &global_average_pooling_op->params.qu8_gavgpool,
    sizeof(global_average_pooling_op->params.qu8_gavgpool),
    update_params_qu8,
    threadpool);
}

static void update_params_qs8(
  xnn_operator_t global_average_pooling_op,
  size_t width)
{
  const int32_t bias = -((int32_t) width * global_average_pooling_op->input_zero_point);
  const float scale = global_average_pooling_op->input_scale / (global_average_pooling_op->output_scale * (float) width);
  xnn_params.qs8.gavgpool.update.qs8(&global_average_pooling_op->params.qs8_gavgpool, bias, scale);
}

enum xnn_status xnn_setup_global_average_pooling_nwc_qs8(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const int8_t* input,
    int8_t* output,
    pthreadpool_t threadpool)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width,
    input, output,
    0 /* log2(sizeof(int8_t)) */,
    &xnn_params.qs8.gavgpool,
    XNN_INIT_FLAG_QS8,
    xnn_operator_type_global_average_pooling_nwc_qs8,
    &global_average_pooling_op->params.qs8_gavgpool,
    sizeof(global_average_pooling_op->params.qs8_gavgpool),
    update_params_qs8,
    threadpool);
}

static void update_params_f16(
  xnn_operator_t global_average_pooling_op,
  size_t width)
{
  xnn_params.f16.gavgpool.update.f16(
    &global_average_pooling_op->params.f16_scaleminmax,
    fp16_ieee_from_fp32_value(1.0f / (float) width));
}

enum xnn_status xnn_setup_global_average_pooling_nwc_f16(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width,
    input, output,
    1 /* log2(sizeof(uint16_t)) */,
    &xnn_params.f16.gavgpool,
    XNN_INIT_FLAG_F16,
    xnn_operator_type_global_average_pooling_nwc_f16,
    &global_average_pooling_op->params.f16_scaleminmax,
    sizeof(global_average_pooling_op->params.f16_scaleminmax),
    update_params_f16,
    threadpool);
}

static void update_params_f32(
  xnn_operator_t global_average_pooling_op,
  size_t width)
{
  xnn_params.f32.gavgpool.update.f32(&global_average_pooling_op->params.f32_scaleminmax, 1.0f / (float) width);
}

enum xnn_status xnn_setup_global_average_pooling_nwc_f32(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width,
    input, output,
    2 /* log2(sizeof(float)) */,
    &xnn_params.f32.gavgpool,
    XNN_INIT_FLAG_F32,
    xnn_operator_type_global_average_pooling_nwc_f32,
    &global_average_pooling_op->params.f32_scaleminmax,
    sizeof(global_average_pooling_op->params.f32_scaleminmax),
    update_params_f32,
    threadpool);
}
