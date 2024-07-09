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

#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/compute.h"
#include "xnnpack/config-types.h"
#include "xnnpack/config.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static enum xnn_status create_global_average_pooling_nwc(
    uint32_t flags,
    uint32_t log2_element_size,
    size_t params_offset,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    const struct xnn_gavgpool_config* gavgpool_config,
    xnn_operator_t* global_average_pooling_op_out)
{
  xnn_operator_t global_average_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
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

  memcpy((void*) ((uintptr_t) global_average_pooling_op + params_offset), params, params_size);

  global_average_pooling_op->type = operator_type;
  global_average_pooling_op->flags = flags;
  global_average_pooling_op->gavgpool_config  = gavgpool_config;

  global_average_pooling_op->state = xnn_run_state_invalid;

  *global_average_pooling_op_out = global_average_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(global_average_pooling_op);
  return status;
}

enum xnn_status xnn_create_global_average_pooling_nwc_qu8(
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

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
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

  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  assert(gavgpool_config != NULL);

  union xnn_qu8_avgpool_minmax_params params;
  if (gavgpool_config->init.qu8 != NULL) {
    gavgpool_config->init.qu8(&params, 0 /* bias */, 1.0f /* scale */, output_zero_point, output_min, output_max);
  }
  const enum xnn_status status = create_global_average_pooling_nwc(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    offsetof(struct xnn_operator, params.qu8_gavgpool),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_nwc_qu8,
    gavgpool_config,
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

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
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

  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  assert(gavgpool_config != NULL);

  union xnn_qs8_avgpool_minmax_params params;
  if (gavgpool_config->init.qs8 != NULL) {
    gavgpool_config->init.qs8(&params, 0 /* bias */, 1.0f /* scale */, output_zero_point, output_min, output_max);
  }
  const enum xnn_status status = create_global_average_pooling_nwc(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    offsetof(struct xnn_operator, params.qs8_gavgpool),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_nwc_qs8,
    gavgpool_config,
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

  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_scaleminmax_params params;
  if (gavgpool_config->init.f16 != NULL) {
    gavgpool_config->init.f16(
      &params, 0 /* scale */, fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max));
  }
  return create_global_average_pooling_nwc(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
    offsetof(struct xnn_operator, params.f16_scaleminmax),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_nwc_f16,
    gavgpool_config,
    global_average_pooling_op_out);
}

enum xnn_status xnn_create_global_average_pooling_nwc_f32(
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

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  if (gavgpool_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_nwc_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_scaleminmax_params params;
  if (gavgpool_config->init.f32 != NULL) {
    gavgpool_config->init.f32(&params, 0.0f /* scale */, output_min, output_max);
  }
  return create_global_average_pooling_nwc(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    offsetof(struct xnn_operator, params.f32_scaleminmax),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_nwc_f32,
    gavgpool_config,
    global_average_pooling_op_out);
}

enum xnn_status xnn_create_global_sum_pooling_nwc_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* global_sum_pooling_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)) >= fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max))) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f16),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max)));
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_scaleminmax_params params;
  if (gavgpool_config->init.f16 != NULL) {
    gavgpool_config->init.f16(
      &params,
      /*scale=*/UINT16_C(0x3C00) /* 1.0h */,
      fp16_ieee_from_fp32_value(output_min),
      fp16_ieee_from_fp32_value(output_max));
  }
  return create_global_average_pooling_nwc(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
    offsetof(struct xnn_operator, params.f16_scaleminmax),
    &params, sizeof(params),
    xnn_operator_type_global_sum_pooling_nwc_f16,
    gavgpool_config,
    global_sum_pooling_op_out);
}

enum xnn_status xnn_create_global_sum_pooling_nwc_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* global_sum_pooling_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  if (gavgpool_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_global_sum_pooling_nwc_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_scaleminmax_params params;
  if (gavgpool_config->init.f32 != NULL) {
    gavgpool_config->init.f32(&params, /*scale=*/1.0f, output_min, output_max);
  }
  return create_global_average_pooling_nwc(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    offsetof(struct xnn_operator, params.f32_scaleminmax),
    &params, sizeof(params),
    xnn_operator_type_global_sum_pooling_nwc_f32,
    gavgpool_config,
    global_sum_pooling_op_out);
}

static enum xnn_status reshape_global_average_pooling_nwc(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t log2_data_element_size,
    size_t log2_accumulator_element_size,
    const struct xnn_gavgpool_config gavgpool[restrict XNN_MIN_ELEMENTS(1)],
    enum xnn_operator_type expected_operator_type,
    const void* params,
    size_t params_size,
    void (*update_params)(xnn_operator_t, size_t),
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_uninitialized;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(expected_operator_type), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(expected_operator_type), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  global_average_pooling_op->channels = channels;
  global_average_pooling_op->input_pixel_stride = input_stride;
  global_average_pooling_op->output_pixel_stride = output_stride;

  if (width == 0) {
    xnn_log_error("failed to reshape %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(global_average_pooling_op->type), width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  global_average_pooling_op->batch_size = batch_size;
  global_average_pooling_op->input_width = width;

  if (update_params != NULL) {
    update_params(global_average_pooling_op, width);
  }

  const bool input_size_changed =  (channels != global_average_pooling_op->last_input_channels);
  if (input_size_changed) {
    const size_t zero_bytes = (channels << log2_data_element_size) + XNN_EXTRA_BYTES;
    void* zero_buffer = global_average_pooling_op->zero_buffer;
    xnn_release_simd_memory(zero_buffer);
    zero_buffer =
      (void*) xnn_allocate_zero_simd_memory(zero_bytes);
    global_average_pooling_op->zero_buffer = zero_buffer;
    if (zero_buffer == NULL) {
      xnn_log_error(
          "failed to allocate %zu bytes for %s operator zero padding",
          zero_bytes, xnn_operator_type_to_string(expected_operator_type));
      return xnn_status_out_of_memory;
    }
    global_average_pooling_op->zero_buffer = zero_buffer;
    global_average_pooling_op->last_input_channels = channels;
  }

  assert(gavgpool->row_tile != 0);

  const size_t input_stride_in_bytes = global_average_pooling_op->input_pixel_stride << log2_data_element_size;
  global_average_pooling_op->context.global_average_pooling_nwc = (struct global_average_pooling_nwc_context) {
      .zero = global_average_pooling_op->zero_buffer,
      .input_pixel_stride = input_stride_in_bytes,
      .input_batch_stride = input_stride_in_bytes * width,
      .input_elements = width,
      .channels = channels,
      .output_batch_stride = (global_average_pooling_op->output_pixel_stride << log2_data_element_size),
  };
  memcpy(&global_average_pooling_op->context.global_average_pooling_nwc.params, params, params_size);
  global_average_pooling_op->compute[0].range[0] = batch_size;


  if (width <= gavgpool->row_tile) {
    *workspace_size = 0;
    *workspace_alignment = 1;
    global_average_pooling_op->compute[0].type = xnn_parallelization_type_1d;
    global_average_pooling_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_unipass;
    global_average_pooling_op->context.global_average_pooling_nwc.unipass_ukernel = gavgpool->unipass;
  } else {
    const size_t multipass_batch_stride =
        round_up_po2(
        (channels + (XNN_MULTIPASS_EXTRA_BYTES >> log2_data_element_size)) << log2_accumulator_element_size,
          XNN_ALLOCATION_ALIGNMENT);
    global_average_pooling_op->context.global_average_pooling_nwc.multipass_batch_stride = multipass_batch_stride;

    const size_t num_threads = pthreadpool_get_threads_count(threadpool);
    const bool use_threads_workspace_size = num_threads < batch_size;
    if (use_threads_workspace_size) {
      *workspace_size = num_threads * multipass_batch_stride;
      *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
      global_average_pooling_op->compute[0].type = xnn_parallelization_type_1d_with_thread;
      global_average_pooling_op->compute[0].task_1d_with_thread =
        (pthreadpool_task_1d_with_thread_t) xnn_compute_global_average_pooling_nwc_multipass_with_thread;
    } else {
      *workspace_size = batch_size * multipass_batch_stride;
      *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
      global_average_pooling_op->compute[0].type = xnn_parallelization_type_1d;
      global_average_pooling_op->compute[0].task_1d =
        (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_multipass;
    }

    global_average_pooling_op->context.global_average_pooling_nwc.multipass_ukernel = gavgpool->multipass;
  }
  global_average_pooling_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static void update_params_qu8(
  xnn_operator_t global_average_pooling_op,
  size_t width)
{
  const int32_t bias = -((int32_t) width * global_average_pooling_op->input_zero_point);
  const float scale = global_average_pooling_op->input_scale / (global_average_pooling_op->output_scale * (float) width);
  global_average_pooling_op->gavgpool_config->update.qu8(&global_average_pooling_op->params.qu8_gavgpool, bias, scale);
}

enum xnn_status xnn_reshape_global_average_pooling_nwc_qu8(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width, channels, input_stride, output_stride,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
    global_average_pooling_op->gavgpool_config,
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
  global_average_pooling_op->gavgpool_config->update.qs8(&global_average_pooling_op->params.qs8_gavgpool, bias, scale);
}

enum xnn_status xnn_reshape_global_average_pooling_nwc_qs8(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width, channels, input_stride, output_stride,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_INT32_T,
    global_average_pooling_op->gavgpool_config,
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
  global_average_pooling_op->gavgpool_config->update.f16(
    &global_average_pooling_op->params.f16_scaleminmax,
    fp16_ieee_from_fp32_value(1.0f / (float) width));
}

enum xnn_status xnn_reshape_global_average_pooling_nwc_f16(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width, channels, input_stride, output_stride,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_HALF,
    global_average_pooling_op->gavgpool_config,
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
  global_average_pooling_op->gavgpool_config->update.f32(
    &global_average_pooling_op->params.f32_scaleminmax, 1.0f / (float) width);
}

enum xnn_status xnn_reshape_global_average_pooling_nwc_f32(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_global_average_pooling_nwc(
    global_average_pooling_op,
    batch_size, width, channels, input_stride, output_stride,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    global_average_pooling_op->gavgpool_config,
    xnn_operator_type_global_average_pooling_nwc_f32,
    &global_average_pooling_op->params.f32_scaleminmax,
    sizeof(global_average_pooling_op->params.f32_scaleminmax),
    update_params_f32,
    threadpool);
}

static enum xnn_status setup_global_average_pooling_nwc(
    xnn_operator_t global_average_pooling_op,
    enum xnn_operator_type expected_operator_type,
    void* workspace,
    const void* input,
    void* output)
{
  if (global_average_pooling_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (global_average_pooling_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(global_average_pooling_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  struct global_average_pooling_nwc_context* context = &global_average_pooling_op->context.global_average_pooling_nwc;
  if (context->multipass_batch_stride != 0 && workspace == NULL) {
    xnn_log_error(
      "failed to setup %s operator: workspace of size %zu required but workspace is NULL",
      xnn_operator_type_to_string(global_average_pooling_op->type), context->multipass_batch_stride);
    return xnn_status_invalid_state;
  }

  context->input = input;
  context->output = output;
  context->multipass_buffer = workspace;

  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_global_average_pooling_nwc_qu8(
    xnn_operator_t global_average_pooling_op,
    void* workspace,
    const uint8_t* input,
    uint8_t* output)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    xnn_operator_type_global_average_pooling_nwc_qu8,
    workspace,
    input, output);
}

enum xnn_status xnn_setup_global_average_pooling_nwc_qs8(
  xnn_operator_t global_average_pooling_op,
  void* workspace,
  const int8_t* input,
  int8_t* output)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    xnn_operator_type_global_average_pooling_nwc_qs8,
    workspace,
    input, output);
}

enum xnn_status xnn_setup_global_average_pooling_nwc_f16(
  xnn_operator_t global_average_pooling_op,
  void* workspace,
  const void* input,
  void* output)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    xnn_operator_type_global_average_pooling_nwc_f16,
    workspace,
    input, output);
}

enum xnn_status xnn_setup_global_average_pooling_nwc_f32(
    xnn_operator_t global_average_pooling_op,
    void* workspace,
    const float* input,
    float* output)
{
  return setup_global_average_pooling_nwc(
    global_average_pooling_op,
    xnn_operator_type_global_average_pooling_nwc_f32,
    workspace,
    input, output);
}

enum xnn_status xnn_reshape_global_sum_pooling_nwc_f16(
  xnn_operator_t global_sum_pooling_op,
  size_t batch_size,
  size_t width,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool)
{
  return reshape_global_average_pooling_nwc(
    global_sum_pooling_op,
    batch_size, width, channels, input_stride, output_stride,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_HALF,
    global_sum_pooling_op->gavgpool_config,
    xnn_operator_type_global_sum_pooling_nwc_f16,
    &global_sum_pooling_op->params.f16_scaleminmax,
    sizeof(global_sum_pooling_op->params.f16_scaleminmax),
    /*update_params=*/NULL,
    threadpool);
}

enum xnn_status xnn_reshape_global_sum_pooling_nwc_f32(
  xnn_operator_t global_sum_pooling_op,
  size_t batch_size,
  size_t width,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool)
{
  return reshape_global_average_pooling_nwc(
    global_sum_pooling_op,
    batch_size, width, channels, input_stride, output_stride,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    global_sum_pooling_op->gavgpool_config,
    xnn_operator_type_global_sum_pooling_nwc_f32,
    &global_sum_pooling_op->params.f32_scaleminmax,
    sizeof(global_sum_pooling_op->params.f32_scaleminmax),
    /*update_params=*/NULL,
    threadpool);
}

enum xnn_status xnn_setup_global_sum_pooling_nwc_f16(
  xnn_operator_t global_sum_pooling_op,
  void* workspace,
  const void* input,
  void* output)
{
  return setup_global_average_pooling_nwc(
    global_sum_pooling_op,
    xnn_operator_type_global_sum_pooling_nwc_f16,
    workspace,
    input, output);
}

enum xnn_status xnn_setup_global_sum_pooling_nwc_f32(
  xnn_operator_t global_sum_pooling_op,
  void* workspace,
  const float* input,
  float* output)
{
  return setup_global_average_pooling_nwc(
    global_sum_pooling_op,
    xnn_operator_type_global_sum_pooling_nwc_f32,
    workspace,
    input, output);
}
