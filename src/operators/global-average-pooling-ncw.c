// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
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
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static enum xnn_status create_global_average_pooling_ncw(
    uint32_t flags,
    uint32_t log2_element_size,
    size_t params_offset,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    const struct xnn_gavgpool_cw_config* gavgpool_cw_config,
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

  global_average_pooling_op->state = xnn_run_state_invalid;
  global_average_pooling_op->gavgpool_cw_config = gavgpool_cw_config;

  *global_average_pooling_op_out = global_average_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(global_average_pooling_op);
  return status;
}

enum xnn_status xnn_create_global_average_pooling_ncw_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* global_average_pooling_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16));
    return xnn_status_invalid_parameter;
  }

  if (fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)) >= fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max))) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max)));
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gavgpool_cw_config* gavgpool_cw_config = xnn_init_f16_gavgpool_cw_config();
  if (gavgpool_cw_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_gavgpool_params params;
  if (gavgpool_cw_config->init.f16 != NULL) {
    gavgpool_cw_config->init.f16(
      &params, 0 /* scale */, fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max), 0);
  }

  return create_global_average_pooling_ncw(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
    offsetof(struct xnn_operator, params.f16_gavgpool),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_ncw_f16,
    gavgpool_cw_config,
    global_average_pooling_op_out);
}

enum xnn_status xnn_create_global_average_pooling_ncw_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* global_average_pooling_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gavgpool_cw_config* gavgpool_cw_config = xnn_init_f32_gavgpool_cw_config();
  if (gavgpool_cw_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_gavgpool_params params;
  assert(gavgpool_cw_config->init.f32 != NULL);
  gavgpool_cw_config->init.f32(&params, nanf(""), output_min, output_max, 0);

  return create_global_average_pooling_ncw(
    flags, /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    offsetof(struct xnn_operator, params.f32_gavgpool),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_ncw_f32,
    gavgpool_cw_config,
    global_average_pooling_op_out);
}

enum xnn_status xnn_reshape_global_average_pooling_ncw_f32(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    size_t channels,
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != xnn_operator_type_global_average_pooling_ncw_f32) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32));
    return xnn_status_uninitialized;
  }

  if (width == 0) {
    xnn_log_error(
      "failed to reshape %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32), width);
    return xnn_status_invalid_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32), channels);
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->channels = channels;

  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  xnn_update_f32_gavgpool_params(&global_average_pooling_op->params.f32_gavgpool,
    1.0f / (float) width, width);

  global_average_pooling_op->context.global_average_pooling_ncw = (struct global_average_pooling_ncw_context) {
    .input_elements = width * sizeof(float),
    .input_channel_stride = width * sizeof(float),
    .input_batch_stride = channels * width * sizeof(float),
    .output_channel_stride = sizeof(float),
    .output_batch_stride = channels * sizeof(float),
    .ukernel = global_average_pooling_op->gavgpool_cw_config->ukernel,
    .params.f32 = global_average_pooling_op->params.f32_gavgpool,
  };

  global_average_pooling_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
  global_average_pooling_op->compute[0].task_2d_tile_1d =
    (pthreadpool_task_2d_tile_1d_t) xnn_compute_global_average_pooling_ncw;
  global_average_pooling_op->compute[0].range[0] = batch_size;
  global_average_pooling_op->compute[0].range[1] = channels;

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  if (num_threads > 1) {
    const size_t target_channels_per_thread = 8;
    global_average_pooling_op->compute[0].tile[0] =
        divide_round_up(channels, num_threads * target_channels_per_thread);
  } else {
    global_average_pooling_op->compute[0].tile[0] = channels;
  }

  global_average_pooling_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_global_average_pooling_ncw_f16(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    size_t channels,
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != xnn_operator_type_global_average_pooling_ncw_f16) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16));
    return xnn_status_uninitialized;
  }

  if (width == 0) {
    xnn_log_error(
      "failed to reshape %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16), width);
    return xnn_status_invalid_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16), channels);
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->channels = channels;
  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  if (global_average_pooling_op->gavgpool_cw_config->update.f16 != NULL) {
    global_average_pooling_op->gavgpool_cw_config->update.f16(
      &global_average_pooling_op->params.f16_gavgpool, fp16_ieee_from_fp32_value(1.0f / (float) width), width);
  }

  global_average_pooling_op->context.global_average_pooling_ncw = (struct global_average_pooling_ncw_context) {
    .input_elements = width * sizeof(uint16_t),
    .input_channel_stride = width * sizeof(uint16_t),
    .input_batch_stride = channels * width * sizeof(uint16_t),
    .output_channel_stride = sizeof(uint16_t),
    .output_batch_stride = channels * sizeof(uint16_t),
    .ukernel = global_average_pooling_op->gavgpool_cw_config->ukernel,
    .params.f16 = global_average_pooling_op->params.f16_gavgpool,
  };

  global_average_pooling_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
  global_average_pooling_op->compute[0].task_2d_tile_1d =
    (pthreadpool_task_2d_tile_1d_t) xnn_compute_global_average_pooling_ncw;
  global_average_pooling_op->compute[0].range[0] = batch_size;
  global_average_pooling_op->compute[0].range[1] = channels;

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  if (num_threads > 1) {
    const size_t target_channels_per_thread = 8;
    global_average_pooling_op->compute[0].tile[0] =
        divide_round_up(channels, num_threads * target_channels_per_thread);
  } else {
    global_average_pooling_op->compute[0].tile[0] = channels;
  }

  global_average_pooling_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static enum xnn_status setup_global_average_pooling_ncw(
  xnn_operator_t global_average_pooling_op,
  enum xnn_operator_type expected_operator_type,
  const float* input,
  float* output)
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

  global_average_pooling_op->context.global_average_pooling_ncw.input = input;
  global_average_pooling_op->context.global_average_pooling_ncw.output = output;

  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_global_average_pooling_ncw_f32(
    xnn_operator_t global_average_pooling_op,
    const float* input,
    float* output)
{
  return setup_global_average_pooling_ncw(
    global_average_pooling_op, xnn_operator_type_global_average_pooling_ncw_f32, input, output);
}

enum xnn_status xnn_setup_global_average_pooling_ncw_f16(
    xnn_operator_t global_average_pooling_op,
    const void* input,
    void* output)
{
  return setup_global_average_pooling_ncw(
    global_average_pooling_op, xnn_operator_type_global_average_pooling_ncw_f16, input, output);
}
