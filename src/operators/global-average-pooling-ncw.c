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
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>


static enum xnn_status create_global_average_pooling_ncw(
    size_t channels,
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

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
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

  global_average_pooling_op->channels = channels;
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
    size_t channels,
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
    channels, flags,
    1 /* log2(sizeof(uint16_t)) */,
    offsetof(struct xnn_operator, params.f16_gavgpool),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_ncw_f16,
    gavgpool_cw_config,
    global_average_pooling_op_out);
}

enum xnn_status xnn_create_global_average_pooling_ncw_f32(
    size_t channels,
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  union xnn_f32_gavgpool_params params;
  xnn_init_f32_gavgpool_params(&params, nanf(""), output_min, output_max, 0);

  const struct xnn_gavgpool_cw_config* gavgpool_cw_config = xnn_init_f32_gavgpool_cw_config();
  if (gavgpool_cw_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32));
    return xnn_status_unsupported_hardware;
  }

  return create_global_average_pooling_ncw(
    channels, flags,
    2 /* log2(sizeof(float)) */,
    offsetof(struct xnn_operator, params.f32_gavgpool),
    &params, sizeof(params),
    xnn_operator_type_global_average_pooling_ncw_f32,
    gavgpool_cw_config,
    global_average_pooling_op_out);
}

enum xnn_status xnn_setup_global_average_pooling_ncw_f32(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != xnn_operator_type_global_average_pooling_ncw_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32));
    return xnn_status_uninitialized;
  }

  if (width == 0) {
    xnn_log_error(
      "failed to setup %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f32), width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  xnn_update_f32_gavgpool_params(&global_average_pooling_op->params.f32_gavgpool,
    1.0f / (float) width, width);

  global_average_pooling_op->context.global_average_pooling_ncw = (struct global_average_pooling_ncw_context) {
    .input_elements = width * sizeof(float),
    .input = input,
    .input_channel_stride = width * sizeof(float),
    .input_batch_stride = global_average_pooling_op->channels * width * sizeof(float),
    .output = output,
    .output_channel_stride = sizeof(float),
    .output_batch_stride = global_average_pooling_op->channels * sizeof(float),
    .ukernel = global_average_pooling_op->gavgpool_cw_config->ukernel,
    .params.f32 = global_average_pooling_op->params.f32_gavgpool,
  };

  global_average_pooling_op->compute.type = xnn_parallelization_type_2d_tile_1d;
  global_average_pooling_op->compute.task_2d_tile_1d =
    (pthreadpool_task_2d_tile_1d_t) xnn_compute_global_average_pooling_ncw;
  global_average_pooling_op->compute.range[0] = batch_size;
  global_average_pooling_op->compute.range[1] = global_average_pooling_op->channels;
  global_average_pooling_op->compute.tile[0] = global_average_pooling_op->channels; //xnn_params.f32.gavgpool_cw.channel_tile;

  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_global_average_pooling_ncw_f16(
    xnn_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (global_average_pooling_op->type != xnn_operator_type_global_average_pooling_ncw_f16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16),
      xnn_operator_type_to_string(global_average_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16));
    return xnn_status_uninitialized;
  }

  if (width == 0) {
    xnn_log_error(
      "failed to setup %s operator with width %zu: width must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_global_average_pooling_ncw_f16), width);
    return xnn_status_invalid_parameter;
  }

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
    .input = input,
    .input_channel_stride = width * sizeof(uint16_t),
    .input_batch_stride = global_average_pooling_op->channels * width * sizeof(uint16_t),
    .output = output,
    .output_channel_stride = sizeof(uint16_t),
    .output_batch_stride = global_average_pooling_op->channels * sizeof(uint16_t),
    .ukernel = global_average_pooling_op->gavgpool_cw_config->ukernel,
    .params.f16 = global_average_pooling_op->params.f16_gavgpool,
  };

  global_average_pooling_op->compute.type = xnn_parallelization_type_2d_tile_1d;
  global_average_pooling_op->compute.task_2d_tile_1d =
    (pthreadpool_task_2d_tile_1d_t) xnn_compute_global_average_pooling_ncw;
  global_average_pooling_op->compute.range[0] = batch_size;
  global_average_pooling_op->compute.range[1] = global_average_pooling_op->channels;
  global_average_pooling_op->compute.tile[0] = global_average_pooling_op->channels; //xnn_params.f16.gavgpool_cw.channel_tile;

  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
