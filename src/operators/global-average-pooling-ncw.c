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


enum xnn_status xnn_create_global_average_pooling_ncw_f32(
    size_t channels,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* global_average_pooling_op_out)
{
  xnn_operator_t global_average_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Global Average Pooling operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Global Average Pooling operator with %zu channels: number of channels must be non-zero",
      channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create Global Average Pooling operator with NaN output lower bound: lower bound must be non-NaN");
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create Global Average Pooling operator with NaN output upper bound: upper bound must be non-NaN");
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Global Average Pooling operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    goto error;
  }

  status = xnn_status_unsupported_parameter;
  if (xnn_params.f32.spchw_gavgpool.ukernel == NULL) {
    xnn_log_error(
      "failed to create Global Average Pooling operator: "
      "only selected configurations parameters are supported");
    goto error;
  }

  status = xnn_status_out_of_memory;

  global_average_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (global_average_pooling_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Global Average Pooling operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  global_average_pooling_op->channels = channels;
  global_average_pooling_op->f32_gavgpool_params = xnn_init_f32_gavgpool_params(nanf(""), output_min, output_max, 0);

  global_average_pooling_op->type = xnn_operator_type_global_average_pooling_ncw_f32;
  global_average_pooling_op->ukernel.type = xnn_ukernel_type_global_average_pooling;

  global_average_pooling_op->state = xnn_run_state_invalid;

  *global_average_pooling_op_out = global_average_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(global_average_pooling_op);
  return status;
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
    xnn_log_error("failed to setup Global Average Pooling (F32, NCW) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  global_average_pooling_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Global Average Pooling operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (width == 0) {
    xnn_log_error("failed to setup Global Average Pooling operator with width %zu: width must be non-zero", width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    global_average_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  xnn_update_f32_gavgpool_params(&global_average_pooling_op->f32_gavgpool_params,
    1.0f / (float) width, width);

  global_average_pooling_op->context.global_average_pooling_ncw = (struct global_average_pooling_ncw_context) {
    .input_elements = width * sizeof(float),
    .input = input,
    .input_channel_stride = width * sizeof(float),
    .input_batch_stride = global_average_pooling_op->channels * width * sizeof(float),
    .output = output,
    .output_channel_stride = sizeof(float),
    .output_batch_stride = global_average_pooling_op->channels * sizeof(float),
    .ukernel = xnn_params.f32.spchw_gavgpool.ukernel,
    .params.f32 = global_average_pooling_op->f32_gavgpool_params,
  };

  global_average_pooling_op->compute.type = xnn_parallelization_type_2d_tile_1d;
  global_average_pooling_op->compute.task_2d_tile_1d =
    (pthreadpool_task_2d_tile_1d_t) xnn_compute_global_average_pooling_ncw;
  global_average_pooling_op->compute.range[0] = batch_size;
  global_average_pooling_op->compute.range[1] = global_average_pooling_op->channels;
  global_average_pooling_op->compute.tile[0] = global_average_pooling_op->channels; //xnn_params.f32.spchw_gavgpool.channel_tile;

  global_average_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
