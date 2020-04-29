// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


enum xnn_status xnn_create_prelu_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    const float* negative_slope,
    uint32_t flags,
    xnn_operator_t* prelu_op_out)
{
  xnn_operator_t prelu_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create PReLU operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create PReLU operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create PReLU operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create PReLU operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  prelu_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (prelu_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for PReLU operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  const size_t packed_channels = round_up_po2(channels, XNN_EXTRA_BYTES / sizeof(float));
  prelu_op->packed_weights = xnn_allocate_simd_memory(packed_channels * sizeof(float));
  if (prelu_op->packed_weights == NULL) {
    xnn_log_error("failed to allocate %zu bytes for packed slope data",
      packed_channels * sizeof(float));
    goto error;
  }
  memcpy(prelu_op->packed_weights, negative_slope, channels * sizeof(float));

  prelu_op->channels = channels;
  prelu_op->input_pixel_stride = input_stride;
  prelu_op->output_pixel_stride = output_stride;

  prelu_op->type = xnn_operator_type_prelu_nc_f32;
  prelu_op->ukernel.type = xnn_ukernel_type_prelu;

  prelu_op->state = xnn_run_state_invalid;

  *prelu_op_out = prelu_op;
  return xnn_status_success;

error:
  xnn_delete_operator(prelu_op);
  return status;
}

enum xnn_status xnn_setup_prelu_nc_f32(
    xnn_operator_t prelu_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (prelu_op->type != xnn_operator_type_prelu_nc_f32) {
    xnn_log_error("failed to setup PReLU (NC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  prelu_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup PReLU operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    prelu_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = prelu_op->channels;
  prelu_op->context.prelu = (struct prelu_context) {
    .n = channels * sizeof(float),
    .x = input,
    .x_stride = prelu_op->input_pixel_stride * sizeof(float),
    .w = prelu_op->packed_weights,
    .y = output,
    .y_stride = prelu_op->output_pixel_stride * sizeof(float),
    .ukernel = xnn_params.f32.prelu.ukernel,
  };

  size_t batch_tile = batch_size;
  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  if (num_threads > 1) {
    const size_t target_tiles_per_thread = 5;
    const size_t max_batch_tile = divide_round_up(batch_size, num_threads * target_tiles_per_thread);
    if (max_batch_tile < batch_tile) {
      const uint32_t row_tile = xnn_params.f32.prelu.row_tile;
      batch_tile = min(batch_tile, divide_round_up(batch_tile, max_batch_tile * row_tile) * row_tile);
    }
  }
  prelu_op->compute.type = xnn_parallelization_type_1d_tile_1d;
  prelu_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_prelu;
  prelu_op->compute.range[0] = batch_size;
  prelu_op->compute.tile[0] = batch_tile;
  prelu_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
