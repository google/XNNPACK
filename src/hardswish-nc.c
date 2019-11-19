// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

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


enum xnn_status xnn_create_hardswish_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* hardswish_op_out)
{
  xnn_operator_t hardswish_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create HardSwish operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create HardSwish operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create HardSwish operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create HardSwish operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  hardswish_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (hardswish_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for xnn_operator structure", sizeof(struct xnn_operator));
    goto error;
  }

  hardswish_op->channels = channels;
  hardswish_op->input_pixel_stride = input_stride;
  hardswish_op->output_pixel_stride = output_stride;
  hardswish_op->f32_hswish_params = xnn_init_f32_hswish_params();

  hardswish_op->type = xnn_operator_type_hardswish_nc_f32;
  hardswish_op->ukernel.type = xnn_ukernel_type_hswish;

  hardswish_op->state = xnn_run_state_invalid;

  *hardswish_op_out = hardswish_op;
  return xnn_status_success;

error:
  xnn_delete_operator(hardswish_op);
  return status;
}

enum xnn_status xnn_setup_hardswish_nc_f32(
    xnn_operator_t hardswish_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (hardswish_op->type != xnn_operator_type_hardswish_nc_f32) {
    xnn_log_error("failed to setup HardSwish (F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  hardswish_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup HardSwish operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    hardswish_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = hardswish_op->channels;
  const size_t input_stride = hardswish_op->input_pixel_stride;
  const size_t output_stride = hardswish_op->output_pixel_stride;
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    hardswish_op->context.univector_contiguous = (struct univector_contiguous_context) {
      .x = input,
      .x_stride = input_stride * sizeof(float),
      .y = output,
      .y_stride = output_stride * sizeof(float),
      .ukernel = xnn_params.f32.hswish,
      .params.f32_hswish = hardswish_op->f32_hswish_params,
    };
    hardswish_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    hardswish_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
    hardswish_op->compute.range[0] = batch_size * channels * sizeof(float);
    hardswish_op->compute.tile[0] = block_size;
  } else {
    hardswish_op->context.univector_strided = (struct univector_strided_context) {
      .n = channels * sizeof(float),
      .x = input,
      .x_stride = input_stride * sizeof(float),
      .y = output,
      .y_stride = output_stride * sizeof(float),
      .ukernel = xnn_params.f32.hswish,
      .params.f32_hswish = hardswish_op->f32_hswish_params,
    };
    hardswish_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    hardswish_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
    hardswish_op->compute.range[0] = batch_size;
    hardswish_op->compute.tile[0] = 1;
  }
  hardswish_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
