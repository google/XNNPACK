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
#include <xnnpack/params.h>


enum xnn_status xnn_create_channel_pad_nc_x32(
    size_t input_channels,
    size_t pad_before_channels,
    size_t pad_after_channels,
    size_t input_stride,
    size_t output_stride,
    const void* pad_value,
    uint32_t flags,
    xnn_operator_t* channel_pad_op_out)
{
  xnn_operator_t channel_pad_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Channel Pad operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (input_channels == 0) {
    xnn_log_error(
      "failed to create Channel Pad operator with %zu input channels: number of channels must be non-zero",
      input_channels);
    goto error;
  }

  if (input_stride < input_channels) {
    xnn_log_error(
      "failed to create Channel Pad operator with input element stride of %zu: "
      "stride must be at least as large as the number of input channels (%zu)",
      input_stride, input_channels);
    goto error;
  }

  const size_t output_channels = pad_before_channels + input_channels + pad_after_channels;
  if (output_stride < output_channels) {
    xnn_log_error(
      "failed to create Channel Pad operator with output element stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu+%zu+%zu)",
      output_stride, pad_before_channels, input_channels, pad_after_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  channel_pad_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (channel_pad_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Channel Pad operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  channel_pad_op->channels = input_channels;
  channel_pad_op->pad_before_channels = pad_before_channels;
  channel_pad_op->pad_after_channels = pad_after_channels;
  channel_pad_op->input_pixel_stride = input_stride;
  channel_pad_op->output_pixel_stride = output_stride;
  channel_pad_op->pad_value = *((const uint32_t*) pad_value);

  channel_pad_op->type = xnn_operator_type_channel_pad_nc_x32;
  channel_pad_op->ukernel.type = xnn_ukernel_type_pad;

  channel_pad_op->state = xnn_run_state_invalid;

  *channel_pad_op_out = channel_pad_op;
  return xnn_status_success;

error:
  xnn_delete_operator(channel_pad_op);
  return status;
}

enum xnn_status xnn_setup_channel_pad_nc_x32(
    xnn_operator_t channel_pad_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (channel_pad_op->type != xnn_operator_type_channel_pad_nc_x32) {
    xnn_log_error("failed to setup Channel Pad (X32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  channel_pad_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Channel Pad operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    channel_pad_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  channel_pad_op->batch_size = batch_size;
  channel_pad_op->input = input;
  channel_pad_op->output = output;

  channel_pad_op->context.channel_pad = (struct channel_pad_context) {
    .x = input,
    .x_stride = channel_pad_op->input_pixel_stride * sizeof(uint32_t),
    .y = output,
    .y_stride = channel_pad_op->output_pixel_stride * sizeof(uint32_t),
    .n = channel_pad_op->channels * sizeof(uint32_t),
    .l = channel_pad_op->pad_before_channels * sizeof(uint32_t),
    .r = channel_pad_op->pad_after_channels * sizeof(uint32_t),
    .c = channel_pad_op->pad_value,
    .ukernel = xnn_params.x32.pad.ukernel,
  };
  channel_pad_op->compute.type = xnn_parallelization_type_1d_tile_1d;
  channel_pad_op->compute.task_1d_tile_1d =
      (pthreadpool_task_1d_tile_1d_t) xnn_compute_channel_pad;
  channel_pad_op->compute.range[0] = batch_size;
  channel_pad_op->compute.tile[0] = xnn_params.x32.pad.mr;
  channel_pad_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
