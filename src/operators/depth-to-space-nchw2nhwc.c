// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/operator.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>


enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x32(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  xnn_operator_t depth_to_space_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (output_channels == 0) {
    xnn_log_error("failed to create %s operator with %zu output channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), output_channels);
    goto error;
  }

  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32),
      output_channel_stride, output_channels);
    goto error;
  }

  if (block_size <= 1) {
    xnn_log_error("failed to create %s operator with %u block size: block size must be greater than 1",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32),
      block_size);
    goto error;
  }

  const size_t input_channels = output_channels * block_size * block_size;
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32),
      input_channel_stride, block_size, block_size, input_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  depth_to_space_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (depth_to_space_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32));
    goto error;
  }

  depth_to_space_op->channels = output_channels;
  depth_to_space_op->input_pixel_stride = input_channel_stride;
  depth_to_space_op->output_pixel_stride = output_channel_stride;
  depth_to_space_op->block_size = block_size;

  depth_to_space_op->type = xnn_operator_type_depth_to_space_nchw2nhwc_x32;
  depth_to_space_op->flags = flags;

  depth_to_space_op->state = xnn_run_state_invalid;

  *depth_to_space_op_out = depth_to_space_op;
  return xnn_status_success;

error:
  xnn_delete_operator(depth_to_space_op);
  return status;
}

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x32(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (depth_to_space_op->type != xnn_operator_type_depth_to_space_nchw2nhwc_x32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }
  depth_to_space_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error("failed to setup %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    depth_to_space_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const uint32_t block_size = depth_to_space_op->block_size;
  const size_t output_height = input_height * block_size;
  const size_t output_width = input_width * block_size;
  depth_to_space_op->context.depthtospace2d_chw = (struct depthtospace2d_chw2hwc_context) {
    .output_channels = depth_to_space_op->channels,
    .input_height = input_height,
    .input_width = input_width,
    .block_size = block_size,
    .input = input,
    .output = output,
    .input_batch_stride = depth_to_space_op->input_pixel_stride * input_height * input_width * sizeof(float),
    .output_batch_stride = depth_to_space_op->output_pixel_stride * output_height * output_width * sizeof(float),
    .output_channel_stride = depth_to_space_op->output_pixel_stride,
    .ukernel = xnn_params.x32.depthtospace2d_chw2hwc.ukernel,
  };

  depth_to_space_op->compute.type = xnn_parallelization_type_1d;
  depth_to_space_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_depthtospace2d_chw2hwc;
  depth_to_space_op->compute.range[0] = batch_size;
  depth_to_space_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
