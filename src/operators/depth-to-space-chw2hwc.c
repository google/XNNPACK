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
#include <xnnpack/math.h>
#include <xnnpack/params.h>

enum xnn_status xnn_create_depth_to_space_chw2hwc_x32(
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  xnn_operator_t depth_to_space_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw_f32));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
        "failed to create %s operator with %zu channels: number of channels must be non-zero",
        xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), channels);
    goto error;
  }

  if (block_size < 2) {
    xnn_log_error(
        "failed to create %s operator with %u block size: block size must be greater than 1",
        xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), block_size);
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

  depth_to_space_op->channels = channels;
  depth_to_space_op->input_pixel_stride = input_pixel_stride;
  depth_to_space_op->output_pixel_stride = output_pixel_stride;
  depth_to_space_op->block_size = block_size;

  depth_to_space_op->type = xnn_operator_type_depth_to_space_nchw2nhwc_x32;
  depth_to_space_op->ukernel.type = xnn_ukernel_type_depth_to_space_chw2hwc;
  depth_to_space_op->flags = flags;

  depth_to_space_op->state = xnn_run_state_invalid;

  *depth_to_space_op_out = depth_to_space_op;
  return xnn_status_success;

error:
  xnn_delete_operator(depth_to_space_op);
  return status;
}

enum xnn_status xnn_setup_depth_to_space_chw2hwc_x32(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t output_height,
    size_t output_width,
    const float* input,
    float* output,
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
    xnn_log_error(
        "failed to setup %s operator with %zux%zu input: input dimensions must be greater than 1",
        xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (max(input_width, input_height) >= 16777216) {
    xnn_log_error(
        "failed to setup %s operator with %zux%zu input: input dimensions must be below 2**24",
        xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), input_width, input_height);
    return xnn_status_unsupported_parameter;
  }

  if (output_width == 0 || output_height == 0) {
    xnn_log_error(
        "failed to setup %s operator with %zux%zu output: output dimensions must be non-zero",
        xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), output_width, output_height);
    return xnn_status_invalid_parameter;
  }

  if (max(output_width, output_height) >= 16777216) {
    xnn_log_error(
        "failed to setup %s operator with %zux%zu output: output dimensions must be below 2**24",
        xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32), output_width, output_height);
    return xnn_status_unsupported_parameter;
  }

  if (batch_size == 0) {
    depth_to_space_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  depth_to_space_op->context.depth_to_space_chw = (struct depth_to_space_chw2hwc_context) {
    .output_channels = depth_to_space_op->output_pixel_stride,
    .input_height = input_height,
    .input_width = input_width,
    .block_size = depth_to_space_op->block_size,
    .input = input,
    .output = output,
    // TODO(artsiom,kartynnik): Check with maratek@ for additional padding at the end of the image
    .input_batch_stride = depth_to_space_op->input_pixel_stride * input_height * input_width * sizeof(float),
    .output_batch_stride = depth_to_space_op->output_pixel_stride * output_height * output_width * sizeof(float),
    .input_channel_stride = input_height * input_width * sizeof(float),
    .input_height_stride = input_width * sizeof(float),
    .output_height_stride = output_width * depth_to_space_op->output_pixel_stride * sizeof(float),
    .output_width_stride = depth_to_space_op->output_pixel_stride * sizeof(float),
    .ukernel = xnn_params.x32.depth_to_space_chw2hwc.ukernel,
  };

  depth_to_space_op->compute.type = xnn_parallelization_type_1d;
  depth_to_space_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_depth_to_space_chw2hwc;
  depth_to_space_op->compute.range[0] = batch_size;
  depth_to_space_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
