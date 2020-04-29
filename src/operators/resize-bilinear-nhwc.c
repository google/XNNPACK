// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/operator.h>
#include <xnnpack/log.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/indirection.h>

enum xnn_status xnn_create_resize_bilinear2d_nhwc_f32(
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  xnn_operator_t resize_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Resize Bilinear operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create Resize Bilinear operator with %zu channels: number of channels must be non-zero",
      channels);
    goto error;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create Resize Bilinear operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_pixel_stride, channels);
    goto error;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create Resize Bilinear operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_pixel_stride, channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  resize_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (resize_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Resize Bilinear operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  resize_op->channels = channels;
  resize_op->input_pixel_stride = input_pixel_stride;
  resize_op->output_pixel_stride = output_pixel_stride;

  resize_op->type = xnn_operator_type_resize_bilinear_nhwc_f32;
  resize_op->ukernel.type = xnn_ukernel_type_unpooling;
  resize_op->flags = flags;

  resize_op->state = xnn_run_state_invalid;

  *resize_op_out = resize_op;
  return xnn_status_success;

error:
  xnn_delete_operator(resize_op);
  return status;
}

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_f32(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t output_height,
    size_t output_width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (resize_op->type != xnn_operator_type_resize_bilinear_nhwc_f32) {
    xnn_log_error("failed to setup Resize Bilinear (NHWC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  resize_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Resize Bilinear operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup Resize Bilinear operator with %zux%zu input: input dimensions must be non-zero",
      input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (max(input_width, input_height) >= 16777216) {
    xnn_log_error(
      "failed to setup Resize Bilinear operator with %zux%zu input: "
      "input dimensions must be below 2**24",
      input_width, input_height);
    return xnn_status_unsupported_parameter;
  }

  if (output_width == 0 || output_height == 0) {
    xnn_log_error(
      "failed to setup Resize Bilinear operator with %zux%zu output: output dimensions must be non-zero",
      output_width, output_height);
    return xnn_status_invalid_parameter;
  }

  if (max(output_width, output_height) >= 16777216) {
    xnn_log_error(
      "failed to setup Resize Bilinear operator with %zux%zu output: "
      "output dimensions must be below 2**24",
      output_width, output_height);
    return xnn_status_unsupported_parameter;
  }

  if (batch_size == 0) {
    resize_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  if (output_height * output_width != resize_op->last_output_height * resize_op->last_output_width) {
    const size_t indirection_buffer_size = sizeof(void*) * (output_height * output_width * 4);
    const size_t packed_weights_size = sizeof(float) * (output_height * output_width * 2);

    const void** indirection_buffer = (const void**) xnn_reallocate_memory(resize_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for indirection buffer", indirection_buffer_size);
      return xnn_status_out_of_memory;
    }
    resize_op->indirection_buffer = indirection_buffer;

    // Note: packed weights must be SIMD-aligned, so we can't use xnn_reallocate_memory
    xnn_release_simd_memory(resize_op->packed_weights);
    resize_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
    if (resize_op->packed_weights == NULL) {
      xnn_log_error("failed to allocate %zu bytes for packed weights", packed_weights_size);
      return xnn_status_out_of_memory;
    }
  }

  const size_t input_pixel_stride_in_bytes = resize_op->input_pixel_stride * sizeof(float);
  if (input_height != resize_op->last_input_height ||
      input_width != resize_op->last_input_width ||
      output_height != resize_op->last_output_height ||
      output_width != resize_op->last_output_width)
  {
    const uint32_t flags = resize_op->flags;
    xnn_indirection_init_resize_bilinear2d_f32(
      input_pixel_stride_in_bytes,
      input_height, input_width,
      output_height, output_width,
      input, resize_op->indirection_buffer, resize_op->packed_weights,
      !!(flags & XNN_FLAG_ALIGN_CORNERS),
      !!(flags & XNN_FLAG_TENSORFLOW_LEGACY_MODE));

    resize_op->last_input = input;
    resize_op->last_input_height = input_height;
    resize_op->last_input_width = input_width;
    resize_op->last_output_height = output_height;
    resize_op->last_output_width = output_width;
  }

  const size_t output_pixel_stride_in_bytes = resize_op->output_pixel_stride * sizeof(float);
  resize_op->context.resize_bilinear = (struct resize_bilinear_context) {
    .scaled_channels = resize_op->channels * sizeof(float),
    .indirect_input = resize_op->indirection_buffer,
    .input_offset = (size_t) ((uintptr_t) input - (uintptr_t) resize_op->last_input),
    .input_batch_stride = input_pixel_stride_in_bytes * input_height * input_width,
    .packed_weights = resize_op->packed_weights,
    .output = output,
    .output_pixel_stride = output_pixel_stride_in_bytes,
    .output_batch_stride = output_pixel_stride_in_bytes * output_height * output_width,
    .log2_wsize = 3 /* log2(2 * sizeof(float)) */,
    .ukernel = xnn_params.f32.ibilinear.ukernel,
  };

  const size_t output_size = output_height * output_width;
  size_t output_size_tile = output_size;
  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  if (num_threads > 1) {
    const size_t target_tiles_per_thread = 5;
    const size_t max_output_size_tile = divide_round_up(output_size, num_threads * target_tiles_per_thread);
    if (max_output_size_tile < output_size_tile) {
      const uint32_t output_size_subtile = xnn_params.f32.ibilinear.pixel_tile;
      output_size_tile =
        min(output_size_tile,
          divide_round_up(output_size_tile, max_output_size_tile * output_size_subtile) * output_size_subtile);
    }
  }
  resize_op->compute.type = xnn_parallelization_type_2d_tile_1d;
  resize_op->compute.task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_resize_bilinear;
  resize_op->compute.range[0] = batch_size;
  resize_op->compute.range[1] = output_size;
  resize_op->compute.tile[0] = output_size_tile;
  resize_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
