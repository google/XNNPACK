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


static inline size_t compute_output_dimension(
    size_t input_dimension,
    size_t input_padding_dimension,
    size_t kernel_dimension)
{
  return doz(kernel_dimension * input_dimension, input_padding_dimension);
}

enum xnn_status xnn_create_unpooling2d_nhwc_x32(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint32_t flags,
    xnn_operator_t* unpooling_op_out)
{
  xnn_operator_t unpooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Unpooling operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to create Unpooling operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      pooling_width, pooling_height);
    goto error;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to create Unpooling operator with 1 pooling element: 1x1 unpooling is meaningless");
    goto error;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create Unpooling operator with %zu channels: number of channels must be non-zero",
      channels);
    goto error;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create Unpooling operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_pixel_stride, channels);
    goto error;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create Unpooling operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_pixel_stride, channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  unpooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (unpooling_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Unpooling operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  unpooling_op->padding_top = input_padding_top;
  unpooling_op->padding_right = input_padding_right;
  unpooling_op->padding_bottom = input_padding_bottom;
  unpooling_op->padding_left = input_padding_left;

  unpooling_op->kernel_height = pooling_height;
  unpooling_op->kernel_width = pooling_width;
  unpooling_op->channels = channels;
  unpooling_op->input_pixel_stride = input_pixel_stride;
  unpooling_op->output_pixel_stride = output_pixel_stride;

  unpooling_op->type = xnn_operator_type_unpooling_nhwc_x32;
  unpooling_op->ukernel.type = xnn_ukernel_type_unpooling;

  unpooling_op->state = xnn_run_state_invalid;

  *unpooling_op_out = unpooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(unpooling_op);
  return status;
}

enum xnn_status xnn_setup_unpooling2d_nhwc_x32(
    xnn_operator_t unpooling_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    const uint32_t* index,
    void* output,
    pthreadpool_t threadpool)
{
  if (unpooling_op->type != xnn_operator_type_unpooling_nhwc_x32) {
    xnn_log_error("failed to setup Unpooling (X32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  unpooling_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Unpooling operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup Unpooling operator with %zux%zu input: input dimensions must be non-zero",
      input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    unpooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  unpooling_op->batch_size = batch_size;
  unpooling_op->input_height = input_height;
  unpooling_op->input_width = input_width;
  unpooling_op->input = input;

  unpooling_op->output_height = compute_output_dimension(
    input_height, unpooling_op->padding_top + unpooling_op->padding_bottom,
    unpooling_op->kernel_height);
  unpooling_op->output_width = compute_output_dimension(
    input_width, unpooling_op->padding_left + unpooling_op->padding_right,
    unpooling_op->kernel_width);
  unpooling_op->output = output;

  size_t valid_batch_size = 0;
  if (output == unpooling_op->last_output &&
      input_height == unpooling_op->last_input_height &&
      input_width == unpooling_op->last_input_width)
  {
    valid_batch_size = unpooling_op->valid_batch_size;
    if (batch_size <= valid_batch_size) {
      unpooling_op->compute.range[0] = batch_size * input_height;
      unpooling_op->state = xnn_run_state_ready;
      return xnn_status_success;
    }
  }

  const size_t pooling_height = unpooling_op->kernel_height;
  const size_t pooling_width = unpooling_op->kernel_width;
  const size_t pooling_size = pooling_height * pooling_width;

  const size_t indirection_buffer_size = sizeof(void*) * (batch_size * input_height * input_width * pooling_size);

  void** indirection_buffer = (void**) xnn_reallocate_memory(unpooling_op->indirection_buffer, indirection_buffer_size);
  if (indirection_buffer == NULL) {
    xnn_log_error("failed to allocate %zu bytes for indirection buffer", indirection_buffer_size);
    return xnn_status_out_of_memory;
  }
  unpooling_op->indirection_buffer = (const void**) indirection_buffer;

  xnn_indirection_init_unpool2d(unpooling_op, valid_batch_size, 2 /* log2(sizeof(type32)) */);

  const size_t channels = unpooling_op->channels;
  const size_t input_pixel_stride_in_bytes = unpooling_op->input_pixel_stride * sizeof(float);
  unpooling_op->context.unpooling = (struct unpooling_context) {
    .input = input,
    .input_height_stride = input_width * input_pixel_stride_in_bytes,
    .input_width_stride = input_pixel_stride_in_bytes,
    .index = index,
    .index_height_stride = input_width * channels * sizeof(uint32_t),
    .index_width_stride = channels * sizeof(uint32_t),
    .indirect_output = indirection_buffer,
    .indirect_output_height_stride = input_width * pooling_size * sizeof(void*),
    .indirect_output_width_stride = pooling_size * sizeof(void*),
    .pooling_size = pooling_size,
    .channels = channels,
    .fill_value = 0,
    .ukernel = xnn_params.x32.unpool,
  };
  unpooling_op->compute.type = xnn_parallelization_type_2d;
  unpooling_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_unpooling;
  unpooling_op->compute.range[0] = batch_size * input_height;
  unpooling_op->compute.range[1] = input_width;
  unpooling_op->state = xnn_run_state_ready;

  unpooling_op->last_output = output;
  unpooling_op->last_input_height = input_height;
  unpooling_op->last_input_width = input_width;
  unpooling_op->valid_batch_size = max(valid_batch_size, batch_size);

  return xnn_status_success;
}
