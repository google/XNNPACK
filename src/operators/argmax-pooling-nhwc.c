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
#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/indirection.h>


static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension)
{
  return padded_input_dimension / kernel_dimension;
}

static const struct argmaxpool_parameters* select_ukernel(
    size_t pooling_size,
    const struct argmaxpool_parameters* ukernel)
{
  while (ukernel->qr == 0 && ukernel->mr < pooling_size) {
    ukernel++;
  }
  return ukernel;
}

enum xnn_status xnn_create_argmax_pooling2d_nhwc_f32(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* argmax_pooling_op_out)
{
  xnn_operator_t argmax_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Argmax Pooling operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      pooling_width, pooling_height);
    goto error;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with 1 pooling element: "
      "1x1 pooling is meaningless");
    goto error;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with %zu channels: "
      "number of channels must be non-zero",
      channels);
    goto error;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_pixel_stride, channels);
    goto error;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_pixel_stride, channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with NaN output lower bound: "
      "lower bound must be non-NaN");
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with NaN output upper bound: "
      "upper bound must be non-NaN");
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Argmax Pooling operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    goto error;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to create Argmax Pooling operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      goto error;
    }
  }

  status = xnn_status_out_of_memory;

  argmax_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (argmax_pooling_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Argmax Pooling operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  argmax_pooling_op->padding_top = input_padding_top;
  argmax_pooling_op->padding_right = input_padding_right;
  argmax_pooling_op->padding_bottom = input_padding_bottom;
  argmax_pooling_op->padding_left = input_padding_left;

  argmax_pooling_op->kernel_height = pooling_height;
  argmax_pooling_op->kernel_width = pooling_width;
  argmax_pooling_op->stride_height = pooling_height;
  argmax_pooling_op->stride_width = pooling_width;
  argmax_pooling_op->dilation_height = 1;
  argmax_pooling_op->dilation_width = 1;
  argmax_pooling_op->channels = channels;
  argmax_pooling_op->input_pixel_stride = input_pixel_stride;
  argmax_pooling_op->output_pixel_stride = output_pixel_stride;

  argmax_pooling_op->f32_minmax_params = xnn_init_f32_minmax_params(output_min, output_max);

  argmax_pooling_op->type = xnn_operator_type_argmax_pooling_nhwc_f32;
  argmax_pooling_op->ukernel.type = xnn_ukernel_type_argmax_pooling;
  argmax_pooling_op->flags = flags;

  argmax_pooling_op->state = xnn_run_state_invalid;

  *argmax_pooling_op_out = argmax_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(argmax_pooling_op);
  return status;
}

enum xnn_status xnn_setup_argmax_pooling2d_nhwc_f32(
    xnn_operator_t argmax_pooling_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const float* input,
    float* output,
    uint32_t* index,
    pthreadpool_t threadpool)
{
  if (argmax_pooling_op->type != xnn_operator_type_argmax_pooling_nhwc_f32) {
    xnn_log_error("failed to setup Argmax Pooling (NHWC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  argmax_pooling_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Argmax Pooling operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup Argmax Pooling operator with %zux%zu input: input dimensions must be non-zero",
      input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    argmax_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  argmax_pooling_op->batch_size = batch_size;
  argmax_pooling_op->input_height = input_height;
  argmax_pooling_op->input_width = input_width;
  argmax_pooling_op->input = input;

  const size_t pooling_height = argmax_pooling_op->kernel_height;
  const size_t pooling_width = argmax_pooling_op->kernel_width;

  if (argmax_pooling_op->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    argmax_pooling_op->output_height = divide_round_up(input_height, pooling_height);
    argmax_pooling_op->output_width = divide_round_up(input_width, pooling_width);

    const uint32_t padding_height = argmax_pooling_op->output_height * pooling_height - input_height;
    const uint32_t padding_width = argmax_pooling_op->output_width * pooling_width - input_width;
    argmax_pooling_op->padding_top = padding_height / 2;
    argmax_pooling_op->padding_left = padding_width / 2;
    argmax_pooling_op->padding_bottom = padding_height - argmax_pooling_op->padding_top;
    argmax_pooling_op->padding_right = padding_width - argmax_pooling_op->padding_left;
  } else {
    argmax_pooling_op->output_height = compute_output_dimension(
        argmax_pooling_op->padding_top + input_height + argmax_pooling_op->padding_bottom,
        argmax_pooling_op->kernel_height);
    argmax_pooling_op->output_width = compute_output_dimension(
        argmax_pooling_op->padding_left + input_width + argmax_pooling_op->padding_right,
        argmax_pooling_op->kernel_width);
  }

  const size_t pooling_size = pooling_height * pooling_width;
  const size_t output_height = argmax_pooling_op->output_height;
  const size_t output_width = argmax_pooling_op->output_width;
  const struct argmaxpool_parameters* ukernel = select_ukernel(pooling_size, xnn_params.f32.argmaxpool);
  const uint32_t mr = ukernel->mr;

  const size_t step_width = pooling_width;
  const size_t step_height = pooling_size + (output_width - 1) * step_width * pooling_height;

  if (input_height != argmax_pooling_op->last_input_height ||
      input_width != argmax_pooling_op->last_input_width)
  {
    // Micro-kernel may read up to (mr - 1) elements after the end of indirection buffer.
    const size_t indirection_buffer_size = sizeof(void*) * ((mr - 1) + output_height * step_height);

    const void** indirection_buffer = (const void**) xnn_reallocate_memory(argmax_pooling_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for indirection buffer", indirection_buffer_size);
      return xnn_status_out_of_memory;
    }
    argmax_pooling_op->indirection_buffer = indirection_buffer;

    xnn_indirection_init_maxpool2d(argmax_pooling_op, step_height, step_width, 2 /* log2(sizeof(float)) */);

    argmax_pooling_op->last_input = input;
    argmax_pooling_op->last_input_height = input_height;
    argmax_pooling_op->last_input_width = input_width;
  }

  const size_t channels = argmax_pooling_op->channels;

  const size_t indirect_input_height_stride = step_height * sizeof(void*);
  const size_t output_width_stride = argmax_pooling_op->output_pixel_stride * sizeof(float);
  const size_t output_height_stride = output_width * output_width_stride;
  const size_t index_height_stride = output_width * channels * sizeof(uint32_t);

  const uint32_t qr = ukernel->qr;
  const size_t multipass_adjustment = qr == 0 ? 0 : round_up(pooling_size - mr, qr) + mr - qr;
  argmax_pooling_op->context.argmax_pooling = (struct argmax_pooling_context) {
    .indirect_input = argmax_pooling_op->indirection_buffer,
    .indirect_input_height_stride = indirect_input_height_stride,
    .input_offset = (size_t) ((uintptr_t) input - (uintptr_t) argmax_pooling_op->last_input),
    .input_batch_stride = input_height * input_width * argmax_pooling_op->input_pixel_stride * sizeof(float),
    .output = output,
    .output_batch_stride = output_height * output_height_stride,
    .output_height_stride = output_height_stride,
    .output_width = output_width,
    .index = index,
    .index_batch_stride = output_height * index_height_stride,
    .index_height_stride = index_height_stride,
    .pooling_size = pooling_size,
    .channels = channels,
    .input_increment = (pooling_height * step_width - multipass_adjustment) * sizeof(void*),
    .output_increment = output_width_stride - channels * sizeof(float),
    .params.f32 = argmax_pooling_op->f32_minmax_params,
  };
  argmax_pooling_op->compute.type = xnn_parallelization_type_2d;
  argmax_pooling_op->compute.range[0] = batch_size;
  argmax_pooling_op->compute.range[1] = output_height;

  if (pooling_size <= mr) {
    argmax_pooling_op->context.argmax_pooling.unipass_ukernel = ukernel->up;
    argmax_pooling_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_argmax_pooling_unipass;
  } else {
    argmax_pooling_op->context.argmax_pooling.multipass_ukernel = ukernel->mp;
    argmax_pooling_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_argmax_pooling_multipass;
  }
  argmax_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

