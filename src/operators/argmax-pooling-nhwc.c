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
#include <xnnpack/config.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>
#include <xnnpack/indirection.h>


static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension)
{
  return padded_input_dimension / kernel_dimension;
}

static const struct xnn_argmaxpool_config* select_ukernel(
    size_t pooling_size,
    const struct xnn_argmaxpool_config* ukernel)
{
  while (ukernel->remainder_pass_tile_size == 0 && ukernel->first_pass_tile_size < pooling_size) {
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
    uint32_t flags,
    xnn_operator_t* argmax_pooling_op_out)
{
  xnn_operator_t argmax_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  const struct xnn_argmaxpool_config* argmaxpool_config = xnn_init_f32_argmaxpool_config();
  if (argmaxpool_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32), pooling_width, pooling_height);
    goto error;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to create %s operator with 1 pooling element: 1x1 pooling is meaningless",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32));
    goto error;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to create %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32),
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      goto error;
    }
  }

  status = xnn_status_out_of_memory;

  argmax_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (argmax_pooling_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32));
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

  argmax_pooling_op->type = xnn_operator_type_argmax_pooling_nhwc_f32;
  argmax_pooling_op->flags = flags;
  argmax_pooling_op->argmaxpool_config = argmaxpool_config;

  argmax_pooling_op->state = xnn_run_state_invalid;

  *argmax_pooling_op_out = argmax_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(argmax_pooling_op);
  return status;
}

enum xnn_status xnn_reshape_argmax_pooling2d_nhwc_f32(
    xnn_operator_t argmax_pooling_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t* output_height_out,
    size_t* output_width_out,
    pthreadpool_t threadpool)
{
  if (argmax_pooling_op->type != xnn_operator_type_argmax_pooling_nhwc_f32) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32),
      xnn_operator_type_to_string(argmax_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  argmax_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32), input_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32), output_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    argmax_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  argmax_pooling_op->channels = channels;
  argmax_pooling_op->input_pixel_stride = input_pixel_stride;
  argmax_pooling_op->output_pixel_stride = output_pixel_stride;
  argmax_pooling_op->batch_size = batch_size;
  argmax_pooling_op->input_height = input_height;
  argmax_pooling_op->input_width = input_width;

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

  const size_t output_height = argmax_pooling_op->output_height;
  const size_t output_width = argmax_pooling_op->output_width;
  if (output_height_out != NULL) {
    *output_height_out = output_height;
  }
  if (output_width_out != NULL) {
    *output_width_out = output_width;
  }
  const size_t pooling_size = pooling_height * pooling_width;
  const struct xnn_argmaxpool_config* argmaxpool_config = argmax_pooling_op->argmaxpool_config;
  const struct xnn_argmaxpool_config* ukernel = select_ukernel(pooling_size, argmaxpool_config);
  const uint32_t first_pass_tile_size = ukernel->first_pass_tile_size;

  const size_t step_width = pooling_width;
  const size_t step_height = pooling_size + (output_width - 1) * step_width * pooling_height;

  // Micro-kernel may read up to (first_pass_tile_size - 1) elements after the end of indirection buffer.
  const size_t indirection_buffer_size = sizeof(void*) * ((first_pass_tile_size - 1) + output_height * step_height);

  // Allocate indirection buffer as size is known here. We initialize the buffer in setup, when input pointer is known.
  const void** indirection_buffer =
    (const void**) xnn_reallocate_memory(argmax_pooling_op->indirection_buffer, indirection_buffer_size);
  if (indirection_buffer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator indirection buffer",
      indirection_buffer_size, xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32));
    return xnn_status_out_of_memory;
  }
  argmax_pooling_op->indirection_buffer = indirection_buffer;
  xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
    indirection_buffer_size, xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32));

  const size_t indirect_input_height_stride = step_height * sizeof(void*);
  const size_t output_width_stride = output_pixel_stride * sizeof(float);
  const size_t output_height_stride = output_width * output_width_stride;
  const size_t index_height_stride = output_width * channels * sizeof(uint32_t);

  const uint32_t remainder_pass_tile_size = ukernel->remainder_pass_tile_size;
  const size_t multipass_adjustment = remainder_pass_tile_size == 0 ? 0 : round_up(pooling_size - first_pass_tile_size, remainder_pass_tile_size) + first_pass_tile_size - remainder_pass_tile_size;
  argmax_pooling_op->context.argmax_pooling = (struct argmax_pooling_context) {
    .indirect_input = argmax_pooling_op->indirection_buffer,
    .indirect_input_height_stride = indirect_input_height_stride,
    .input_batch_stride = input_height * input_width * input_pixel_stride * sizeof(float),
    .output_batch_stride = output_height * output_height_stride,
    .output_height_stride = output_height_stride,
    .output_height = output_height,
    .output_width = output_width,
    .index_batch_stride = output_height * index_height_stride,
    .index_height_stride = index_height_stride,
    .pooling_size = pooling_size,
    .channels = channels,
    .input_increment = (pooling_height * step_width - multipass_adjustment) * sizeof(void*),
    .output_increment = output_width_stride - channels * sizeof(float),
  };
  argmax_pooling_op->compute[0].range[0] = batch_size;
  argmax_pooling_op->compute[0].range[1] = output_height;

  if (pooling_size <= first_pass_tile_size) {
    *workspace_size = 0;
    *workspace_alignment = 1;
    argmax_pooling_op->compute[0].type = xnn_parallelization_type_2d;
    argmax_pooling_op->context.argmax_pooling.unipass_ukernel = ukernel->up;
    argmax_pooling_op->compute[0].task_2d = (pthreadpool_task_2d_t) xnn_compute_argmax_pooling_unipass;
  } else {
    const size_t accumulation_buffer_size =
      round_up_po2((channels + XNN_MULTIPASS_EXTRA_BYTES / sizeof(float)) * sizeof(float), XNN_ALLOCATION_ALIGNMENT);
    const size_t index_buffer_size =
      round_up_po2((channels + XNN_MULTIPASS_EXTRA_BYTES / sizeof(float)) * sizeof(uint32_t), XNN_ALLOCATION_ALIGNMENT);
    const size_t accumulation_and_index_buffer_size = accumulation_buffer_size + index_buffer_size;

    argmax_pooling_op->context.argmax_pooling.accumulation_buffer_size = accumulation_buffer_size;
    argmax_pooling_op->context.argmax_pooling.accumulation_and_index_buffer_size = accumulation_and_index_buffer_size;

    const size_t num_threads = pthreadpool_get_threads_count(threadpool);
    const bool use_threads_workspace = num_threads < batch_size * output_height;

    if (use_threads_workspace) {
      *workspace_size = num_threads * accumulation_and_index_buffer_size;
      *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
      argmax_pooling_op->compute[0].type = xnn_parallelization_type_2d_with_thread;
      argmax_pooling_op->compute[0].task_2d_with_thread =
        (pthreadpool_task_2d_with_thread_t) xnn_compute_argmax_pooling_multipass_with_thread;
    } else {
      *workspace_size = batch_size * output_height * accumulation_and_index_buffer_size;
      *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
      argmax_pooling_op->compute[0].type = xnn_parallelization_type_2d;
      argmax_pooling_op->compute[0].task_2d = (pthreadpool_task_2d_t) xnn_compute_argmax_pooling_multipass;
    }

    argmax_pooling_op->context.argmax_pooling.multipass_ukernel = ukernel->mp;
  }
  argmax_pooling_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_setup_argmax_pooling2d_nhwc_f32(
    xnn_operator_t argmax_pooling_op,
    void* workspace,
    const float* input,
    float* output,
    uint32_t* index)
{
  if (argmax_pooling_op->type != xnn_operator_type_argmax_pooling_nhwc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_argmax_pooling_nhwc_f32),
      xnn_operator_type_to_string(argmax_pooling_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (argmax_pooling_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(argmax_pooling_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  // Set input before initializing indirection buffers.
  argmax_pooling_op->input = input;
  argmax_pooling_op->context.argmax_pooling.output = output;
  argmax_pooling_op->context.argmax_pooling.index = index;

  if ((argmax_pooling_op->context.argmax_pooling.accumulation_buffer_size != 0) && workspace == NULL) {
    xnn_log_error(
        "failed to setup %s operator: workspace is NULL",
        xnn_operator_type_to_string(argmax_pooling_op->type));
  }
  argmax_pooling_op->context.argmax_pooling.multipass_buffer = workspace;

  const size_t pooling_height = argmax_pooling_op->kernel_height;
  const size_t pooling_width = argmax_pooling_op->kernel_width;
  const size_t pooling_size = pooling_height * pooling_width;
  const size_t output_width = argmax_pooling_op->output_width;
  // TODO(zhin): Consider storing step_width and step_height in operator, this is already calculated in reshape.
  const size_t step_width = pooling_width;
  const size_t step_height = pooling_size + (output_width - 1) * step_width * pooling_height;

  xnn_indirection_init_maxpool2d(argmax_pooling_op, step_height, step_width, /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT);

  argmax_pooling_op->context.argmax_pooling.indirect_input = argmax_pooling_op->indirection_buffer,

  argmax_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

