// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
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
#include <xnnpack/log.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/indirection.h>


enum xnn_status create_resize_bilinear2d_nchw(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct xnn_ibilinear_chw_config* ibilinear_chw_config,
    xnn_operator_t* resize_op_out)
{
  xnn_operator_t resize_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (output_width == 0 || output_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu output: output dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), output_width, output_height);
    goto error;
  }

  if (max(output_width, output_height) >= (1L << 24)) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu output: output dimensions must be below 2**24",
      xnn_operator_type_to_string(operator_type), output_width, output_height);
    goto error;
  }

  status = xnn_status_out_of_memory;

  resize_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (resize_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  resize_op->output_height = output_height;
  resize_op->output_width = output_width;

  resize_op->type = operator_type;
  resize_op->flags = flags;
  resize_op->ibilinear_chw_config = ibilinear_chw_config;

  resize_op->state = xnn_run_state_invalid;

  *resize_op_out = resize_op;
  return xnn_status_success;

error:
  xnn_delete_operator(resize_op);
  return status;
}

enum xnn_status xnn_create_resize_bilinear2d_nchw_f16(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_chw_config* ibilinear_chw_config = xnn_init_f16_ibilinear_chw_config();
  if (ibilinear_chw_config == NULL) {
    printf("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw_f16));
    return xnn_status_unsupported_hardware;
  }

  return create_resize_bilinear2d_nchw(
    output_height,
    output_width,
    flags,
    xnn_operator_type_resize_bilinear_nchw_f16,
    ibilinear_chw_config,
    resize_op_out);
}

enum xnn_status xnn_create_resize_bilinear2d_nchw_f32(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_chw_config* ibilinear_chw_config = xnn_init_f32_ibilinear_chw_config();
  if (ibilinear_chw_config == NULL) {
    printf("QQQfailed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw_f32));
    return xnn_status_unsupported_hardware;
  }

  return create_resize_bilinear2d_nchw(
    output_height,
    output_width,
    flags,
    xnn_operator_type_resize_bilinear_nchw_f32,
    ibilinear_chw_config,
    resize_op_out);
}

static enum xnn_status reshape_resize_bilinear2d_nchw(
    xnn_operator_t resize_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint32_t log2_data_element_size,
    uint32_t log2_weight_element_size,
    xnn_indirection_init_resize_bilinear2d_chw_fn indirection_init,
    pthreadpool_t threadpool)
{
  if (resize_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(resize_op->type));
    return xnn_status_invalid_parameter;
  }
  resize_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (input_width <= 1 || input_height <= 1) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be greater than 1",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (max(input_width, input_height) >= (1L << 24)) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be below 2**24",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height);
    return xnn_status_unsupported_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(expected_operator_type), input_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(expected_operator_type), output_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    resize_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t output_height = resize_op->output_height;
  const size_t output_width = resize_op->output_width;
  if (output_height * output_width != resize_op->last_output_height * resize_op->last_output_width) {
    const size_t indirection_buffer_size = sizeof(void*) * (output_height * output_width * 4);
    const size_t packed_weights_size = (output_height * output_width * 2) << log2_weight_element_size;

    const void** indirection_buffer = (const void**) xnn_reallocate_memory(resize_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator indirection buffer",
        indirection_buffer_size, xnn_operator_type_to_string(expected_operator_type));
      return xnn_status_out_of_memory;
    }
    resize_op->indirection_buffer = indirection_buffer;
    xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
      indirection_buffer_size, xnn_operator_type_to_string(expected_operator_type));

    // Note: packed weights must be SIMD-aligned, so we can't use xnn_reallocate_memory
    xnn_release_simd_memory(resize_op->packed_weights.pointer);
    resize_op->packed_weights.pointer = xnn_allocate_simd_memory(packed_weights_size);
    if (resize_op->packed_weights.pointer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator packed weights",
        packed_weights_size, xnn_operator_type_to_string(expected_operator_type));
      return xnn_status_out_of_memory;
    }
  }

  const size_t input_pixel_stride_in_bytes = 1 << log2_data_element_size; // Since the layout in CHW the pixels
  if (input_height != resize_op->last_input_height ||
      input_width != resize_op->last_input_width ||
      output_height != resize_op->last_output_height ||
      output_width != resize_op->last_output_width)
  {
    const uint32_t flags = resize_op->flags;
    // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
    void* dummy_input = (void*) XNN_ALLOCATION_ALIGNMENT;
    indirection_init(
        input_pixel_stride_in_bytes,
        input_height, input_width,
        output_height, output_width,
        dummy_input, resize_op->indirection_buffer, resize_op->packed_weights.pointer,
        !!(flags & XNN_FLAG_ALIGN_CORNERS),
        !!(flags & XNN_FLAG_TENSORFLOW_LEGACY_MODE));

    resize_op->last_input = dummy_input;
    resize_op->last_input_height = input_height;
    resize_op->last_input_width = input_width;
    resize_op->last_output_height = output_height;
    resize_op->last_output_width = output_width;
  }

  const struct xnn_ibilinear_chw_config* ibilinear_chw = resize_op->ibilinear_chw_config;
  // Resize bilinear packed weights can change when the operator is resized, we will not use weights cache.
  assert(resize_op->weights_cache == NULL);
  resize_op->context.resize_bilinear_chw = (struct resize_bilinear_chw_context) {
    .output_pixels = output_height * output_width,
    .channels = resize_op->channels,
    .input_channel_stride = (input_height * input_width) << log2_data_element_size,
    .indirect_input = resize_op->indirection_buffer,
    .input_batch_stride = (input_pixel_stride * input_height * input_width) << log2_data_element_size,
    .packed_weights = resize_op->packed_weights.pointer,
    .output_batch_stride = (output_pixel_stride * output_height * output_width) << log2_data_element_size,
    .output_channel_stride = (output_height * output_width) << log2_data_element_size,
    .ukernel = ibilinear_chw->ukernel,
  };

  #if XNN_TEST_MODE
    const size_t output_channel_tile = ibilinear_chw->channel_tile;
  #else
    size_t output_channel_tile = channels;
    const size_t num_threads = pthreadpool_get_threads_count(threadpool);
    if (num_threads > 1) {
      const size_t target_tiles_per_thread = 4;
      const size_t max_channel_tile = divide_round_up(output_channel_tile, num_threads * target_tiles_per_thread);
      if (max_channel_tile < output_channel_tile) {
        const uint32_t output_channel_subtile = ibilinear_chw->channel_tile;
        output_channel_tile =
          min(output_channel_tile,
            divide_round_up(output_channel_tile, max_channel_tile * output_channel_subtile) * output_channel_subtile);
      }
    }
  #endif
  resize_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
  resize_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_resize_bilinear_chw;
  resize_op->compute[0].range[0] = batch_size;
  resize_op->compute[0].range[1] = channels;
  resize_op->compute[0].tile[0] = output_channel_tile;
  resize_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_resize_bilinear2d_nchw_f16(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    pthreadpool_t threadpool)
{
  return reshape_resize_bilinear2d_nchw(
    resize_op,
    xnn_operator_type_resize_bilinear_nchw_f16,
    batch_size,
    input_height,
    input_width,
    channels,
    input_pixel_stride,
    output_pixel_stride,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_HALF,
    (xnn_indirection_init_resize_bilinear2d_chw_fn) xnn_indirection_init_resize_bilinear2d_chw_f16,
    threadpool);
}

enum xnn_status xnn_reshape_resize_bilinear2d_nchw_f32(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    pthreadpool_t threadpool)
{
  return reshape_resize_bilinear2d_nchw(
    resize_op,
    xnn_operator_type_resize_bilinear_nchw_f32,
    batch_size,
    input_height,
    input_width,
    channels,
    input_pixel_stride,
    output_pixel_stride,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    (xnn_indirection_init_resize_bilinear2d_chw_fn) xnn_indirection_init_resize_bilinear2d_chw_f32,
    threadpool);
}

static enum xnn_status setup_resize_bilinear2d_nchw(
    xnn_operator_t resize_op,
    enum xnn_operator_type expected_operator_type,
    const void* input,
    void* output)
{
  if (resize_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(resize_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (resize_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(resize_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  resize_op->context.resize_bilinear_chw.input_offset =
    (size_t) ((uintptr_t) input - (uintptr_t) resize_op->last_input);
  resize_op->context.resize_bilinear_chw.output = output;

  resize_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_resize_bilinear2d_nchw_f16(
    xnn_operator_t resize_op,
    const void* input,
    void* output)
{
  return setup_resize_bilinear2d_nchw(
    resize_op,
    xnn_operator_type_resize_bilinear_nchw_f16,
    input,
    output);
}

enum xnn_status xnn_setup_resize_bilinear2d_nchw_f32(
    xnn_operator_t resize_op,
    const float* input,
    float* output)
{
  return setup_resize_bilinear2d_nchw(
    resize_op,
    xnn_operator_type_resize_bilinear_nchw_f32,
    input,
    output);
}
