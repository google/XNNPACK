// Copyright 2020 Google LLC
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


enum xnn_status create_resize_bilinear2d_nchw(
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint32_t flags,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* resize_op_out)
{
  xnn_operator_t resize_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
    goto error;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), input_pixel_stride, channels);
    goto error;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), output_pixel_stride, channels);
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

  resize_op->channels = channels;
  resize_op->input_pixel_stride = input_pixel_stride;
  resize_op->output_pixel_stride = output_pixel_stride;

  resize_op->type = operator_type;
  resize_op->flags = flags;

  resize_op->state = xnn_run_state_invalid;

  *resize_op_out = resize_op;
  return xnn_status_success;

error:
  xnn_delete_operator(resize_op);
  return status;
}

enum xnn_status xnn_create_resize_bilinear2d_nchw_f16(
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  return create_resize_bilinear2d_nchw(
    channels,
    input_pixel_stride,
    output_pixel_stride,
    flags,
    XNN_INIT_FLAG_F16 | XNN_INIT_FLAG_F16_NATIVE,
    xnn_operator_type_resize_bilinear_nchw_f16,
    resize_op_out);
}

enum xnn_status xnn_create_resize_bilinear2d_nchw_f32(
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  return create_resize_bilinear2d_nchw(
    channels,
    input_pixel_stride,
    output_pixel_stride,
    flags,
    XNN_INIT_FLAG_F32,
    xnn_operator_type_resize_bilinear_nchw_f32,
    resize_op_out);
}


static enum xnn_status setup_resize_bilinear2d_nchw(
    xnn_operator_t resize_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t output_height,
    size_t output_width,
    const void* input,
    void* output,
    uint32_t log2_element_size,
    uint32_t log2_weight_element_size,
    xnn_indirection_init_resize_bilinear2d_chw_fn indirection_init,
    const struct ibilinear_chw_parameters ibilinear_chw[restrict XNN_MIN_ELEMENTS(1)],
    size_t num_threads)
{
  if (resize_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(resize_op->type));
    return xnn_status_invalid_parameter;
  }
  resize_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (input_width <= 1 || input_height <= 1) {
    xnn_log_error(
      "failed to setup %s operator with %zux%zu input: input dimensions must be greater than 1",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (max(input_width, input_height) >= 16777216) {
    xnn_log_error(
      "failed to setup %s operator with %zux%zu input: input dimensions must be below 2**24",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height);
    return xnn_status_unsupported_parameter;
  }

  if (output_width == 0 || output_height == 0) {
    xnn_log_error(
      "failed to setup %s operator with %zux%zu output: output dimensions must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), output_width, output_height);
    return xnn_status_invalid_parameter;
  }

  if (max(output_width, output_height) >= 16777216) {
    xnn_log_error(
      "failed to setup %s operator with %zux%zu output: output dimensions must be below 2**24",
      xnn_operator_type_to_string(expected_operator_type), output_width, output_height);
    return xnn_status_unsupported_parameter;
  }

  if (batch_size == 0) {
    resize_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

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

  const size_t input_pixel_stride_in_bytes = 1 << log2_element_size; // Since the layout in CHW the pixels
  if (input_height != resize_op->last_input_height ||
      input_width != resize_op->last_input_width ||
      output_height != resize_op->last_output_height ||
      output_width != resize_op->last_output_width)
  {
    const uint32_t flags = resize_op->flags;
    indirection_init(
        input_pixel_stride_in_bytes,
        input_height, input_width,
        output_height, output_width,
        input, resize_op->indirection_buffer, resize_op->packed_weights.pointer,
        !!(flags & XNN_FLAG_ALIGN_CORNERS),
        !!(flags & XNN_FLAG_TENSORFLOW_LEGACY_MODE));

    resize_op->last_input = input;
    resize_op->last_input_height = input_height;
    resize_op->last_input_width = input_width;
    resize_op->last_output_height = output_height;
    resize_op->last_output_width = output_width;
  }

  // Resize bilinear packed weights can change when the operator is resized, we will not use weights cache.
  assert(resize_op->weights_cache == NULL);
  resize_op->context.resize_bilinear_chw = (struct resize_bilinear_chw_context) {
    .output_pixels = output_height * output_width,
    .channels = resize_op->channels,
    .input_channel_stride = (input_height * input_width) << log2_element_size,
    .indirect_input = resize_op->indirection_buffer,
    .input_offset = (size_t) ((uintptr_t) input - (uintptr_t) resize_op->last_input),
    .input_batch_stride = (resize_op->input_pixel_stride * input_height * input_width) << log2_element_size,
    .packed_weights = resize_op->packed_weights.pointer,
    .output = output,
    .output_batch_stride = (resize_op->output_pixel_stride * output_height * output_width) << log2_element_size,
    .output_channel_stride = (output_height * output_width) << log2_element_size,
    .ukernel = ibilinear_chw->ukernel,
  };

  #if XNN_TEST_MODE
    const size_t output_channel_tile = ibilinear_chw->channel_tile;
  #else
    size_t output_channel_tile = resize_op->channels;
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
  resize_op->compute.type = xnn_parallelization_type_2d_tile_1d;
  resize_op->compute.task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_resize_bilinear_chw;
  resize_op->compute.range[0] = batch_size;
  resize_op->compute.range[1] = resize_op->channels;
  resize_op->compute.tile[0] = output_channel_tile;
  resize_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_resize_bilinear2d_nchw_f16(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t output_height,
    size_t output_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_resize_bilinear2d_nchw(
    resize_op,
    xnn_operator_type_resize_bilinear_nchw_f16,
    batch_size,
    input_height,
    input_width,
    output_height,
    output_width,
    input,
    output,
    1 /* log2(element size) == log2(sizeof(uint16_t)) */,
    1 /* log2(weight element size) == log2(sizeof(int16_t)) */,
    (xnn_indirection_init_resize_bilinear2d_chw_fn) xnn_indirection_init_resize_bilinear2d_chw_f16,
    &xnn_params.f16.ibilinear_chw,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_resize_bilinear2d_nchw_f32(
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
  return setup_resize_bilinear2d_nchw(
    resize_op,
    xnn_operator_type_resize_bilinear_nchw_f32,
    batch_size,
    input_height,
    input_width,
    output_height,
    output_width,
    input,
    output,
    2 /* log2(element size) == log2(sizeof(float)) */,
    2 /* log2(weight element size) == log2(sizeof(float)) */,
    (xnn_indirection_init_resize_bilinear2d_chw_fn) xnn_indirection_init_resize_bilinear2d_chw_f32,
    &xnn_params.f32.ibilinear_chw,
    pthreadpool_get_threads_count(threadpool));
}
