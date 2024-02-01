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
#include <xnnpack/log.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/indirection.h>


static enum xnn_status create_resize_bilinear2d_nhwc(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct xnn_ibilinear_config* ibilinear_config,
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
      xnn_operator_type_to_string(resize_op->type), output_width, output_height);
    return xnn_status_invalid_parameter;
  }

  if (max(output_width, output_height) >= 16777216) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu output: output dimensions must be below 2**24",
      xnn_operator_type_to_string(resize_op->type), output_width, output_height);
    return xnn_status_unsupported_parameter;
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
  resize_op->ibilinear_config = ibilinear_config;

  resize_op->state = xnn_run_state_invalid;

  *resize_op_out = resize_op;
  return xnn_status_success;

error:
  xnn_delete_operator(resize_op);
  return status;
}

enum xnn_status xnn_create_resize_bilinear2d_nhwc_f16(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_config* ibilinear_config = xnn_init_f16_ibilinear_config();
  if (ibilinear_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc_f16));
    return xnn_status_unsupported_hardware;
  }

  return create_resize_bilinear2d_nhwc(
    output_height,
    output_width,
    flags,
    xnn_operator_type_resize_bilinear_nhwc_f16,
    ibilinear_config,
    resize_op_out);
}

enum xnn_status xnn_create_resize_bilinear2d_nhwc_f32(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_config* ibilinear_config = xnn_init_f32_ibilinear_config();
  if (ibilinear_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  return create_resize_bilinear2d_nhwc(
    output_height,
    output_width,
    flags,
    xnn_operator_type_resize_bilinear_nhwc_f32,
    ibilinear_config,
    resize_op_out);
}

enum xnn_status xnn_create_resize_bilinear2d_nhwc_s8(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_config* ibilinear_config = xnn_init_s8_ibilinear_config();
  assert(ibilinear_config != NULL);

  return create_resize_bilinear2d_nhwc(
    output_height,
    output_width,
    flags,
    xnn_operator_type_resize_bilinear_nhwc_s8,
    ibilinear_config,
    resize_op_out);
}

enum xnn_status xnn_create_resize_bilinear2d_nhwc_u8(
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_config* ibilinear_config = xnn_init_u8_ibilinear_config();
  assert(ibilinear_config != NULL);

  return create_resize_bilinear2d_nhwc(
    output_height,
    output_width,
    flags,
    xnn_operator_type_resize_bilinear_nhwc_u8,
    ibilinear_config,
    resize_op_out);
}

static enum xnn_status reshape_resize_bilinear2d_nhwc(
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
    xnn_indirection_init_resize_bilinear2d_hwc_fn indirection_init,
    size_t* workspace_size,
    size_t* workspace_alignment,
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
      xnn_operator_type_to_string(resize_op->type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(resize_op->type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (max(input_width, input_height) >= 16777216) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be below 2**24",
      xnn_operator_type_to_string(resize_op->type), input_width, input_height);
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
  const bool enable_transient_indirection = !!(resize_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER);
  const size_t input_pixel_stride_in_bytes = input_pixel_stride << log2_data_element_size;
  const size_t indirection_buffer_size = sizeof(void*) * (output_height * output_width * 4);
  const size_t packed_weights_size = (output_height * output_width * 2) << log2_weight_element_size;

  #if !XNN_TEST_MODE
    const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  #endif
  size_t resize_bilinear_compute_index = 0;
  if (enable_transient_indirection) {
    *workspace_size = indirection_buffer_size + packed_weights_size;
    *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;

    resize_bilinear_compute_index++;
    resize_op->context.resize_nhwc_indirection_init = (struct resize_bilinear_nhwc_indirection_init_context) {
      .input_pixel_stride = input_pixel_stride_in_bytes,
      .input_height = input_height, .input_width = input_width,
      .output_height = output_height, .output_width = output_width,
      .align_corners = !!(resize_op->flags & XNN_FLAG_ALIGN_CORNERS),
      .tensorflow_legacy_mode = !!(resize_op->flags & XNN_FLAG_TENSORFLOW_LEGACY_MODE),
      .indirection_init = indirection_init,
      .packed_weight_size = packed_weights_size,
    };
    resize_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
    resize_op->compute[0].context_offset = offsetof(struct xnn_operator, context.resize_nhwc_indirection_init) - offsetof(struct xnn_operator, context);
    resize_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_resize_bilinear_indirection;
    resize_op->compute[0].range[0] = output_height;
    #if XNN_TEST_MODE
      resize_op->compute[0].tile[0] = 1;
    #else
      if (num_threads > 1) {
        const size_t target_tiles_per_thread = 5;
        resize_op->compute[0].tile[0] = divide_round_up(output_height, num_threads * target_tiles_per_thread);
      } else {
        resize_op->compute[0].tile[0] = output_height;
      }
    #endif
  } else {
    *workspace_size = 0;
    *workspace_alignment = 1;

    if (output_height * output_width != resize_op->last_output_height * resize_op->last_output_width) {
      const void** indirection_buffer = (const void**) xnn_reallocate_memory(resize_op->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator indirection buffer",
          indirection_buffer_size, xnn_operator_type_to_string(resize_op->type));
        return xnn_status_out_of_memory;
      }
      resize_op->indirection_buffer = indirection_buffer;
      xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
        indirection_buffer_size, xnn_operator_type_to_string(resize_op->type));

      // Note: packed weights must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(resize_op->packed_weights.pointer);
      resize_op->packed_weights.pointer = xnn_allocate_simd_memory(packed_weights_size);
      if (resize_op->packed_weights.pointer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_weights_size, xnn_operator_type_to_string(resize_op->type));
        return xnn_status_out_of_memory;
      }
    }

    if (input_height != resize_op->last_input_height ||
        input_width != resize_op->last_input_width ||
        output_height != resize_op->last_output_height ||
        output_width != resize_op->last_output_width)
    {
      const uint32_t flags = resize_op->flags;
      // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
      void* dummy_input = (void*) XNN_ALLOCATION_ALIGNMENT;
      indirection_init(
        /*output_y_start=*/0, /*output_y_end=*/output_height,
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
  }

  const struct xnn_ibilinear_config* ibilinear = resize_op->ibilinear_config;
  const size_t output_pixel_stride_in_bytes = output_pixel_stride << log2_data_element_size;
  // Resize bilinear packed weights can change when the operator is resized, we will not use weights cache.
  assert(resize_op->weights_cache == NULL);
  resize_op->context.resize_bilinear = (struct resize_bilinear_context) {
    .scaled_channels = channels << log2_data_element_size,
    .indirect_input = resize_op->indirection_buffer,
    .input_batch_stride = input_pixel_stride_in_bytes * input_height * input_width,
    .packed_weights = resize_op->packed_weights.pointer,
    .output_pixel_stride = output_pixel_stride_in_bytes,
    .output_batch_stride = output_pixel_stride_in_bytes * output_height * output_width,
    .log2_wsize = 1 + log2_weight_element_size /* log2(2 * sizeof(weight)) */,
    .input_offset = (size_t) 0,
    .ukernel = ibilinear->ukernel,
  };

  const size_t output_size = output_height * output_width;
  #if XNN_TEST_MODE
    const size_t output_size_tile = ibilinear->pixel_tile;
  #else
    size_t output_size_tile = output_size;
    if (num_threads > 1) {
      const size_t target_tiles_per_thread = 5;
      const size_t max_output_size_tile = divide_round_up(output_size, num_threads * target_tiles_per_thread);
      if (max_output_size_tile < output_size_tile) {
        const uint32_t output_size_subtile = ibilinear->pixel_tile;
        output_size_tile =
          min(output_size_tile,
            divide_round_up(output_size_tile, max_output_size_tile * output_size_subtile) * output_size_subtile);
      }
    }
  #endif
  resize_op->compute[resize_bilinear_compute_index].type = xnn_parallelization_type_2d_tile_1d;
  resize_op->compute[resize_bilinear_compute_index].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_resize_bilinear;
  resize_op->compute[resize_bilinear_compute_index].range[0] = batch_size;
  resize_op->compute[resize_bilinear_compute_index].range[1] = output_size;
  resize_op->compute[resize_bilinear_compute_index].tile[0] = output_size_tile;
  resize_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_resize_bilinear2d_nhwc_f16(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_f16,
    batch_size,
    input_height,
    input_width,
    channels,
    input_pixel_stride,
    output_pixel_stride,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_HALF,
    (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_f16,
    workspace_size, workspace_alignment,
    threadpool);
}

enum xnn_status xnn_reshape_resize_bilinear2d_nhwc_f32(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_f32,
    batch_size,
    input_height,
    input_width,
    channels,
    input_pixel_stride,
    output_pixel_stride,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_f32,
    workspace_size, workspace_alignment,
    threadpool);
}

enum xnn_status xnn_reshape_resize_bilinear2d_nhwc_s8(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_s8,
    batch_size,
    input_height,
    input_width,
    channels,
    input_pixel_stride,
    output_pixel_stride,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_INT16_T,
    (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_q11,
    workspace_size, workspace_alignment,
    threadpool);
}

enum xnn_status xnn_reshape_resize_bilinear2d_nhwc_u8(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_u8,
    batch_size,
    input_height,
    input_width,
    channels,
    input_pixel_stride,
    output_pixel_stride,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_INT16_T,
    (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_q11,
    workspace_size, workspace_alignment,
    threadpool);
}

static enum xnn_status setup_resize_bilinear2d_nhwc(
    xnn_operator_t resize_op,
    enum xnn_operator_type expected_operator_type,
    void* workspace,
    const void* input,
    void* output,
    uint32_t log2_weight_element_size)
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

  const size_t output_height = resize_op->context.resize_nhwc_indirection_init.output_height;
  const size_t output_width = resize_op->context.resize_nhwc_indirection_init.output_width;
  const size_t packed_weights_size = (output_height * output_width * 2) << log2_weight_element_size;
  if (resize_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    resize_op->context.resize_bilinear.packed_weights = (const void*) workspace;
    resize_op->context.resize_bilinear.indirect_input = (const void**) ((uintptr_t) workspace + packed_weights_size);
    resize_op->context.resize_nhwc_indirection_init.buffer = (const void**) workspace;
    resize_op->context.resize_nhwc_indirection_init.input = input;
  } else {
    resize_op->context.resize_bilinear.input_offset = (size_t) ((uintptr_t) input - (uintptr_t) resize_op->last_input);
  }
  resize_op->context.resize_bilinear.output = output;

  resize_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_f16(
    xnn_operator_t resize_op,
    void* workspace,
    const void* input,
    void* output)
{
  return setup_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_f16,
    workspace,
    input,
    output,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_HALF);
}

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_f32(
    xnn_operator_t resize_op,
    void* workspace,
    const float* input,
    float* output)
{
  return setup_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_f32,
    workspace,
    input,
    output,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_FLOAT);
}

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_s8(
    xnn_operator_t resize_op,
    void* workspace,
    const int8_t* input,
    int8_t* output)
{
  return setup_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_s8,
    workspace,
    input,
    output,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_UINT16_T);
}

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_u8(
    xnn_operator_t resize_op,
    void* workspace,
    const uint8_t* input,
    uint8_t* output)
{
  return setup_resize_bilinear2d_nhwc(
    resize_op,
    xnn_operator_type_resize_bilinear_nhwc_u8,
    workspace,
    input,
    output,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_UINT16_T);
}
