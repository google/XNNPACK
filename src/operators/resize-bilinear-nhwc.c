// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

static const struct xnn_ibilinear_config* get_ibilinear_nhwc_config(enum xnn_datatype datatype)
{
  switch (datatype) {
    case xnn_datatype_qint8:
      return xnn_init_s8_ibilinear_config();
    case xnn_datatype_quint8:
      return xnn_init_u8_ibilinear_config();
    case xnn_datatype_fp16:
      return xnn_init_f16_ibilinear_config();
    case xnn_datatype_fp32:
      return xnn_init_f32_ibilinear_config();
    default:
      return NULL;
  }
}

enum xnn_status xnn_create_resize_bilinear2d_nhwc(
    enum xnn_datatype datatype,
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_config* ibilinear_config = get_ibilinear_nhwc_config(datatype);
  if (ibilinear_config == NULL || ibilinear_config->ukernel == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc));
    return xnn_status_unsupported_hardware;
  }

  xnn_operator_t resize_op = NULL;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc));
    return xnn_status_uninitialized;
  }

  if (output_width == 0 || output_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu output: output dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc), output_width, output_height);
    return xnn_status_invalid_parameter;
  }

  if (max(output_width, output_height) >= 16777216) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu output: output dimensions must be below 2**24",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc), output_width, output_height);
    return xnn_status_unsupported_parameter;
  }

  resize_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (resize_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc));
    return xnn_status_out_of_memory;
  }

  resize_op->output_height = output_height;
  resize_op->output_width = output_width;

  resize_op->type = xnn_operator_type_resize_bilinear_nhwc;
  resize_op->flags = flags;
  resize_op->ibilinear_config = ibilinear_config;

  resize_op->state = xnn_run_state_invalid;

  *resize_op_out = resize_op;
  return xnn_status_success;
}

enum xnn_status xnn_reshape_resize_bilinear2d_nhwc(
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
  if (resize_op->type != xnn_operator_type_resize_bilinear_nhwc) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc),
        xnn_operator_type_to_string_v2(resize_op));
    return xnn_status_invalid_parameter;
  }
  resize_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string_v2(resize_op));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zux%zu input: input dimensions "
        "must be non-zero",
        xnn_operator_type_to_string_v2(resize_op), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (max(input_width, input_height) >= 16777216) {
    xnn_log_error(
        "failed to reshape %s operator with %zux%zu input: input dimensions "
        "must be below 2**24",
        xnn_operator_type_to_string_v2(resize_op), input_width, input_height);
    return xnn_status_unsupported_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc), input_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc), output_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    resize_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t log2_data_element_size = resize_op->ibilinear_config->log2_data_element_size;
  const size_t log2_weight_element_size = resize_op->ibilinear_config->log2_weight_element_size;
  const size_t output_height = resize_op->output_height;
  const size_t output_width = resize_op->output_width;
  const bool enable_transient_indirection = !!(resize_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER);
  const size_t input_pixel_stride_in_bytes = input_pixel_stride << log2_data_element_size;
  const size_t indirection_buffer_size = sizeof(void*) * (output_height * output_width * 4);
  const size_t packed_weights_size = (output_height * output_width * 2) << log2_weight_element_size;

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);

  size_t resize_bilinear_compute_index = 0;
  if (enable_transient_indirection) {
    // Round up to a multiple of pointer size
    const size_t indirect_input_offset = (packed_weights_size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
    *workspace_size = indirection_buffer_size + indirect_input_offset;
    *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;

    resize_bilinear_compute_index++;
    resize_op->context.resize_nhwc_indirection_init = (struct resize_bilinear_nhwc_indirection_init_context) {
      .input_pixel_stride = input_pixel_stride_in_bytes,
      .input_height = input_height, .input_width = input_width,
      .output_height = output_height, .output_width = output_width,
      .align_corners = !!(resize_op->flags & XNN_FLAG_ALIGN_CORNERS),
      .tensorflow_legacy_mode = !!(resize_op->flags & XNN_FLAG_TENSORFLOW_LEGACY_MODE),
      .indirection_init = resize_op->ibilinear_config->indirection_init,
      .indirect_input_offset = indirect_input_offset,
    };
    resize_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
    resize_op->compute[0].context_offset = offsetof(struct xnn_operator, context.resize_nhwc_indirection_init) - offsetof(struct xnn_operator, context);
    resize_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_resize_bilinear_indirection;
    resize_op->compute[0].range[0] = output_height;

    if (num_threads > 1) {
      const size_t target_tiles_per_thread = 5;
      resize_op->compute[0].tile[0] = divide_round_up(output_height, num_threads * target_tiles_per_thread);
    } else {
      resize_op->compute[0].tile[0] = output_height;
    }
  } else {
    *workspace_size = 0;
    *workspace_alignment = 1;

    if (output_height * output_width != resize_op->last_output_height * resize_op->last_output_width ||
        channels != resize_op->last_input_channels) {
      const void** indirection_buffer = (const void**) xnn_reallocate_memory(resize_op->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error(
            "failed to allocate %zu bytes for %s operator indirection buffer",
            indirection_buffer_size, xnn_operator_type_to_string_v2(resize_op));
        return xnn_status_out_of_memory;
      }
      resize_op->indirection_buffer = indirection_buffer;
      xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
                    indirection_buffer_size,
                    xnn_operator_type_to_string_v2(resize_op));

      // Note: packed weights must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(resize_op->packed_weights.pointer);
      resize_op->packed_weights.pointer = xnn_allocate_simd_memory(packed_weights_size);
      if (resize_op->packed_weights.pointer == NULL) {
        xnn_log_error(
            "failed to allocate %zu bytes for %s operator packed weights",
            packed_weights_size, xnn_operator_type_to_string_v2(resize_op));
        return xnn_status_out_of_memory;
      }
    }

    if (input_height != resize_op->last_input_height ||
        input_width != resize_op->last_input_width ||
        output_height != resize_op->last_output_height ||
        output_width != resize_op->last_output_width ||
        channels != resize_op->last_input_channels)
    {
      const uint32_t flags = resize_op->flags;
      // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
      void* dummy_input = (void*) XNN_ALLOCATION_ALIGNMENT;
      resize_op->ibilinear_config->indirection_init(
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
      resize_op->last_input_channels = channels;
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

  resize_op->compute[resize_bilinear_compute_index].type = xnn_parallelization_type_2d_tile_1d;
  resize_op->compute[resize_bilinear_compute_index].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_resize_bilinear;
  resize_op->compute[resize_bilinear_compute_index].range[0] = batch_size;
  resize_op->compute[resize_bilinear_compute_index].range[1] = output_size;
  resize_op->compute[resize_bilinear_compute_index].tile[0] = output_size_tile;
  resize_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_setup_resize_bilinear2d_nhwc(
    xnn_operator_t resize_op,
    void* workspace,
    const void* input,
    void* output)
{
  if (resize_op->type != xnn_operator_type_resize_bilinear_nhwc) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nhwc),
        xnn_operator_type_to_string_v2(resize_op));
    return xnn_status_invalid_parameter;
  }

  switch (resize_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(resize_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  const size_t log2_weight_element_size = resize_op->ibilinear_config->log2_weight_element_size;
  const size_t output_height = resize_op->context.resize_nhwc_indirection_init.output_height;
  const size_t output_width = resize_op->context.resize_nhwc_indirection_init.output_width;
  const size_t packed_weights_size = (output_height * output_width * 2) << log2_weight_element_size;
  if (resize_op->flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    // indirect_input should start at a multiple of pointer size to avoid ubsan failures
    const size_t indirect_input_offset = (packed_weights_size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
    resize_op->context.resize_bilinear.packed_weights = (const void*) workspace;
    resize_op->context.resize_bilinear.indirect_input = (const void**) ((uintptr_t) workspace + indirect_input_offset);
    resize_op->context.resize_nhwc_indirection_init.buffer = (const void**) workspace;
    resize_op->context.resize_nhwc_indirection_init.input = input;
  } else {
    resize_op->context.resize_bilinear.input_offset = (size_t) ((uintptr_t) input - (uintptr_t) resize_op->last_input);
  }
  resize_op->context.resize_bilinear.output = output;

  resize_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
