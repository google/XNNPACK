// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/normalization.h>
#include <xnnpack/operator.h>

/// Reorder the data in array using the indices in loop_order.
///
/// Changing the loop order can have dramatic performance implications.
static void reorder_array(
    size_t num_dims,
    const size_t loop_order[restrict XNN_MIN_ELEMENTS(1) ],
    size_t array[restrict XNN_MIN_ELEMENTS(1)])
{
  size_t tmp[XNN_MAX_TENSOR_DIMS];
  memcpy(tmp, array, sizeof(size_t) * num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    array[i] = tmp[loop_order[i]];
  }
}

static void init_transpose_nd(
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t transpose_op)
{
  transpose_op->flags = flags;
  transpose_op->type = operator_type;
}

static enum xnn_status create_transpose_nd(
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* transpose_op_out)
{
  xnn_operator_t transpose_op = NULL;

  enum xnn_status status = xnn_status_uninitialized;
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;
  transpose_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (transpose_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  init_transpose_nd(flags, operator_type, transpose_op);
  *transpose_op_out = transpose_op;

  return xnn_status_success;

error:
  xnn_delete_operator(transpose_op);
  return status;
}

/// input_stride and output_stride are the number of elements between each
/// dimension, not the size of the dimension. This is because depth to space
/// splits the input channel dimension into three dimensions - block_size *
/// block_size * output_channels but gives input_channel_stride the stride over
/// all three dimensions. This must be multiplied by the product of the previous
/// dimensions to get the stride in elements. input_channel_stride is not
/// requried to be a multiple of block_size * block_size * output_channels so
/// the stride in number of elements must be supplied.
/// An interface for sub-tensors can easily be built on top of this.
static enum xnn_status setup_transpose_nd(
  xnn_operator_t transpose_op,
  const void* input,
  void* output,
  const size_t num_dims,
  const size_t* input_shape,
  const size_t* perm,
  const size_t* input_stride,
  const size_t* output_stride,
  size_t element_size,
  const bool prevent_perm_normalization)
{
  transpose_op->state = xnn_run_state_invalid;
  enum xnn_status status = xnn_status_invalid_parameter;
  if (num_dims == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu num_dims: num_dims must be non-zero",
      xnn_operator_type_to_string(transpose_op->type), num_dims);
    goto error;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create %s operator with %zu num_dims: num_dims must be <= %d",
      xnn_operator_type_to_string(transpose_op->type), num_dims, XNN_MAX_TENSOR_DIMS);
    goto error;
  }

  for (size_t i = 0; i < num_dims; ++i) {
    if (perm[i] >= num_dims) {
      xnn_log_error(
          "failed to create %s operator with %zu perm and %zu num_dims: 0 <= perm < num_dims",
          xnn_operator_type_to_string(transpose_op->type), perm[i], num_dims);
      goto error;
    }
  }

  for (size_t i = 0; i < num_dims - 1; ++i) {
    for (size_t j = i + 1; j < num_dims; ++j) {
      if (perm[i] == perm[j]) {
        xnn_log_error(
            "failed to create %s operator with duplicate entries in perm",
            xnn_operator_type_to_string(transpose_op->type));
        goto error;
      }
    }
  }

  if (input_stride != NULL) {
    if (input_stride[num_dims - 1] != 1) {
      xnn_log_error(
          "failed to create %s operator with %zu input_stride[num_dims - 1]: input_stride[num_dims - 1] == 1",
          xnn_operator_type_to_string(transpose_op->type), input_stride[num_dims - 1]);
      goto error;
    }
    size_t current_stride = 1;
    for (size_t i = num_dims - 1; i > 0; --i) {
      if ((input_stride[i - 1] < input_stride[i] * input_shape[i]) || (input_stride[i - 1] < current_stride)) {
        xnn_log_error(
            "failed to create %s operator with %zu input_shape and %zu input_stride: input_stride >= input_shape",
            xnn_operator_type_to_string(transpose_op->type), input_shape[i], input_stride[i]);
        goto error;
      }
      current_stride *= input_shape[i];
    }
  }

  if (output_stride != NULL) {
    if (output_stride[num_dims - 1] != 1) {
      xnn_log_error(
          "failed to create %s operator with %zu output_stride[num_dims - 1]: output_stride[num_dims - 1] == 1",
          xnn_operator_type_to_string(transpose_op->type), output_stride[num_dims - 1]);
      goto error;
    }
    size_t current_stride = 1;
    for (size_t i = num_dims - 1; i > 0; --i) {
      if ((output_stride[i - 1] < output_stride[i] * input_shape[perm[i]]) || (output_stride[i - 1] < current_stride)) {
        xnn_log_error(
            "failed to create %s operator with %zu output_shape and %zu output_stride: output_stride >= output_shape",
            xnn_operator_type_to_string(transpose_op->type), input_shape[perm[i]], output_stride[i]);
        goto error;
      }
      current_stride *= input_shape[perm[i]];
    }
  }

  // Early exit without setting up context if any shape dimension is zero.
  bool degenerate_shape = false;
  for (size_t i = 0; i < num_dims; ++i) {
    degenerate_shape |= input_shape[i] == 0;
  }

  if (degenerate_shape) {
    transpose_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  transpose_op->channels = num_dims;

  struct transpose_context* context = &transpose_op->context.transpose;
  size_t normalized_dims;
  size_t normalized_shape[XNN_MAX_TENSOR_DIMS];
  size_t normalized_perm[XNN_MAX_TENSOR_DIMS];
  size_t normalized_element_size;
  if (!prevent_perm_normalization) {
    xnn_normalize_transpose_permutation(num_dims, element_size, perm, input_shape, input_stride, output_stride,
                                        &normalized_dims, &normalized_element_size, normalized_perm, normalized_shape,
                                        context->input_stride, context->output_stride);
  } else {
    normalized_dims = num_dims;
    memcpy(normalized_shape, input_shape, num_dims * sizeof(size_t));
    memcpy(normalized_perm, perm, num_dims * sizeof(size_t));
    normalized_element_size = element_size;
    context->input_stride[normalized_dims - 1] = normalized_element_size;
    context->output_stride[normalized_dims - 1] = normalized_element_size;
    for (size_t i = normalized_dims - 1; i > 0; --i) {
      context->input_stride[i - 1] = context->input_stride[i] * normalized_shape[i];
      context->output_stride[i - 1] = context->output_stride[i] * normalized_shape[normalized_perm[i]];
    }
  }

  size_t loop_order[XNN_MAX_TENSOR_DIMS];
  memcpy(loop_order, normalized_perm, sizeof(size_t) * normalized_dims);

  /// The innermost loop must iterate over the contiguous input dimension and the second most inner loop over the
  /// contiguous output dimension.
  if (normalized_dims > 1) {
    for (size_t i = 0; i < normalized_dims - 2; ++i) {
      if (loop_order[i] == normalized_dims - 1) {
        size_t tmp = loop_order[i];
        loop_order[i] = loop_order[normalized_dims - 2];
        loop_order[normalized_dims - 2] = tmp;
        tmp = context->output_stride[i];
        context->output_stride[i] = context->output_stride[normalized_dims - 2];
        context->output_stride[normalized_dims - 2] = tmp;
        break;
      }
    }
  }

  for (size_t i = 0; i < normalized_dims; ++i) {
    transpose_op->compute.range[i] = normalized_shape[i];
  }
  reorder_array(normalized_dims, loop_order, context->input_stride);
  reorder_array(normalized_dims, loop_order, transpose_op->compute.range);

  const struct xnn_transpose_config* transpose_config = xnn_init_transpose_config();
  assert(transpose_config != NULL);

  bool variable_size_ukernel = false;
  switch (normalized_element_size) {
    case 1:
      context->const_size_ukernel = transpose_config->x8.const_size_ukernel;
      transpose_op->compute.tile[0] = transpose_config->x8.tile_size;
      transpose_op->compute.tile[1] = transpose_config->x8.tile_size;
      if (transpose_config->x8.init.x16 != NULL) {
        transpose_config->x8.init.x8(&context->params.x8_params);
      }
      break;
    case 2:
      transpose_op->compute.tile[0] = transpose_config->x16.tile_size;
      transpose_op->compute.tile[1] = transpose_config->x16.tile_size;
      context->const_size_ukernel = transpose_config->x16.const_size_ukernel;
      if (transpose_config->x16.init.x16 != NULL) {
        transpose_config->x16.init.x16(&context->params.x16_params);
      }
      break;
    case 3:
      transpose_op->compute.tile[0] = transpose_config->x24.tile_size;
      transpose_op->compute.tile[1] = transpose_config->x24.tile_size;
      context->const_size_ukernel = transpose_config->x24.const_size_ukernel;
      if (transpose_config->x24.init.x24 != NULL) {
        transpose_config->x24.init.x24(&context->params.x24_params);
      }
      break;
    case 4:
      transpose_op->compute.tile[0] = transpose_config->x32.tile_size;
      transpose_op->compute.tile[1] = transpose_config->x32.tile_size;
      context->const_size_ukernel = transpose_config->x32.const_size_ukernel;
      if (transpose_config->x32.init.x32 != NULL) {
        transpose_config->x32.init.x32(&context->params.x32_params);
      }
      break;
    default:
      transpose_op->compute.tile[0] = transpose_config->xx.tile_size;
      transpose_op->compute.tile[1] = transpose_config->xx.tile_size;
      context->variable_size_ukernel = transpose_config->xx.variable_size_ukernel;
      variable_size_ukernel = true;
  }

  struct univector_contiguous_context* univector_context = &transpose_op->context.univector_contiguous;
  switch (normalized_dims) {
    case 1:
      transpose_op->compute.type = xnn_parallelization_type_1d_tile_1d;
      transpose_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
      transpose_op->compute.range[0] = normalized_element_size;
      transpose_op->compute.tile[0] = normalized_element_size;
      univector_context->ukernel = transpose_config->copy;
      univector_context->log2_xsize = 0;
      univector_context->log2_ysize = 0;
      break;
    case 2:
      transpose_op->compute.type = xnn_parallelization_type_2d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_transposev_2d;
      } else {
        transpose_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_transposec_2d;
      }
      break;
    case 3:
      transpose_op->compute.type = xnn_parallelization_type_3d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_transposev_3d;
      } else {
        transpose_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_transposec_3d;
      }
      break;
    case 4:
      transpose_op->compute.type = xnn_parallelization_type_4d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_transposev_4d;
      } else {
        transpose_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_transposec_4d;
      }
      break;
    case 5:
      transpose_op->compute.type = xnn_parallelization_type_5d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute.task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_transposev_5d;
      } else {
        transpose_op->compute.task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_transposec_5d;
      }
      break;
    case 6:
      transpose_op->compute.type = xnn_parallelization_type_6d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute.task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_transposev_6d;
      } else {
        transpose_op->compute.task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_transposec_6d;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }

  context->variable_ukernel = variable_size_ukernel;
  if (transpose_op->channels == 1) {
    transpose_op->context.univector_contiguous.x = input;
    transpose_op->context.univector_contiguous.y = output;
  } else {
    transpose_op->context.transpose.x = input;
    transpose_op->context.transpose.y = output;
  }
  transpose_op->state = xnn_run_state_ready;

  return xnn_status_success;

error:
  xnn_delete_operator(transpose_op);
  return status;
}

enum xnn_status xnn_create_transpose_nd_x32(
  uint32_t flags,
  xnn_operator_t* transpose_op_out)
{
  return create_transpose_nd(
    flags,
    xnn_operator_type_transpose_nd_x32,
    transpose_op_out);
}

enum xnn_status xnn_create_transpose_nd_x16(
  uint32_t flags,
  xnn_operator_t* transpose_op_out)
{
  return create_transpose_nd(
    flags,
    xnn_operator_type_transpose_nd_x16,
    transpose_op_out);
}

enum xnn_status xnn_create_transpose_nd_x8(
  uint32_t flags,
  xnn_operator_t* transpose_op_out)
{
  return create_transpose_nd(
    flags,
    xnn_operator_type_transpose_nd_x8,
    transpose_op_out);
}

enum xnn_status xnn_setup_transpose_nd_x32(
    xnn_operator_t transpose_op,
    const void* input,
    void* output,
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x32),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(
    transpose_op,
    input, output,
    num_dims, shape, perm, NULL, NULL,
    sizeof(uint32_t), false);
}

enum xnn_status xnn_setup_transpose_nd_x16(
    xnn_operator_t transpose_op,
    const void* input,
    void* output,
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x16),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(
    transpose_op,
    input, output,
    num_dims, shape, perm, NULL, NULL,
    sizeof(uint16_t), false);
}

enum xnn_status xnn_setup_transpose_nd_x8(
    xnn_operator_t transpose_op,
    const void* input,
    void* output,
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x8),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(
    transpose_op,
    input, output,
    num_dims, shape, perm, NULL, NULL,
    sizeof(uint8_t), false);
}

enum xnn_status run_transpose_nd(
    uint32_t flags,
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    size_t element_size,
    enum xnn_operator_type operator_type,
    pthreadpool_t threadpool) {

  struct xnn_operator transpose_op;
  memset(&transpose_op, 0, sizeof(transpose_op));
  init_transpose_nd(flags, operator_type, &transpose_op);

  const enum xnn_status status = setup_transpose_nd(
    &transpose_op,
    input, output,
    num_dims, input_shape, output_perm, NULL, NULL,
    element_size, false);
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_run_operator(&transpose_op, threadpool);
}

enum xnn_status xnn_run_transpose_nd_x8(
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  return run_transpose_nd(
    flags,
    input, output,
    num_dims, input_shape, output_perm,
    sizeof(uint8_t), xnn_operator_type_transpose_nd_x8,
    threadpool);
}

enum xnn_status xnn_run_transpose_nd_x16(
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  return run_transpose_nd(
    flags,
    input, output,
    num_dims, input_shape, output_perm,
    sizeof(uint16_t), xnn_operator_type_transpose_nd_x16,
    threadpool);
}

enum xnn_status xnn_run_transpose_nd_x32(
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  return run_transpose_nd(
    flags,
    input, output,
    num_dims, input_shape, output_perm,
    sizeof(uint32_t), xnn_operator_type_transpose_nd_x32,
    threadpool);
}

enum xnn_status create_depth_to_space_nchw2nhwc(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* depth_to_space_op_out)
{
  xnn_operator_t depth_to_space_op = NULL;

  enum xnn_status status = xnn_status_uninitialized;
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;
  if (output_channels == 0) {
    xnn_log_error("failed to create %s operator with %zu output channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), output_channels);
    goto error;
  }

  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu)",
      xnn_operator_type_to_string(operator_type),
      output_channel_stride, output_channels);
    goto error;
  }

  if (block_size <= 1) {
    xnn_log_error("failed to create %s operator with %" PRIu32 " block size: block size must be greater than 1",
      xnn_operator_type_to_string(operator_type),
      block_size);
    goto error;
  }

  const size_t input_channels = output_channels * block_size * block_size;
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      input_channel_stride, block_size, block_size, input_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  depth_to_space_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (depth_to_space_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  depth_to_space_op->channels = output_channels;
  depth_to_space_op->input_pixel_stride = input_channel_stride;
  depth_to_space_op->output_pixel_stride = output_channel_stride;
  depth_to_space_op->block_size = block_size;

  depth_to_space_op->type = operator_type;
  depth_to_space_op->flags = flags;

  depth_to_space_op->state = xnn_run_state_invalid;

  *depth_to_space_op_out = depth_to_space_op;
  return xnn_status_success;

error:
  xnn_delete_operator(depth_to_space_op);
  return status;
}

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x16(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nchw2nhwc(
    output_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nchw2nhwc_x16,
    depth_to_space_op_out);
}

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x32(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nchw2nhwc(
    output_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nchw2nhwc_x32,
    depth_to_space_op_out);
}

enum xnn_status setup_depth_to_space_nchw2nhwc(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    enum xnn_operator_type operator_type,
    size_t element_size)
{
  depth_to_space_op->state = xnn_run_state_invalid;

  if (input_width == 0 || input_height == 0) {
    xnn_log_error("failed to setup %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    depth_to_space_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const uint32_t block_size = depth_to_space_op->block_size;
  const size_t channels = depth_to_space_op->channels;

  const size_t input_shape[6] = {batch_size, block_size, block_size, channels, input_height, input_width};
  const size_t perm[6] = {0, 4, 1, 5, 2, 3};
  const size_t area = input_height * input_width;
  const size_t elements_per_batch = area * channels;
  const size_t input_stride[6] = {
    depth_to_space_op->input_pixel_stride * area,
    block_size * elements_per_batch,
    elements_per_batch,
    area,
    input_width,
    1
  };
  const size_t output_stride[6] = {
    input_height * block_size * input_width * block_size * depth_to_space_op->output_pixel_stride,
    block_size * input_width * block_size * depth_to_space_op->output_pixel_stride,
    input_width * block_size * depth_to_space_op->output_pixel_stride,
    block_size * depth_to_space_op->output_pixel_stride,
    depth_to_space_op->output_pixel_stride,
    1
  };

  return setup_transpose_nd(
    depth_to_space_op,
    input, output,
    6, input_shape, perm, input_stride, output_stride,
    element_size, false);
}

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x16(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (depth_to_space_op->type != xnn_operator_type_depth_to_space_nchw2nhwc_x16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x16),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_depth_to_space_nchw2nhwc(
    depth_to_space_op,
    batch_size, input_height, input_width,
    input, output,
    xnn_operator_type_depth_to_space_nchw2nhwc_x16, /*element_size=*/2);
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

  return setup_depth_to_space_nchw2nhwc(
    depth_to_space_op,
    batch_size, input_height, input_width,
    input, output,
    xnn_operator_type_depth_to_space_nchw2nhwc_x32, /*element_size=*/4);
}


enum xnn_status xnn_create_batch_to_space_nhwc(
  xnn_operator_t* const batch_to_space_op_out,
  const enum xnn_operator_type operator_type)
{
  xnn_operator_t batch_to_space_op = NULL;
  enum xnn_status status = xnn_status_out_of_memory;

  batch_to_space_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (batch_to_space_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  batch_to_space_op->state = xnn_run_state_invalid;
  batch_to_space_op->type = operator_type;

  *batch_to_space_op_out = batch_to_space_op;
  return xnn_status_success;

error:
  xnn_delete_operator(batch_to_space_op);
  return status;
}

enum xnn_status xnn_create_batch_to_space_nhwc_x8(xnn_operator_t* batch_to_space_op_out)
{
  return xnn_create_batch_to_space_nhwc(
      batch_to_space_op_out,
      xnn_operator_type_batch_to_space_nhwc_x8);
}

enum xnn_status xnn_create_batch_to_space_nhwc_x16(xnn_operator_t* batch_to_space_op_out)
{
  return xnn_create_batch_to_space_nhwc(
      batch_to_space_op_out,
      xnn_operator_type_batch_to_space_nhwc_x16);
}

enum xnn_status xnn_create_batch_to_space_nhwc_x32(xnn_operator_t* batch_to_space_op_out)
{
  return xnn_create_batch_to_space_nhwc(
      batch_to_space_op_out,
      xnn_operator_type_batch_to_space_nhwc_x32);
}

void setup_batch_to_space_identity(xnn_operator_t op)
{
  op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_batch_to_space_v1d;
  struct batch_to_space_context* const context = &op->context.batch_to_space;
  const size_t output_batch = context->input_batch / context->block_height / context->block_width;
  const size_t input_row_stride = context->output_width_nocrop * context->element_size;
  context->identity = (struct batch_to_space_context_identity){
    .input_batch_stride
        = context->output_height_nocrop * context->output_width_nocrop * context->element_size,
    .input_row_stride = input_row_stride,
    .output_row_stride = context->output_width * context->element_size,
    .output_batch = output_batch,
    .initial_input_offset = context->crop_top * input_row_stride + context->crop_left * context->element_size,
  };
}

void setup_batch_to_space_tile_2d_h(
    xnn_operator_t op,
    size_t vertical_crop,
    size_t horizontal_crop)
{
  struct batch_to_space_context* const context = &op->context.batch_to_space;
  if (context->transpose.variable_ukernel) {
    op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_batch_to_space_v2dh;
  } else {
    op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_batch_to_space_c2dh;
  }
  const size_t tile_element_size = context->transpose.input_stride[0];
  context->tile_2d = (struct batch_to_space_context_tile_2d) {
    .left_crop_col = context->crop_left,
    .right_crop_col = context->output_width_nocrop - context->crop_right,
    .top_crop_row = context->crop_top,
    .bottom_crop_row = context->output_height_nocrop - context->crop_bottom,
    .crop_top_bytes = context->crop_top * context->output_width * context->element_size,
    .crop_left_bytes = context->crop_left * context->element_size,
    .vertical_crop = vertical_crop,
    .vertical_crop_bytes = vertical_crop * context->output_width * context->element_size,
    .horizontal_crop = horizontal_crop,
    .horizontal_crop_bytes = horizontal_crop * context->element_size,
    .ld_input = context->transpose.input_stride[1],
    .ld_output = context->transpose.output_stride[0],
    .input_element_stride = tile_element_size,
    .output_element_stride = tile_element_size,
    .element_size = tile_element_size,
  };
}

void setup_batch_to_space_tile_2d_v(
    xnn_operator_t op,
    const struct xnn_transpose_config* const xnn_transpose_conf,
    size_t vertical_crop,
    size_t horizontal_crop)
{
  // Because cropping affects the elements size in this case, we force the use of the variable size ukernel.
  op->compute.task_2d_tile_2d
      = (pthreadpool_task_2d_tile_2d_t) xnn_compute_batch_to_space_2dv;
  op->context.transpose.variable_size_ukernel
      = xnn_transpose_conf->xx.variable_size_ukernel;
  struct batch_to_space_context* const context = &op->context.batch_to_space;
  const size_t tile_element_size = context->transpose.input_stride[0];
  const size_t output_element_stride = context->transpose.output_stride[1]
            - horizontal_crop * context->element_size;
  context->tile_2d = (struct batch_to_space_context_tile_2d) {
    .top_crop_row = context->crop_top,
    .bottom_crop_row = context->output_height_nocrop - context->crop_bottom,
    .crop_top_bytes = context->crop_top * tile_element_size,
    .vertical_crop = vertical_crop,
    .vertical_crop_bytes = vertical_crop * tile_element_size,
    .horizontal_crop = horizontal_crop,
    .horizontal_crop_bytes = horizontal_crop * context->element_size,
    .base_input_offset = context->crop_left * context->element_size,
    .output_batch_stride_compensation =
        vertical_crop * horizontal_crop * context->element_size
        - vertical_crop * tile_element_size,
    .ld_input = context->transpose.input_stride[1],
    .ld_output = context->transpose.output_stride[0]
        - horizontal_crop * context->element_size * context->block_height,
    .input_element_stride = context->transpose.input_stride[0],
    .output_element_stride = output_element_stride,
    .element_size = output_element_stride,
  };
}

void setup_batch_to_space_tile_4d(
    xnn_operator_t op,
    size_t vertical_crop,
    size_t horizontal_crop)
{
  struct batch_to_space_context* const context = &op->context.batch_to_space;
  if (context->transpose.variable_ukernel) {
    op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_batch_to_space_v4d;
  } else {
    op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_batch_to_space_c4d;
  }
  context->tile_4d =  (struct batch_to_space_context_tile_4d) {
    .top_crop_row = context->crop_top,
    .bottom_crop_row = context->output_height_nocrop - context->crop_bottom,
    .k_output_stride = context->transpose.output_stride[2] / context->element_size,
    .crop_left_bytes = context->crop_left * context->element_size,
    .crop_top_bytes = context->output_width * context->crop_top * context->element_size,
    .vertical_crop_bytes = context->output_width * vertical_crop * context->element_size,
    .horizontal_crop_bytes = horizontal_crop * context->element_size,
    .ld_input = context->transpose.input_stride[3],
    .ld_output = context->transpose.output_stride[2],
  };
}

enum xnn_status xnn_setup_batch_to_space_nhwc(
  const size_t type_size,
  xnn_operator_t batch_to_space_op,
  size_t input_batch,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t block_height,
  size_t block_width,
  size_t crop_top,
  size_t crop_bottom,
  size_t crop_left,
  size_t crop_right,
  const void* input,
  void* output)
{
  enum xnn_status status = xnn_status_unsupported_hardware;
  batch_to_space_op->state = xnn_run_state_invalid;
  const struct xnn_transpose_config* const xnn_transpose_conf = xnn_init_transpose_config();
  if (xnn_transpose_conf == NULL) {
    xnn_log_error("failed to create %s operator: hardware is not supported.",
                  xnn_operator_type_to_string(batch_to_space_op->type));
    goto error;
  }

  const size_t output_height_nocrop = input_height * block_height;
  const size_t output_width_nocrop = input_width * block_width;
  const size_t vertical_crop = crop_top + crop_bottom;
  const size_t horizontal_crop = crop_left + crop_right;

  if (output_height_nocrop <= vertical_crop || output_width_nocrop <= horizontal_crop) {
    batch_to_space_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  // Set up the underlying transposition.
  const size_t input_shape[4] = {
    block_height,
    block_width,
    input_batch / (block_height * block_width) * input_height,
    input_width};
  const size_t perm[4] = {2, 0, 3, 1};

  const bool cropped = crop_top || crop_bottom || crop_left || crop_right;
  const bool prevent_perm_normalization = cropped && block_height == 1;

  status = setup_transpose_nd(
      batch_to_space_op,
      input,
      output,
      4,
      input_shape,
      perm,
      NULL,
      NULL,
      type_size * input_channels,
      prevent_perm_normalization);

  if (status != xnn_status_success) {
    // Don't goto error, setup_transpose_nd has already cleaned up.
    return status;
  }

  // If there is no crop specification, a simple transpose will do the right thing.
  if (!cropped) {
    return xnn_status_success;
  }

  const size_t output_height = output_height_nocrop - vertical_crop;
  const size_t output_width = output_width_nocrop - horizontal_crop;

  struct batch_to_space_context* const context = &batch_to_space_op->context.batch_to_space;
  context->type_size = type_size;
  context->element_size = type_size * input_channels;
  context->input_batch = input_batch;
  context->input_height = input_height;
  context->input_width = input_width;
  context->input_channels = input_channels;
  context->block_height = block_height;
  context->block_width = block_width;
  context->crop_top = crop_top;
  context->crop_bottom = crop_bottom;
  context->crop_left = crop_left;
  context->crop_right = crop_right;
  context->output_height_nocrop = output_height_nocrop;
  context->output_width_nocrop = output_width_nocrop;
  context->output_height = output_height;
  context->output_width = output_width;

  const enum xnn_parallelization_type compute_type = batch_to_space_op->compute.type;
  if (compute_type == xnn_parallelization_type_1d_tile_1d) {
    setup_batch_to_space_identity(batch_to_space_op);
  } else if (compute_type == xnn_parallelization_type_2d_tile_2d && block_height == 1) {
    setup_batch_to_space_tile_2d_h(batch_to_space_op, vertical_crop, horizontal_crop);
  } else if (compute_type == xnn_parallelization_type_2d_tile_2d && block_width == 1) {
    // TODO(qkhan): test for const size transpose handling
    setup_batch_to_space_tile_2d_v(batch_to_space_op, xnn_transpose_conf, vertical_crop, horizontal_crop);
  } else if (compute_type == xnn_parallelization_type_4d_tile_2d) {
    setup_batch_to_space_tile_4d(batch_to_space_op, vertical_crop, horizontal_crop);
  } else {
    XNN_UNREACHABLE;
  }
  return xnn_status_success;

error:
  xnn_delete_operator(batch_to_space_op);
  return status;
}

enum xnn_status xnn_setup_batch_to_space_nhwc_x8(
  xnn_operator_t batch_to_space_op,
  size_t input_batch,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t block_height,
  size_t block_width,
  size_t crop_top,
  size_t crop_bottom,
  size_t crop_left,
  size_t crop_right,
  const void* input,
  void* output)
{
  return xnn_setup_batch_to_space_nhwc(
      sizeof(int8_t),
      batch_to_space_op,
      input_batch, input_height, input_width, input_channels,
      block_height, block_width,
      crop_top, crop_bottom, crop_left, crop_right,
      input, output);
}

enum xnn_status xnn_setup_batch_to_space_nhwc_x16(
  xnn_operator_t batch_to_space_op,
  size_t input_batch,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t block_height,
  size_t block_width,
  size_t crop_top,
  size_t crop_bottom,
  size_t crop_left,
  size_t crop_right,
  const void* input,
  void* output)
{
  return xnn_setup_batch_to_space_nhwc(
      sizeof(int16_t),
      batch_to_space_op,
      input_batch, input_height, input_width, input_channels,
      block_height, block_width,
      crop_top, crop_bottom, crop_left, crop_right,
      input, output);
}

enum xnn_status xnn_setup_batch_to_space_nhwc_x32(
  xnn_operator_t batch_to_space_op,
  size_t input_batch,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t block_height,
  size_t block_width,
  size_t crop_top,
  size_t crop_bottom,
  size_t crop_left,
  size_t crop_right,
  const void* input,
  void* output)
{
  return xnn_setup_batch_to_space_nhwc(
      sizeof(int32_t),
      batch_to_space_op,
      input_batch, input_height, input_width, input_channels,
      block_height, block_width,
      crop_top, crop_bottom, crop_left, crop_right,
      input, output);
}

static enum xnn_status create_depth_to_space_nhwc(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* depth_to_space_op_out)
{
  xnn_operator_t depth_to_space_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (output_channels == 0) {
    xnn_log_error("failed to create %s operator with %zu output channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), output_channels);
    goto error;
  }

  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu)",
      xnn_operator_type_to_string(operator_type),
      output_channel_stride, output_channels);
    goto error;
  }

  if (block_size <= 1) {
    xnn_log_error("failed to create %s operator with %" PRIu32 " block size: block size must be greater than 1",
      xnn_operator_type_to_string(operator_type),
      block_size);
    goto error;
  }

  const size_t input_channels = output_channels * block_size * block_size;
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      input_channel_stride, block_size, block_size, input_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  depth_to_space_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (depth_to_space_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  depth_to_space_op->channels = output_channels;
  depth_to_space_op->input_pixel_stride = input_channel_stride;
  depth_to_space_op->output_pixel_stride = output_channel_stride;
  depth_to_space_op->block_size = block_size;

  depth_to_space_op->type = operator_type;
  depth_to_space_op->flags = flags;

  depth_to_space_op->state = xnn_run_state_invalid;

  *depth_to_space_op_out = depth_to_space_op;
  return xnn_status_success;

error:
  xnn_delete_operator(depth_to_space_op);
  return status;
}

enum xnn_status xnn_create_depth_to_space_nhwc_x8(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nhwc(
    output_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nhwc_x8,
    depth_to_space_op_out);
}

enum xnn_status xnn_create_depth_to_space_nhwc_x16(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nhwc(
    output_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nhwc_x16,
    depth_to_space_op_out);
}

enum xnn_status xnn_create_depth_to_space_nhwc_x32(
    size_t output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nhwc(
    output_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nhwc_x32,
    depth_to_space_op_out);
}

static enum xnn_status setup_depth_to_space_nhwc(
    xnn_operator_t depth_to_space_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    uint32_t element_size)
{
  if (depth_to_space_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }
  depth_to_space_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error("failed to setup %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    depth_to_space_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const uint32_t block_size = depth_to_space_op->block_size;
  const size_t channels = depth_to_space_op->channels;
  const size_t input_pixel_stride = depth_to_space_op->input_pixel_stride;
  const size_t output_pixel_stride = depth_to_space_op->output_pixel_stride;
  const size_t block_output_pixel_stride = block_size * depth_to_space_op->output_pixel_stride;

  const size_t input_shape[5] = {batch_size * input_height, input_width, block_size, block_size, channels};
  const size_t perm[5] = {0, 2, 1, 3, 4};
  const size_t input_stride[5] = {
    input_width * input_pixel_stride,
    input_pixel_stride,
    block_size * channels,
    channels,
    1
  };
  const size_t output_stride[5] = {
    block_size * input_width * block_output_pixel_stride,
    input_width * block_output_pixel_stride,
    block_output_pixel_stride,
    output_pixel_stride,
    1
  };

  return setup_transpose_nd(
    depth_to_space_op,
    input, output,
    5, input_shape, perm, input_stride, output_stride,
    element_size, false);
}

enum xnn_status xnn_setup_depth_to_space_nhwc_x8(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x8,
    batch_size, input_height, input_width,
    input, output, 1);
}

enum xnn_status xnn_setup_depth_to_space_nhwc_x16(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x16,
    batch_size, input_height, input_width,
    input, output, 2);
}

enum xnn_status xnn_setup_depth_to_space_nhwc_x32(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x32,
    batch_size, input_height, input_width,
    input, output, 4);
}

static enum xnn_status create_space_to_depth_nhwc(
    size_t input_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* space_to_depth_op_out)
{
  xnn_operator_t space_to_depth_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (input_channels == 0) {
    xnn_log_error("failed to create %s operator with %zu input channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), input_channels);
    goto error;
  }

  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%zu)",
      xnn_operator_type_to_string(operator_type),
      input_channel_stride, input_channels);
    goto error;
  }

  if (block_size <= 1) {
    xnn_log_error("failed to create %s operator with %" PRIu32 " block size: block size must be greater than 1",
      xnn_operator_type_to_string(operator_type),
      block_size);
    goto error;
  }

  const size_t output_channels = input_channels * block_size * block_size;
  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      output_channel_stride, block_size, block_size, input_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  space_to_depth_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (space_to_depth_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  space_to_depth_op->channels = input_channels;
  space_to_depth_op->input_pixel_stride = input_channel_stride;
  space_to_depth_op->output_pixel_stride = output_channel_stride;
  space_to_depth_op->block_size = block_size;

  space_to_depth_op->type = operator_type;
  space_to_depth_op->flags = flags;

  space_to_depth_op->state = xnn_run_state_invalid;

  *space_to_depth_op_out = space_to_depth_op;
  return xnn_status_success;

error:
  xnn_delete_operator(space_to_depth_op);
  return status;
}

enum xnn_status xnn_create_space_to_depth_nhwc_x8(
    size_t input_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* space_to_depth_op_out)
{
  return create_space_to_depth_nhwc(
    input_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_space_to_depth_nhwc_x8,
    space_to_depth_op_out);
}

enum xnn_status xnn_create_space_to_depth_nhwc_x16(
    size_t input_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* space_to_depth_op_out)
{
  return create_space_to_depth_nhwc(
    input_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_space_to_depth_nhwc_x16,
    space_to_depth_op_out);
}

enum xnn_status xnn_create_space_to_depth_nhwc_x32(
    size_t input_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* space_to_depth_op_out)
{
  return create_space_to_depth_nhwc(
    input_channels,
    input_channel_stride,
    output_channel_stride,
    block_size,
    flags,
    xnn_operator_type_space_to_depth_nhwc_x32,
    space_to_depth_op_out);
}

static enum xnn_status setup_space_to_depth_nhwc(
    xnn_operator_t space_to_depth_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    uint32_t element_size)
{
  if (space_to_depth_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(space_to_depth_op->type));
    return xnn_status_invalid_parameter;
  }
  space_to_depth_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error("failed to setup %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  const uint32_t block_size = space_to_depth_op->block_size;
  if (input_width % block_size != 0) {
    xnn_log_error(
        "failed to setup %s operator with %zu input width and %u block size: input width must be divisible by block "
        "size",
      xnn_operator_type_to_string(expected_operator_type), input_width, block_size);
    return xnn_status_invalid_parameter;
  }

  if (input_height % block_size != 0) {
    xnn_log_error(
        "failed to setup %s operator with %zu input height and %u block size: input height must be divisible by block "
        "size",
      xnn_operator_type_to_string(expected_operator_type), input_height, block_size);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    space_to_depth_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = space_to_depth_op->channels;
  const size_t input_shape[5] = {
    batch_size * (input_height / block_size),
    block_size,
    input_width / block_size,
    block_size,
    channels
  };
  const size_t perm[5] = {0, 2, 1, 3, 4};

  const size_t input_stride[5] = {
    block_size * input_width * space_to_depth_op->input_pixel_stride,
    input_width * space_to_depth_op->input_pixel_stride,
    block_size * space_to_depth_op->input_pixel_stride,
    space_to_depth_op->input_pixel_stride,
    1
  };
  const size_t output_stride[5] = {
    (input_width/block_size) * space_to_depth_op->output_pixel_stride,
    space_to_depth_op->output_pixel_stride,
    block_size * channels,
    channels,
    1
  };

  return setup_transpose_nd(
    space_to_depth_op,
    input, output,
    5, input_shape, perm, input_stride, output_stride,
    element_size, false);
}

enum xnn_status xnn_setup_space_to_depth_nhwc_x8(
    xnn_operator_t space_to_depth_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x8,
    batch_size, input_height, input_width,
    input, output, sizeof(uint8_t));
}

enum xnn_status xnn_setup_space_to_depth_nhwc_x16(
    xnn_operator_t space_to_depth_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x16,
    batch_size, input_height, input_width,
    input, output, sizeof(uint16_t));
}

enum xnn_status xnn_setup_space_to_depth_nhwc_x32(
    xnn_operator_t space_to_depth_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x32,
    batch_size, input_height, input_width,
    input, output, sizeof(uint32_t));
}
