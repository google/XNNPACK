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
    const struct xnn_transpose_config* transpose_config,
    enum xnn_operator_type operator_type,
    xnn_operator_t transpose_op)
{
  transpose_op->flags = flags;
  transpose_op->transpose_config = transpose_config;
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

  const struct xnn_transpose_config* transpose_config = xnn_init_transpose_config();
  assert(transpose_config != NULL);

  status = xnn_status_out_of_memory;
  transpose_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (transpose_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  init_transpose_nd(flags, transpose_config, operator_type, transpose_op);
  *transpose_op_out = transpose_op;

  return xnn_status_success;

error:
  xnn_delete_operator(transpose_op);
  return status;
}

enum xnn_status xnn_create_transpose_nd_x64(
  uint32_t flags,
  xnn_operator_t* transpose_op_out)
{
  return create_transpose_nd(
    flags,
    xnn_operator_type_transpose_nd_x64,
    transpose_op_out);
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

/// input_stride and output_stride are the number of elements between each
/// dimension, not the size of the dimension. This is because depth to space
/// splits the input channel dimension into three dimensions - block_size *
/// block_size * output_channels but gives input_channel_stride the stride over
/// all three dimensions. This must be multiplied by the product of the previous
/// dimensions to get the stride in elements. input_channel_stride is not
/// requried to be a multiple of block_size * block_size * output_channels so
/// the stride in number of elements must be supplied.
/// An interface for sub-tensors can easily be built on top of this.
static enum xnn_status reshape_transpose_nd(
  xnn_operator_t transpose_op,
  const size_t num_dims,
  const size_t* input_shape,
  const size_t* perm,
  const size_t* input_stride,
  const size_t* output_stride,
  size_t element_size)
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
            "failed to create %s operator with duplicate entries in perm %zu %zu",
            xnn_operator_type_to_string(transpose_op->type), perm[i], perm[j]);
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

  struct transpose_context* context = &transpose_op->context.transpose;
  size_t normalized_dims;
  size_t normalized_shape[XNN_MAX_TENSOR_DIMS];
  size_t normalized_perm[XNN_MAX_TENSOR_DIMS];
  size_t normalized_element_size;
  xnn_normalize_transpose_permutation(num_dims, element_size, perm, input_shape, input_stride, output_stride, &normalized_dims,
                                      &normalized_element_size, normalized_perm, normalized_shape, context->input_stride, context->output_stride);
  assert(normalized_dims);

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
    transpose_op->compute[0].range[i] = normalized_shape[i];
  }
  reorder_array(normalized_dims, loop_order, context->input_stride);
  reorder_array(normalized_dims, loop_order, transpose_op->compute[0].range);

  const struct xnn_transpose_config* transpose_config = transpose_op->transpose_config;

  bool variable_size_ukernel = false;
  const size_t ukernel_selector = normalized_perm[normalized_dims-1] == normalized_dims-1 ? 0 : normalized_element_size;
  switch (ukernel_selector) {
    case 1:
      context->const_size_ukernel = transpose_config->x8.const_size_ukernel;
      transpose_op->compute[0].tile[0] = transpose_config->x8.tile_size;
      transpose_op->compute[0].tile[1] = transpose_config->x8.tile_size;
      if (transpose_config->x8.init.x16 != NULL) {
        transpose_config->x8.init.x8(&context->params.x8_params);
      }
      break;
    case 2:
      transpose_op->compute[0].tile[0] = transpose_config->x16.tile_size;
      transpose_op->compute[0].tile[1] = transpose_config->x16.tile_size;
      context->const_size_ukernel = transpose_config->x16.const_size_ukernel;
      if (transpose_config->x16.init.x16 != NULL) {
        transpose_config->x16.init.x16(&context->params.x16_params);
      }
      break;
    case 3:
      transpose_op->compute[0].tile[0] = transpose_config->x24.tile_size;
      transpose_op->compute[0].tile[1] = transpose_config->x24.tile_size;
      context->const_size_ukernel = transpose_config->x24.const_size_ukernel;
      if (transpose_config->x24.init.x24 != NULL) {
        transpose_config->x24.init.x24(&context->params.x24_params);
      }
      break;
    case 4:
      transpose_op->compute[0].tile[0] = transpose_config->x32.tile_size;
      transpose_op->compute[0].tile[1] = transpose_config->x32.tile_size;
      context->const_size_ukernel = transpose_config->x32.const_size_ukernel;
      if (transpose_config->x32.init.x32 != NULL) {
        transpose_config->x32.init.x32(&context->params.x32_params);
      }
      break;
    default:
      transpose_op->compute[0].tile[0] = transpose_config->xx.tile_size;
      transpose_op->compute[0].tile[1] = transpose_config->xx.tile_size;
      context->variable_size_ukernel = transpose_config->xx.variable_size_ukernel;
      variable_size_ukernel = true;
  }

  struct univector_contiguous_context* univector_context = &transpose_op->context.univector_contiguous;
  switch (normalized_dims) {
    case 1:
      transpose_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
      transpose_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
      transpose_op->compute[0].range[0] = normalized_element_size;
      transpose_op->compute[0].tile[0] = normalized_element_size;
      univector_context->ukernel = transpose_config->copy;
      univector_context->log2_xsize = 0;
      univector_context->log2_ysize = 0;
      break;
    case 2:
      transpose_op->compute[0].type = xnn_parallelization_type_2d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute[0].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_transposev_2d;
      } else {
        transpose_op->compute[0].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_transposec_2d;
      }
      break;
    case 3:
      transpose_op->compute[0].type = xnn_parallelization_type_3d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_transposev_3d;
      } else {
        transpose_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_transposec_3d;
      }
      break;
    case 4:
      transpose_op->compute[0].type = xnn_parallelization_type_4d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute[0].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_transposev_4d;
      } else {
        transpose_op->compute[0].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_transposec_4d;
      }
      break;
    case 5:
      transpose_op->compute[0].type = xnn_parallelization_type_5d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute[0].task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_transposev_5d;
      } else {
        transpose_op->compute[0].task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_transposec_5d;
      }
      break;
    case 6:
      transpose_op->compute[0].type = xnn_parallelization_type_6d_tile_2d;
      if (variable_size_ukernel) {
        transpose_op->compute[0].task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_transposev_6d;
      } else {
        transpose_op->compute[0].task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_transposec_6d;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (num_dims == 1) {
    transpose_op->ukernel.type = xnn_microkernel_type_default;
  } else {
    transpose_op->ukernel.type = xnn_microkernel_type_transpose;
  }

  transpose_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;

error:
  xnn_delete_operator(transpose_op);
  return status;
}

enum xnn_status xnn_reshape_transpose_nd_x64(
    xnn_operator_t transpose_op,
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x64) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x64),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_transpose_nd(
    transpose_op,
    num_dims, shape, perm, NULL, NULL,
    sizeof(uint64_t));
}

enum xnn_status xnn_reshape_transpose_nd_x32(
    xnn_operator_t transpose_op,
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x32) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x32),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_transpose_nd(
    transpose_op,
    num_dims, shape, perm, NULL, NULL,
    sizeof(uint32_t));
}

enum xnn_status xnn_reshape_transpose_nd_x16(
    xnn_operator_t transpose_op,
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x16) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x16),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_transpose_nd(
    transpose_op,
    num_dims, shape, perm, NULL, NULL,
    sizeof(uint16_t));
}

enum xnn_status xnn_reshape_transpose_nd_x8(
    xnn_operator_t transpose_op,
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x8) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x8),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_transpose_nd(
    transpose_op,
    num_dims, shape, perm, NULL, NULL,
    sizeof(uint8_t));
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
  void* output)
{
  switch (transpose_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(transpose_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  if (transpose_op->ukernel.type == xnn_microkernel_type_default) {
    transpose_op->context.univector_contiguous.x = input;
    transpose_op->context.univector_contiguous.y = output;
  } else {
    assert(transpose_op->ukernel.type == xnn_microkernel_type_transpose);
    transpose_op->context.transpose.x = input;
    transpose_op->context.transpose.y = output;
  }
  transpose_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_transpose_nd_x64(
    xnn_operator_t transpose_op,
    const void* input,
    void* output)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x64) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x64),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(transpose_op, input, output);
}

enum xnn_status xnn_setup_transpose_nd_x32(
    xnn_operator_t transpose_op,
    const void* input,
    void* output)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x32),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(transpose_op, input, output);
}

enum xnn_status xnn_setup_transpose_nd_x16(
    xnn_operator_t transpose_op,
    const void* input,
    void* output)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x16),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(transpose_op, input, output);
}

enum xnn_status xnn_setup_transpose_nd_x8(
    xnn_operator_t transpose_op,
    const void* input,
    void* output)
{
  if (transpose_op->type != xnn_operator_type_transpose_nd_x8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_nd_x8),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(transpose_op, input, output);
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

  const struct xnn_transpose_config* transpose_config = xnn_init_transpose_config();
  assert(transpose_config != NULL);

  init_transpose_nd(flags, transpose_config, operator_type, &transpose_op);

  enum xnn_status status = reshape_transpose_nd(
    &transpose_op,
    num_dims, input_shape, output_perm, NULL, NULL,
    element_size);
  if (status != xnn_status_success) {
    return status;
  }

  status = setup_transpose_nd(&transpose_op, input, output);
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

enum xnn_status xnn_run_transpose_nd_x64(
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
    sizeof(uint64_t), xnn_operator_type_transpose_nd_x64,
    threadpool);
}

enum xnn_status create_depth_to_space_nchw2nhwc(
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
  if (block_size <= 1) {
    xnn_log_error("failed to create %s operator with %" PRIu32 " block size: block size must be greater than 1",
      xnn_operator_type_to_string(operator_type),
      block_size);
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

  const struct xnn_transpose_config* transpose_config = xnn_init_transpose_config();
  assert(transpose_config != NULL);

  depth_to_space_op->block_size = block_size;

  depth_to_space_op->type = operator_type;
  depth_to_space_op->flags = flags;
  depth_to_space_op->transpose_config = transpose_config;

  depth_to_space_op->state = xnn_run_state_invalid;

  *depth_to_space_op_out = depth_to_space_op;
  return xnn_status_success;

error:
  xnn_delete_operator(depth_to_space_op);
  return status;
}

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x16(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nchw2nhwc(
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nchw2nhwc_x16,
    depth_to_space_op_out);
}

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x32(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nchw2nhwc(
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nchw2nhwc_x32,
    depth_to_space_op_out);
}

enum xnn_status reshape_depth_to_space_nchw2nhwc(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    enum xnn_operator_type operator_type,
    size_t element_size,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out)
{
  depth_to_space_op->state = xnn_run_state_invalid;

  if (input_width == 0 || input_height == 0 || input_channels == 0) {
    xnn_log_error("failed to reshape %s operator with %zux%zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), input_width, input_height, input_channels);
    return xnn_status_invalid_parameter;
  }

  const uint32_t block_size = depth_to_space_op->block_size;
  if (input_channels % (block_size * block_size) != 0) {
    xnn_log_error("failed to reshape %s operator with %zu input_channels and %zu block_sizex: "
                  "input channels must be divisible by block_size * block_size",
      xnn_operator_type_to_string(operator_type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  const size_t output_channels = input_channels / block_size / block_size;

  if (batch_size == 0) {
    depth_to_space_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t input_shape[6] = {batch_size, block_size, block_size, output_channels, input_height, input_width};
  const size_t perm[6] = {0, 4, 1, 5, 2, 3};
  const size_t area = input_height * input_width;
  const size_t elements_per_batch = area * output_channels;
  const size_t input_stride[6] = {
    input_channels * area,
    block_size * elements_per_batch,
    elements_per_batch,
    area,
    input_width,
    1
  };

  if (output_height_out != NULL) {
    *output_height_out = input_height * block_size;
  }
  if (output_width_out != NULL) {
    *output_width_out = input_width * block_size;
  }
  if (output_channels_out != NULL) {
    *output_channels_out = output_channels;
  }

  const size_t output_stride[6] = {
    input_height * block_size * input_width * block_size * output_channels,
    block_size * input_width * block_size * output_channels,
    input_width * block_size * output_channels,
    block_size * output_channels,
    output_channels,
    1
  };

  return reshape_transpose_nd(
    depth_to_space_op,
    6, input_shape, perm, input_stride, output_stride,
    element_size);
}

enum xnn_status xnn_reshape_depth_to_space_nchw2nhwc_x16(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  if (depth_to_space_op->type != xnn_operator_type_depth_to_space_nchw2nhwc_x16) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x16),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_depth_to_space_nchw2nhwc(
    depth_to_space_op,
    batch_size, input_height, input_width, input_channels,
    xnn_operator_type_depth_to_space_nchw2nhwc_x16, /*element_size=*/2,
    output_height_out, output_width_out, output_channels_out);
}

enum xnn_status xnn_reshape_depth_to_space_nchw2nhwc_x32(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  if (depth_to_space_op->type != xnn_operator_type_depth_to_space_nchw2nhwc_x32) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_depth_to_space_nchw2nhwc(
    depth_to_space_op,
    batch_size, input_height, input_width, input_channels,
    xnn_operator_type_depth_to_space_nchw2nhwc_x32, /*element_size=*/4,
    output_height_out, output_width_out, output_channels_out);
}

enum xnn_status setup_depth_to_space_nchw2nhwc(
    xnn_operator_t depth_to_space_op,
    const void* input,
    void* output)
{
  return setup_transpose_nd(depth_to_space_op, input, output);
}

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x16(
    xnn_operator_t depth_to_space_op,
    const void* input,
    void* output)
{
  if (depth_to_space_op->type != xnn_operator_type_depth_to_space_nchw2nhwc_x16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x16),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_depth_to_space_nchw2nhwc(
    depth_to_space_op,
    input, output);
}

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x32(
    xnn_operator_t depth_to_space_op,
    const void* input,
    void* output)
{
  if (depth_to_space_op->type != xnn_operator_type_depth_to_space_nchw2nhwc_x32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_depth_to_space_nchw2nhwc_x32),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_depth_to_space_nchw2nhwc(
    depth_to_space_op,
    input, output);
}

static enum xnn_status create_depth_to_space_nhwc(
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

  if (block_size <= 1) {
    xnn_log_error("failed to create %s operator with %" PRIu32 " block size: block size must be greater than 1",
      xnn_operator_type_to_string(operator_type),
      block_size);
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

  const struct xnn_transpose_config* transpose_config = xnn_init_transpose_config();
  assert(transpose_config != NULL);

  depth_to_space_op->block_size = block_size;
  depth_to_space_op->type = operator_type;
  depth_to_space_op->flags = flags;
  depth_to_space_op->transpose_config = transpose_config;

  depth_to_space_op->state = xnn_run_state_invalid;

  *depth_to_space_op_out = depth_to_space_op;
  return xnn_status_success;

error:
  xnn_delete_operator(depth_to_space_op);
  return status;
}

enum xnn_status xnn_create_depth_to_space_nhwc_x8(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nhwc(
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nhwc_x8,
    depth_to_space_op_out);
}

enum xnn_status xnn_create_depth_to_space_nhwc_x16(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nhwc(
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nhwc_x16,
    depth_to_space_op_out);
}

enum xnn_status xnn_create_depth_to_space_nhwc_x32(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* depth_to_space_op_out)
{
  return create_depth_to_space_nhwc(
    block_size,
    flags,
    xnn_operator_type_depth_to_space_nhwc_x32,
    depth_to_space_op_out);
}

static enum xnn_status reshape_depth_to_space_nhwc(
    xnn_operator_t depth_to_space_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    uint32_t element_size,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out)
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

  if (input_width == 0 || input_height == 0 || input_channels == 0) {
    xnn_log_error("failed to setup %s operator with %zux%zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height, input_channels);
    return xnn_status_invalid_parameter;
  }

  const uint32_t block_size = depth_to_space_op->block_size;
  if (input_channels % (block_size * block_size) != 0) {
    xnn_log_error("failed to reshape %s operator with %zu input_channels and %u block_size: "
                  "input channels must be divisible by block_size * block_size",
      xnn_operator_type_to_string(expected_operator_type), input_channels, block_size);
    return xnn_status_invalid_parameter;
  }

  const size_t output_channels = input_channels / block_size / block_size;

  if (batch_size == 0) {
    depth_to_space_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t block_output_pixel_stride = block_size * output_channels;

  const size_t input_shape[5] = {batch_size * input_height, input_width, block_size, block_size, output_channels};
  const size_t perm[5] = {0, 2, 1, 3, 4};
  const size_t input_stride[5] = {
    input_width * input_channels,
    input_channels,
    block_size * output_channels,
    output_channels,
    1
  };

  if (output_height_out != NULL) {
    *output_height_out = input_height * block_size;
  }
  if (output_width_out != NULL) {
    *output_width_out = input_width * block_size;
  }
  if (output_channels_out != NULL) {
    *output_channels_out = output_channels;
  }

  const size_t output_stride[5] = {
    block_size * input_width * block_output_pixel_stride,
    input_width * block_output_pixel_stride,
    block_output_pixel_stride,
    output_channels,
    1
  };

  return reshape_transpose_nd(
    depth_to_space_op,
    5, input_shape, perm, input_stride, output_stride,
    element_size);
}

enum xnn_status xnn_reshape_depth_to_space_nhwc_x8(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  return reshape_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x8,
    batch_size, input_height, input_width, input_channels,
    /*element_size=*/1,
    output_height_out, output_width_out, output_channels_out);
}

enum xnn_status xnn_reshape_depth_to_space_nhwc_x16(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  return reshape_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x16,
    batch_size, input_height, input_width, input_channels,
    /*element_size=*/2,
    output_height_out, output_width_out, output_channels_out);
}

enum xnn_status xnn_reshape_depth_to_space_nhwc_x32(
    xnn_operator_t depth_to_space_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  return reshape_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x32,
    batch_size, input_height, input_width, input_channels,
    /*element_size=*/4,
    output_height_out, output_width_out, output_channels_out);
}

static enum xnn_status setup_depth_to_space_nhwc(
    xnn_operator_t depth_to_space_op,
    enum xnn_operator_type expected_operator_type,
    const void* input,
    void* output)
{
  if (depth_to_space_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(depth_to_space_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(depth_to_space_op, input, output);
}

enum xnn_status xnn_setup_depth_to_space_nhwc_x8(
    xnn_operator_t depth_to_space_op,
    const void* input,
    void* output)
{
  return setup_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x8,
    input, output);
}

enum xnn_status xnn_setup_depth_to_space_nhwc_x16(
    xnn_operator_t depth_to_space_op,
    const void* input,
    void* output)
{
  return setup_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x16,
    input, output);
}

enum xnn_status xnn_setup_depth_to_space_nhwc_x32(
    xnn_operator_t depth_to_space_op,
    const void* input,
    void* output)
{
  return setup_depth_to_space_nhwc(
    depth_to_space_op,
    xnn_operator_type_depth_to_space_nhwc_x32,
    input, output);
}

static enum xnn_status create_space_to_depth_nhwc(
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

  if (block_size <= 1) {
    xnn_log_error("failed to create %s operator with %" PRIu32 " block size: block size must be greater than 1",
      xnn_operator_type_to_string(operator_type),
      block_size);
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

  const struct xnn_transpose_config* transpose_config = xnn_init_transpose_config();
  assert(transpose_config != NULL);

  space_to_depth_op->block_size = block_size;

  space_to_depth_op->type = operator_type;
  space_to_depth_op->flags = flags;
  space_to_depth_op->transpose_config = transpose_config;

  space_to_depth_op->state = xnn_run_state_invalid;

  *space_to_depth_op_out = space_to_depth_op;
  return xnn_status_success;

error:
  xnn_delete_operator(space_to_depth_op);
  return status;
}

enum xnn_status xnn_create_space_to_depth_nhwc_x8(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* space_to_depth_op_out)
{
  return create_space_to_depth_nhwc(
    block_size,
    flags,
    xnn_operator_type_space_to_depth_nhwc_x8,
    space_to_depth_op_out);
}

enum xnn_status xnn_create_space_to_depth_nhwc_x16(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* space_to_depth_op_out)
{
  return create_space_to_depth_nhwc(
    block_size,
    flags,
    xnn_operator_type_space_to_depth_nhwc_x16,
    space_to_depth_op_out);
}

enum xnn_status xnn_create_space_to_depth_nhwc_x32(
    uint32_t block_size,
    uint32_t flags,
    xnn_operator_t* space_to_depth_op_out)
{
  return create_space_to_depth_nhwc(
    block_size,
    flags,
    xnn_operator_type_space_to_depth_nhwc_x32,
    space_to_depth_op_out);
}

static enum xnn_status reshape_space_to_depth_nhwc(
    xnn_operator_t space_to_depth_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    uint32_t element_size,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out)
{
  if (space_to_depth_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(space_to_depth_op->type));
    return xnn_status_invalid_parameter;
  }
  space_to_depth_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0 || input_channels == 0) {
    xnn_log_error("failed to reshape %s operator with %zux%zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), input_width, input_height, input_channels);
    return xnn_status_invalid_parameter;
  }

  const uint32_t block_size = space_to_depth_op->block_size;
  const size_t output_channels = input_channels * block_size * block_size;

  if (input_width % block_size != 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu input width and %u block size: input width must be divisible by block "
        "size",
      xnn_operator_type_to_string(expected_operator_type), input_width, block_size);
    return xnn_status_invalid_parameter;
  }

  if (input_height % block_size != 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu input height and %u block size: input height must be divisible by block "
        "size",
      xnn_operator_type_to_string(expected_operator_type), input_height, block_size);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    space_to_depth_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t input_shape[5] = {
    batch_size * (input_height / block_size),
    block_size,
    input_width / block_size,
    block_size,
    input_channels
  };
  const size_t perm[5] = {0, 2, 1, 3, 4};

  if (output_height_out != NULL) {
    *output_height_out = input_height / block_size;
  }
  if (output_width_out != NULL) {
    *output_width_out = input_width / block_size;
  }
  if (output_channels_out != NULL) {
    *output_channels_out = output_channels;
  }

  const size_t input_stride[5] = {
    block_size * input_width * input_channels,
    input_width * input_channels,
    block_size * input_channels,
    input_channels,
    1
  };
  const size_t output_stride[5] = {
    (input_width/block_size) * output_channels,
    output_channels,
    block_size * input_channels,
    input_channels,
    1
  };

  return reshape_transpose_nd(
    space_to_depth_op,
    5, input_shape, perm, input_stride, output_stride,
    element_size);
}

enum xnn_status xnn_reshape_space_to_depth_nhwc_x8(
    xnn_operator_t space_to_depth_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  return reshape_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x8,
    batch_size, input_height, input_width, input_channels,
    sizeof(uint8_t),
    output_height_out, output_width_out, output_channels_out);
}

enum xnn_status xnn_reshape_space_to_depth_nhwc_x16(
    xnn_operator_t space_to_depth_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  return reshape_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x16,
    batch_size, input_height, input_width, input_channels,
    sizeof(uint16_t),
    output_height_out, output_width_out, output_channels_out);
}

enum xnn_status xnn_reshape_space_to_depth_nhwc_x32(
    xnn_operator_t space_to_depth_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t* output_height_out,
    size_t* output_width_out,
    size_t* output_channels_out,
    pthreadpool_t threadpool)
{
  return reshape_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x32,
    batch_size, input_height, input_width, input_channels,
    sizeof(uint32_t),
    output_height_out, output_width_out, output_channels_out);
}

static enum xnn_status setup_space_to_depth_nhwc(
    xnn_operator_t space_to_depth_op,
    enum xnn_operator_type expected_operator_type,
    const void* input,
    void* output)
{
  if (space_to_depth_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(space_to_depth_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose_nd(space_to_depth_op, input, output);
}

enum xnn_status xnn_setup_space_to_depth_nhwc_x8(
    xnn_operator_t space_to_depth_op,
    const void* input,
    void* output)
{
  return setup_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x8,
    input, output);
}

enum xnn_status xnn_setup_space_to_depth_nhwc_x16(
    xnn_operator_t space_to_depth_op,
    const void* input,
    void* output)
{
  return setup_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x16,
    input, output);
}

enum xnn_status xnn_setup_space_to_depth_nhwc_x32(
    xnn_operator_t space_to_depth_op,
    const void* input,
    void* output)
{
  return setup_space_to_depth_nhwc(
    space_to_depth_op,
    xnn_operator_type_space_to_depth_nhwc_x32,
    input, output);
}
