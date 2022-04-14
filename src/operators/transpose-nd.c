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

static enum xnn_status create_transpose_nd(
    uint32_t flags,
    uint32_t datatype_init_flags,
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

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error(
      "failed to create %s operator: operations on data type are not supported",
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

  transpose_op->flags = flags;
  transpose_op->type = operator_type;
  *transpose_op_out = transpose_op;

  return xnn_status_success;

error:
  xnn_delete_operator(transpose_op);
  return status;
}

static enum xnn_status setup_transpose(
  xnn_operator_t transpose_op,
  const void* input,
  void* output,
  const size_t num_dims,
  const size_t* input_shape,
  const size_t* perm,
  size_t element_size)
{
  enum xnn_status status = xnn_status_invalid_parameter;
  transpose_op->state = xnn_run_state_invalid;

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

  transpose_op->channels = num_dims;

  struct transpose_context* context = &transpose_op->context.transpose;
  size_t normalized_dims;
  size_t normalized_shape[XNN_MAX_TENSOR_DIMS];
  size_t normalized_perm[XNN_MAX_TENSOR_DIMS];
  xnn_normalize_transpose_permutation(num_dims, element_size, perm, input_shape, &normalized_dims,
                                      &context->element_size, normalized_perm, normalized_shape);

  size_t loop_order[XNN_MAX_TENSOR_DIMS];
  size_t n = normalized_shape[normalized_dims - 1];
  context->input_stride[normalized_dims - 1] = context->element_size;
  context->output_stride[normalized_perm[normalized_dims - 1]] = context->element_size;
  for(size_t i = normalized_dims - 1; i-- > 0;) {
    context->input_stride[i] = context->input_stride[i + 1] * normalized_shape[i + 1];
    context->output_stride[normalized_perm[i]] = context->output_stride[normalized_perm[i + 1]] * normalized_shape[normalized_perm[i + 1]];
    n *= normalized_shape[i];
  }

  memcpy(loop_order, normalized_perm, sizeof(size_t) * normalized_dims);
  /// The innermost loop must iterate over the contiguous input dimension and the second most inner loop over the
  /// contiguous output dimension.
  if (normalized_dims > 1) {
    for (size_t i = 0; i < normalized_dims - 2; ++i) {
      if (loop_order[i] == normalized_dims - 1) {
        size_t tmp = loop_order[i];
        loop_order[i] = loop_order[normalized_dims - 2];
        loop_order[normalized_dims - 2] = tmp;
        break;
      }
    }
  }

  for (size_t i = 0; i < normalized_dims; ++i) {
    transpose_op->compute.range[i] = normalized_shape[i];
  }
  reorder_array(normalized_dims, loop_order, context->input_stride);
  reorder_array(normalized_dims, loop_order, context->output_stride);
  reorder_array(normalized_dims, loop_order, transpose_op->compute.range);

  bool variable_size_ukernel = false;
  switch (context->element_size) {
    case 1:
      context->log2_element_size = 0;
      context->const_size_ukernel = xnn_params.x8.transpose.const_size_ukernel;
      transpose_op->compute.tile[0] = xnn_params.x8.transpose.tile_size;
      transpose_op->compute.tile[1] = xnn_params.x8.transpose.tile_size;
      break;
    case 2:
      context->log2_element_size = 1;
      transpose_op->compute.tile[0] = xnn_params.x16.transpose.tile_size;
      transpose_op->compute.tile[1] = xnn_params.x16.transpose.tile_size;
      context->const_size_ukernel = xnn_params.x16.transpose.const_size_ukernel;
      break;
    case 4:
      context->log2_element_size = 2;
      transpose_op->compute.tile[0] = xnn_params.x32.transpose.tile_size;
      transpose_op->compute.tile[1] = xnn_params.x32.transpose.tile_size;
      context->const_size_ukernel = xnn_params.x32.transpose.const_size_ukernel;
      break;
    default:
      transpose_op->compute.tile[0] = xnn_params.xx.transpose.tile_size;
      transpose_op->compute.tile[1] = xnn_params.xx.transpose.tile_size;
      context->variable_size_ukernel = xnn_params.xx.transpose.variable_size_ukernel;
      variable_size_ukernel = true;
  }

  struct univector_contiguous_context* univector_context = &transpose_op->context.univector_contiguous;
  switch (normalized_dims) {
    case 1:
      transpose_op->compute.type = xnn_parallelization_type_1d_tile_1d;
      transpose_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_univector_contiguous;
      transpose_op->compute.range[0] = context->element_size;
      univector_context->ukernel = xnn_params.xx.copy;
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
    XNN_INIT_FLAG_X32,
    xnn_operator_type_transpose_nd_x32,
    transpose_op_out);
}

enum xnn_status xnn_create_transpose_nd_x16(
  uint32_t flags,
  xnn_operator_t* transpose_op_out)
{
  return create_transpose_nd(
    flags,
    XNN_INIT_FLAG_X16,
    xnn_operator_type_transpose_nd_x16,
    transpose_op_out);
}

enum xnn_status xnn_create_transpose_nd_x8(
  uint32_t flags,
  xnn_operator_t* transpose_op_out)
{
  return create_transpose_nd(
    flags,
    XNN_INIT_FLAG_X8,
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

  return setup_transpose(
    transpose_op,
    input, output,
    num_dims, shape, perm,
    sizeof(uint32_t));
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

  return setup_transpose(
    transpose_op,
    input, output,
    num_dims, shape, perm,
    sizeof(uint16_t));
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

  return setup_transpose(
    transpose_op,
    input, output,
    num_dims, shape, perm,
    sizeof(uint8_t));
}
