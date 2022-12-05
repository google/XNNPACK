// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/normalization.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>

static void init_slice_nd(
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t slice_op)
{
  slice_op->type = operator_type;
  slice_op->flags = flags;
  slice_op->state = xnn_run_state_invalid;
}

static enum xnn_status create_slice_nd(
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* slice_op_out)
{
  xnn_operator_t slice_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  slice_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (slice_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  init_slice_nd(
    flags,
    operator_type,
    slice_op);

  *slice_op_out = slice_op;
  return xnn_status_success;

error:
  xnn_delete_operator(slice_op);
  return status;
}

enum xnn_status xnn_create_slice_nd_x8(
    uint32_t flags,
    xnn_operator_t *slice_op_out)
{
  return create_slice_nd(flags, xnn_operator_type_slice_nd_x8, slice_op_out);
}

enum xnn_status xnn_create_slice_nd_x16(
    uint32_t flags,
    xnn_operator_t *slice_op_out)
{
  return create_slice_nd(flags, xnn_operator_type_slice_nd_x16, slice_op_out);
}

enum xnn_status xnn_create_slice_nd_x32(
    uint32_t flags,
    xnn_operator_t *slice_op_out)
{
  return create_slice_nd(flags, xnn_operator_type_slice_nd_x32, slice_op_out);
}

static enum xnn_status setup_slice_nd(
    xnn_operator_t slice_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* offsets,
    const size_t* sizes,
    const void* input,
    void* output,
    uint32_t log2_element_size,
    size_t num_threads)
{
  if (slice_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(slice_op->type));
    return xnn_status_invalid_parameter;
  }
  slice_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(slice_op->type));
    return xnn_status_uninitialized;
  }

  if (num_dims == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu num_dims: num_dims must be non-zero",
      xnn_operator_type_to_string(slice_op->type), num_dims);
    return xnn_status_unsupported_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create %s operator with %zu num_dims: num_dims must be <= %d",
      xnn_operator_type_to_string(slice_op->type), num_dims, XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  for (size_t i = 0; i < num_dims; i++) {
    if (input_shape[i] == 0) {
      xnn_log_error(
        "failed to setup %s operator: input shape dimension #%zu is zero",
        xnn_operator_type_to_string(slice_op->type), i);
      return xnn_status_invalid_parameter;
    }
    if (offsets[i] >= input_shape[i]) {
      xnn_log_error(
          "failed to create %s operator with %zu offsets[%zu]: 0 <= offset < %zu",
          xnn_operator_type_to_string(slice_op->type), offsets[i], i, input_shape[i]);
      return xnn_status_unsupported_parameter;
    }
    if (sizes[i] == 0 || sizes[i] > input_shape[i]) {
      xnn_log_error(
          "failed to create %s operator with %zu sizes[%zu]: 0 < size <= %zu",
          xnn_operator_type_to_string(slice_op->type), sizes[i], i, input_shape[i]);
      return xnn_status_unsupported_parameter;
    }
    if (offsets[i] + sizes[i] > input_shape[i]) {
      xnn_log_error(
          "failed to create %s operator with %zu offsets[%zu] and %zu sizes[%zu]: offset + size <= %zu",
          xnn_operator_type_to_string(slice_op->type), offsets[i], i, sizes[i], i, input_shape[i]);
      return xnn_status_unsupported_parameter;
    }
  }

  size_t normalized_offsets[XNN_MAX_TENSOR_DIMS];
  size_t normalized_input_shape[XNN_MAX_TENSOR_DIMS];
  size_t normalized_output_shape[XNN_MAX_TENSOR_DIMS];
  size_t num_normalized_dims;

  xnn_normalize_slice(
      num_dims,
      offsets,
      sizes,
      input_shape,
      normalized_offsets,
      normalized_input_shape,
      normalized_output_shape,
      &num_normalized_dims);
  assert(num_normalized_dims <= XNN_MAX_TENSOR_DIMS);

  slice_op->context.slice = (struct slice_context) {
    .input = input,
    .output = output,
    .ukernel = xnn_params.xx.copy,
  };

  // TODO(b/246969669): move strides calculation into normalization to simplify code here.
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    slice_op->context.slice.offsets[i] = normalized_offsets[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  slice_op->context.slice.offsets[0] <<= log2_element_size;
  size_t input_stride = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1];
  size_t output_stride = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1];
  for (size_t i = 1; i < XNN_MAX_TENSOR_DIMS; i++) {
    slice_op->context.slice.input_stride[i - 1] = input_stride << log2_element_size;
    slice_op->context.slice.output_stride[i - 1] = output_stride << log2_element_size;
    input_stride *= normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
    output_stride *= normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  slice_op->context.slice.contiguous_size = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1] << log2_element_size;

  slice_op->context.slice.input =
      (void*) ((uintptr_t) slice_op->context.slice.input + slice_op->context.slice.offsets[0]);

  // Pre-calculate offsets into input pointer.
  for (size_t i = 1; i < num_normalized_dims; i++) {
    slice_op->context.slice.input =
        (void*) ((uintptr_t) slice_op->context.slice.input +
                 slice_op->context.slice.offsets[i] * slice_op->context.slice.input_stride[i-1]);
  }

  switch (num_normalized_dims) {
    case 1:
    case 2:
      slice_op->compute.type = xnn_parallelization_type_1d;
      slice_op->compute.task_1d = (pthreadpool_task_1d_t)xnn_compute_slice_1d;
      slice_op->compute.range[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 2];
      break;
    case 3:
      slice_op->compute.type = xnn_parallelization_type_2d;
      slice_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_slice_2d;
      slice_op->compute.range[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 3];
      slice_op->compute.range[1] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 2];
      break;
    case 4:
      slice_op->compute.type = xnn_parallelization_type_3d;
      slice_op->compute.task_3d = (pthreadpool_task_3d_t) xnn_compute_slice_3d;
      slice_op->compute.range[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 4];
      slice_op->compute.range[1] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 3];
      slice_op->compute.range[2] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 2];
      break;
    case 5:
      slice_op->compute.type = xnn_parallelization_type_4d;
      slice_op->compute.task_4d = (pthreadpool_task_4d_t) xnn_compute_slice_4d;
      slice_op->compute.range[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 5];
      slice_op->compute.range[1] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 4];
      slice_op->compute.range[2] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 3];
      slice_op->compute.range[3] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 2];
      break;
    case 6:
      // TODO(b/246969669): write normalized_output_shape in reverse order to simplify code here.
      slice_op->compute.type = xnn_parallelization_type_5d;
      slice_op->compute.task_5d = (pthreadpool_task_5d_t) xnn_compute_slice_5d;
      slice_op->compute.range[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 6];
      slice_op->compute.range[1] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 5];
      slice_op->compute.range[2] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 4];
      slice_op->compute.range[3] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 3];
      slice_op->compute.range[4] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 2];
      break;
    default:
      XNN_UNREACHABLE;
  }
  slice_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_slice_nd_x8(
    xnn_operator_t slice_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* offsets,
    const size_t* sizes,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_slice_nd(
    slice_op, xnn_operator_type_slice_nd_x8,
    num_dims, input_shape, offsets, sizes,
    input, output, 0 /* log2(element size) */,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_slice_nd_x16(
    xnn_operator_t slice_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* offsets,
    const size_t* sizes,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_slice_nd(
    slice_op, xnn_operator_type_slice_nd_x16,
    num_dims, input_shape, offsets, sizes,
    input, output, 1 /* log2(element size) */,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_slice_nd_x32(
    xnn_operator_t slice_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* offsets,
    const size_t* sizes,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_slice_nd(
    slice_op, xnn_operator_type_slice_nd_x32,
    num_dims, input_shape, offsets, sizes,
    input, output, 2 /* log2(element size) */,
    pthreadpool_get_threads_count(threadpool));
}

static enum xnn_status xnn_run_slice_nd(
    enum xnn_operator_type operator_type,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* offsets,
    const size_t* sizes,
    const void* input,
    void* output,
    uint32_t log2_element_size,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  struct xnn_operator slice_op;
  memset(&slice_op, 0, sizeof(slice_op));

  init_slice_nd(
    flags,
    operator_type,
    &slice_op);

  const enum xnn_status status = setup_slice_nd(
    &slice_op, operator_type,
    num_dims, input_shape, offsets, sizes,
    input, output,
    log2_element_size,
    pthreadpool_get_threads_count(threadpool));

  if (status != xnn_status_success){
    return status;
  }

  return xnn_run_operator(&slice_op, threadpool);
}

enum xnn_status xnn_run_slice_nd_x32(
    size_t num_dims,
    const size_t* input_shape,
    const size_t* offsets,
    const size_t* sizes,
    const void* input,
    void* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_slice_nd_x32));
    return xnn_status_uninitialized;
  }
  return xnn_run_slice_nd(
    xnn_operator_type_slice_nd_x32,
    num_dims, input_shape, offsets, sizes,
    input, output,
    2 /* log2(element size) */,
    flags,
    threadpool);
}
