// Copyright 2020 Google LLC
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
#include <xnnpack/operator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static enum xnn_status create_constant_pad_nd(
    uint32_t padding_value,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* constant_pad_op_out)
{
  xnn_operator_t constant_pad_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_constant_pad_nd_x32));
    goto error;
  }

  status = xnn_status_out_of_memory;

  constant_pad_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (constant_pad_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_constant_pad_nd_x32));
    goto error;
  }

  constant_pad_op->pad_value = padding_value;

  constant_pad_op->type = operator_type;
  constant_pad_op->flags = flags;

  constant_pad_op->state = xnn_run_state_invalid;

  *constant_pad_op_out = constant_pad_op;
  return xnn_status_success;

error:
  xnn_delete_operator(constant_pad_op);
  return status;
}

enum xnn_status xnn_create_constant_pad_nd_x32(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out)
{
  return create_constant_pad_nd(
    *((uint32_t*) padding_value), flags, xnn_operator_type_constant_pad_nd_x32, constant_pad_op_out);
}

static enum xnn_status setup_constant_pad_nd(
    xnn_operator_t constant_pad_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_paddings,
    const size_t* post_paddings,
    const void* input,
    void* output,
    size_t num_threads)
{
  if (constant_pad_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(constant_pad_op->type));
    return xnn_status_invalid_parameter;
  }
  constant_pad_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(constant_pad_op->type));
    return xnn_status_uninitialized;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to setup %s operator with %zu dimensions in input shape: "
      "the number of input dimensions must not exceed %d",
      xnn_operator_type_to_string(constant_pad_op->type), num_dims, XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  for (size_t i = 0; i < num_dims; i++) {
    if (input_shape[i] == 0) {
      xnn_log_error(
        "failed to setup %s operator: input shape dimension #%zu is zero",
        xnn_operator_type_to_string(constant_pad_op->type), i);
      return xnn_status_invalid_parameter;
    }
  }

  size_t num_squeezed_dims = 0;
  size_t normalized_pre_paddings[XNN_MAX_TENSOR_DIMS];
  size_t normalized_input_shape[XNN_MAX_TENSOR_DIMS];
  size_t normalized_output_shape[XNN_MAX_TENSOR_DIMS];
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    normalized_pre_paddings[i] = 0;
    normalized_input_shape[i] = 1;
    normalized_output_shape[i] = 1;
  }

  bool is_previous_dim_padded = true;
  for (size_t i = 0; i < num_dims; i++) {
    const size_t pre_padding = pre_paddings[num_dims - 1 - i];
    const size_t post_padding = post_paddings[num_dims - 1 - i];
    const size_t input_dim = input_shape[num_dims - 1 - i];

    const bool is_current_dim_padded = (pre_padding | post_padding) != 0;
    if (is_current_dim_padded || is_previous_dim_padded) {
      normalized_pre_paddings[XNN_MAX_TENSOR_DIMS - 1 - num_squeezed_dims] = pre_padding;
      normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - num_squeezed_dims] = input_dim;
      normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - num_squeezed_dims] = pre_padding + input_dim + post_padding;

      num_squeezed_dims += 1;
      is_previous_dim_padded = is_current_dim_padded;
    } else {
      assert(!is_previous_dim_padded);
      assert(pre_padding == 0);
      assert(post_padding == 0);
      assert(i != 0);

      normalized_input_shape[XNN_MAX_TENSOR_DIMS - num_squeezed_dims] *= input_dim;
      normalized_output_shape[XNN_MAX_TENSOR_DIMS - num_squeezed_dims] *= input_dim;
    }
  }

  constant_pad_op->context.pad = (struct pad_context) {
    .input = input,
    .output = output,
    .padding_value = constant_pad_op->pad_value,
    .fill_ukernel = xnn_params.xx.fill.ukernel,
    .pad_ukernel = xnn_params.xx.pad.ukernel,
  };

  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    constant_pad_op->context.pad.pre_paddings[i] = normalized_pre_paddings[XNN_MAX_TENSOR_DIMS - 1 - i];
    constant_pad_op->context.pad.input_size[i] = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  size_t input_stride = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1];
  size_t output_stride = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1];
  for (size_t i = 1; i < XNN_MAX_TENSOR_DIMS; i++) {
    constant_pad_op->context.pad.input = (const void*)
      ((uintptr_t) constant_pad_op->context.pad.input - constant_pad_op->context.pad.pre_paddings[i] * input_stride * sizeof(float));
    constant_pad_op->context.pad.input_stride[i - 1] = input_stride * sizeof(float);
    constant_pad_op->context.pad.output_stride[i - 1] = output_stride * sizeof(float);
    input_stride *= normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
    output_stride *= normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  constant_pad_op->context.pad.input_size[0] *= sizeof(float);
  constant_pad_op->context.pad.output_size[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1] * sizeof(float);
  constant_pad_op->context.pad.pre_paddings[0] *= sizeof(float);
  constant_pad_op->context.pad.post_paddings[0] =
    constant_pad_op->context.pad.output_size[0] - constant_pad_op->context.pad.pre_paddings[0] - constant_pad_op->context.pad.input_size[0];

  constant_pad_op->compute.type = xnn_parallelization_type_5d;
  constant_pad_op->compute.task_5d = (pthreadpool_task_5d_t) xnn_compute_pad_5d;
  constant_pad_op->compute.range[0] = normalized_output_shape[0];
  constant_pad_op->compute.range[1] = normalized_output_shape[1];
  constant_pad_op->compute.range[2] = normalized_output_shape[2];
  constant_pad_op->compute.range[3] = normalized_output_shape[3];
  constant_pad_op->compute.range[4] = normalized_output_shape[4];
  constant_pad_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_constant_pad_nd_x32(
    xnn_operator_t constant_pad_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_padding,
    const size_t* post_padding,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_constant_pad_nd(
    constant_pad_op, xnn_operator_type_constant_pad_nd_x32,
    num_dims, input_shape, pre_padding, post_padding,
    input, output,
    pthreadpool_get_threads_count(threadpool));
}
