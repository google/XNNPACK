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


static enum xnn_status create_pad_nd(
    uint32_t padding_value,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* pad_op_out)
{
  xnn_operator_t pad_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Pad operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_out_of_memory;

  pad_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (pad_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Pad operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  pad_op->pad_value = padding_value;

  pad_op->type = operator_type;
  pad_op->ukernel.type = xnn_ukernel_type_pad;

  pad_op->state = xnn_run_state_invalid;

  *pad_op_out = pad_op;
  return xnn_status_success;

error:
  xnn_delete_operator(pad_op);
  return status;
}

enum xnn_status xnn_create_pad_nd_x32(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* pad_op_out)
{
  return create_pad_nd(
    *((uint32_t*) padding_value), flags, xnn_operator_type_pad_nd_x32, pad_op_out);
}

static enum xnn_status setup_pad_nd(
    xnn_operator_t pad_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_paddings,
    const size_t* post_paddings,
    const void* input,
    void* output,
    size_t num_threads)
{
  if (pad_op->type != expected_operator_type) {
    xnn_log_error("failed to setup Pad (ND, X32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  pad_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Pad operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to setup Pad operator with %zu dimensions in input shape: "
      "the number of input dimensions must not exceed %d",
      num_dims, XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  for (size_t i = 0; i < num_dims; i++) {
    if (input_shape[i] == 0) {
      xnn_log_error("failed to setup Pad operator: input shape dimension #%zu is zero", i);
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

  pad_op->context.pad = (struct pad_context) {
    .input = input,
    .output = output,
    .padding_value = pad_op->pad_value,
    .fill_ukernel = xnn_params.x32.fill.ukernel,
    .pad_ukernel = xnn_params.x32.pad.ukernel,
  };

  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    pad_op->context.pad.pre_paddings[i] = normalized_pre_paddings[XNN_MAX_TENSOR_DIMS - 1 - i];
    pad_op->context.pad.input_size[i] = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  size_t input_stride = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1];
  size_t output_stride = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1];
  for (size_t i = 1; i < XNN_MAX_TENSOR_DIMS; i++) {
    pad_op->context.pad.input -= pad_op->context.pad.pre_paddings[i] * input_stride * sizeof(float);
    pad_op->context.pad.input_stride[i - 1] = input_stride * sizeof(float);
    pad_op->context.pad.output_stride[i - 1] = output_stride * sizeof(float);
    input_stride *= normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
    output_stride *= normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  pad_op->context.pad.input_size[0] *= sizeof(float);
  pad_op->context.pad.output_size[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1] * sizeof(float);
  pad_op->context.pad.pre_paddings[0] *= sizeof(float);
  pad_op->context.pad.post_paddings[0] =
    pad_op->context.pad.output_size[0] - pad_op->context.pad.pre_paddings[0] - pad_op->context.pad.input_size[0];

  pad_op->compute.type = xnn_parallelization_type_5d_tile_2d;
  pad_op->compute.task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_pad_5d;
  pad_op->compute.range[0] = normalized_output_shape[0];
  pad_op->compute.range[1] = normalized_output_shape[1];
  pad_op->compute.range[2] = normalized_output_shape[2];
  pad_op->compute.range[3] = normalized_output_shape[3];
  pad_op->compute.range[4] = normalized_output_shape[4];
  pad_op->compute.tile[0] = 1;
  pad_op->compute.tile[1] = 1;
  pad_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_pad_nd_x32(
    xnn_operator_t pad_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_padding,
    const size_t* post_padding,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_pad_nd(
    pad_op, xnn_operator_type_pad_nd_x32,
    num_dims, input_shape, pre_padding, post_padding,
    input, output,
    pthreadpool_get_threads_count(threadpool));
}
