// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/compute.h"
#include "xnnpack/config-types.h"
#include "xnnpack/config.h"
#include "xnnpack/log.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static void init_constant_pad_nd(
    uint32_t padding_value,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct xnn_xx_fill_config* fill_config,
    const struct xnn_xx_pad_config* pad_config,
    xnn_operator_t constant_pad_op)
{
  constant_pad_op->pad_value = padding_value;

  constant_pad_op->type = operator_type;
  constant_pad_op->flags = flags;
  constant_pad_op->fill_config = fill_config;
  constant_pad_op->pad_config = pad_config;

  constant_pad_op->state = xnn_run_state_invalid;
}

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
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  constant_pad_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (constant_pad_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  const struct xnn_xx_fill_config* fill_config = xnn_init_xx_fill_config();
  if (fill_config == NULL) {
    xnn_log_error(
      "failed to create fill operator: unsupported hardware configuration");
    goto error;
  }

  const struct xnn_xx_pad_config* pad_config = xnn_init_xx_pad_config();
  if (pad_config == NULL) {
    xnn_log_error(
      "failed to create pad operator: unsupported hardware configuration");
    goto error;
  }

  init_constant_pad_nd(padding_value, flags, operator_type, fill_config, pad_config, constant_pad_op);
  *constant_pad_op_out = constant_pad_op;

  return xnn_status_success;

error:
  xnn_delete_operator(constant_pad_op);
  return status;
}

enum xnn_status xnn_create_constant_pad_nd_x8(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out)
{
  const uint32_t padding_pattern = *((const uint8_t*) padding_value);
  return create_constant_pad_nd(
    padding_pattern * UINT32_C(0x01010101), flags, xnn_operator_type_constant_pad_nd_x8, constant_pad_op_out);
}

enum xnn_status xnn_create_constant_pad_nd_x16(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out)
{
  const uint32_t padding_pattern = *((const uint16_t*) padding_value);
  return create_constant_pad_nd(
    padding_pattern * UINT32_C(0x00010001), flags, xnn_operator_type_constant_pad_nd_x16, constant_pad_op_out);
}

enum xnn_status xnn_create_constant_pad_nd_x32(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out)
{
  return create_constant_pad_nd(
    *((const uint32_t*) padding_value), flags, xnn_operator_type_constant_pad_nd_x32, constant_pad_op_out);
}

static enum xnn_status reshape_constant_pad_nd(
    xnn_operator_t constant_pad_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_paddings,
    const size_t* post_paddings,
    uint32_t log2_element_size,
    pthreadpool_t threadpool)
{
  if (constant_pad_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(constant_pad_op->type));
    return xnn_status_invalid_parameter;
  }
  constant_pad_op->state = xnn_run_state_invalid;

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

  const struct xnn_xx_fill_config* xx_fill_config = constant_pad_op->fill_config;
  const struct xnn_xx_pad_config* xx_pad_config = constant_pad_op->pad_config;

  constant_pad_op->context.pad = (struct pad_context) {
    .padding_value = constant_pad_op->pad_value,
    .fill_ukernel = xx_fill_config->ukernel,
    .pad_ukernel = xx_pad_config->ukernel,
  };

  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    constant_pad_op->context.pad.pre_paddings[i] = normalized_pre_paddings[XNN_MAX_TENSOR_DIMS - 1 - i];
    constant_pad_op->context.pad.input_size[i] = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  size_t input_stride = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1];
  size_t output_stride = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1];
  for (size_t i = 1; i < XNN_MAX_TENSOR_DIMS; i++) {
    constant_pad_op->context.pad.input_stride[i - 1] = input_stride << log2_element_size;
    constant_pad_op->context.pad.output_stride[i - 1] = output_stride << log2_element_size;
    input_stride *= normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
    output_stride *= normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
  }
  constant_pad_op->context.pad.input_size[0] <<= log2_element_size;
  constant_pad_op->context.pad.output_size[0] = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1] << log2_element_size;
  constant_pad_op->context.pad.pre_paddings[0] <<= log2_element_size;
  constant_pad_op->context.pad.post_paddings[0] =
    constant_pad_op->context.pad.output_size[0] - constant_pad_op->context.pad.pre_paddings[0] - constant_pad_op->context.pad.input_size[0];

  constant_pad_op->compute[0].type = xnn_parallelization_type_5d;
  constant_pad_op->compute[0].task_5d = (pthreadpool_task_5d_t) xnn_compute_pad_5d;
  constant_pad_op->compute[0].range[0] = normalized_output_shape[0];
  constant_pad_op->compute[0].range[1] = normalized_output_shape[1];
  constant_pad_op->compute[0].range[2] = normalized_output_shape[2];
  constant_pad_op->compute[0].range[3] = normalized_output_shape[3];
  constant_pad_op->compute[0].range[4] = normalized_output_shape[4];
  constant_pad_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_constant_pad_nd_x8(
    xnn_operator_t constant_pad_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_padding,
    const size_t* post_padding,
    pthreadpool_t threadpool)
{
  return reshape_constant_pad_nd(
    constant_pad_op, xnn_operator_type_constant_pad_nd_x8,
    num_dims, input_shape, pre_padding, post_padding,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    threadpool);
}

enum xnn_status xnn_reshape_constant_pad_nd_x16(
    xnn_operator_t constant_pad_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_padding,
    const size_t* post_padding,
    pthreadpool_t threadpool)
{
  return reshape_constant_pad_nd(
    constant_pad_op, xnn_operator_type_constant_pad_nd_x16,
    num_dims, input_shape, pre_padding, post_padding,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT16_T,
    threadpool);
}

enum xnn_status xnn_reshape_constant_pad_nd_x32(
    xnn_operator_t constant_pad_op,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_padding,
    const size_t* post_padding,
    pthreadpool_t threadpool)
{
  return reshape_constant_pad_nd(
    constant_pad_op, xnn_operator_type_constant_pad_nd_x32,
    num_dims, input_shape, pre_padding, post_padding,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT32_T,
    threadpool);
}

static enum xnn_status setup_constant_pad_nd(
    xnn_operator_t constant_pad_op,
    enum xnn_operator_type expected_operator_type,
    const void* input,
    void* output)
{
  if (constant_pad_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(constant_pad_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (constant_pad_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(constant_pad_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  constant_pad_op->context.pad.input = input;
  constant_pad_op->context.pad.output = output;

  for (size_t i = 1; i < XNN_MAX_TENSOR_DIMS; i++) {
    constant_pad_op->context.pad.input =
      (const void*) ((uintptr_t) constant_pad_op->context.pad.input -
                     (constant_pad_op->context.pad.pre_paddings[i] * constant_pad_op->context.pad.input_stride[i - 1]));
  }
  constant_pad_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_constant_pad_nd_x8(
    xnn_operator_t constant_pad_op,
    const void* input,
    void* output)
{
  return setup_constant_pad_nd(
    constant_pad_op, xnn_operator_type_constant_pad_nd_x8,
    input, output);
}

enum xnn_status xnn_setup_constant_pad_nd_x16(
    xnn_operator_t constant_pad_op,
    const void* input,
    void* output)
{
  return setup_constant_pad_nd(
    constant_pad_op, xnn_operator_type_constant_pad_nd_x16,
    input, output);
}

enum xnn_status xnn_setup_constant_pad_nd_x32(
    xnn_operator_t constant_pad_op,
    const void* input,
    void* output)
{
  return setup_constant_pad_nd(
    constant_pad_op, xnn_operator_type_constant_pad_nd_x32,
    input, output);
}

enum xnn_status run_constant_pad_nd(
    uint32_t flags,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_paddings,
    const size_t* post_paddings,
    const void* input,
    void* output,
    uint32_t log2_element_size,
    const uint32_t padding_value,
    enum xnn_operator_type operator_type,
    pthreadpool_t threadpool)
{
  struct xnn_operator constant_pad_op;
  memset(&constant_pad_op, 0, sizeof(constant_pad_op));

  const struct xnn_xx_fill_config* fill_config = xnn_init_xx_fill_config();
  if (fill_config == NULL) {
    xnn_log_error(
      "failed to create fill operator: unsupported hardware configuration");
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_xx_pad_config* pad_config = xnn_init_xx_pad_config();
  if (pad_config == NULL) {
    xnn_log_error(
      "failed to create pad operator: unsupported hardware configuration");
    return xnn_status_unsupported_hardware;
  }

  init_constant_pad_nd(
      padding_value,
      flags,
      operator_type,
      fill_config,
      pad_config,
      &constant_pad_op);

  enum xnn_status status = reshape_constant_pad_nd(
    &constant_pad_op, operator_type,
    num_dims, input_shape, pre_paddings, post_paddings,
    log2_element_size,
    threadpool);

  if (status != xnn_status_success) {
    return status;
  }

  status = setup_constant_pad_nd(
    &constant_pad_op, operator_type,
    input, output);

  if (status != xnn_status_success) {
    return status;
  }

  return xnn_run_operator(&constant_pad_op, threadpool);
}

enum xnn_status xnn_run_constant_pad_nd_x8(
    uint32_t flags,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_paddings,
    const size_t* post_paddings,
    const void* input,
    void* output,
    const void* padding_value,
    pthreadpool_t threadpool)
{
  const uint32_t padding_pattern = *((const uint8_t*) padding_value);
  return run_constant_pad_nd(
    flags,
    num_dims, input_shape, pre_paddings, post_paddings,
    input, output, /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    padding_pattern * UINT32_C(0x01010101),
    xnn_operator_type_constant_pad_nd_x32,
    threadpool);
}

enum xnn_status xnn_run_constant_pad_nd_x16(
    uint32_t flags,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_paddings,
    const size_t* post_paddings,
    const void* input,
    void* output,
    const void* padding_value,
    pthreadpool_t threadpool)
{
  const uint32_t padding_pattern = *((const uint16_t*) padding_value);
  return run_constant_pad_nd(
    flags,
    num_dims, input_shape, pre_paddings, post_paddings,
    input, output, /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT16_T,
    padding_pattern * UINT32_C(0x00010001),
    xnn_operator_type_constant_pad_nd_x32,
    threadpool);
}

enum xnn_status xnn_run_constant_pad_nd_x32(
    uint32_t flags,
    size_t num_dims,
    const size_t* input_shape,
    const size_t* pre_paddings,
    const size_t* post_paddings,
    const void* input,
    void* output,
    const void* padding_value,
    pthreadpool_t threadpool)
{
  const uint32_t padding_pattern = *((const uint32_t*) padding_value);
  return run_constant_pad_nd(
    flags,
    num_dims, input_shape, pre_paddings, post_paddings,
    input, output, /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT32_T,
    padding_pattern,
    xnn_operator_type_constant_pad_nd_x32,
    threadpool);
}
