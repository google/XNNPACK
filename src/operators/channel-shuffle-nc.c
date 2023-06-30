// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
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
#include <xnnpack/config.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>


static enum xnn_status create_channel_shuffle_nc(
  size_t groups,
  size_t group_channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  const struct xnn_zip_config* zip_config,
  enum xnn_operator_type operator_type,
  xnn_operator_t* channel_shuffle_op_out)
{
  xnn_operator_t channel_shuffle_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (groups <= 1) {
    xnn_log_error(
      "failed to create %s operator with %zu groups: at least two groups required",
      xnn_operator_type_to_string(operator_type), groups);
    goto error;
  }

  if (group_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu group channels: number of group channels must be non-zero",
      xnn_operator_type_to_string(operator_type), group_channels);
    goto error;
  }

  const size_t channels = groups * group_channels;
  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zux%zu)",
      xnn_operator_type_to_string(operator_type), input_stride, groups, group_channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zux%zu)",
      xnn_operator_type_to_string(operator_type), output_stride, groups, group_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  channel_shuffle_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (channel_shuffle_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  channel_shuffle_op->groups = groups;
  channel_shuffle_op->group_channels = group_channels;
  channel_shuffle_op->input_pixel_stride = input_stride;
  channel_shuffle_op->output_pixel_stride = output_stride;

  channel_shuffle_op->type = operator_type;
  channel_shuffle_op->flags = flags;
  channel_shuffle_op->zip_config = zip_config;

  channel_shuffle_op->state = xnn_run_state_invalid;

  *channel_shuffle_op_out = channel_shuffle_op;
  return xnn_status_success;

error:
  xnn_delete_operator(channel_shuffle_op);
  return status;
}


enum xnn_status xnn_create_channel_shuffle_nc_x8(
    size_t groups,
    size_t group_channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* channel_shuffle_op_out)
{
  const struct xnn_zip_config* zip_config = xnn_init_x8_zip_config();
  assert(zip_config != NULL);
  return create_channel_shuffle_nc(
    groups,
    group_channels,
    input_stride,
    output_stride,
    flags,
    zip_config,
    xnn_operator_type_channel_shuffle_nc_x8,
    channel_shuffle_op_out);
}

enum xnn_status xnn_create_channel_shuffle_nc_x32(
    size_t groups,
    size_t group_channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* channel_shuffle_op_out)
{
  const struct xnn_zip_config* zip_config = xnn_init_x32_zip_config();
  if (zip_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_channel_shuffle_nc_x32));
    return xnn_status_unsupported_hardware;
  }
  return create_channel_shuffle_nc(
    groups,
    group_channels,
    input_stride,
    output_stride,
    flags,
    zip_config,
    xnn_operator_type_channel_shuffle_nc_x32,
    channel_shuffle_op_out);
}

static enum xnn_status reshape_channel_shuffle_nc(
    xnn_operator_t channel_shuffle_op,
    size_t batch_size,
    uint32_t log2_element_size,
    const struct xnn_zip_config zip[restrict XNN_MIN_ELEMENTS(1)])
{
  channel_shuffle_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(channel_shuffle_op->type));
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    channel_shuffle_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  channel_shuffle_op->batch_size = batch_size;

  const size_t groups = channel_shuffle_op->groups;
  channel_shuffle_op->context.channel_shuffle = (struct channel_shuffle_context) {
    .x_stride = channel_shuffle_op->input_pixel_stride << log2_element_size,
    .y_stride = channel_shuffle_op->output_pixel_stride << log2_element_size,
    .n = channel_shuffle_op->group_channels << log2_element_size,
    .m = groups,
  };
  channel_shuffle_op->compute[0].type = xnn_parallelization_type_1d;
  channel_shuffle_op->compute[0].range[0] = batch_size;
  switch (groups) {
    case 2:
      channel_shuffle_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_fixed;
      channel_shuffle_op->context.channel_shuffle.fixed_ukernel = zip->x2;
      break;
    case 3:
      channel_shuffle_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_fixed;
      channel_shuffle_op->context.channel_shuffle.fixed_ukernel = zip->x3;
      break;
    case 4:
      channel_shuffle_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_fixed;
      channel_shuffle_op->context.channel_shuffle.fixed_ukernel = zip->x4;
      break;
    default:
      channel_shuffle_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_variable;
      channel_shuffle_op->context.channel_shuffle.variable_ukernel = zip->xm;
      break;
    case 0:
    case 1:
      XNN_UNREACHABLE;
  }
  channel_shuffle_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_channel_shuffle_nc_x8(
    xnn_operator_t channel_shuffle_op,
    size_t batch_size,
    pthreadpool_t threadpool)
{
  if (channel_shuffle_op->type != xnn_operator_type_channel_shuffle_nc_x8) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_channel_shuffle_nc_x8),
      xnn_operator_type_to_string(channel_shuffle_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_channel_shuffle_nc(
    channel_shuffle_op,
    batch_size,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    channel_shuffle_op->zip_config);
}

enum xnn_status xnn_reshape_channel_shuffle_nc_x32(
    xnn_operator_t channel_shuffle_op,
    size_t batch_size,
    pthreadpool_t threadpool)
{
  if (channel_shuffle_op->type != xnn_operator_type_channel_shuffle_nc_x32) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_channel_shuffle_nc_x32),
      xnn_operator_type_to_string(channel_shuffle_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_channel_shuffle_nc(
    channel_shuffle_op,
    batch_size,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT32_T,
    channel_shuffle_op->zip_config);
}

static enum xnn_status setup_channel_shuffle_nc(
    xnn_operator_t channel_shuffle_op,
    const void* input,
    void* output)
{
  switch (channel_shuffle_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(channel_shuffle_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  channel_shuffle_op->context.channel_shuffle.x = input;
  channel_shuffle_op->context.channel_shuffle.y = output;

  channel_shuffle_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_channel_shuffle_nc_x8(
    xnn_operator_t channel_shuffle_op,
    const void* input,
    void* output)
{
  if (channel_shuffle_op->type != xnn_operator_type_channel_shuffle_nc_x8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_channel_shuffle_nc_x8),
      xnn_operator_type_to_string(channel_shuffle_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_channel_shuffle_nc(
    channel_shuffle_op,
    input,
    output);
}

enum xnn_status xnn_setup_channel_shuffle_nc_x32(
    xnn_operator_t channel_shuffle_op,
    const void* input,
    void* output)
{
  if (channel_shuffle_op->type != xnn_operator_type_channel_shuffle_nc_x32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_channel_shuffle_nc_x32),
      xnn_operator_type_to_string(channel_shuffle_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_channel_shuffle_nc(
    channel_shuffle_op,
    input,
    output);
}
