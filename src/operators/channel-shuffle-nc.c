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
#include <xnnpack/operator.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>


static enum xnn_status create_channel_shuffle_nc(
  size_t groups,
  size_t group_channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  enum xnn_operator_type operator_type,
  xnn_operator_t* channel_shuffle_op_out)
{
  xnn_operator_t channel_shuffle_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Channel Shuffle operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (groups <= 1) {
    xnn_log_error(
      "failed to create Channel Shuffle operator with %zu groups: at least two groups required", groups);
    goto error;
  }

  if (group_channels == 0) {
    xnn_log_error(
      "failed to create Channel Shuffle operator with %zu group channels: number of group channels must be non-zero",
      group_channels);
    goto error;
  }

  const size_t channels = groups * group_channels;
  if (input_stride < channels) {
    xnn_log_error(
      "failed to create Channel Shuffle operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zux%zu)",
      input_stride, groups, group_channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create Channel Shuffle operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zux%zu)",
      output_stride, groups, group_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  channel_shuffle_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (channel_shuffle_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Channel Shuffle operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  channel_shuffle_op->groups = groups;
  channel_shuffle_op->group_channels = group_channels;
  channel_shuffle_op->input_pixel_stride = input_stride;
  channel_shuffle_op->output_pixel_stride = output_stride;

  channel_shuffle_op->type = operator_type;
  channel_shuffle_op->ukernel.type = xnn_ukernel_type_channel_shuffle;

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
  return create_channel_shuffle_nc(
    groups,
    group_channels,
    input_stride,
    output_stride,
    flags,
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
  return create_channel_shuffle_nc(
    groups,
    group_channels,
    input_stride,
    output_stride,
    flags,
    xnn_operator_type_channel_shuffle_nc_x32,
    channel_shuffle_op_out);
}

static enum xnn_status setup_channel_shuffle_nc(
    xnn_operator_t channel_shuffle_op,
    size_t batch_size,
    const void* input,
    void* output,
    uint32_t log2_element_size,
    const struct zip_parameters zip[restrict XNN_MIN_ELEMENTS(1)])
{
  channel_shuffle_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Channel Shuffle operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    channel_shuffle_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  channel_shuffle_op->batch_size = batch_size;
  channel_shuffle_op->input = input;
  channel_shuffle_op->output = output;

  const size_t groups = channel_shuffle_op->groups;
  channel_shuffle_op->context.channel_shuffle = (struct channel_shuffle_context) {
    .x = input,
    .x_stride = channel_shuffle_op->input_pixel_stride << log2_element_size,
    .y = output,
    .y_stride = channel_shuffle_op->output_pixel_stride << log2_element_size,
    .n = channel_shuffle_op->group_channels << log2_element_size,
    .m = groups,
  };
  channel_shuffle_op->compute.type = xnn_parallelization_type_1d;
  channel_shuffle_op->compute.range[0] = batch_size;
  switch (groups) {
    case 2:
      channel_shuffle_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_fixed;
      channel_shuffle_op->context.channel_shuffle.fixed_ukernel = zip->x2;
      break;
    case 3:
      channel_shuffle_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_fixed;
      channel_shuffle_op->context.channel_shuffle.fixed_ukernel = zip->x3;
      break;
    case 4:
      channel_shuffle_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_fixed;
      channel_shuffle_op->context.channel_shuffle.fixed_ukernel = zip->x4;
      break;
    default:
      channel_shuffle_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_channel_shuffle_variable;
      channel_shuffle_op->context.channel_shuffle.variable_ukernel = zip->xm;
      break;
    case 0:
    case 1:
      XNN_UNREACHABLE;
  }
  channel_shuffle_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_channel_shuffle_nc_x8(
    xnn_operator_t channel_shuffle_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (channel_shuffle_op->type != xnn_operator_type_channel_shuffle_nc_x8) {
    xnn_log_error("failed to setup Channel Shuffle (NC, X8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_channel_shuffle_nc(
    channel_shuffle_op,
    batch_size,
    input,
    output,
    0 /* log2(sizeof(element)) = log2(sizeof(uint8_t)) */,
    &xnn_params.x8.zip);
}

enum xnn_status xnn_setup_channel_shuffle_nc_x32(
    xnn_operator_t channel_shuffle_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (channel_shuffle_op->type != xnn_operator_type_channel_shuffle_nc_x32) {
    xnn_log_error("failed to setup Channel Shuffle (NC, X32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_channel_shuffle_nc(
    channel_shuffle_op,
    batch_size,
    input,
    output,
    2 /* log2(sizeof(element)) = log2(sizeof(uint32_t)) */,
    &xnn_params.x32.zip);
}
