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


enum xnn_status xnn_create_softargmax_nc_q8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint32_t flags,
    xnn_operator_t* softargmax_op_out)
{
  xnn_operator_t softargmax_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create SoftArgMax operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create SoftArgMax operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create Sigmoid operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create Sigmoid operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create SoftArgMax operator with %.7g input scale: scale must be finite, normalized, and positive",
      input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create SoftArgMax operator with %.7g output scale: scale must be finite, normalized, and positive",
      output_scale);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  if (output_scale != 0x1.0p-8f) {
    xnn_log_error(
      "failed to create SoftArgMax operator with %.7g output scale: only output scale of 1/256 is supported",
      output_scale);
    goto error;
  }

  if (output_zero_point != 0) {
    xnn_log_error(
      "failed to create SoftArgMax operator with %" PRIu8 " output zero point: "
      "only output zero point of 0 is supported",
      output_zero_point);
    goto error;
  }

  status = xnn_status_out_of_memory;

  softargmax_op = xnn_allocate_zero_memory(sizeof(struct xnn_operator));
  if (softargmax_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for SoftArgMax operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  softargmax_op->lookup_table = xnn_allocate_memory(256 * sizeof(uint32_t));
  if (softargmax_op->lookup_table == NULL) {
    xnn_log_error("failed to allocate 256 bytes for SoftArgMax lookup table");
    goto error;
  }

  uint32_t* lookup_table = softargmax_op->lookup_table;
  const double qscale = fmin(((double) UINT32_MAX) / (double) channels, 8388607.0);
  for (int32_t i = 0; i < 256; i++) {
    const double scaled_exp_xi = qscale * exp((double) (i - 255) * (double) input_scale);
    lookup_table[(uint32_t) i] = (uint32_t) lrint(scaled_exp_xi);
  }

  softargmax_op->channels = channels;
  softargmax_op->input_pixel_stride = input_stride;
  softargmax_op->output_pixel_stride = output_stride;

  softargmax_op->type = xnn_operator_type_softargmax_nc_q8;
  softargmax_op->ukernel.type = xnn_ukernel_type_softargmax;

  softargmax_op->state = xnn_run_state_invalid;

  *softargmax_op_out = softargmax_op;
  return xnn_status_success;

error:
  xnn_delete_operator(softargmax_op);
  return status;
}

enum xnn_status xnn_setup_softargmax_nc_q8(
    xnn_operator_t softargmax_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (softargmax_op->type != xnn_operator_type_softargmax_nc_q8) {
    xnn_log_error("failed to setup SoftArgMax (NC, Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  softargmax_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup SoftArgMax operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    softargmax_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  softargmax_op->batch_size = batch_size;
  softargmax_op->input = input;
  softargmax_op->output = output;

  softargmax_op->context.u8_softargmax = (struct u8_softargmax_context) {
    .n = softargmax_op->channels,
    .x = input,
    .x_stride = softargmax_op->input_pixel_stride * sizeof(uint8_t),
    .t = softargmax_op->lookup_table,
    .y = output,
    .y_stride = softargmax_op->output_pixel_stride * sizeof(uint8_t),
    .rmax_ukernel = xnn_params.u8.rmax,
    .lut_norm_ukernel = xnn_params.u8.lut32norm,
  };
  softargmax_op->compute.type = xnn_parallelization_type_1d;
  softargmax_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_u8_softargmax;
  softargmax_op->compute.range[0] = batch_size;
  softargmax_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
