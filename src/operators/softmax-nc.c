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
#include <xnnpack/params-init.h>


enum xnn_status xnn_create_softmax_nc_q8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint32_t flags,
    xnn_operator_t* softmax_op_out)
{
  xnn_operator_t softmax_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create SoftMax operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create SoftMax operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create SoftMax operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create SoftMax operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create SoftMax operator with %.7g input scale: scale must be finite, normalized, and positive",
      input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create SoftMax operator with %.7g output scale: scale must be finite, normalized, and positive",
      output_scale);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  if (output_scale != 0x1.0p-8f) {
    xnn_log_error(
      "failed to create SoftMax operator with %.7g output scale: only output scale of 1/256 is supported",
      output_scale);
    goto error;
  }

  if (output_zero_point != 0) {
    xnn_log_error(
      "failed to create SoftMax operator with %" PRIu8 " output zero point: "
      "only output zero point of 0 is supported",
      output_zero_point);
    goto error;
  }

  status = xnn_status_out_of_memory;

  softmax_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (softmax_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for SoftMax operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  softmax_op->lookup_table = xnn_allocate_simd_memory(256 * sizeof(uint32_t));
  if (softmax_op->lookup_table == NULL) {
    xnn_log_error("failed to allocate 256 bytes for SoftMax lookup table");
    goto error;
  }

  uint32_t* lookup_table = softmax_op->lookup_table;
  const double qscale = fmin(((double) UINT32_MAX) / (double) channels, 8388607.0);
  for (int32_t i = 0; i < 256; i++) {
    const double scaled_exp_xi = qscale * exp((double) (i - 255) * (double) input_scale);
    lookup_table[(uint32_t) i] = (uint32_t) lrint(scaled_exp_xi);
  }

  softmax_op->channels = channels;
  softmax_op->input_pixel_stride = input_stride;
  softmax_op->output_pixel_stride = output_stride;

  softmax_op->type = xnn_operator_type_softmax_nc_q8;
  softmax_op->ukernel.type = xnn_ukernel_type_softmax;

  softmax_op->state = xnn_run_state_invalid;

  *softmax_op_out = softmax_op;
  return xnn_status_success;

error:
  xnn_delete_operator(softmax_op);
  return status;
}

enum xnn_status xnn_setup_softmax_nc_q8(
    xnn_operator_t softmax_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (softmax_op->type != xnn_operator_type_softmax_nc_q8) {
    xnn_log_error("failed to setup SoftMax (NC, Q8) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  softmax_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup SoftMax operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    softmax_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  softmax_op->batch_size = batch_size;
  softmax_op->input = input;
  softmax_op->output = output;

  softmax_op->context.u8_softmax = (struct u8_softmax_context) {
    .n = softmax_op->channels,
    .x = input,
    .x_stride = softmax_op->input_pixel_stride * sizeof(uint8_t),
    .t = softmax_op->lookup_table,
    .y = output,
    .y_stride = softmax_op->output_pixel_stride * sizeof(uint8_t),
    .rmax_ukernel = xnn_params.u8.rmax,
    .lut_norm_ukernel = xnn_params.u8.lut32norm,
  };
  softmax_op->compute.type = xnn_parallelization_type_1d;
  softmax_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_u8_softmax;
  softmax_op->compute.range[0] = batch_size;
  softmax_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_create_softmax_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* softmax_op_out)
{
  xnn_operator_t softmax_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create SoftMax operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create SoftMax operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create SoftMax operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create SoftMax operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      output_stride, channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  softmax_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (softmax_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for SoftMax operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  softmax_op->channels = channels;
  softmax_op->input_pixel_stride = input_stride;
  softmax_op->output_pixel_stride = output_stride;

  softmax_op->type = xnn_operator_type_softmax_nc_f32;
  softmax_op->ukernel.type = xnn_ukernel_type_softmax;

  softmax_op->state = xnn_run_state_invalid;

  *softmax_op_out = softmax_op;
  return xnn_status_success;

error:
  xnn_delete_operator(softmax_op);
  return status;
}

enum xnn_status xnn_setup_softmax_nc_f32(
    xnn_operator_t softmax_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (softmax_op->type != xnn_operator_type_softmax_nc_f32) {
    xnn_log_error("failed to setup SoftMax (NC, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }
  softmax_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup SoftMax operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    softmax_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  softmax_op->batch_size = batch_size;
  softmax_op->input = input;
  softmax_op->output = output;

  softmax_op->context.f32_three_pass_softmax = (struct f32_three_pass_softmax_context) {
    .n = softmax_op->channels * sizeof(float),
    .x = input,
    .x_stride = softmax_op->input_pixel_stride * sizeof(float),
    .y = output,
    .y_stride = softmax_op->output_pixel_stride * sizeof(float),
    .rmax_ukernel = xnn_params.f32.rmax,
    .raddstoreexpminusmax_ukernel = xnn_params.f32.raddstoreexpminusmax,
    .vmulc_ukernel = xnn_params.f32.vmul.opc_ukernel,
    .params = xnn_init_f32_minmax_params(-INFINITY, INFINITY),
  };
  softmax_op->compute.type = xnn_parallelization_type_1d;
  softmax_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_f32_three_pass_softmax;
  softmax_op->compute.range[0] = batch_size;
  softmax_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
