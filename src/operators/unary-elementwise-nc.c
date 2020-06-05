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


static enum xnn_status create_unary_elementwise_nc(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    xnn_operator_t* unary_elementwise_op_out)
{
  xnn_operator_t unary_elementwise_op = NULL;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  unary_elementwise_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (unary_elementwise_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }

  unary_elementwise_op->channels = channels;
  unary_elementwise_op->input_pixel_stride = input_stride;
  unary_elementwise_op->output_pixel_stride = output_stride;
  if (params_size != 0) {
    memcpy(&unary_elementwise_op->params, params, params_size);
  }

  unary_elementwise_op->type = operator_type;
  unary_elementwise_op->ukernel.type = xnn_ukernel_type_unary_elementwise;

  unary_elementwise_op->state = xnn_run_state_invalid;

  *unary_elementwise_op_out = unary_elementwise_op;
  return xnn_status_success;
}

static enum xnn_status setup_unary_elementwise_nc(
    xnn_operator_t unary_elementwise_op,
    size_t batch_size,
    const void* input,
    void* output,
    xnn_univector_ukernel_function ukernel,
    uint32_t log2_element_size,
    const void* params,
    size_t params_size)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(unary_elementwise_op->type));
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    unary_elementwise_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = unary_elementwise_op->channels;
  const size_t input_stride = unary_elementwise_op->input_pixel_stride;
  const size_t output_stride = unary_elementwise_op->output_pixel_stride;
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    unary_elementwise_op->context.univector_contiguous = (struct univector_contiguous_context) {
      .x = input,
      .x_stride = input_stride << log2_element_size,
      .y = output,
      .y_stride = output_stride << log2_element_size,
      .ukernel = ukernel,
    };
    if (params_size != 0) {
      memcpy(&unary_elementwise_op->context.univector_contiguous.params, params, params_size);
    }
    unary_elementwise_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    unary_elementwise_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
    unary_elementwise_op->compute.range[0] = (batch_size * channels) << log2_element_size;
    unary_elementwise_op->compute.tile[0] = block_size;
  } else {
    unary_elementwise_op->context.univector_strided = (struct univector_strided_context) {
      .n = channels << log2_element_size,
      .x = input,
      .x_stride = input_stride << log2_element_size,
      .y = output,
      .y_stride = output_stride << log2_element_size,
      .ukernel = ukernel,
    };
    if (params_size != 0) {
      memcpy(&unary_elementwise_op->context.univector_strided.params, params, params_size);
    }
    unary_elementwise_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    unary_elementwise_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
    unary_elementwise_op->compute.range[0] = batch_size;
    unary_elementwise_op->compute.tile[0] = 1;
  }
  unary_elementwise_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_create_clamp_nc_u8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* clamp_op_out)
{
  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_u8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const union xnn_u8_minmax_params params = xnn_init_u8_minmax_params(output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_u8,
    clamp_op_out);
}

enum xnn_status xnn_create_clamp_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* clamp_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const union xnn_f32_minmax_params params = xnn_init_f32_minmax_params(output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_f32,
    clamp_op_out);
}

enum xnn_status xnn_create_copy_nc_x32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* copy_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_copy_nc_x32,
    copy_op_out);
}

enum xnn_status xnn_create_hardswish_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* hardswish_op_out)
{
  const union xnn_f32_hswish_params params = xnn_init_f32_hswish_params();
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_hardswish_nc_f32,
    hardswish_op_out);
}

enum xnn_status xnn_create_sigmoid_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* sigmoid_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_sigmoid_nc_f32,
    sigmoid_op_out);
}

enum xnn_status xnn_setup_clamp_nc_u8(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (clamp_op->type != xnn_operator_type_clamp_nc_u8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_u8),
      xnn_operator_type_to_string(clamp_op->type));
    return xnn_status_invalid_parameter;
  }
  clamp_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    clamp_op,
    batch_size, input, output,
    xnn_params.u8.clamp,
    0 /* log2(sizeof(uint8_t)) */,
    &clamp_op->params.u8_minmax, sizeof(clamp_op->params.u8_minmax));
}

enum xnn_status xnn_setup_clamp_nc_f32(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (clamp_op->type != xnn_operator_type_clamp_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32),
      xnn_operator_type_to_string(clamp_op->type));
    return xnn_status_invalid_parameter;
  }
  clamp_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    clamp_op,
    batch_size, input, output,
    xnn_params.f32.clamp,
    2 /* log2(sizeof(float)) */,
    &clamp_op->params.f32_minmax, sizeof(clamp_op->params.f32_minmax));
}

static void memcpy_ukernel(size_t size, const void* input, void* output, const void* params) {
  memcpy(output, input, size);
}

enum xnn_status xnn_setup_copy_nc_x32(
    xnn_operator_t copy_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (copy_op->type != xnn_operator_type_copy_nc_x32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_copy_nc_x32),
      xnn_operator_type_to_string(copy_op->type));
    return xnn_status_invalid_parameter;
  }
  copy_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    copy_op,
    batch_size, input, output,
    memcpy_ukernel,
    2 /* log2(sizeof(uint32_t)) */,
    NULL, 0);
}

enum xnn_status xnn_setup_hardswish_nc_f32(
    xnn_operator_t hardswish_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (hardswish_op->type != xnn_operator_type_hardswish_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_hardswish_nc_f32),
      xnn_operator_type_to_string(hardswish_op->type));
    return xnn_status_invalid_parameter;
  }
  hardswish_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    hardswish_op,
    batch_size, input, output,
    xnn_params.f32.hswish,
    2 /* log2(sizeof(float)) */,
    &hardswish_op->params.f32_hswish, sizeof(hardswish_op->params.f32_hswish));
}

enum xnn_status xnn_setup_sigmoid_nc_f32(
    xnn_operator_t sigmoid_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (sigmoid_op->type != xnn_operator_type_sigmoid_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_sigmoid_nc_f32),
      xnn_operator_type_to_string(sigmoid_op->type));
    return xnn_status_invalid_parameter;
  }
  sigmoid_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    sigmoid_op,
    batch_size, input, output,
    xnn_params.f32.sigmoid,
    2 /* log2(sizeof(float)) */,
    NULL, 0);
}
