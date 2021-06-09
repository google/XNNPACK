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
    xnn_univector_ukernel_function ukernel,
    xnn_operator_t* unary_elementwise_op_out)
{
  xnn_operator_t unary_elementwise_op = NULL;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
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

  unary_elementwise_op->ukernel.vunary.function = ukernel;
  unary_elementwise_op->type = operator_type;
  unary_elementwise_op->flags = flags;

  unary_elementwise_op->state = xnn_run_state_invalid;

  *unary_elementwise_op_out = unary_elementwise_op;
  return xnn_status_success;
}

static enum xnn_status setup_unary_elementwise_nc(
    xnn_operator_t unary_elementwise_op,
    size_t batch_size,
    const void* input,
    void* output,
    uint32_t log2_element_size,
    const void* params,
    size_t params_size)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
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

  xnn_univector_ukernel_function ukernel = unary_elementwise_op->ukernel.vunary.function;

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

  union xnn_u8_minmax_params params;
  xnn_init_u8_minmax_params(&params, output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_u8,
    xnn_params.u8.clamp,
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

  const bool relu_activation = (output_max == INFINITY) && (output_min == 0.0f);
  xnn_univector_ukernel_function clamp_ukernel = (relu_activation && (xnn_params.f32.relu != NULL)) ?
    xnn_params.f32.relu : xnn_params.f32.clamp;

  union xnn_f32_minmax_params params;
  xnn_init_f32_minmax_params(&params, output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_f32,
    clamp_ukernel,
    clamp_op_out);
}

enum xnn_status xnn_create_abs_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* abs_op_out)
{
  union xnn_f32_abs_params params;
  xnn_init_f32_abs_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_abs_nc_f32,
    xnn_params.f32.abs,
    abs_op_out);
}

enum xnn_status xnn_create_bankers_rounding_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* rounding_op_out)
{
  union xnn_f32_rnd_params params;
  xnn_init_f32_rnd_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_bankers_rounding_nc_f32,
    xnn_params.f32.rndne,
    rounding_op_out);
}

enum xnn_status xnn_create_ceiling_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* ceiling_op_out)
{
  union xnn_f32_rnd_params params;
  xnn_init_f32_rnd_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_ceiling_nc_f32,
    xnn_params.f32.rndu,
    ceiling_op_out);
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
    xnn_params.xx.copy,
    copy_op_out);
}

enum xnn_status xnn_create_elu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float alpha,
  uint32_t flags,
  xnn_operator_t* elu_op_out)
{
  if (alpha <= 0.0f || !isnormal(alpha)) {
    xnn_log_error(
      "failed to create %s operator with %.7g alpha parameter: alpha must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_elu_nc_f32), alpha);
    return xnn_status_invalid_parameter;
  }

  union xnn_f32_elu_params params;
  xnn_init_f32_elu_params(&params, 1.0f /* prescale */, alpha, 1.0f /* beta */);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_elu_nc_f32,
    xnn_params.f32.elu,
    elu_op_out);
}

enum xnn_status xnn_create_floor_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* floor_op_out)
{
  union xnn_f32_rnd_params params;
  xnn_init_f32_rnd_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_floor_nc_f32,
    xnn_params.f32.rndd,
    floor_op_out);
}

enum xnn_status xnn_create_hardswish_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* hardswish_op_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_hardswish_nc_f16));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & XNN_INIT_FLAG_F16) != XNN_INIT_FLAG_F16) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(xnn_operator_type_hardswish_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  struct xnn_f16_hswish_params params;
  xnn_init_f16_hswish_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_hardswish_nc_f16,
    xnn_params.f16.hswish,
    hardswish_op_out);
}

enum xnn_status xnn_create_hardswish_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* hardswish_op_out)
{
  union xnn_f32_hswish_params params;
  xnn_init_f32_hswish_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_hardswish_nc_f32,
    xnn_params.f32.hswish,
    hardswish_op_out);
}

enum xnn_status xnn_create_leaky_relu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out)
{
  if (!isfinite(negative_slope)) {
    xnn_log_error(
      "failed to create %s operator with %f negative slope: finite number expected",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_f32),
      negative_slope);
    return xnn_status_invalid_parameter;
  }

  union xnn_f32_lrelu_params params;
  xnn_init_f32_lrelu_params(&params, negative_slope);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_leaky_relu_nc_f32,
    xnn_params.f32.lrelu,
    leaky_relu_op_out);
}

enum xnn_status xnn_create_negate_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* negate_op_out)
{
  union xnn_f32_neg_params params;
  xnn_init_f32_neg_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_negate_nc_f32,
    xnn_params.f32.neg,
    negate_op_out);
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
    xnn_params.f32.sigmoid,
    sigmoid_op_out);
}

enum xnn_status xnn_create_square_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* square_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_square_nc_f32,
    xnn_params.f32.sqr,
    square_op_out);
}

enum xnn_status xnn_create_square_root_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* sqrt_op_out)
{
  union xnn_f32_sqrt_params params;
  xnn_init_f32_sqrt_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_square_root_nc_f32,
    xnn_params.f32.sqrt,
    sqrt_op_out);
}

enum xnn_status xnn_create_truncation_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* truncation_op_out)
{
  union xnn_f32_rnd_params params;
  xnn_init_f32_rnd_params(&params);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_truncation_nc_f32,
    xnn_params.f32.rndz,
    truncation_op_out);
}

enum xnn_status xnn_setup_abs_nc_f32(
    xnn_operator_t abs_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (abs_op->type != xnn_operator_type_abs_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_abs_nc_f32),
      xnn_operator_type_to_string(abs_op->type));
    return xnn_status_invalid_parameter;
  }
  abs_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    abs_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &abs_op->params.f32_abs, sizeof(abs_op->params.f32_abs));
}

enum xnn_status xnn_setup_bankers_rounding_nc_f32(
    xnn_operator_t rounding_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (rounding_op->type != xnn_operator_type_bankers_rounding_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_bankers_rounding_nc_f32),
      xnn_operator_type_to_string(rounding_op->type));
    return xnn_status_invalid_parameter;
  }
  rounding_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    rounding_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &rounding_op->params.f32_rnd, sizeof(rounding_op->params.f32_rnd));
}

enum xnn_status xnn_setup_ceiling_nc_f32(
    xnn_operator_t ceiling_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (ceiling_op->type != xnn_operator_type_ceiling_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_ceiling_nc_f32),
      xnn_operator_type_to_string(ceiling_op->type));
    return xnn_status_invalid_parameter;
  }
  ceiling_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    ceiling_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &ceiling_op->params.f32_rnd, sizeof(ceiling_op->params.f32_rnd));
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
    2 /* log2(sizeof(float)) */,
    &clamp_op->params.f32_minmax, sizeof(clamp_op->params.f32_minmax));
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
    2 /* log2(sizeof(uint32_t)) */,
    NULL, 0);
}

enum xnn_status xnn_setup_elu_nc_f32(
    xnn_operator_t elu_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (elu_op->type != xnn_operator_type_elu_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_elu_nc_f32),
      xnn_operator_type_to_string(elu_op->type));
    return xnn_status_invalid_parameter;
  }
  elu_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    elu_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &elu_op->params.f32_elu, sizeof(elu_op->params.f32_elu));
}

enum xnn_status xnn_setup_floor_nc_f32(
    xnn_operator_t floor_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (floor_op->type != xnn_operator_type_floor_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_floor_nc_f32),
      xnn_operator_type_to_string(floor_op->type));
    return xnn_status_invalid_parameter;
  }
  floor_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    floor_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &floor_op->params.f32_rnd, sizeof(floor_op->params.f32_rnd));
}

enum xnn_status xnn_setup_hardswish_nc_f16(
    xnn_operator_t hardswish_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (hardswish_op->type != xnn_operator_type_hardswish_nc_f16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_hardswish_nc_f16),
      xnn_operator_type_to_string(hardswish_op->type));
    return xnn_status_invalid_parameter;
  }
  hardswish_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    hardswish_op,
    batch_size, input, output,
    1 /* log2(sizeof(half)) */,
    &hardswish_op->params.f16_hswish, sizeof(hardswish_op->params.f16_hswish));
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
    2 /* log2(sizeof(float)) */,
    &hardswish_op->params.f32_hswish, sizeof(hardswish_op->params.f32_hswish));
}

enum xnn_status xnn_setup_leaky_relu_nc_f32(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool)
{
  if (leaky_relu_op->type != xnn_operator_type_leaky_relu_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_f32),
      xnn_operator_type_to_string(leaky_relu_op->type));
    return xnn_status_invalid_parameter;
  }
  leaky_relu_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    leaky_relu_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &leaky_relu_op->params.f32_lrelu, sizeof(leaky_relu_op->params.f32_lrelu));
}

enum xnn_status xnn_setup_negate_nc_f32(
    xnn_operator_t negate_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (negate_op->type != xnn_operator_type_negate_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_negate_nc_f32),
      xnn_operator_type_to_string(negate_op->type));
    return xnn_status_invalid_parameter;
  }
  negate_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    negate_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &negate_op->params.f32_neg, sizeof(negate_op->params.f32_neg));
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
    2 /* log2(sizeof(float)) */,
    NULL, 0);
}

enum xnn_status xnn_setup_square_nc_f32(
    xnn_operator_t square_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (square_op->type != xnn_operator_type_square_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_square_nc_f32),
      xnn_operator_type_to_string(square_op->type));
    return xnn_status_invalid_parameter;
  }
  square_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    square_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    NULL, 0);
}

enum xnn_status xnn_setup_square_root_nc_f32(
    xnn_operator_t sqrt_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (sqrt_op->type != xnn_operator_type_square_root_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_square_root_nc_f32),
      xnn_operator_type_to_string(sqrt_op->type));
    return xnn_status_invalid_parameter;
  }
  sqrt_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    sqrt_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    NULL, 0);
}

enum xnn_status xnn_setup_truncation_nc_f32(
    xnn_operator_t truncation_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (truncation_op->type != xnn_operator_type_truncation_nc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_truncation_nc_f32),
      xnn_operator_type_to_string(truncation_op->type));
    return xnn_status_invalid_parameter;
  }
  truncation_op->state = xnn_run_state_invalid;

  return setup_unary_elementwise_nc(
    truncation_op,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    &truncation_op->params.f32_rnd, sizeof(truncation_op->params.f32_rnd));
}