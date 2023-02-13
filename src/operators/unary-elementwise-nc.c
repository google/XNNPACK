// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/microparams-init.h>


static void init_unary_elementwise_nc(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    xnn_vunary_ukernel_fn ukernel,
    xnn_operator_t unary_elementwise_op)
{
  assert(ukernel != NULL);
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
}

static enum xnn_status create_unary_elementwise_nc(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    const struct xnn_unary_elementwise_config* config,
    xnn_operator_t* unary_elementwise_op_out)
{
  xnn_operator_t unary_elementwise_op = NULL;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  if (config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  unary_elementwise_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (unary_elementwise_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }

  init_unary_elementwise_nc(
    channels,
    input_stride, output_stride,
    flags,
    params, params_size,
    operator_type,
    config->ukernel,
    unary_elementwise_op);

  *unary_elementwise_op_out = unary_elementwise_op;
  return xnn_status_success;
}

static bool is_copy_operator(enum xnn_operator_type operator_type) {
  switch (operator_type) {
    case xnn_operator_type_copy_nc_x8:
    case xnn_operator_type_copy_nc_x16:
    case xnn_operator_type_copy_nc_x32:
      return true;
    default:
      return false;
  }
}

static enum xnn_status setup_unary_elementwise_nc(
    xnn_operator_t unary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    const void* input,
    void* output,
    uint32_t log2_input_size,
    uint32_t log2_output_size,
    const void* params,
    size_t params_size,
    size_t num_threads)
{
  if (unary_elementwise_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(unary_elementwise_op->type));
    return xnn_status_invalid_parameter;
  }
  unary_elementwise_op->state = xnn_run_state_invalid;

  if (batch_size == 0 || (input == output && is_copy_operator(expected_operator_type))) {
    unary_elementwise_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = unary_elementwise_op->channels;
  const size_t input_stride = unary_elementwise_op->input_pixel_stride;
  const size_t output_stride = unary_elementwise_op->output_pixel_stride;

  xnn_vunary_ukernel_fn ukernel = unary_elementwise_op->ukernel.vunary.function;

  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;
    unary_elementwise_op->context.univector_contiguous = (struct univector_contiguous_context) {
      .x = input,
      .y = output,
      .log2_xsize = log2_input_size,
      .log2_ysize = log2_output_size,
      .ukernel = ukernel,
    };
    if (params_size != 0) {
      memcpy(&unary_elementwise_op->context.univector_contiguous.params, params, params_size);
    }

    const size_t range = (batch_size * channels) << log2_input_size;
    unary_elementwise_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    unary_elementwise_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
    unary_elementwise_op->compute.range[0] = range;
    unary_elementwise_op->compute.tile[0] = (num_threads == 1) ? range : block_size;
  } else {
    unary_elementwise_op->context.univector_strided = (struct univector_strided_context) {
      .n = channels << log2_input_size,
      .x = input,
      .x_stride = input_stride << log2_input_size,
      .y = output,
      .y_stride = output_stride << log2_output_size,
      .ukernel = ukernel,
    };
    if (params_size != 0) {
      memcpy(&unary_elementwise_op->context.univector_strided.params, params, params_size);
    }
    unary_elementwise_op->compute.type = xnn_parallelization_type_1d_tile_1d;
    unary_elementwise_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
    unary_elementwise_op->compute.range[0] = batch_size;
    unary_elementwise_op->compute.tile[0] = (num_threads == 1) ? batch_size : 1;
  }
  unary_elementwise_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_create_clamp_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* clamp_op_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f16));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & XNN_INIT_FLAG_F16) != XNN_INIT_FLAG_F16) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f16));
    return xnn_status_invalid_parameter;
  }

  const uint16_t output_min_as_half = fp16_ieee_from_fp32_value(output_min);
  const uint16_t output_max_as_half = fp16_ieee_from_fp32_value(output_max);
  output_min = fp16_ieee_to_fp32_value(output_min_as_half);
  output_max = fp16_ieee_to_fp32_value(output_max_as_half);
  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f16), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_unary_elementwise_config* f16_clamp_config = xnn_init_f16_clamp_config();
  if (f16_clamp_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params params;
  assert(f16_clamp_config->init.f16_minmax != NULL);
  f16_clamp_config->init.f16_minmax(&params, output_min_as_half, output_max_as_half);

  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_f16,
    f16_clamp_config,
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

  const struct xnn_unary_elementwise_config* f32_clamp_config = xnn_init_f32_clamp_config();
  const struct xnn_unary_elementwise_config* f32_relu_config = xnn_init_f32_relu_config();
  if (f32_clamp_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const bool relu_activation = (output_max == INFINITY) && (output_min == 0.0f);
  const struct xnn_unary_elementwise_config* config = f32_clamp_config;
  if (relu_activation && f32_relu_config != NULL && f32_relu_config->ukernel != NULL) {
    config = f32_relu_config;
  }

  union xnn_f32_minmax_params params;
  assert(f32_clamp_config->init.f32_minmax != NULL);
  f32_clamp_config->init.f32_minmax(&params, output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_f32,
    config,
    clamp_op_out);
}

enum xnn_status xnn_create_clamp_nc_s8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* clamp_op_out)
{
  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_clamp_nc_s8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* s8_clamp_config = xnn_init_s8_clamp_config();
  if (s8_clamp_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_clamp_nc_s8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_s8_minmax_params params;
  assert(s8_clamp_config->init.s8_minmax != NULL);
  s8_clamp_config->init.s8_minmax(&params, output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_s8,
    s8_clamp_config,
    clamp_op_out);
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
  const struct xnn_unary_elementwise_config* u8_clamp_config = xnn_init_u8_clamp_config();
  if (u8_clamp_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_clamp_nc_u8));
    return xnn_status_unsupported_hardware;
  }

  union xnn_u8_minmax_params params;
  assert(u8_clamp_config->init.u8_minmax != NULL);
  u8_clamp_config->init.u8_minmax(&params, output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_clamp_nc_u8,
    u8_clamp_config,
    clamp_op_out);
}

enum xnn_status xnn_create_abs_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* abs_op_out)
{
  const struct xnn_unary_elementwise_config* f16_abs_config = xnn_init_f16_abs_config();
  if (f16_abs_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_abs_nc_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f16_abs_params params;
  if (f16_abs_config->init.f16_abs != NULL) {
    f16_abs_config->init.f16_abs(&params);
  }
  return create_unary_elementwise_nc(
      channels, input_stride, output_stride, flags, &params, sizeof(params),
      xnn_operator_type_abs_nc_f16, f16_abs_config,
      abs_op_out);
}

enum xnn_status xnn_create_abs_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* abs_op_out)
{
  const struct xnn_unary_elementwise_config* f32_abs_config = xnn_init_f32_abs_config();
  if (f32_abs_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_abs_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_abs_params params;
  if (f32_abs_config->init.f32_abs != NULL) {
    f32_abs_config->init.f32_abs(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_abs_nc_f32,
    f32_abs_config,
    abs_op_out);
}

enum xnn_status xnn_create_bankers_rounding_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* rounding_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_bankers_rounding_nc_f16,
    xnn_init_f16_rndne_config(),
    rounding_op_out);
}

enum xnn_status xnn_create_bankers_rounding_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* rounding_op_out)
{
  const struct xnn_unary_elementwise_config* f32_rndne_config = xnn_init_f32_rndne_config();
  if (f32_rndne_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_bankers_rounding_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndne_config->init.f32_rnd != NULL) {
    f32_rndne_config->init.f32_rnd(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_bankers_rounding_nc_f32,
    f32_rndne_config,
    rounding_op_out);
}

enum xnn_status xnn_create_ceiling_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* ceiling_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_ceiling_nc_f16,
    xnn_init_f16_rndu_config(),
    ceiling_op_out);
}

enum xnn_status xnn_create_ceiling_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* ceiling_op_out)
{
  const struct xnn_unary_elementwise_config* f32_rndu_config = xnn_init_f32_rndu_config();
  if (f32_rndu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_ceiling_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndu_config->init.f32_rnd != NULL) {
    f32_rndu_config->init.f32_rnd(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_ceiling_nc_f32,
    f32_rndu_config,
    ceiling_op_out);
}

enum xnn_status xnn_create_convert_nc_f16_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  const struct xnn_unary_elementwise_config* f16_to_f32_cvt_config = xnn_init_f16_to_f32_cvt_config();
  if (f16_to_f32_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f16_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_f32_cvt_params params;
  if (f16_to_f32_cvt_config->init.f16_f32_cvt != NULL) {
    f16_to_f32_cvt_config->init.f16_f32_cvt(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_f16_f32,
    f16_to_f32_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_convert_nc_f32_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  const struct xnn_unary_elementwise_config* f32_to_f16_cvt_config = xnn_init_f32_to_f16_cvt_config();
  if (f32_to_f16_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_f16_cvt_params params;
  if (f32_to_f16_cvt_config->init.f32_f16_cvt != NULL) {
    f32_to_f16_cvt_config->init.f32_f16_cvt(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_f32_f16,
    f32_to_f16_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_convert_nc_f32_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* f32_to_qs8_cvt_config = xnn_init_f32_to_qs8_cvt_config();
  if (f32_to_qs8_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qs8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_qs8_cvt_params params;
  assert(f32_to_qs8_cvt_config->init.f32_qs8_cvt != NULL);
  f32_to_qs8_cvt_config->init.f32_qs8_cvt(&params, 1.0f / output_scale, output_zero_point, output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_f32_qs8,
    f32_to_qs8_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_convert_nc_f32_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* f32_to_qu8_cvt_config = xnn_init_f32_to_qu8_cvt_config();
  if (f32_to_qu8_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qu8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_qu8_cvt_params params;
  assert(f32_to_qu8_cvt_config->init.f32_qu8_cvt != NULL);
  f32_to_qu8_cvt_config->init.f32_qu8_cvt(&params, 1.0f / output_scale, output_zero_point, output_min, output_max);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_f32_qu8,
    f32_to_qu8_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_convert_nc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  int8_t input_zero_point,
  float output_scale,
  int8_t output_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale > 0x1.0p+7f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input-to-output scale ratio: scale ratio must be in [2**-8, 2**7] range",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8), input_output_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* qs8_cvt_config = xnn_init_qs8_cvt_config();
  if (qs8_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_qs8_cvt_params params;
  assert(qs8_cvt_config->init.qs8_cvt != NULL);
  qs8_cvt_config->init.qs8_cvt(&params, input_output_scale, input_zero_point, output_zero_point);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_qs8,
    qs8_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_convert_nc_qs8_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  int8_t input_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8_f32), input_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* qs8_to_f32_cvt_config = xnn_init_qs8_to_f32_cvt_config();
  if (qs8_to_f32_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_qs8_f32_cvt_params params;
  assert(qs8_to_f32_cvt_config->init.qs8_f32_cvt != NULL);
  qs8_to_f32_cvt_config->init.qs8_f32_cvt(&params, input_scale, input_zero_point);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_qs8_f32,
    qs8_to_f32_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_convert_nc_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  uint8_t input_zero_point,
  float output_scale,
  uint8_t output_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale > 0x1.0p+7f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input-to-output scale ratio: scale ratio must be in [2**-8, 2**7] range",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8), input_output_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* qu8_cvt_config = xnn_init_qu8_cvt_config();
  if (qu8_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_qu8_cvt_params params;
  assert(qu8_cvt_config->init.qu8_cvt != NULL);
  qu8_cvt_config->init.qu8_cvt(&params, input_output_scale, input_zero_point, output_zero_point);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_qu8,
    qu8_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_convert_nc_qu8_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  uint8_t input_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8_f32), input_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* qu8_to_f32_cvt_config = xnn_init_qu8_to_f32_cvt_config();
  if (qu8_to_f32_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_qu8_f32_cvt_params params;
  assert(qu8_to_f32_cvt_config->init.qu8_f32_cvt != NULL);
  qu8_to_f32_cvt_config->init.qu8_f32_cvt(&params, input_scale, input_zero_point);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_convert_nc_qu8_f32,
    qu8_to_f32_cvt_config,
    convert_op_out);
}

enum xnn_status xnn_create_copy_nc_x8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* copy_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_copy_nc_x8,
    xnn_init_xx_copy_config(),
    copy_op_out);
}

enum xnn_status xnn_create_copy_nc_x16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* copy_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_copy_nc_x16,
    xnn_init_xx_copy_config(),
    copy_op_out);
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
    xnn_init_xx_copy_config(),
    copy_op_out);
}

enum xnn_status xnn_create_elu_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float alpha,
  uint32_t flags,
  xnn_operator_t* elu_op_out)
{
  const uint16_t alpha_as_half = fp16_ieee_from_fp32_value(alpha);
  alpha = fp16_ieee_to_fp32_value(alpha_as_half);
  if (alpha <= 0.0f || !isnormal(alpha)) {
    xnn_log_error(
      "failed to create %s operator with %.7g alpha parameter: alpha must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_elu_nc_f16), alpha);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_unary_elementwise_config* f16_elu_config = xnn_init_f16_elu_config();
  if (f16_elu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_elu_nc_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f16_elu_params params;
  assert(f16_elu_config->init.f16_elu != NULL);
  f16_elu_config->init.f16_elu(&params,
    UINT16_C(0x3C00)  /* prescale = 1.0h */, alpha_as_half, UINT16_C(0x3C00)  /* beta = 1.0h */);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_elu_nc_f16,
    f16_elu_config,
    elu_op_out);
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
  const struct xnn_unary_elementwise_config* f32_elu_config = xnn_init_f32_elu_config();
  if (f32_elu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_elu_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_elu_params params;
  assert(f32_elu_config->init.f32_elu != NULL);
  f32_elu_config->init.f32_elu(&params, 1.0f /* prescale */, alpha, 1.0f /* beta */);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_elu_nc_f32,
    f32_elu_config,
    elu_op_out);
}

enum xnn_status xnn_create_floor_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* floor_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_floor_nc_f16,
    xnn_init_f16_rndd_config(),
    floor_op_out);
}

enum xnn_status xnn_create_floor_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* floor_op_out)
{
  const struct xnn_unary_elementwise_config* f32_rndd_config = xnn_init_f32_rndd_config();
  if (f32_rndd_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_floor_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndd_config->init.f32_rnd != NULL) {
    f32_rndd_config->init.f32_rnd(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_floor_nc_f32,
    f32_rndd_config,
    floor_op_out);
}

enum xnn_status xnn_create_hardswish_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* hardswish_op_out)
{
  const struct xnn_unary_elementwise_config* f16_hswish_config = xnn_init_f16_hswish_config();
  if (f16_hswish_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_hardswish_nc_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f16_hswish_params params;
  if (f16_hswish_config->init.f16_hswish != NULL) {
    f16_hswish_config->init.f16_hswish(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_hardswish_nc_f16,
    f16_hswish_config,
    hardswish_op_out);
}

enum xnn_status xnn_create_hardswish_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* hardswish_op_out)
{
  const struct xnn_unary_elementwise_config* f32_hswish_config = xnn_init_f32_hswish_config();
  if (f32_hswish_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_hardswish_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_hswish_params params;
  if (f32_hswish_config->init.f32_hswish != NULL) {
    f32_hswish_config->init.f32_hswish(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_hardswish_nc_f32,
    f32_hswish_config,
    hardswish_op_out);
}

enum xnn_status xnn_create_leaky_relu_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out)
{
  const uint16_t negative_slope_as_half = fp16_ieee_from_fp32_value(negative_slope);
  negative_slope = fp16_ieee_to_fp32_value(negative_slope_as_half);
  if (!isfinite(negative_slope)) {
    xnn_log_error(
      "failed to create %s operator with %f negative slope: finite number expected",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_f32),
      negative_slope);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* f16_lrelu_config = xnn_init_f16_lrelu_config();
  if (f16_lrelu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f16_lrelu_params params;
  assert(f16_lrelu_config->init.f16_lrelu != NULL);
  f16_lrelu_config->init.f16_lrelu(&params, negative_slope_as_half);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_leaky_relu_nc_f16,
    f16_lrelu_config,
    leaky_relu_op_out);
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
  const struct xnn_unary_elementwise_config* f32_lrelu_config = xnn_init_f32_lrelu_config();
  if (f32_lrelu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_lrelu_params params;
  assert(f32_lrelu_config->init.f32_lrelu != NULL);
  f32_lrelu_config->init.f32_lrelu(&params, negative_slope);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_leaky_relu_nc_f32,
    f32_lrelu_config,
    leaky_relu_op_out);
}

enum xnn_status xnn_create_leaky_relu_nc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  int8_t input_zero_point,
  float input_scale,
  int8_t output_zero_point,
  float output_scale,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out)
{
  if (!isfinite(negative_slope)) {
    xnn_log_error(
      "failed to create %s operator with %f negative slope: finite number expected",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qs8),
      negative_slope);
    return xnn_status_invalid_parameter;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  const float positive_input_output_scale = input_scale / output_scale;
  if (positive_input_output_scale < 0x1.0p-8f || positive_input_output_scale > 0x1.0p+7f) {
    xnn_log_error(
      "failed to create %s operator with %.7g positive-input-to-output scale ratio: scale ratio must be in [2**-8, 2**7] range",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qs8), positive_input_output_scale);
    return xnn_status_invalid_parameter;
  }

  const float negative_input_output_scale = positive_input_output_scale * negative_slope;
  if (negative_input_output_scale < -0x1.FFFC00p+6f || negative_input_output_scale > 0x1.0p+7f) {
    xnn_log_error(
      "failed to create %s operator with %.7g negative-input-to-output scale ratio: scale ratio must be in (-2**7, 2**7] range and ",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qs8), negative_input_output_scale);
    return xnn_status_invalid_parameter;
  }

  if (fabsf(negative_input_output_scale) < 0x1.0p-8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g negative-input-to-output scale ratio: scale ratio must be at least 2**-8 in absolute value",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qs8), negative_input_output_scale);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_unary_elementwise_config* qs8_lrelu_config = xnn_init_qs8_lrelu_config();
  if (qs8_lrelu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qs8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_qs8_lrelu_params params;
  assert(qs8_lrelu_config->init.qs8_lrelu != NULL);
  qs8_lrelu_config->init.qs8_lrelu(&params, positive_input_output_scale, negative_input_output_scale, input_zero_point, output_zero_point);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_leaky_relu_nc_qs8,
    qs8_lrelu_config,
    leaky_relu_op_out);
}

enum xnn_status xnn_create_leaky_relu_nc_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out)
{
  if (!isfinite(negative_slope)) {
    xnn_log_error(
      "failed to create %s operator with %f negative slope: finite number expected",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qu8),
      negative_slope);
    return xnn_status_invalid_parameter;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  const float positive_input_output_scale = input_scale / output_scale;
  if (positive_input_output_scale < 0x1.0p-8f || positive_input_output_scale > 0x1.0p+7f) {
    xnn_log_error(
      "failed to create %s operator with %.7g positive-input-to-output scale ratio: scale ratio must be in [2**-8, 2**7] range",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qu8), positive_input_output_scale);
    return xnn_status_invalid_parameter;
  }

  const float negative_input_output_scale = positive_input_output_scale * negative_slope;
  if (negative_input_output_scale < -0x1.FFFC00p+6f || negative_input_output_scale > 0x1.0p+7f) {
    xnn_log_error(
      "failed to create %s operator with %.7g negative-input-to-output scale ratio: scale ratio must be in (-2**7, 2**7] range and ",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qu8), negative_input_output_scale);
    return xnn_status_invalid_parameter;
  }

  if (fabsf(negative_input_output_scale) < 0x1.0p-8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g negative-input-to-output scale ratio: scale ratio must be at least 2**-8 in absolute value",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qu8), negative_input_output_scale);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_unary_elementwise_config* qu8_lrelu_config = xnn_init_qu8_lrelu_config();
  if (qu8_lrelu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_qu8));
    return xnn_status_unsupported_hardware;
  }

  union xnn_qu8_lrelu_params params;
  assert(qu8_lrelu_config->init.qu8_lrelu != NULL);
  qu8_lrelu_config->init.qu8_lrelu(&params, positive_input_output_scale, negative_input_output_scale, input_zero_point, output_zero_point);
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_leaky_relu_nc_qu8,
    qu8_lrelu_config,
    leaky_relu_op_out);
}

enum xnn_status xnn_create_negate_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* negate_op_out)
{
  const struct xnn_unary_elementwise_config* f16_neg_config = xnn_init_f16_neg_config();
  if (f16_neg_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_negate_nc_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f16_neg_params params;
  if (f16_neg_config->init.f16_neg != NULL) {
    f16_neg_config->init.f16_neg(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_negate_nc_f16,
    f16_neg_config,
    negate_op_out);
}

enum xnn_status xnn_create_negate_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* negate_op_out)
{
  const struct xnn_unary_elementwise_config* f32_neg_config = xnn_init_f32_neg_config();
  if (f32_neg_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_negate_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_neg_params params;
  if (f32_neg_config->init.f32_neg != NULL) {
    f32_neg_config->init.f32_neg(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_negate_nc_f32,
    f32_neg_config,
    negate_op_out);
}

enum xnn_status xnn_create_sigmoid_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* sigmoid_op_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_F16) != XNN_INIT_FLAG_F16) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(xnn_operator_type_sigmoid_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_unary_elementwise_config* f16_sigmoid_config = xnn_init_f16_sigmoid_config();
  if (f16_sigmoid_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_sigmoid_nc_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f16_sigmoid_params params;
  if (f16_sigmoid_config->init.f16_sigmoid != NULL) {
    f16_sigmoid_config->init.f16_sigmoid(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_sigmoid_nc_f16,
    f16_sigmoid_config,
    sigmoid_op_out);
}

enum xnn_status xnn_create_sigmoid_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* sigmoid_op_out)
{
  const struct xnn_unary_elementwise_config* f32_sigmoid_config = xnn_init_f32_sigmoid_config();
  if (f32_sigmoid_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_sigmoid_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_sigmoid_params params;
  if (f32_sigmoid_config->init.f32_sigmoid != NULL) {
    f32_sigmoid_config->init.f32_sigmoid(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_sigmoid_nc_f32,
    f32_sigmoid_config,
    sigmoid_op_out);
}

enum xnn_status xnn_create_square_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* square_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_square_nc_f16,
    xnn_init_f16_sqr_config(),
    square_op_out);
}

enum xnn_status xnn_create_square_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* square_op_out)
{
  const struct xnn_unary_elementwise_config* f32_sqr_config = xnn_init_f32_sqr_config();
  if (f32_sqr_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_square_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_default_params params;
  if (f32_sqr_config->init.f32_default != NULL) {
    f32_sqr_config->init.f32_default(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_square_nc_f32,
    f32_sqr_config,
    square_op_out);
}

enum xnn_status xnn_create_square_root_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* sqrt_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_square_root_nc_f16,
    xnn_init_f16_sqrt_config(),
    sqrt_op_out);
}

enum xnn_status xnn_create_square_root_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* sqrt_op_out)
{
  const struct xnn_unary_elementwise_config* f32_sqrt_config = xnn_init_f32_sqrt_config();
  if (f32_sqrt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_square_root_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_sqrt_params params;
  if (f32_sqrt_config->init.f32_sqrt != NULL) {
    f32_sqrt_config->init.f32_sqrt(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_square_root_nc_f32,
    f32_sqrt_config,
    sqrt_op_out);
}

enum xnn_status xnn_create_truncation_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* truncation_op_out)
{
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    xnn_operator_type_truncation_nc_f16,
    xnn_init_f16_rndz_config(),
    truncation_op_out);
}

enum xnn_status xnn_create_truncation_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t flags,
    xnn_operator_t* truncation_op_out)
{
  const struct xnn_unary_elementwise_config* f32_rndz_config = xnn_init_f32_rndz_config();
  if (f32_rndz_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_truncation_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndz_config->init.f32_rnd != NULL) {
    f32_rndz_config->init.f32_rnd(&params);
  }
  return create_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    &params, sizeof(params),
    xnn_operator_type_truncation_nc_f32,
    f32_rndz_config,
    truncation_op_out);
}

enum xnn_status xnn_setup_abs_nc_f16(
    xnn_operator_t abs_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    abs_op, xnn_operator_type_abs_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &abs_op->params.f16_abs, sizeof(abs_op->params.f16_abs),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_abs_nc_f32(
    xnn_operator_t abs_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    abs_op, xnn_operator_type_abs_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &abs_op->params.f32_abs, sizeof(abs_op->params.f32_abs),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_bankers_rounding_nc_f16(
    xnn_operator_t rounding_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    rounding_op, xnn_operator_type_bankers_rounding_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_bankers_rounding_nc_f32(
    xnn_operator_t rounding_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    rounding_op, xnn_operator_type_bankers_rounding_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &rounding_op->params.f32_rnd, sizeof(rounding_op->params.f32_rnd),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_ceiling_nc_f16(
    xnn_operator_t ceiling_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    ceiling_op, xnn_operator_type_ceiling_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_ceiling_nc_f32(
    xnn_operator_t ceiling_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    ceiling_op, xnn_operator_type_ceiling_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &ceiling_op->params.f32_rnd, sizeof(ceiling_op->params.f32_rnd),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_clamp_nc_f16(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    clamp_op, xnn_operator_type_clamp_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &clamp_op->params.f16_minmax, sizeof(clamp_op->params.f16_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_clamp_nc_f32(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    clamp_op, xnn_operator_type_clamp_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &clamp_op->params.f32_minmax, sizeof(clamp_op->params.f32_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_clamp_nc_s8(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const int8_t* input,
    int8_t* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    clamp_op, xnn_operator_type_clamp_nc_s8,
    batch_size, input, output,
    0 /* log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(int8_t)) */,
    &clamp_op->params.s8_minmax, sizeof(clamp_op->params.s8_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_clamp_nc_u8(
    xnn_operator_t clamp_op,
    size_t batch_size,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    clamp_op, xnn_operator_type_clamp_nc_u8,
    batch_size, input, output,
    0 /* log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(uint8_t)) */,
    &clamp_op->params.u8_minmax, sizeof(clamp_op->params.u8_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_f16_f32(
  xnn_operator_t convert_op,
  size_t batch_size,
  const void* input,
  float* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_f16_f32,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    2 /* log2(sizeof(float)) */,
    &convert_op->params.f16_f32_cvt, sizeof(convert_op->params.f16_f32_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_f32_f16(
  xnn_operator_t convert_op,
  size_t batch_size,
  const float* input,
  void* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_f32_f16,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &convert_op->params.f32_f16_cvt, sizeof(convert_op->params.f32_f16_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_f32_qs8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const float* input,
  int8_t* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_f32_qs8,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    0 /* log2(sizeof(int8_t)) */,
    &convert_op->params.f32_qs8_cvt, sizeof(convert_op->params.f32_qs8_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_f32_qu8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const float* input,
  uint8_t* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_f32_qu8,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    0 /* log2(sizeof(uint8_t)) */,
    &convert_op->params.f32_qu8_cvt, sizeof(convert_op->params.f32_qu8_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_qs8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_qs8,
    batch_size, input, output,
    0 /* log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(int8_t)) */,
    &convert_op->params.qs8_cvt, sizeof(convert_op->params.qs8_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_qs8_f32(
  xnn_operator_t convert_op,
  size_t batch_size,
  const int8_t* input,
  float* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_qs8_f32,
    batch_size, input, output,
    0 /* log2(sizeof(int8_t)) */,
    2 /* log2(sizeof(float)) */,
    &convert_op->params.qs8_f32_cvt, sizeof(convert_op->params.qs8_f32_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_qu8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_qu8,
    batch_size, input, output,
    0 /* log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(uint8_t)) */,
    &convert_op->params.qu8_cvt, sizeof(convert_op->params.qu8_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convert_nc_qu8_f32(
  xnn_operator_t convert_op,
  size_t batch_size,
  const uint8_t* input,
  float* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    convert_op, xnn_operator_type_convert_nc_qu8_f32,
    batch_size, input, output,
    0 /* log2(sizeof(uint8_t)) */,
    2 /* log2(sizeof(float)) */,
    &convert_op->params.qu8_f32_cvt, sizeof(convert_op->params.qu8_f32_cvt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_copy_nc_x8(
    xnn_operator_t copy_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x8,
    batch_size, input, output,
    0 /* log2(sizeof(uint16_t)) */,
    0 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_copy_nc_x16(
    xnn_operator_t copy_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_copy_nc_x32(
    xnn_operator_t copy_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x32,
    batch_size, input, output,
    2 /* log2(sizeof(uint32_t)) */,
    2 /* log2(sizeof(uint32_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_elu_nc_f16(
    xnn_operator_t elu_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    elu_op, xnn_operator_type_elu_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &elu_op->params.f16_elu, sizeof(elu_op->params.f16_elu),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_elu_nc_f32(
    xnn_operator_t elu_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    elu_op, xnn_operator_type_elu_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &elu_op->params.f32_elu, sizeof(elu_op->params.f32_elu),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_floor_nc_f16(
    xnn_operator_t floor_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    floor_op, xnn_operator_type_floor_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_floor_nc_f32(
    xnn_operator_t floor_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    floor_op, xnn_operator_type_floor_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &floor_op->params.f32_rnd, sizeof(floor_op->params.f32_rnd),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_hardswish_nc_f16(
    xnn_operator_t hardswish_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    hardswish_op, xnn_operator_type_hardswish_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &hardswish_op->params.f16_hswish, sizeof(hardswish_op->params.f16_hswish),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_hardswish_nc_f32(
    xnn_operator_t hardswish_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    hardswish_op, xnn_operator_type_hardswish_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &hardswish_op->params.f32_hswish, sizeof(hardswish_op->params.f32_hswish),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_leaky_relu_nc_f16(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    leaky_relu_op, xnn_operator_type_leaky_relu_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &leaky_relu_op->params.f16_lrelu, sizeof(leaky_relu_op->params.f16_lrelu),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_leaky_relu_nc_f32(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    leaky_relu_op, xnn_operator_type_leaky_relu_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &leaky_relu_op->params.f32_lrelu, sizeof(leaky_relu_op->params.f32_lrelu),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_leaky_relu_nc_qs8(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    leaky_relu_op, xnn_operator_type_leaky_relu_nc_qs8,
    batch_size, input, output,
    0 /* log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(int8_t)) */,
    &leaky_relu_op->params.qs8_lrelu, sizeof(leaky_relu_op->params.qs8_lrelu),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_leaky_relu_nc_qu8(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    leaky_relu_op, xnn_operator_type_leaky_relu_nc_qu8,
    batch_size, input, output,
    0 /* log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(uint8_t)) */,
    &leaky_relu_op->params.qu8_lrelu, sizeof(leaky_relu_op->params.qu8_lrelu),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_negate_nc_f16(
    xnn_operator_t negate_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    negate_op, xnn_operator_type_negate_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &negate_op->params.f16_neg, sizeof(negate_op->params.f16_neg),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_negate_nc_f32(
    xnn_operator_t negate_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    negate_op, xnn_operator_type_negate_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &negate_op->params.f32_neg, sizeof(negate_op->params.f32_neg),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_sigmoid_nc_f16(
    xnn_operator_t sigmoid_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    sigmoid_op, xnn_operator_type_sigmoid_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    &sigmoid_op->params.f16_sigmoid, sizeof(sigmoid_op->params.f16_sigmoid),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_sigmoid_nc_f32(
    xnn_operator_t sigmoid_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    sigmoid_op, xnn_operator_type_sigmoid_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &sigmoid_op->params.f32_sigmoid, sizeof(sigmoid_op->params.f32_sigmoid),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_square_nc_f16(
    xnn_operator_t square_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    square_op, xnn_operator_type_square_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_square_nc_f32(
    xnn_operator_t square_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    square_op, xnn_operator_type_square_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &square_op->params.f32_default, sizeof(square_op->params.f32_default),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_square_root_nc_f16(
    xnn_operator_t sqrt_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    sqrt_op, xnn_operator_type_square_root_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_square_root_nc_f32(
    xnn_operator_t sqrt_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    sqrt_op, xnn_operator_type_square_root_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &sqrt_op->params.f32_sqrt, sizeof(sqrt_op->params.f32_sqrt),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_truncation_nc_f16(
    xnn_operator_t truncation_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    truncation_op, xnn_operator_type_truncation_nc_f16,
    batch_size, input, output,
    1 /* log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(uint16_t)) */,
    NULL, 0,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_truncation_nc_f32(
    xnn_operator_t truncation_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_unary_elementwise_nc(
    truncation_op, xnn_operator_type_truncation_nc_f32,
    batch_size, input, output,
    2 /* log2(sizeof(float)) */,
    2 /* log2(sizeof(float)) */,
    &truncation_op->params.f32_rnd, sizeof(truncation_op->params.f32_rnd),
    pthreadpool_get_threads_count(threadpool));
}

static enum xnn_status run_unary_elementwise_nc(
    enum xnn_operator_type operator_type,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const void* input,
    void* output,
    const struct xnn_unary_elementwise_config* config,
    const void* params,
    size_t params_size,
    uint32_t log2_input_size,
    uint32_t log2_output_size,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  if (config == NULL) {
    xnn_log_error(
          "failed to create %s operator: unsupported hardware configuration",
          xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to run %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to run %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to run %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  struct xnn_operator unary_elementwise_op;
  memset(&unary_elementwise_op, 0, sizeof(unary_elementwise_op));

  init_unary_elementwise_nc(
    channels, input_stride, output_stride, flags,
    NULL, 0,
    operator_type,
    config->ukernel,
    &unary_elementwise_op);

  const enum xnn_status status = setup_unary_elementwise_nc(
    &unary_elementwise_op, operator_type,
    batch_size, input, output,
    log2_input_size,
    log2_output_size,
    params, params_size,
    pthreadpool_get_threads_count(threadpool));

  if (status != xnn_status_success){
    return status;
  }

  return xnn_run_operator(&unary_elementwise_op, threadpool);
}

enum xnn_status xnn_run_abs_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const float* input,
    float* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_abs_config = xnn_init_f32_abs_config();
  if (f32_abs_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_abs_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_abs_params params;
  if (f32_abs_config->init.f32_abs != NULL) {
    f32_abs_config->init.f32_abs(&params);
  }

  return run_unary_elementwise_nc(
    xnn_operator_type_abs_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_abs_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_bankers_rounding_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const float* input,
    float* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_rndne_config = xnn_init_f32_rndne_config();
  if (f32_rndne_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_bankers_rounding_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndne_config->init.f32_rnd != NULL) {
    f32_rndne_config->init.f32_rnd(&params);
  }

  return run_unary_elementwise_nc(
      xnn_operator_type_bankers_rounding_nc_f32,
      channels,
      input_stride, output_stride,
      batch_size,
      input, output,
      f32_rndne_config,
      &params, sizeof(params),
      2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
      flags,
      threadpool);
}

enum xnn_status xnn_run_ceiling_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_rndu_config = xnn_init_f32_rndu_config();
  if (f32_rndu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_ceiling_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndu_config->init.f32_rnd != NULL) {
    f32_rndu_config->init.f32_rnd(&params);
  }

  return run_unary_elementwise_nc(
    xnn_operator_type_ceiling_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_rndu_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_clamp_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
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
  const struct xnn_unary_elementwise_config* f32_clamp_config = xnn_init_f32_clamp_config();
  const struct xnn_unary_elementwise_config* f32_relu_config = xnn_init_f32_relu_config();
  if (f32_clamp_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_clamp_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const bool relu_activation = (output_max == INFINITY) && (output_min == 0.0f);
  const struct xnn_unary_elementwise_config* config = f32_clamp_config;
  if (relu_activation && f32_relu_config->ukernel != NULL) {
    config = f32_relu_config;
  }

  union xnn_f32_minmax_params params;
  assert(f32_clamp_config->init.f32_minmax != NULL);
  f32_clamp_config->init.f32_minmax(&params, output_min, output_max);

  return run_unary_elementwise_nc(
    xnn_operator_type_clamp_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_convert_nc_f16_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const void* input,
    float* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f16_to_f32_cvt_config = xnn_init_f16_to_f32_cvt_config();
  if (f16_to_f32_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f16_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f16_f32_cvt_params params;
  if (f16_to_f32_cvt_config->init.f16_f32_cvt != NULL) {
    f16_to_f32_cvt_config->init.f16_f32_cvt(&params);
  }

  return run_unary_elementwise_nc(
    xnn_operator_type_convert_nc_f16_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f16_to_f32_cvt_config,
    &params, sizeof(params),
    1 /* log2(sizeof(uint16_t)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_convert_nc_f32_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const float* input,
    void* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_to_f16_cvt_config = xnn_init_f32_to_f16_cvt_config();
  if (f32_to_f16_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_f16));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_f16_cvt_params params;
  if (f32_to_f16_cvt_config->init.f32_f16_cvt != NULL) {
    f32_to_f16_cvt_config->init.f32_f16_cvt(&params);
  }

  return run_unary_elementwise_nc(
    xnn_operator_type_convert_nc_f32_f16,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_to_f16_cvt_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 1 /* log2(sizeof(uint16_t)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_convert_nc_f32_qs8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const float* input,
    int8_t* output,
    float output_scale,
    int8_t output_zero_point,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* f32_to_qs8_cvt_config = xnn_init_f32_to_qs8_cvt_config();
  if (f32_to_qs8_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qs8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_qs8_cvt_params params;
  assert(f32_to_qs8_cvt_config->init.f32_qs8_cvt != NULL);
  f32_to_qs8_cvt_config->init.f32_qs8_cvt(&params, 1.0f / output_scale, output_zero_point, INT8_MIN, INT8_MAX);
  return run_unary_elementwise_nc(
    xnn_operator_type_convert_nc_f32_qs8,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_to_qs8_cvt_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 0 /* log2(sizeof(int8_t)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_convert_nc_f32_qu8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const float* input,
    uint8_t* output,
    float output_scale,
    uint8_t output_zero_point,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* f32_to_qu8_cvt_config = xnn_init_f32_to_qu8_cvt_config();
  if (f32_to_qu8_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qu8));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_qu8_cvt_params params;
  assert(f32_to_qu8_cvt_config->init.f32_qu8_cvt != NULL);
  f32_to_qu8_cvt_config->init.f32_qu8_cvt(&params, 1.0f / output_scale, output_zero_point, 0, UINT8_MAX);
  return run_unary_elementwise_nc(
    xnn_operator_type_convert_nc_f32_qu8,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_to_qu8_cvt_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 0 /* log2(sizeof(uint8_t)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_convert_nc_qs8_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const int8_t* input,
    float* output,
    float input_scale,
    int8_t input_zero_point,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8_f32), input_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* qs8_to_f32_cvt_config = xnn_init_qs8_to_f32_cvt_config();
  if (qs8_to_f32_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_qs8_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_qs8_f32_cvt_params params;
  assert(qs8_to_f32_cvt_config->init.qs8_f32_cvt != NULL);
  qs8_to_f32_cvt_config->init.qs8_f32_cvt(&params, input_scale, input_zero_point);
  return run_unary_elementwise_nc(
    xnn_operator_type_convert_nc_qs8_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    qs8_to_f32_cvt_config,
    &params, sizeof(params),
    0 /* log2(sizeof(int8_t)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_convert_nc_qu8_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const uint8_t* input,
    float* output,
    float input_scale,
    uint8_t input_zero_point,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale parameter: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8_f32), input_scale);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* qu8_to_f32_cvt_config = xnn_init_qu8_to_f32_cvt_config();
  if (qu8_to_f32_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_qu8_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_qu8_f32_cvt_params params;
  assert(qu8_to_f32_cvt_config->init.qu8_f32_cvt != NULL);
  qu8_to_f32_cvt_config->init.qu8_f32_cvt(&params, input_scale, input_zero_point);
  return run_unary_elementwise_nc(
    xnn_operator_type_convert_nc_qu8_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    qu8_to_f32_cvt_config,
    &params, sizeof(params),
    0 /* log2(sizeof(uint8_t)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_copy_nc_x32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const uint32_t* input,
    uint32_t* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  return run_unary_elementwise_nc(
    xnn_operator_type_copy_nc_x32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    xnn_init_xx_copy_config(),
    NULL, 0,
    2 /* log2(sizeof(uint32_t)) */, 2 /* log2(sizeof(uint32_t)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_elu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  float alpha,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (alpha <= 0.0f || !isnormal(alpha)) {
    xnn_log_error(
      "failed to create %s operator with %.7g alpha parameter: alpha must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_elu_nc_f32), alpha);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* f32_elu_config = xnn_init_f32_elu_config();
  if (f32_elu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_elu_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_elu_params params;
  assert(f32_elu_config->init.f32_elu != NULL);
  f32_elu_config->init.f32_elu(&params, 1.0f /* prescale */, alpha, 1.0f /* beta */);
  return run_unary_elementwise_nc(
    xnn_operator_type_elu_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_elu_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_floor_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_rndd_config = xnn_init_f32_rndd_config();
  if (f32_rndd_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_floor_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndd_config->init.f32_rnd != NULL) {
    f32_rndd_config->init.f32_rnd(&params);
  }
  return run_unary_elementwise_nc(
    xnn_operator_type_floor_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_rndd_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_hardswish_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const float* input,
    float* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_hswish_config = xnn_init_f32_hswish_config();
  if (f32_hswish_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_hardswish_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_hswish_params params;
  if (f32_hswish_config->init.f32_hswish != NULL) {
    f32_hswish_config->init.f32_hswish(&params);
  }
  return run_unary_elementwise_nc(
    xnn_operator_type_hardswish_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_hswish_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_leaky_relu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  float negative_slope,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (!isfinite(negative_slope)) {
    xnn_log_error(
      "failed to create %s operator with %f negative slope: finite number expected",
      xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_f32),
      negative_slope);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_unary_elementwise_config* f32_lrelu_config = xnn_init_f32_lrelu_config();
  if (f32_lrelu_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_leaky_relu_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_lrelu_params params;
  assert(f32_lrelu_config->init.f32_lrelu != NULL);
  f32_lrelu_config->init.f32_lrelu(&params, negative_slope);
  return run_unary_elementwise_nc(
    xnn_operator_type_leaky_relu_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_lrelu_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_negate_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_neg_config = xnn_init_f32_neg_config();
  if (f32_neg_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_negate_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_neg_params params;
  if (f32_neg_config->init.f32_neg != NULL) {
    f32_neg_config->init.f32_neg(&params);
  }
  return run_unary_elementwise_nc(
    xnn_operator_type_negate_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_neg_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_sigmoid_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_sigmoid_config = xnn_init_f32_sigmoid_config();
  if (f32_sigmoid_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_sigmoid_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_sigmoid_params params;
  if (f32_sigmoid_config->init.f32_sigmoid != NULL) {
    f32_sigmoid_config->init.f32_sigmoid(&params);
  }
  return run_unary_elementwise_nc(
    xnn_operator_type_sigmoid_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_sigmoid_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_square_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_sqr_config = xnn_init_f32_sqr_config();
  if (f32_sqr_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_square_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_default_params params;
  if (f32_sqr_config->init.f32_default != NULL) {
    f32_sqr_config->init.f32_default(&params);
  }
  return run_unary_elementwise_nc(
    xnn_operator_type_square_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_sqr_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_square_root_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_sqrt_config = xnn_init_f32_sqrt_config();
  if (f32_sqrt_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_square_root_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_sqrt_params params;
  if (f32_sqrt_config->init.f32_sqrt != NULL) {
    f32_sqrt_config->init.f32_sqrt(&params);
  }
  return run_unary_elementwise_nc(
    xnn_operator_type_square_root_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_sqrt_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}

enum xnn_status xnn_run_truncation_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_unary_elementwise_config* f32_rndz_config = xnn_init_f32_rndz_config();
  if (f32_rndz_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_truncation_nc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_rnd_params params;
  if (f32_rndz_config->init.f32_rnd != NULL) {
    f32_rndz_config->init.f32_rnd(&params);
  }
  return run_unary_elementwise_nc(
    xnn_operator_type_truncation_nc_f32,
    channels,
    input_stride, output_stride,
    batch_size,
    input, output,
    f32_rndz_config,
    &params, sizeof(params),
    2 /* log2(sizeof(float)) */, 2 /* log2(sizeof(float)) */,
    flags,
    threadpool);
}
