// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/operator.h>

static void init_binary_elementwise_nd(
  const void* params,
  size_t params_size,
  uint32_t flags,
  enum xnn_operator_type operator_type,
  const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig,
  xnn_operator_t binary_elementwise_op)
{
  if (params_size != 0) {
    memcpy(&binary_elementwise_op->params, params, params_size);
  }

  binary_elementwise_op->binary_elementwise_subconfig = binary_elementwise_subconfig;

  binary_elementwise_op->type = operator_type;
  binary_elementwise_op->flags = flags;

  binary_elementwise_op->state = xnn_run_state_invalid;
}

static enum xnn_status create_binary_elementwise_nd(
    uint32_t flags,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig,
    xnn_operator_t* binary_elementwise_op_out)
{
  if (binary_elementwise_subconfig == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  xnn_operator_t binary_elementwise_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (binary_elementwise_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }

  init_binary_elementwise_nd(
    params,
    params_size,
    flags,
    operator_type,
    binary_elementwise_subconfig,
    binary_elementwise_op);

  *binary_elementwise_op_out = binary_elementwise_op;
  return xnn_status_success;
}

static enum xnn_status create_binary_elementwise_nd_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct xnn_binary_elementwise_config* config,
    xnn_operator_t* binary_elementwise_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_invalid_parameter;
  }

  if (fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)) >= fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max))) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(operator_type),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min)),
      fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max)));
    return xnn_status_invalid_parameter;
  }

  if (config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params params;
  assert(config->init.f16_minmax != NULL);
  config->init.f16_minmax(&params,
    fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max));

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    operator_type,
    &config->minmax,
    binary_elementwise_op_out);
}

static enum xnn_status create_binary_elementwise_nd_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct xnn_binary_elementwise_config* config,
    xnn_operator_t* binary_elementwise_op_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(operator_type), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig = &config->minmax;
  if (linear_activation && config->linear.op_ukernel != NULL) {
    binary_elementwise_subconfig = &config->linear;
  }

  union xnn_f32_minmax_params params;
  assert(config->init.f32_minmax != NULL);
  config->init.f32_minmax(&params, output_min, output_max);

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    operator_type,
    binary_elementwise_subconfig,
    binary_elementwise_op_out);
}

enum xnn_status xnn_create_add_nd_qs8(
    int8_t input1_zero_point,
    float input1_scale,
    int8_t input2_zero_point,
    float input2_scale,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* add_op_out)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qs8_vadd_config = xnn_init_qs8_vadd_config();
  if (qs8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qs8_add_minmax_params qs8_add;
    union xnn_qs8_add_minmax_params qs8_radd;
  } params;
  assert(qs8_vadd_config->init.qs8_add != NULL);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, input2_output_scale, output_min, output_max);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    input2_output_scale, input1_output_scale, output_min, output_max);

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_add_nd_qs8,
    &qs8_vadd_config->minmax,
    add_op_out);
}

enum xnn_status xnn_create_add_nd_qu8(
    uint8_t input1_zero_point,
    float input1_scale,
    uint8_t input2_zero_point,
    float input2_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* add_op_out)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qu8_vadd_config = xnn_init_qu8_vadd_config();
  if (qu8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qu8_add_minmax_params qu8_add;
    union xnn_qu8_add_minmax_params qu8_radd;
  } params;
  assert(qu8_vadd_config->init.qu8_add != NULL);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, input2_output_scale, output_min, output_max);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    input2_output_scale, input1_output_scale, output_min, output_max);

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_add_nd_qu8,
    &qu8_vadd_config->minmax,
    add_op_out);
}

enum xnn_status xnn_create_add_nd_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* add_op_out)
{
  return create_binary_elementwise_nd_f16(
    output_min,
    output_max,
    flags,
    xnn_operator_type_add_nd_f16,
    xnn_init_f16_vadd_config(),
    add_op_out);
}

enum xnn_status xnn_create_add_nd_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* add_op_out)
{
  return create_binary_elementwise_nd_f32(
    output_min,
    output_max,
    flags,
    xnn_operator_type_add_nd_f32,
    xnn_init_f32_vadd_config(),
    add_op_out);
}

enum xnn_status xnn_create_divide_nd_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* divide_op_out)
{
  return create_binary_elementwise_nd_f16(
    output_min,
    output_max,
    flags,
    xnn_operator_type_divide_nd_f16,
    xnn_init_f16_vdiv_config(),
    divide_op_out);
}

enum xnn_status xnn_create_divide_nd_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* divide_op_out)
{
  return create_binary_elementwise_nd_f32(
    output_min,
    output_max,
    flags,
    xnn_operator_type_divide_nd_f32,
    xnn_init_f32_vdiv_config(),
    divide_op_out);
}

enum xnn_status xnn_create_maximum_nd_f16(
    uint32_t flags,
    xnn_operator_t* maximum_op_out)
{
  const struct xnn_binary_elementwise_config* f16_vmax_config = xnn_init_f16_vmax_config();
  if (f16_vmax_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_maximum_nd_f16));
    return xnn_status_unsupported_hardware;
  }
  return create_binary_elementwise_nd(
    flags,
    NULL,
    0,
    xnn_operator_type_maximum_nd_f16,
    &f16_vmax_config->minmax,
    maximum_op_out);
}

enum xnn_status xnn_create_maximum_nd_f32(
    uint32_t flags,
    xnn_operator_t* maximum_op_out)
{
  const struct xnn_binary_elementwise_config* f32_vmax_config = xnn_init_f32_vmax_config();
  if (f32_vmax_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_maximum_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_default_params params;
  if (f32_vmax_config->init.f32_default != NULL) {
    f32_vmax_config->init.f32_default(&params);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_maximum_nd_f32,
    &f32_vmax_config->minmax,
    maximum_op_out);
}

enum xnn_status xnn_create_minimum_nd_f16(
    uint32_t flags,
    xnn_operator_t* minimum_op_out)
{
  const struct xnn_binary_elementwise_config* f16_vmin_config = xnn_init_f16_vmin_config();
  if (f16_vmin_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_minimum_nd_f16));
  }
  return create_binary_elementwise_nd(
    flags,
    NULL,
    0,
    xnn_operator_type_minimum_nd_f16,
    &f16_vmin_config->minmax,
    minimum_op_out);
}

enum xnn_status xnn_create_minimum_nd_f32(
    uint32_t flags,
    xnn_operator_t* minimum_op_out)
{
  const struct xnn_binary_elementwise_config* f32_vmin_config = xnn_init_f32_vmin_config();
  if (f32_vmin_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_minimum_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_default_params params;
  if (f32_vmin_config->init.f32_default != NULL) {
    f32_vmin_config->init.f32_default(&params);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_minimum_nd_f32,
    &f32_vmin_config->minmax,
    minimum_op_out);
}

enum xnn_status xnn_create_multiply_nd_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* multiply_op_out)
{
  return create_binary_elementwise_nd_f16(
    output_min,
    output_max,
    flags,
    xnn_operator_type_multiply_nd_f16,
    xnn_init_f16_vmul_config(),
    multiply_op_out);
}

enum xnn_status xnn_create_multiply_nd_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* multiply_op_out)
{
  return create_binary_elementwise_nd_f32(
    output_min,
    output_max,
    flags,
    xnn_operator_type_multiply_nd_f32,
    xnn_init_f32_vmul_config(),
    multiply_op_out);
}

enum xnn_status xnn_create_multiply_nd_qs8(
    int8_t input1_zero_point,
    float input1_scale,
    int8_t input2_zero_point,
    float input2_scale,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* multiply_op_out)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float product_scale = input1_scale * input2_scale;
  const float product_output_scale = product_scale / output_scale;
  if (product_output_scale < 0x1.0p-16f || product_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g product-to-output scale ratio: scale ratio must be in [2**-16, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), product_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qs8_vmul_config = xnn_init_qs8_vmul_config();
  if (qs8_vmul_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qs8_mul_minmax_params qs8_mul;
    union xnn_qs8_mul_minmax_params qs8_rmul;
  } params;
  assert(qs8_vmul_config->init.qs8_mul != NULL);
  qs8_vmul_config->init.qs8_mul(
    &params.qs8_mul, input1_zero_point, input2_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);
  qs8_vmul_config->init.qs8_mul(
    &params.qs8_rmul, input2_zero_point, input1_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_multiply_nd_qs8,
    &qs8_vmul_config->minmax,
    multiply_op_out);
}

enum xnn_status xnn_create_multiply_nd_qu8(
    uint8_t input1_zero_point,
    float input1_scale,
    uint8_t input2_zero_point,
    float input2_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* multiply_op_out)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float product_scale = input1_scale * input2_scale;
  const float product_output_scale = product_scale / output_scale;
  if (product_output_scale < 0x1.0p-16f || product_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g product-to-output scale ratio: scale ratio must be in [2**-16, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), product_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qu8_vmul_config = xnn_init_qu8_vmul_config();
  if (qu8_vmul_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qu8_mul_minmax_params qu8_mul;
    union xnn_qu8_mul_minmax_params qu8_rmul;
  } params;
  assert(qu8_vmul_config->init.qu8_mul != NULL);
  qu8_vmul_config->init.qu8_mul(
    &params.qu8_mul, input1_zero_point, input2_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);
  qu8_vmul_config->init.qu8_mul(
    &params.qu8_rmul, input2_zero_point, input1_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_multiply_nd_qu8,
    &qu8_vmul_config->minmax,
    multiply_op_out);
}

enum xnn_status xnn_create_squared_difference_nd_f16(
    uint32_t flags,
    xnn_operator_t* squared_difference_op_out)
{
  const struct xnn_binary_elementwise_config* f16_vqsrdiff_config = xnn_init_f16_vsqrdiff_config();
  if (f16_vqsrdiff_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_squared_difference_nd_f16));
    return xnn_status_unsupported_hardware;
  }
  return create_binary_elementwise_nd(
    flags,
    NULL,
    0,
    xnn_operator_type_squared_difference_nd_f16,
    &f16_vqsrdiff_config->minmax,
    squared_difference_op_out);
}

enum xnn_status xnn_create_squared_difference_nd_f32(
    uint32_t flags,
    xnn_operator_t* squared_difference_op_out)
{
  const struct xnn_binary_elementwise_config* f32_vsqrdiff_config = xnn_init_f32_vsqrdiff_config();
  if (f32_vsqrdiff_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_squared_difference_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_default_params params;
  if (f32_vsqrdiff_config->init.f32_default != NULL) {
    f32_vsqrdiff_config->init.f32_default(&params);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_squared_difference_nd_f32,
    &f32_vsqrdiff_config->minmax,
    squared_difference_op_out);
}

enum xnn_status xnn_create_subtract_nd_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* subtract_op_out)
{
  return create_binary_elementwise_nd_f16(
    output_min,
    output_max,
    flags,
    xnn_operator_type_subtract_nd_f16,
    xnn_init_f16_vsub_config(),
    subtract_op_out);
}

enum xnn_status xnn_create_subtract_nd_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* subtract_op_out)
{
  return create_binary_elementwise_nd_f32(
    output_min,
    output_max,
    flags,
    xnn_operator_type_subtract_nd_f32,
    xnn_init_f32_vsub_config(),
    subtract_op_out);
}

enum xnn_status xnn_create_subtract_nd_qs8(
    int8_t input1_zero_point,
    float input1_scale,
    int8_t input2_zero_point,
    float input2_scale,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* subtract_op_out)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qs8_vadd_config = xnn_init_qs8_vadd_config();
  if (qs8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qs8_add_minmax_params qs8_add;
    union xnn_qs8_add_minmax_params qs8_radd;
  } params;
  assert(qs8_vadd_config->init.qs8_add != NULL);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, -input2_output_scale, output_min, output_max);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    -input2_output_scale, input1_output_scale, output_min, output_max);

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_subtract_nd_qs8,
    &qs8_vadd_config->minmax,
    subtract_op_out);
}

enum xnn_status xnn_create_subtract_nd_qu8(
    uint8_t input1_zero_point,
    float input1_scale,
    uint8_t input2_zero_point,
    float input2_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* subtract_op_out)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qu8_vadd_config = xnn_init_qu8_vadd_config();
  if (qu8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qu8_add_minmax_params qu8_add;
    union xnn_qu8_add_minmax_params qu8_radd;
  } params;
  assert(qu8_vadd_config->init.qu8_add != NULL);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, -input2_output_scale, output_min, output_max);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    -input2_output_scale, input1_output_scale, output_min, output_max);

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    xnn_operator_type_subtract_nd_qu8,
    &qu8_vadd_config->minmax,
    subtract_op_out);
}

static enum xnn_status reshape_binary_elementwise_nd(
    xnn_operator_t binary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    uint32_t log2_element_size,
    const void* params,
    size_t params_size,
    const void* reversed_params,
    size_t reversed_params_size,
    pthreadpool_t threadpool)
{
  if (binary_elementwise_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(binary_elementwise_op->type));
    return xnn_status_invalid_parameter;
  }
  binary_elementwise_op->state = xnn_run_state_invalid;

  if (max(num_input1_dims, num_input2_dims) > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to reshape %s operator with %zu and %zu dimensions in input shapes: "
      "the number of input dimensions must not exceed %d",
      xnn_operator_type_to_string(binary_elementwise_op->type), num_input1_dims, num_input2_dims, XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  size_t num_compressed_dims = 0;
  size_t compressed_input1_shape[XNN_MAX_TENSOR_DIMS];
  size_t compressed_input2_shape[XNN_MAX_TENSOR_DIMS];
  size_t compressed_output_shape[XNN_MAX_TENSOR_DIMS];
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    compressed_input1_shape[i] = 1;
    compressed_input2_shape[i] = 1;
    compressed_output_shape[i] = 1;
  }
  bool broadcast_input1 = false;
  bool broadcast_input2 = false;
  bool first_nonunit = true;
  bool degenerate_shape = false;
  const size_t num_common_dims = min(num_input1_dims, num_input2_dims);
  for (size_t i = 1; i <= num_common_dims; i++) {
    const size_t input1_dim = input1_shape[num_input1_dims - i];
    const size_t input2_dim = input2_shape[num_input2_dims - i];
    degenerate_shape |= input1_dim == 0;
    degenerate_shape |= input2_dim == 0;
    if (input1_dim == 1 && input2_dim == 1) {
      continue;
    }
    assert(!broadcast_input1 || !broadcast_input2);

    if (input1_dim == 1) {
      if (!broadcast_input1) {
        broadcast_input1 = true;
        broadcast_input2 = false;
        num_compressed_dims++;
      }
      compressed_input2_shape[num_compressed_dims - 1] *= input2_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input2_dim;
    } else if (input2_dim == 1) {
      if (!broadcast_input2) {
        broadcast_input1 = false;
        broadcast_input2 = true;
        num_compressed_dims++;
      }
      compressed_input1_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input1_dim;
    } else if (input1_dim == input2_dim) {
      if (broadcast_input1 || broadcast_input2 || first_nonunit) {
        broadcast_input1 = false;
        broadcast_input2 = false;
        num_compressed_dims++;
      }
      compressed_input1_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_input2_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input1_dim;
    } else {
      xnn_log_error(
        "failed to reshape %s operator: "
        "shape dimension #%zu of input1 (%zu) does not match shape dimension #%zu of input2 (%zu)",
        xnn_operator_type_to_string(binary_elementwise_op->type),
        num_input1_dims - i, input1_dim, num_input2_dims - i, input2_dim);
      return xnn_status_invalid_parameter;
    }
    first_nonunit = false;
  }
  if (num_input1_dims > num_input2_dims) {
    if (!broadcast_input2) {
      num_compressed_dims++;
    }
    for (size_t i = 0; i < num_input1_dims - num_input2_dims; i++) {
      const size_t input1_dim = input1_shape[i];
      degenerate_shape |= input1_dim == 0;
      compressed_input1_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input1_dim;
    }
  } else if (num_input2_dims > num_input1_dims) {
    if (!broadcast_input1) {
      num_compressed_dims++;
    }
    for (size_t i = 0; i < num_input2_dims - num_input1_dims; i++) {
      const size_t input2_dim = input2_shape[i];
      degenerate_shape |= input2_dim == 0;
      compressed_input2_shape[num_compressed_dims - 1] *= input2_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input2_dim;
    }
  }
  num_compressed_dims = max(num_compressed_dims, 1);

  // Early exit without setting up context if any shape dimension is zero.
  if (degenerate_shape) {
    binary_elementwise_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  binary_elementwise_op->context.elementwise_binary = (struct elementwise_binary_context) {
    .elements = compressed_output_shape[0] << log2_element_size,
  };
  if (params_size != 0) {
    memcpy(&binary_elementwise_op->context.elementwise_binary.params, params, params_size);
  }

  const size_t* compressed_a_shape = compressed_input1_shape;
  const size_t* compressed_b_shape = compressed_input2_shape;
  if (compressed_input1_shape[0] == 1) {
    binary_elementwise_op->context.elementwise_binary.flip_a_b = true;
    binary_elementwise_op->context.elementwise_binary.ukernel = binary_elementwise_op->binary_elementwise_subconfig->ropc_ukernel;
    compressed_a_shape = compressed_input2_shape;
    compressed_b_shape = compressed_input1_shape;
    if (reversed_params_size != 0) {
      memcpy(&binary_elementwise_op->context.elementwise_binary.params, reversed_params, reversed_params_size);
    }
  } else if (compressed_input2_shape[0] == 1) {
    binary_elementwise_op->context.elementwise_binary.ukernel = binary_elementwise_op->binary_elementwise_subconfig->opc_ukernel;
  } else if (compressed_input1_shape[0] == compressed_input2_shape[0]) {
    binary_elementwise_op->context.elementwise_binary.ukernel = binary_elementwise_op->binary_elementwise_subconfig->op_ukernel;
  }
  size_t a_stride = compressed_a_shape[0], b_stride = compressed_b_shape[0], y_stride = compressed_output_shape[0];
  for (size_t i = 1; i < num_compressed_dims; i++) {
    if (compressed_a_shape[i] != 1) {
      binary_elementwise_op->context.elementwise_binary.a_stride[XNN_MAX_TENSOR_DIMS - 1 - i] = a_stride << log2_element_size;
    }
    if (compressed_b_shape[i] != 1) {
      binary_elementwise_op->context.elementwise_binary.b_stride[XNN_MAX_TENSOR_DIMS - 1 - i] = b_stride << log2_element_size;
    }
    binary_elementwise_op->context.elementwise_binary.y_stride[XNN_MAX_TENSOR_DIMS - 1 - i] = y_stride << log2_element_size;
    a_stride *= compressed_a_shape[i];
    b_stride *= compressed_b_shape[i];
    y_stride *= compressed_output_shape[i];
  }

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  const size_t element_tile = binary_elementwise_op->binary_elementwise_subconfig->element_tile;
  if (compressed_output_shape[5] == 1) {
    if (compressed_output_shape[4] == 1) {
      if (compressed_output_shape[3] == 1) {
        if (compressed_output_shape[2] == 1) {
          if (compressed_output_shape[1] == 1) {
            binary_elementwise_op->context.elementwise_binary.a_stride[4] = compressed_a_shape[0] == 1 ? 0 : (1 << log2_element_size);
            binary_elementwise_op->context.elementwise_binary.b_stride[4] = compressed_b_shape[0] == 1 ? 0 : (1 << log2_element_size);
            binary_elementwise_op->context.elementwise_binary.y_stride[4] = (1 << log2_element_size);
            binary_elementwise_op->context.elementwise_binary.elements = (1 << log2_element_size);
            binary_elementwise_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
            binary_elementwise_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_elementwise_binary_1d_tile;
            binary_elementwise_op->compute[0].range[0] = compressed_output_shape[0] * (1 << log2_element_size);
            binary_elementwise_op->compute[0].tile[0] = max(element_tile, round_up_po2(binary_elementwise_op->compute[0].range[0] / num_threads, (1 << log2_element_size)));
          } else {
            binary_elementwise_op->compute[0].type = xnn_parallelization_type_1d;
            binary_elementwise_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_elementwise_binary_1d;
            binary_elementwise_op->compute[0].range[0] = compressed_output_shape[1];
          }
        } else {
          binary_elementwise_op->compute[0].type = xnn_parallelization_type_2d;
          binary_elementwise_op->compute[0].task_2d = (pthreadpool_task_2d_t) xnn_compute_elementwise_binary_2d;
          binary_elementwise_op->compute[0].range[0] = compressed_output_shape[2];
          binary_elementwise_op->compute[0].range[1] = compressed_output_shape[1];
        }
      } else {
        binary_elementwise_op->compute[0].type = xnn_parallelization_type_3d;
        binary_elementwise_op->compute[0].task_3d = (pthreadpool_task_3d_t) xnn_compute_elementwise_binary_3d;
        binary_elementwise_op->compute[0].range[0] = compressed_output_shape[3];
        binary_elementwise_op->compute[0].range[1] = compressed_output_shape[2];
        binary_elementwise_op->compute[0].range[2] = compressed_output_shape[1];
      }
    } else {
      binary_elementwise_op->compute[0].type = xnn_parallelization_type_4d;
      binary_elementwise_op->compute[0].task_4d = (pthreadpool_task_4d_t) xnn_compute_elementwise_binary_4d;
      binary_elementwise_op->compute[0].range[0] = compressed_output_shape[4];
      binary_elementwise_op->compute[0].range[1] = compressed_output_shape[3];
      binary_elementwise_op->compute[0].range[2] = compressed_output_shape[2];
      binary_elementwise_op->compute[0].range[3] = compressed_output_shape[1];
    }
  } else {
    binary_elementwise_op->compute[0].type = xnn_parallelization_type_5d;
    binary_elementwise_op->compute[0].task_5d = (pthreadpool_task_5d_t) xnn_compute_elementwise_binary_5d;
    binary_elementwise_op->compute[0].range[0] = compressed_output_shape[5];
    binary_elementwise_op->compute[0].range[1] = compressed_output_shape[4];
    binary_elementwise_op->compute[0].range[2] = compressed_output_shape[3];
    binary_elementwise_op->compute[0].range[3] = compressed_output_shape[2];
    binary_elementwise_op->compute[0].range[4] = compressed_output_shape[1];
  }
  binary_elementwise_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static enum xnn_status reshape_binary_elementwise_nd_f16(
    xnn_operator_t binary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    binary_elementwise_op,
    expected_operator_type,
    num_input1_dims,
    input1_shape,
    num_input2_dims,
    input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
    &binary_elementwise_op->params.f16_minmax, sizeof(binary_elementwise_op->params.f16_minmax),
    &binary_elementwise_op->params.f16_minmax, sizeof(binary_elementwise_op->params.f16_minmax),
    threadpool);
}

static enum xnn_status reshape_binary_elementwise_nd_f32(
    xnn_operator_t binary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    binary_elementwise_op, expected_operator_type,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &binary_elementwise_op->params.f32_minmax, sizeof(binary_elementwise_op->params.f32_minmax),
    &binary_elementwise_op->params.f32_minmax, sizeof(binary_elementwise_op->params.f32_minmax),
    threadpool);
}

enum xnn_status xnn_reshape_add_nd_f16(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f16(
    add_op, xnn_operator_type_add_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_add_nd_f32(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f32(
    add_op, xnn_operator_type_add_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_add_nd_qs8(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    &add_op->params.qs8_add, sizeof(add_op->params.qs8_add),
    &add_op->params.qs8_radd, sizeof(add_op->params.qs8_radd),
    threadpool);
}

enum xnn_status xnn_reshape_add_nd_qu8(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    &add_op->params.qu8_add, sizeof(add_op->params.qu8_add),
    &add_op->params.qu8_radd, sizeof(add_op->params.qu8_radd),
    threadpool);
}

enum xnn_status xnn_reshape_divide_nd_f16(
    xnn_operator_t divide_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f16(
    divide_op, xnn_operator_type_divide_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_divide_nd_f32(
    xnn_operator_t divide_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f32(
    divide_op, xnn_operator_type_divide_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_maximum_nd_f16(
    xnn_operator_t maximum_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f16(
    maximum_op, xnn_operator_type_maximum_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_maximum_nd_f32(
    xnn_operator_t maximum_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f32(
    maximum_op, xnn_operator_type_maximum_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_minimum_nd_f16(
    xnn_operator_t minimum_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f16(
    minimum_op, xnn_operator_type_minimum_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_minimum_nd_f32(
    xnn_operator_t minimum_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f32(
    minimum_op, xnn_operator_type_minimum_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}


enum xnn_status xnn_reshape_multiply_nd_f16(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f16(
    multiply_op, xnn_operator_type_multiply_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_multiply_nd_f32(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f32(
    multiply_op, xnn_operator_type_multiply_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_multiply_nd_qs8(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    &multiply_op->params.qs8_mul, sizeof(multiply_op->params.qs8_mul),
    &multiply_op->params.qs8_rmul, sizeof(multiply_op->params.qs8_rmul),
    threadpool);
}

enum xnn_status xnn_reshape_multiply_nd_qu8(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    &multiply_op->params.qu8_mul, sizeof(multiply_op->params.qu8_mul),
    &multiply_op->params.qu8_rmul, sizeof(multiply_op->params.qu8_rmul),
    threadpool);
}

enum xnn_status xnn_reshape_squared_difference_nd_f16(
    xnn_operator_t squared_difference_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f16(
    squared_difference_op, xnn_operator_type_squared_difference_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_squared_difference_nd_f32(
    xnn_operator_t squared_difference_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f32(
    squared_difference_op, xnn_operator_type_squared_difference_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_subtract_nd_f16(
    xnn_operator_t subtract_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f16(
    subtract_op, xnn_operator_type_subtract_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_subtract_nd_f32(
    xnn_operator_t subtract_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd_f32(
    subtract_op, xnn_operator_type_subtract_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    threadpool);
}

enum xnn_status xnn_reshape_subtract_nd_qs8(
    xnn_operator_t subtract_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    &subtract_op->params.qs8_add, sizeof(subtract_op->params.qs8_add),
    &subtract_op->params.qs8_radd, sizeof(subtract_op->params.qs8_radd),
    threadpool);
}

enum xnn_status xnn_reshape_subtract_nd_qu8(
    xnn_operator_t subtract_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    pthreadpool_t threadpool)
{
  return reshape_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    &subtract_op->params.qu8_add, sizeof(subtract_op->params.qu8_add),
    &subtract_op->params.qu8_radd, sizeof(subtract_op->params.qu8_radd),
    threadpool);
}

static enum xnn_status setup_binary_elementwise_nd(
    xnn_operator_t binary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    const void* input1,
    const void* input2,
    void* output)
{
  if (binary_elementwise_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(binary_elementwise_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (binary_elementwise_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(binary_elementwise_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  binary_elementwise_op->context.elementwise_binary.a = input1;
  binary_elementwise_op->context.elementwise_binary.b = input2;
  binary_elementwise_op->context.elementwise_binary.y = output;

  if (binary_elementwise_op->context.elementwise_binary.flip_a_b) {
    binary_elementwise_op->context.elementwise_binary.a = input2;
    binary_elementwise_op->context.elementwise_binary.b = input1;
  }

  binary_elementwise_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_add_nd_f16(
    xnn_operator_t add_op,
    const void* input1,
    const void* input2,
    void* output)
{
  return setup_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_f16,
    input1, input2, output);
}

enum xnn_status xnn_setup_add_nd_f32(
    xnn_operator_t add_op,
    const float* input1,
    const float* input2,
    float* output)
{
  return setup_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_f32,
    input1, input2, output);
}

enum xnn_status xnn_setup_add_nd_qs8(
    xnn_operator_t add_op,
    const int8_t* input1,
    const int8_t* input2,
    int8_t* output)
{
  return setup_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_qs8,
    input1, input2, output);
}

enum xnn_status xnn_setup_add_nd_qu8(
    xnn_operator_t add_op,
    const uint8_t* input1,
    const uint8_t* input2,
    uint8_t* output)
{
  return setup_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_qu8,
    input1, input2, output);
}

enum xnn_status xnn_setup_divide_nd_f16(
    xnn_operator_t divide_op,
    const void* input1,
    const void* input2,
    void* output)
{
  return setup_binary_elementwise_nd(
    divide_op, xnn_operator_type_divide_nd_f16,
    input1, input2, output);
}

enum xnn_status xnn_setup_divide_nd_f32(
    xnn_operator_t divide_op,
    const float* input1,
    const float* input2,
    float* output)
{
  return setup_binary_elementwise_nd(
    divide_op, xnn_operator_type_divide_nd_f32,
    input1, input2, output);
}

enum xnn_status xnn_setup_maximum_nd_f16(
    xnn_operator_t maximum_op,
    const void* input1,
    const void* input2,
    void* output)
{
  return setup_binary_elementwise_nd(
    maximum_op, xnn_operator_type_maximum_nd_f16,
    input1, input2, output);
}

enum xnn_status xnn_setup_maximum_nd_f32(
    xnn_operator_t maximum_op,
    const float* input1,
    const float* input2,
    float* output)
{
  return setup_binary_elementwise_nd(
    maximum_op, xnn_operator_type_maximum_nd_f32,
    input1, input2, output);
}

enum xnn_status xnn_setup_minimum_nd_f16(
    xnn_operator_t minimum_op,
    const void* input1,
    const void* input2,
    void* output)
{
  return setup_binary_elementwise_nd(
    minimum_op, xnn_operator_type_minimum_nd_f16,
    input1, input2, output);
}

enum xnn_status xnn_setup_minimum_nd_f32(
    xnn_operator_t minimum_op,
    const float* input1,
    const float* input2,
    float* output)
{
  return setup_binary_elementwise_nd(
    minimum_op, xnn_operator_type_minimum_nd_f32,
    input1, input2, output);
}

enum xnn_status xnn_setup_multiply_nd_f16(
    xnn_operator_t multiply_op,
    const void* input1,
    const void* input2,
    void* output)
{
  return setup_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_f16,
    input1, input2, output);
}

enum xnn_status xnn_setup_multiply_nd_f32(
    xnn_operator_t multiply_op,
    const float* input1,
    const float* input2,
    float* output)
{
  return setup_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_f32,
    input1, input2, output);
}

enum xnn_status xnn_setup_multiply_nd_qs8(
    xnn_operator_t multiply_op,
    const int8_t* input1,
    const int8_t* input2,
    int8_t* output)
{
  return setup_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_qs8,
    input1, input2, output);
}

enum xnn_status xnn_setup_multiply_nd_qu8(
    xnn_operator_t multiply_op,
    const uint8_t* input1,
    const uint8_t* input2,
    uint8_t* output)
{
  return setup_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_qu8,
    input1, input2, output);
}

enum xnn_status xnn_setup_squared_difference_nd_f16(
    xnn_operator_t squared_difference_op,
    const void* input1,
    const void* input2,
    void* output)
{
  return setup_binary_elementwise_nd(
    squared_difference_op, xnn_operator_type_squared_difference_nd_f16,
    input1, input2, output);
}

enum xnn_status xnn_setup_squared_difference_nd_f32(
    xnn_operator_t squared_difference_op,
    const float* input1,
    const float* input2,
    float* output)
{
  return setup_binary_elementwise_nd(
    squared_difference_op, xnn_operator_type_squared_difference_nd_f32,
    input1, input2, output);
}

enum xnn_status xnn_setup_subtract_nd_f16(
    xnn_operator_t subtract_op,
    const void* input1,
    const void* input2,
    void* output)
{
  return setup_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_f16,
    input1, input2, output);
}

enum xnn_status xnn_setup_subtract_nd_f32(
    xnn_operator_t subtract_op,
    const float* input1,
    const float* input2,
    float* output)
{
  return setup_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_f32,
    input1, input2, output);
}

enum xnn_status xnn_setup_subtract_nd_qs8(
    xnn_operator_t subtract_op,
    const int8_t* input1,
    const int8_t* input2,
    int8_t* output)
{
  return setup_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_qs8,
    input1, input2, output);
}

enum xnn_status xnn_setup_subtract_nd_qu8(
    xnn_operator_t subtract_op,
    const uint8_t* input1,
    const uint8_t* input2,
    uint8_t* output)
{
  return setup_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_qu8,
    input1, input2, output);
}

static enum xnn_status run_binary_elementwise_nd(
  enum xnn_operator_type operator_type,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  uint32_t log2_element_size,
  size_t params_offset,
  size_t setup_params_size,
  size_t rparams_offset,
  size_t setup_reversed_params_size,
  const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig,
  const void* create_params,
  size_t create_params_size,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  struct xnn_operator binary_elementwise_op;
  memset(&binary_elementwise_op, 0, sizeof(binary_elementwise_op));

  init_binary_elementwise_nd(
    create_params,
    create_params_size,
    flags,
    operator_type,
    binary_elementwise_subconfig,
    &binary_elementwise_op);

  const void* setup_params = (void*) ((uintptr_t) &binary_elementwise_op + params_offset);
  const void* setup_reversed_params = (void*) ((uintptr_t) &binary_elementwise_op + rparams_offset);

  enum xnn_status status = reshape_binary_elementwise_nd(
    &binary_elementwise_op, operator_type,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    log2_element_size,
    setup_params, setup_params_size,
    setup_reversed_params, setup_reversed_params_size,
    threadpool);


  status = setup_binary_elementwise_nd(
    &binary_elementwise_op, operator_type,
    input1, input2, output);

  if (status != xnn_status_success) {
    return status;
  }

  return xnn_run_operator(&binary_elementwise_op, threadpool);
}

static enum xnn_status run_binary_elementwise_nd_f32(
  enum xnn_operator_type operator_type,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  const struct xnn_binary_elementwise_config* config,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to run %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
      xnn_log_error(
        "failed to run %s operator with NaN output upper bound: upper bound must be non-NaN",
        xnn_operator_type_to_string(operator_type));
      return xnn_status_invalid_parameter;
    }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
        xnn_operator_type_to_string(operator_type), output_min, output_max);
      return xnn_status_invalid_parameter;
    }

  if (config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_minmax_params params;
  assert(config->init.f32_minmax != NULL);
  config->init.f32_minmax(&params, output_min, output_max);

  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig = &config->minmax;
  if (linear_activation && config->linear.op_ukernel != NULL) {
    binary_elementwise_subconfig = &config->linear;
  }

  return run_binary_elementwise_nd(
    operator_type,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    binary_elementwise_subconfig,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_add_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  return run_binary_elementwise_nd_f32(
    xnn_operator_type_add_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    output_min, output_max,
    xnn_init_f32_vadd_config(),
    flags,
    threadpool);
}

enum xnn_status xnn_run_divide_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  return run_binary_elementwise_nd_f32(
    xnn_operator_type_divide_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    output_min, output_max,
    xnn_init_f32_vdiv_config(),
    flags,
    threadpool);
}

enum xnn_status xnn_run_maximum_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_binary_elementwise_config* f32_vmax_config = xnn_init_f32_vmax_config();
  if (f32_vmax_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_maximum_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_default_params params;
  if (f32_vmax_config->init.f32_default != NULL) {
    f32_vmax_config->init.f32_default(&params);
  }

  const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig = &f32_vmax_config->minmax;

  return run_binary_elementwise_nd(
    xnn_operator_type_maximum_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    binary_elementwise_subconfig,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_minimum_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_binary_elementwise_config* f32_vmin_config = xnn_init_f32_vmin_config();
  if (f32_vmin_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_minimum_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_default_params params;
  if (f32_vmin_config->init.f32_default != NULL) {
    f32_vmin_config->init.f32_default(&params);
  }
  const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig = &f32_vmin_config->minmax;

  return run_binary_elementwise_nd(
    xnn_operator_type_minimum_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    binary_elementwise_subconfig,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_multiply_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  return run_binary_elementwise_nd_f32(
    xnn_operator_type_multiply_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    output_min, output_max,
    xnn_init_f32_vmul_config(),
    flags,
    threadpool);
}

enum xnn_status xnn_run_subtract_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  return run_binary_elementwise_nd_f32(
    xnn_operator_type_subtract_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    output_min, output_max,
    xnn_init_f32_vsub_config(),
    flags,
    threadpool);
}

enum xnn_status xnn_run_squared_difference_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  const struct xnn_binary_elementwise_config* f32_vsqrdiff_config = xnn_init_f32_vsqrdiff_config();
  if (f32_vsqrdiff_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_squared_difference_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_default_params params;
  if (f32_vsqrdiff_config->init.f32_default != NULL) {
    f32_vsqrdiff_config->init.f32_default(&params);
  }

  const struct xnn_binary_elementwise_subconfig* binary_elementwise_subconfig = &f32_vsqrdiff_config->minmax;

  return run_binary_elementwise_nd(
    xnn_operator_type_squared_difference_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    offsetof(struct xnn_operator, params.f32_minmax), sizeof(params),
    binary_elementwise_subconfig,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_add_nd_qs8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  int8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  int8_t input2_zero_point,
  float input2_scale,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qs8_vadd_config = xnn_init_qs8_vadd_config();
  if (qs8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qs8));
    return xnn_status_unsupported_hardware;
  }
  struct {
    union xnn_qs8_add_minmax_params qs8_add;
    union xnn_qs8_add_minmax_params qs8_radd;
  } params;
  assert(qs8_vadd_config->init.qs8_add != NULL);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, input2_output_scale, output_min, output_max);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    input2_output_scale, input1_output_scale, output_min, output_max);


  return run_binary_elementwise_nd(
    xnn_operator_type_add_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    offsetof(struct xnn_operator, params.qs8_add), sizeof(params.qs8_add),
    offsetof(struct xnn_operator, params.qs8_radd),  sizeof(params.qs8_radd),
    &qs8_vadd_config->minmax,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_multiply_nd_qs8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  int8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  int8_t input2_zero_point,
  float input2_scale,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float product_scale = input1_scale * input2_scale;
  const float product_output_scale = product_scale / output_scale;

  if (product_output_scale < 0x1.0p-16f || product_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g product-to-output scale ratio: scale ratio must be in [2**-16, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8), product_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qs8_vmul_config = xnn_init_qs8_vmul_config();
  if (qs8_vmul_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qs8));
    return xnn_status_unsupported_hardware;
  }
  struct {
    union xnn_qs8_mul_minmax_params qs8_mul;
    union xnn_qs8_mul_minmax_params qs8_rmul;
  } params;

  assert(qs8_vmul_config->init.qs8_mul != NULL);
  qs8_vmul_config->init.qs8_mul(
    &params.qs8_mul, input1_zero_point, input2_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);
  qs8_vmul_config->init.qs8_mul(
    &params.qs8_rmul, input2_zero_point, input1_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);

  return run_binary_elementwise_nd(
    xnn_operator_type_multiply_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    offsetof(struct xnn_operator, params.qs8_mul), sizeof(params.qs8_mul),
    offsetof(struct xnn_operator, params.qs8_rmul),  sizeof(params.qs8_rmul),
    &qs8_vmul_config->minmax,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_subtract_nd_qs8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  int8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  int8_t input2_zero_point,
  float input2_scale,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qs8_vadd_config = xnn_init_qs8_vadd_config();
  if (qs8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qs8));
    return xnn_status_unsupported_hardware;
  }
  struct {
    union xnn_qs8_add_minmax_params qs8_add;
    union xnn_qs8_add_minmax_params qs8_radd;
  } params;
  assert(qs8_vadd_config->init.qs8_add != NULL);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, -input2_output_scale, output_min, output_max);
  qs8_vadd_config->init.qs8_add(
    &params.qs8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    -input2_output_scale, input1_output_scale, output_min, output_max);

  return run_binary_elementwise_nd(
    xnn_operator_type_subtract_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    offsetof(struct xnn_operator, params.qs8_add), sizeof(params.qs8_add),
    offsetof(struct xnn_operator, params.qs8_radd),  sizeof(params.qs8_radd),
    &qs8_vadd_config->minmax,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_add_nd_qu8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  uint8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  uint8_t input2_zero_point,
  float input2_scale,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qu8_vadd_config = xnn_init_qu8_vadd_config();
  if (qu8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_add_nd_qu8));
    return xnn_status_unsupported_hardware;
  }
  struct {
    union xnn_qu8_add_minmax_params qu8_add;
    union xnn_qu8_add_minmax_params qu8_radd;
  } params;
  assert(qu8_vadd_config->init.qu8_add != NULL);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, input2_output_scale, output_min, output_max);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    input2_output_scale, input1_output_scale, output_min, output_max);

  return run_binary_elementwise_nd(
    xnn_operator_type_add_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    offsetof(struct xnn_operator, params.qu8_add), sizeof(params.qu8_add),
    offsetof(struct xnn_operator, params.qu8_radd),  sizeof(params.qu8_radd),
    &qu8_vadd_config->minmax,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_multiply_nd_qu8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  uint8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  uint8_t input2_zero_point,
  float input2_scale,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
 if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float product_scale = input1_scale * input2_scale;
  const float product_output_scale = product_scale / output_scale;
  if (product_output_scale < 0x1.0p-16f || product_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g product-to-output scale ratio: scale ratio must be in [2**-16, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8), product_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qu8_vmul_config = xnn_init_qu8_vmul_config();
  if (qu8_vmul_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_multiply_nd_qu8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qu8_mul_minmax_params qu8_mul;
    union xnn_qu8_mul_minmax_params qu8_rmul;
  } params;
  assert(qu8_vmul_config->init.qu8_mul != NULL);
  qu8_vmul_config->init.qu8_mul(
    &params.qu8_mul, input1_zero_point, input2_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);
  qu8_vmul_config->init.qu8_mul(
    &params.qu8_rmul, input2_zero_point, input1_zero_point, output_zero_point,
    product_output_scale, output_min, output_max);

  return run_binary_elementwise_nd(
    xnn_operator_type_multiply_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    offsetof(struct xnn_operator, params.qu8_mul), sizeof(params.qu8_mul),
    offsetof(struct xnn_operator, params.qu8_rmul),  sizeof(params.qu8_rmul),
    &qu8_vmul_config->minmax,
    &params,
    sizeof(params),
    flags,
    threadpool);
}

enum xnn_status xnn_run_subtract_nd_qu8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  uint8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  uint8_t input2_zero_point,
  float input2_scale,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool)
{
  if (input1_scale <= 0.0f || !isnormal(input1_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 1 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input1_scale);
    return xnn_status_invalid_parameter;
  }

  if (input2_scale <= 0.0f || !isnormal(input2_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input 2 scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input2_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite and positive",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float input1_output_scale = input1_scale / output_scale;
  if (input1_output_scale < 0x1.0p-10f || input1_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input1-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input1_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const float input2_output_scale = input2_scale / output_scale;
  if (input2_output_scale < 0x1.0p-10f || input2_output_scale >= 0x1.0p+8f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input2-to-output scale ratio: scale ratio must be in [2**-10, 2**8) range",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8), input2_output_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_binary_elementwise_config* qu8_vadd_config = xnn_init_qu8_vadd_config();
  if (qu8_vadd_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_subtract_nd_qu8));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_qu8_add_minmax_params qu8_add;
    union xnn_qu8_add_minmax_params qu8_radd;
  } params;
  assert(qu8_vadd_config->init.qu8_add != NULL);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_add, input1_zero_point, input2_zero_point, output_zero_point,
    input1_output_scale, -input2_output_scale, output_min, output_max);
  qu8_vadd_config->init.qu8_add(
    &params.qu8_radd, input2_zero_point, input1_zero_point, output_zero_point,
    -input2_output_scale, input1_output_scale, output_min, output_max);

  return run_binary_elementwise_nd(
    xnn_operator_type_subtract_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    offsetof(struct xnn_operator, params.qu8_add), sizeof(params.qu8_add),
    offsetof(struct xnn_operator, params.qu8_radd),  sizeof(params.qu8_radd),
    &qu8_vadd_config->minmax,
    &params,
    sizeof(params),
    flags,
    threadpool);
}
