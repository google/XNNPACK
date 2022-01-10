// Copyright 2019 Google LLC
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
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static enum xnn_status create_binary_elementwise_nd(
    uint32_t flags,
    const void* params,
    size_t params_size,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    const struct vbinary_fused_ukernels* vbinary_fused_ukernels,
    xnn_operator_t* binary_elementwise_op_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error("failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  xnn_operator_t binary_elementwise_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (binary_elementwise_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }

  if (params_size != 0) {
    memcpy(&binary_elementwise_op->params, params, params_size);
  }

  binary_elementwise_op->ukernel.vbinary.op_function   = vbinary_fused_ukernels->op_ukernel;
  binary_elementwise_op->ukernel.vbinary.opc_function  = vbinary_fused_ukernels->opc_ukernel;
  binary_elementwise_op->ukernel.vbinary.ropc_function = vbinary_fused_ukernels->ropc_ukernel;

  binary_elementwise_op->type = operator_type;
  binary_elementwise_op->flags = flags;

  binary_elementwise_op->state = xnn_run_state_invalid;

  *binary_elementwise_op_out = binary_elementwise_op;
  return xnn_status_success;
}

static enum xnn_status create_binary_elementwise_nd_f16(
    float output_min,
    float output_max,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct vbinary_parameters vbinary[restrict XNN_MIN_ELEMENTS(1)],
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

  union xnn_f16_minmax_params params;
  if (vbinary->init.f16_minmax != NULL) {
    vbinary->init.f16_minmax(&params,
      fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max));
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_F16,
    operator_type,
    &vbinary->minmax,
    binary_elementwise_op_out);
}

static enum xnn_status create_binary_elementwise_nd_f32(
    float output_min,
    float output_max,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct vbinary_parameters vbinary[restrict XNN_MIN_ELEMENTS(1)],
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(operator_type), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  const struct vbinary_fused_ukernels* vbinary_fused_ukernels = &vbinary->minmax;
  if (linear_activation && vbinary->linear.op_ukernel != NULL) {
    vbinary_fused_ukernels = &vbinary->linear;
  }

  union xnn_f32_minmax_params params;
  if (vbinary->init.f32_minmax != NULL) {
    vbinary->init.f32_minmax(&params, output_min, output_max);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_F32,
    operator_type,
    vbinary_fused_ukernels,
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be below upper bound",
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

  struct {
    union xnn_qs8_addsub_minmax_params qs8_addsub;
    union xnn_qs8_addsub_minmax_params qs8_raddsub;
  } params;
  if (xnn_params.qs8.vadd.init.qs8_addsub != NULL) {
    xnn_params.qs8.vadd.init.qs8_addsub(
      &params.qs8_addsub, input1_zero_point, input2_zero_point, output_zero_point,
      input1_output_scale, input2_output_scale, output_min, output_max);
    xnn_params.qs8.vadd.init.qs8_addsub(
      &params.qs8_raddsub, input2_zero_point, input1_zero_point, output_zero_point,
      input2_output_scale, input1_output_scale, output_min, output_max);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_QS8,
    xnn_operator_type_add_nd_qs8,
    &xnn_params.qs8.vadd.minmax,
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be below upper bound",
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

  struct {
    union xnn_qu8_addsub_minmax_params qu8_addsub;
    union xnn_qu8_addsub_minmax_params qu8_raddsub;
  } params;
  if (xnn_params.qu8.vadd.init.qu8_addsub != NULL) {
    xnn_params.qu8.vadd.init.qu8_addsub(
      &params.qu8_addsub, input1_zero_point, input2_zero_point, output_zero_point,
      input1_output_scale, input2_output_scale, output_min, output_max);
    xnn_params.qu8.vadd.init.qu8_addsub(
      &params.qu8_raddsub, input2_zero_point, input1_zero_point, output_zero_point,
      input2_output_scale, input1_output_scale, output_min, output_max);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_QU8,
    xnn_operator_type_add_nd_qu8,
    &xnn_params.qu8.vadd.minmax,
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
    &xnn_params.f16.vadd,
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
    &xnn_params.f32.vadd,
    add_op_out);
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
    &xnn_params.f32.vdiv,
    divide_op_out);
}

enum xnn_status xnn_create_maximum_nd_f32(
    uint32_t flags,
    xnn_operator_t* maximum_op_out)
{
  union xnn_f32_default_params params;
  if (xnn_params.f32.vmin.init.f32_default != NULL) {
    xnn_params.f32.vmin.init.f32_default(&params);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_F32,
    xnn_operator_type_maximum_nd_f32,
    &xnn_params.f32.vmax.minmax,
    maximum_op_out);
}

enum xnn_status xnn_create_minimum_nd_f32(
    uint32_t flags,
    xnn_operator_t* minimum_op_out)
{
  union xnn_f32_default_params params;
  if (xnn_params.f32.vmin.init.f32_default != NULL) {
    xnn_params.f32.vmin.init.f32_default(&params);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_F32,
    xnn_operator_type_minimum_nd_f32,
    &xnn_params.f32.vmin.minmax,
    minimum_op_out);
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be below upper bound",
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

  struct {
    union xnn_qs8_mul_minmax_params qs8_mul;
    union xnn_qs8_mul_minmax_params qs8_rmul;
  } params;
  if (xnn_params.qs8.vmul.init.qs8_mul != NULL) {
    xnn_params.qs8.vmul.init.qs8_mul(
      &params.qs8_mul, input1_zero_point, input2_zero_point, output_zero_point,
      product_output_scale, output_min, output_max);
    xnn_params.qs8.vmul.init.qs8_mul(
      &params.qs8_rmul, input2_zero_point, input1_zero_point, output_zero_point,
      product_output_scale, output_min, output_max);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_QS8,
    xnn_operator_type_multiply_nd_qs8,
    &xnn_params.qs8.vmul.minmax,
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be below upper bound",
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

  struct {
    union xnn_qu8_mul_minmax_params qu8_mul;
    union xnn_qu8_mul_minmax_params qu8_rmul;
  } params;
  if (xnn_params.qu8.vmul.init.qu8_mul != NULL) {
    xnn_params.qu8.vmul.init.qu8_mul(
      &params.qu8_mul, input1_zero_point, input2_zero_point, output_zero_point,
      product_output_scale, output_min, output_max);
    xnn_params.qu8.vmul.init.qu8_mul(
      &params.qu8_rmul, input2_zero_point, input1_zero_point, output_zero_point,
      product_output_scale, output_min, output_max);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_QU8,
    xnn_operator_type_multiply_nd_qu8,
    &xnn_params.qu8.vmul.minmax,
    multiply_op_out);
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
    &xnn_params.f16.vmul,
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
    &xnn_params.f32.vmul,
    multiply_op_out);
}

enum xnn_status xnn_create_squared_difference_nd_f32(
    uint32_t flags,
    xnn_operator_t* squared_difference_op_out)
{
  union xnn_f32_default_params params;
  if (xnn_params.f32.vmin.init.f32_default != NULL) {
    xnn_params.f32.vmin.init.f32_default(&params);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_F32,
    xnn_operator_type_squared_difference_nd_f32,
    &xnn_params.f32.vsqrdiff.minmax,
    squared_difference_op_out);
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be below upper bound",
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

  struct {
    union xnn_qs8_addsub_minmax_params qs8_addsub;
    union xnn_qs8_addsub_minmax_params qs8_raddsub;
  } params;
  if (xnn_params.qs8.vadd.init.qs8_addsub != NULL) {
    xnn_params.qs8.vadd.init.qs8_addsub(
      &params.qs8_addsub, input1_zero_point, input2_zero_point, output_zero_point,
      input1_output_scale, -input2_output_scale, output_min, output_max);
    xnn_params.qs8.vadd.init.qs8_addsub(
      &params.qs8_raddsub, input2_zero_point, input1_zero_point, output_zero_point,
      -input2_output_scale, input1_output_scale, output_min, output_max);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_QS8,
    xnn_operator_type_subtract_nd_qs8,
    &xnn_params.qs8.vadd.minmax,
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

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be below upper bound",
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

  struct {
    union xnn_qu8_addsub_minmax_params qu8_addsub;
    union xnn_qu8_addsub_minmax_params qu8_raddsub;
  } params;
  if (xnn_params.qu8.vadd.init.qu8_addsub != NULL) {
    xnn_params.qu8.vadd.init.qu8_addsub(
      &params.qu8_addsub, input1_zero_point, input2_zero_point, output_zero_point,
      input1_output_scale, -input2_output_scale, output_min, output_max);
    xnn_params.qu8.vadd.init.qu8_addsub(
      &params.qu8_raddsub, input2_zero_point, input1_zero_point, output_zero_point,
      -input2_output_scale, input1_output_scale, output_min, output_max);
  }
  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_QU8,
    xnn_operator_type_subtract_nd_qu8,
    &xnn_params.qu8.vadd.minmax,
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
    &xnn_params.f32.vsub,
    subtract_op_out);
}

static enum xnn_status setup_binary_elementwise_nd(
    xnn_operator_t binary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const void* input1,
    const void* input2,
    void* output,
    uint32_t datatype_init_flags,
    uint32_t log2_element_size,
    const void* params,
    size_t params_size,
    const void* reversed_params,
    size_t reversed_params_size,
    const struct vbinary_parameters vbinary[restrict XNN_MIN_ELEMENTS(1)],
    size_t num_threads)
{
  binary_elementwise_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(binary_elementwise_op->type));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error("failed to setup %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(binary_elementwise_op->type));
    return xnn_status_unsupported_hardware;
  }

  if (binary_elementwise_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(binary_elementwise_op->type));
    return xnn_status_invalid_parameter;
  }

  if (max(num_input1_dims, num_input2_dims) > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to setup %s operator with %zu and %zu dimensions in input shapes: "
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
        "failed to setup %s operator: "
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
    .a = input1,
    .b = input2,
    .y = output,
    .elements = compressed_output_shape[0] << log2_element_size,
  };
  if (params_size != 0) {
    memcpy(&binary_elementwise_op->context.elementwise_binary.params, params, params_size);
  }

  const size_t* compressed_a_shape = compressed_input1_shape;
  const size_t* compressed_b_shape = compressed_input2_shape;
  if (compressed_input1_shape[0] == 1) {
    binary_elementwise_op->context.elementwise_binary.ukernel = binary_elementwise_op->ukernel.vbinary.ropc_function;
    binary_elementwise_op->context.elementwise_binary.a = input2;
    binary_elementwise_op->context.elementwise_binary.b = input1;
    compressed_a_shape = compressed_input2_shape;
    compressed_b_shape = compressed_input1_shape;
    if (reversed_params_size != 0) {
      memcpy(&binary_elementwise_op->context.elementwise_binary.params, reversed_params, reversed_params_size);
    }
  } else if (compressed_input2_shape[0] == 1) {
    binary_elementwise_op->context.elementwise_binary.ukernel = binary_elementwise_op->ukernel.vbinary.opc_function;
  } else if (compressed_input1_shape[0] == compressed_input2_shape[0]) {
    binary_elementwise_op->context.elementwise_binary.ukernel = binary_elementwise_op->ukernel.vbinary.op_function;
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

  binary_elementwise_op->compute.type = xnn_parallelization_type_5d;
  binary_elementwise_op->compute.task_5d = (pthreadpool_task_5d_t) xnn_compute_elementwise_binary_5d;
  binary_elementwise_op->compute.range[0] = compressed_output_shape[5];
  binary_elementwise_op->compute.range[1] = compressed_output_shape[4];
  binary_elementwise_op->compute.range[2] = compressed_output_shape[3];
  binary_elementwise_op->compute.range[3] = compressed_output_shape[2];
  binary_elementwise_op->compute.range[4] = compressed_output_shape[1];
  binary_elementwise_op->compute.tile[0] = 1;
  binary_elementwise_op->compute.tile[1] = 1;
  binary_elementwise_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_binary_elementwise_nd_f16(
    xnn_operator_t binary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const void* input1,
    const void* input2,
    void* output,
    const struct vbinary_parameters vbinary[restrict XNN_MIN_ELEMENTS(1)],
    size_t num_threads)
{
  return setup_binary_elementwise_nd(
    binary_elementwise_op,
    expected_operator_type,
    num_input1_dims,
    input1_shape,
    num_input2_dims,
    input2_shape,
    input1,
    input2,
    output,
    XNN_INIT_FLAG_F16,
    1 /* log2(sizeof(half)) */,
    &binary_elementwise_op->params.f16_minmax, sizeof(binary_elementwise_op->params.f16_minmax),
    &binary_elementwise_op->params.f16_minmax, sizeof(binary_elementwise_op->params.f16_minmax),
    vbinary,
    num_threads);
}

static enum xnn_status setup_binary_elementwise_nd_f32(
    xnn_operator_t binary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    const struct vbinary_parameters vbinary[restrict XNN_MIN_ELEMENTS(1)],
    size_t num_threads)
{
  return setup_binary_elementwise_nd(
    binary_elementwise_op, expected_operator_type,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    XNN_INIT_FLAG_F32,
    2 /* log2(sizeof(float)) */,
    &binary_elementwise_op->params.f32_minmax, sizeof(binary_elementwise_op->params.f32_minmax),
    &binary_elementwise_op->params.f32_minmax, sizeof(binary_elementwise_op->params.f32_minmax),
    vbinary,
    num_threads);
}

enum xnn_status xnn_setup_add_nd_qs8(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const int8_t* input1,
    const int8_t* input2,
    int8_t* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    XNN_INIT_FLAG_QS8,
    0 /* log2(sizeof(int8_t))) */,
    &add_op->params.qs8_addsub, sizeof(add_op->params.qs8_addsub),
    &add_op->params.qs8_raddsub, sizeof(add_op->params.qs8_raddsub),
    &xnn_params.qs8.vadd,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_add_nd_qu8(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const uint8_t* input1,
    const uint8_t* input2,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd(
    add_op, xnn_operator_type_add_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    XNN_INIT_FLAG_QU8,
    0 /* log2(sizeof(uint8_t))) */,
    &add_op->params.qu8_addsub, sizeof(add_op->params.qu8_addsub),
    &add_op->params.qu8_raddsub, sizeof(add_op->params.qu8_raddsub),
    &xnn_params.qu8.vadd,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_add_nd_f16(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const void* input1,
    const void* input2,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f16(
    add_op, xnn_operator_type_add_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f16.vadd,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_add_nd_f32(
    xnn_operator_t add_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f32(
    add_op, xnn_operator_type_add_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f32.vadd,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_divide_nd_f32(
    xnn_operator_t divide_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f32(
    divide_op, xnn_operator_type_divide_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f32.vdiv,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_maximum_nd_f32(
    xnn_operator_t maximum_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f32(
    maximum_op, xnn_operator_type_maximum_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f32.vmax,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_minimum_nd_f32(
    xnn_operator_t minimum_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f32(
    minimum_op, xnn_operator_type_minimum_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f32.vmin,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_multiply_nd_qs8(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const int8_t* input1,
    const int8_t* input2,
    int8_t* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    XNN_INIT_FLAG_QS8,
    0 /* log2(sizeof(int8_t))) */,
    &multiply_op->params.qs8_mul, sizeof(multiply_op->params.qs8_mul),
    &multiply_op->params.qs8_rmul, sizeof(multiply_op->params.qs8_rmul),
    &xnn_params.qs8.vmul,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_multiply_nd_qu8(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const uint8_t* input1,
    const uint8_t* input2,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd(
    multiply_op, xnn_operator_type_multiply_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    XNN_INIT_FLAG_QU8,
    0 /* log2(sizeof(uint8_t))) */,
    &multiply_op->params.qu8_mul, sizeof(multiply_op->params.qu8_mul),
    &multiply_op->params.qu8_rmul, sizeof(multiply_op->params.qu8_rmul),
    &xnn_params.qu8.vmul,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_multiply_nd_f16(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const void* input1,
    const void* input2,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f16(
    multiply_op, xnn_operator_type_multiply_nd_f16,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f16.vmul,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_multiply_nd_f32(
    xnn_operator_t multiply_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f32(
    multiply_op, xnn_operator_type_multiply_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f32.vmul,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_squared_difference_nd_f32(
    xnn_operator_t squared_difference_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f32(
    squared_difference_op, xnn_operator_type_squared_difference_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f32.vsqrdiff,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_subtract_nd_qs8(
    xnn_operator_t subtract_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const int8_t* input1,
    const int8_t* input2,
    int8_t* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_qs8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    XNN_INIT_FLAG_QS8,
    0 /* log2(sizeof(int8_t))) */,
    &subtract_op->params.qs8_addsub, sizeof(subtract_op->params.qs8_addsub),
    &subtract_op->params.qs8_raddsub, sizeof(subtract_op->params.qs8_raddsub),
    &xnn_params.qs8.vadd,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_subtract_nd_qu8(
    xnn_operator_t subtract_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const uint8_t* input1,
    const uint8_t* input2,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd(
    subtract_op, xnn_operator_type_subtract_nd_qu8,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    XNN_INIT_FLAG_QU8,
    0 /* log2(sizeof(uint8_t))) */,
    &subtract_op->params.qu8_addsub, sizeof(subtract_op->params.qu8_addsub),
    &subtract_op->params.qu8_raddsub, sizeof(subtract_op->params.qu8_raddsub),
    &xnn_params.qu8.vadd,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_subtract_nd_f32(
    xnn_operator_t subtract_op,
    size_t num_input1_dims,
    const size_t* input1_shape,
    size_t num_input2_dims,
    const size_t* input2_shape,
    const float* input1,
    const float* input2,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_binary_elementwise_nd_f32(
    subtract_op, xnn_operator_type_subtract_nd_f32,
    num_input1_dims, input1_shape,
    num_input2_dims, input2_shape,
    input1, input2, output,
    &xnn_params.f32.vsub,
    pthreadpool_get_threads_count(threadpool));
}
