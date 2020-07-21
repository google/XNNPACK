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
  binary_elementwise_op->ukernel.type = xnn_ukernel_type_binary_elementwise;

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

  const struct xnn_f16_minmax_params params = xnn_init_f16_minmax_params(
    fp16_ieee_from_fp32_value(output_min),
    fp16_ieee_from_fp32_value(output_max));

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

  const union xnn_f32_minmax_params params = xnn_init_f32_minmax_params(output_min, output_max);

  return create_binary_elementwise_nd(
    flags,
    &params,
    sizeof(params),
    XNN_INIT_FLAG_F32,
    operator_type,
    vbinary_fused_ukernels,
    binary_elementwise_op_out);
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
  return create_binary_elementwise_nd(
    flags,
    NULL /* params */,
    0 /* params size */,
    XNN_INIT_FLAG_F32,
    xnn_operator_type_maximum_nd_f32,
    &xnn_params.f32.vmax.minmax,
    maximum_op_out);
}

enum xnn_status xnn_create_minimum_nd_f32(
    uint32_t flags,
    xnn_operator_t* minimum_op_out)
{
  return create_binary_elementwise_nd(
    flags,
    NULL /* params */,
    0 /* params size */,
    XNN_INIT_FLAG_F32,
    xnn_operator_type_minimum_nd_f32,
    &xnn_params.f32.vmin.minmax,
    minimum_op_out);
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
  return create_binary_elementwise_nd(
    flags,
    NULL /* params */,
    0 /* params size */,
    XNN_INIT_FLAG_F32,
    xnn_operator_type_squared_difference_nd_f32,
    &xnn_params.f32.vsqrdiff.minmax,
    squared_difference_op_out);
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

  for (size_t i = 0; i < num_input1_dims; i++) {
    if (input1_shape[i] == 0) {
      xnn_log_error(
        "failed to setup %s operator: shape dimension #%zu of input #1 is zero",
        xnn_operator_type_to_string(binary_elementwise_op->type), i);
      return xnn_status_invalid_parameter;
    }
  }

  for (size_t i = 0; i < num_input2_dims; i++) {
    if (input2_shape[i] == 0) {
      xnn_log_error(
        "failed to setup %s operator: shape dimension #%zu of input #2 is zero",
        xnn_operator_type_to_string(binary_elementwise_op->type), i);
      return xnn_status_invalid_parameter;
    }
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
  const size_t num_common_dims = min(num_input1_dims, num_input2_dims);
  for (size_t i = 1; i <= num_common_dims; i++) {
    const size_t input1_dim = input1_shape[num_input1_dims - i];
    const size_t input2_dim = input2_shape[num_input2_dims - i];
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
      compressed_input1_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input1_dim;
    }
  } else if (num_input2_dims > num_input1_dims) {
    if (!broadcast_input1) {
      num_compressed_dims++;
    }
    for (size_t i = 0; i < num_input2_dims - num_input1_dims; i++) {
      const size_t input2_dim = input2_shape[i];
      compressed_input2_shape[num_compressed_dims - 1] *= input2_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input2_dim;
    }
  }
  num_compressed_dims = max(num_compressed_dims, 1);

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

  binary_elementwise_op->compute.type = xnn_parallelization_type_5d_tile_2d;
  binary_elementwise_op->compute.task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_elementwise_binary_5d;
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
    binary_elementwise_op,
    expected_operator_type,
    num_input1_dims,
    input1_shape,
    num_input2_dims,
    input2_shape,
    input1,
    input2,
    output,
    XNN_INIT_FLAG_F32,
    2 /* log2(sizeof(float)) */,
    &binary_elementwise_op->params.f32_minmax, sizeof(binary_elementwise_op->params.f32_minmax),
    vbinary,
    num_threads);
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
    1 /* log2(sizeof(float)) */,
    &binary_elementwise_op->params.f16_minmax, sizeof(binary_elementwise_op->params.f16_minmax),
    vbinary,
    num_threads);
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
