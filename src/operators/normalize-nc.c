// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

static enum xnn_status create_normalize_nc_floating_point(
    enum xnn_norm_type norm_type, float epsilon, uint32_t flags,
    const struct xnn_reduce_config* rsum2_config,
    const struct xnn_binary_elementwise_config* vmul_config,
    enum xnn_operator_type operator_type, xnn_operator_t* normalize_op_out) {
  xnn_operator_t normalize_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  normalize_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (normalize_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_operator),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }
  normalize_op->compute =
      xnn_allocate_zero_memory(sizeof(struct compute_parameters));
  if (normalize_op->compute == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct compute_parameters),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }
  normalize_op->num_compute_invocations = 1;

  normalize_op->type = operator_type;
  normalize_op->flags = flags;
  normalize_op->reduce_config = rsum2_config;
  normalize_op->vmul_config = vmul_config;
  normalize_op->norm_type = norm_type;
  normalize_op->normalize_epsilon = epsilon;

  normalize_op->state = xnn_run_state_invalid;

  *normalize_op_out = normalize_op;
  return xnn_status_success;

error:
  xnn_delete_operator(normalize_op);
  return status;
}

enum xnn_status xnn_create_normalize_nc_f16(enum xnn_norm_type norm_type,
                                            float epsilon, uint32_t flags,
                                            xnn_operator_t* normalize_op_out) {
  const struct xnn_reduce_config* rsum2_config =
      xnn_init_f16_f32acc_rsum2_config();
  if (rsum2_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_normalize_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_binary_elementwise_config* vmul_config =
      xnn_init_f16_vmul_config();
  if (vmul_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_normalize_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  return create_normalize_nc_floating_point(
      norm_type, epsilon, flags, rsum2_config, vmul_config,
      xnn_operator_type_normalize_nc_f16, normalize_op_out);
}

enum xnn_status xnn_create_normalize_nc_f32(enum xnn_norm_type norm_type,
                                            float epsilon, uint32_t flags,
                                            xnn_operator_t* normalize_op_out) {
  const struct xnn_reduce_config* rsum2_config = xnn_init_f32_rsum2_config();
  if (rsum2_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_normalize_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_binary_elementwise_config* vmul_config =
      xnn_init_f32_vmul_config();
  if (vmul_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_normalize_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  return create_normalize_nc_floating_point(
      norm_type, epsilon, flags, rsum2_config, vmul_config,
      xnn_operator_type_normalize_nc_f32, normalize_op_out);
}

static enum xnn_status reshape_normalize_nc_floating_point(
    xnn_operator_t normalize_op, enum xnn_operator_type expected_operator_type,
    size_t channels, size_t input_stride, size_t output_stride,
    size_t batch_size, uint32_t log2_element_size,
    xnn_rsum2_ukernel_fn rsum2_ukernel,
    const struct xnn_binary_elementwise_config* vmul,
    xnn_convert_scale_fn convert_scale, const void* rsum2_params,
    size_t rsum2_params_size, const void* minmax_params,
    size_t minmax_params_size) {
  if (vmul == NULL) {
    return xnn_status_unsupported_hardware;
  }
  if (normalize_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(normalize_op));
    return xnn_status_invalid_parameter;
  }
  normalize_op->state = xnn_run_state_invalid;

  if (channels == 0) {
    xnn_log_error(
        "failed to create %s operator with %zu channels: number of channels "
        "must be non-zero",
        xnn_operator_type_to_string(expected_operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < channels) {
    xnn_log_error(
        "failed to create %s operator with input element stride of %zu: "
        "stride must be at least as large as the number of channels (%zu)",
        xnn_operator_type_to_string(expected_operator_type), input_stride,
        channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
        "failed to create %s operator with output element stride of %zu: "
        "stride must be at least as large as the number of channels (%zu)",
        xnn_operator_type_to_string(expected_operator_type), output_stride,
        channels);
    return xnn_status_invalid_parameter;
  }

  normalize_op->channels = channels;
  normalize_op->input_pixel_stride = input_stride;
  normalize_op->output_pixel_stride = output_stride;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    normalize_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  struct normalize_context* context = &normalize_op->context.normalize;
  *context = (struct normalize_context){
      .n = normalize_op->channels << log2_element_size,
      .x_stride = normalize_op->input_pixel_stride << log2_element_size,
      .y_stride = normalize_op->output_pixel_stride << log2_element_size,
      .num_channels = normalize_op->channels,
      .rsum2_ukernel = rsum2_ukernel,
      .vmul_ukernel = vmul->op_ukernel,
      .vmulc_ukernel = vmul->opc_ukernel,
      .convert_scale = convert_scale,
      .norm_type = normalize_op->norm_type,
      .epsilon = normalize_op->normalize_epsilon,
  };
  if (vmul->opc_ukernel != NULL) {
    context->vmulc_ukernel = vmul->opc_ukernel;
  };
  if (rsum2_params_size > 0) {
    memcpy(&context->rsum2_params, rsum2_params, rsum2_params_size);
  }
  if (minmax_params_size > 0) {
    memcpy(&context->minmax_params, minmax_params, minmax_params_size);
  }
  normalize_op->compute[0].type = xnn_parallelization_type_1d;
  normalize_op->compute[0].task_1d =
      (pthreadpool_task_1d_t)xnn_compute_normalize;
  normalize_op->compute[0].range[0] = batch_size;
  normalize_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static enum xnn_status setup_normalize_nc_floating_point(
    xnn_operator_t normalize_op, enum xnn_operator_type expected_operator_type,
    const void* input, const void* scale, void* output) {
  if (normalize_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(normalize_op));
    return xnn_status_invalid_parameter;
  }

  switch (normalize_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(normalize_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different
      // pointers.
      break;
  }

  normalize_op->context.normalize.x = input;
  normalize_op->context.normalize.y = output;
  normalize_op->context.normalize.scale = scale;
  normalize_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_normalize_nc_f16(xnn_operator_t normalize_op,
                                           const void* input, const void* scale,
                                           void* output) {
  return setup_normalize_nc_floating_point(
      normalize_op, xnn_operator_type_normalize_nc_f16, input, scale, output);
}

enum xnn_status xnn_setup_normalize_nc_f32(xnn_operator_t normalize_op,
                                           const float* input,
                                           const float* scale, float* output) {
  return setup_normalize_nc_floating_point(
      normalize_op, xnn_operator_type_normalize_nc_f32, input, scale, output);
}

static void convert_scale_to_fp16(float input, void* output) {
  *(xnn_float16*)output = xnn_float16_from_float(input);
}

static void convert_scale_to_fp32(float input, void* output) {
  *(float*)output = input;
}

enum xnn_status xnn_reshape_normalize_nc_f16(
    xnn_operator_t normalize_op, size_t channels, size_t input_stride,
    size_t output_stride, size_t batch_size, pthreadpool_t threadpool) {
  const struct xnn_f16_f32acc_scale_params rsum2_params = {
      .scalar = {.scale = 1.0f}};

  return reshape_normalize_nc_floating_point(
      normalize_op, xnn_operator_type_normalize_nc_f16, channels, input_stride,
      output_stride, batch_size,
      /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
      normalize_op->reduce_config->ukernel, normalize_op->vmul_config,
      /*convert_scale=*/convert_scale_to_fp16,
      /*rsum2_params=*/&rsum2_params,
      /*rsum2_params_size=*/sizeof(rsum2_params),
      /*minmax_params=*/NULL, /*minmax_params_size=*/0);
}

enum xnn_status xnn_reshape_normalize_nc_f32(
    xnn_operator_t normalize_op, size_t channels, size_t input_stride,
    size_t output_stride, size_t batch_size, pthreadpool_t threadpool) {
  const struct xnn_f32_scale_params rsum2_params = {.scalar = {.scale = 1.0f}};

  return reshape_normalize_nc_floating_point(
      normalize_op, xnn_operator_type_normalize_nc_f32, channels, input_stride,
      output_stride, batch_size,
      /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      normalize_op->reduce_config->ukernel, normalize_op->vmul_config,
      /*convert_scale=*/convert_scale_to_fp32,
      /*rsum2_params=*/&rsum2_params,
      /*rsum2_params_size=*/sizeof(rsum2_params),
      /*minmax_params=*/NULL, /*minmax_params_size=*/0);
}
