// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/pack.h>
#include <xnnpack/params.h>


static enum xnn_status create_dynamic_fully_connected_nc(
    uint32_t flags,
    uint32_t log2_input_element_size,
    const void* params,
    size_t params_size,
    const struct xnn_gemm_config* gemm_config,
    const struct gemm_fused_ukernels* gemm_ukernels,
    enum xnn_operator_type operator_type,
    xnn_operator_t* dynamic_fully_connected_op_out)
{
  xnn_operator_t dynamic_fully_connected_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  dynamic_fully_connected_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (dynamic_fully_connected_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  memcpy(&dynamic_fully_connected_op->params, params, params_size);
  dynamic_fully_connected_op->type = operator_type;
  dynamic_fully_connected_op->flags = flags;

  const size_t nr = gemm_config->nr;
  const size_t mr = gemm_config->mr;
  dynamic_fully_connected_op->ukernel.type = xnn_microkernel_type_gemm;
  dynamic_fully_connected_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
    .mr = mr,
    .nr = nr,
    .kr = UINT32_C(1) << gemm_config->log2_kr,
    .sr = UINT32_C(1) << gemm_config->log2_sr,
  };

  assert(XNN_MAX_MR >= mr);
  for (size_t i = 0; i < mr; i++) {
    dynamic_fully_connected_op->ukernel.gemm.gemm_cases[i] = gemm_ukernels->gemm[i];
  }
  dynamic_fully_connected_op->ukernel.gemm.packw_gemm_goi = gemm_config->pack_gemm_goi;

  dynamic_fully_connected_op->state = xnn_run_state_invalid;

  *dynamic_fully_connected_op_out = dynamic_fully_connected_op;
  return xnn_status_success;

error:
  xnn_delete_operator(dynamic_fully_connected_op);
  return status;
}

enum xnn_status xnn_create_dynamic_fully_connected_nc_f16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* dynamic_fully_connected_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f16));
    return xnn_status_invalid_parameter;
  }

  const uint16_t fp16_output_min = fp16_ieee_from_fp32_value(output_min);
  const uint16_t fp16_output_max = fp16_ieee_from_fp32_value(output_max);
  const float rounded_output_min = fp16_ieee_to_fp32_value(fp16_output_min);
  const float rounded_output_max = fp16_ieee_to_fp32_value(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f16), rounded_output_min,
      rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, fp16_output_min, fp16_output_max);
  }

  return create_dynamic_fully_connected_nc(
    flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    &params, sizeof(params),
    gemm_config, &gemm_config->minmax,
    xnn_operator_type_dynamic_fully_connected_nc_f16,
    dynamic_fully_connected_op_out);
}

enum xnn_status xnn_create_dynamic_fully_connected_nc_f32(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* dynamic_fully_connected_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr-1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  union xnn_f32_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  return create_dynamic_fully_connected_nc(
    flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &params, sizeof(params),
    gemm_config, gemm_ukernels,
    xnn_operator_type_dynamic_fully_connected_nc_f32,
    dynamic_fully_connected_op_out);
}

static enum xnn_status setup_dynamic_fully_connected_nc(
    xnn_operator_t dynamic_fully_connected_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const void* input,
    const void* kernel,
    const void* bias,
    void* output,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    uint32_t log2_output_element_size,
    const void* params,
    size_t params_size,
    size_t num_threads)
{
  if (dynamic_fully_connected_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(dynamic_fully_connected_op->type));
    return xnn_status_invalid_parameter;
  }
  dynamic_fully_connected_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(dynamic_fully_connected_op->type));
    return xnn_status_uninitialized;
  }

  if (input_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu input channels: number of channels must be non-zero",
      xnn_operator_type_to_string(dynamic_fully_connected_op->type), input_channels);
    return xnn_status_invalid_parameter;
  }

  if (output_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu output channels: number of channels must be non-zero",
      xnn_operator_type_to_string(dynamic_fully_connected_op->type), output_channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of input channels (%zu)",
      xnn_operator_type_to_string(dynamic_fully_connected_op->type), input_stride, input_channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of output channels (%zu)",
      xnn_operator_type_to_string(dynamic_fully_connected_op->type), output_stride, output_channels);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    dynamic_fully_connected_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const uint32_t nr = dynamic_fully_connected_op->ukernel.gemm.nr;
  const uint32_t kr = dynamic_fully_connected_op->ukernel.gemm.kr;
  const uint32_t sr = dynamic_fully_connected_op->ukernel.gemm.sr;

  const size_t n_stride = round_up(output_channels, nr);
  const size_t k_stride = round_up_po2(input_channels, kr * sr);

  if (input_channels != dynamic_fully_connected_op->group_input_channels ||
      output_channels != dynamic_fully_connected_op->group_output_channels)
  {
    const size_t workspace_size = n_stride * bias_element_size + ((n_stride * k_stride) << log2_filter_element_size);

    if (workspace_size > dynamic_fully_connected_op->workspace_size) {
      // Note: packed weights must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(dynamic_fully_connected_op->workspace);
      dynamic_fully_connected_op->workspace = xnn_allocate_simd_memory(workspace_size);
      if (dynamic_fully_connected_op->workspace == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator workspace",
          workspace_size, xnn_operator_type_to_string(dynamic_fully_connected_op->type));
        return xnn_status_out_of_memory;
      }
      dynamic_fully_connected_op->workspace_size = workspace_size;
    }
    dynamic_fully_connected_op->group_input_channels = input_channels;
    dynamic_fully_connected_op->group_output_channels = output_channels;
  }

  uint32_t mr = dynamic_fully_connected_op->ukernel.gemm.mr;
  struct xnn_hmp_gemm_ukernel *gemm_cases = dynamic_fully_connected_op->ukernel.gemm.gemm_cases;

  if (batch_size == 1 && dynamic_fully_connected_op->ukernel.gemm.gemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
    mr = 1;
  }

  assert(mr != 0 && mr <= XNN_MAX_MR);
  struct xnn_hmp_gemm_ukernel gemm_ukernel = gemm_cases[mr-1];

  assert(dynamic_fully_connected_op->ukernel.gemm.packw_gemm_goi != NULL);
  dynamic_fully_connected_op->context.packw_gemm_goi = (struct packw_gemm_goi_context) {
    .k = input_channels,
    .nr = nr,
    .kr = kr,
    .sr = sr,
    .kernel = kernel,
    .k_stride = input_channels << log2_input_element_size,
    .bias = bias,
    .b_stride = bias_element_size,
    .packed_weights = dynamic_fully_connected_op->workspace,
    .w_stride = bias_element_size + (k_stride << log2_input_element_size),
    .packw_gemm_goi = dynamic_fully_connected_op->ukernel.gemm.packw_gemm_goi,
  };
  dynamic_fully_connected_op->context.gemm = (struct gemm_context) {
    .k_scaled = input_channels << log2_input_element_size,
    .w_stride = bias_element_size +
        (round_up_po2(input_channels, dynamic_fully_connected_op->ukernel.gemm.kr * dynamic_fully_connected_op->ukernel.gemm.sr) << log2_input_element_size),
    .a = input,
    .a_stride = input_stride << log2_input_element_size,
    .packed_w = dynamic_fully_connected_op->workspace,
    .c = output,
    .cm_stride = output_stride << log2_output_element_size,
    .cn_stride = nr << log2_output_element_size,
    .log2_csize = log2_output_element_size,
    .ukernel = gemm_ukernel,
  };
  memcpy(&dynamic_fully_connected_op->context.gemm.params, params, params_size);
  dynamic_fully_connected_op->context.gemm.fused_params = &dynamic_fully_connected_op->context.gemm.params;

  dynamic_fully_connected_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
  dynamic_fully_connected_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_packw_gemm_goi;
  dynamic_fully_connected_op->compute[0].context_offset = offsetof(struct xnn_operator, context.packw_gemm_goi) - offsetof(struct xnn_operator, context);
  dynamic_fully_connected_op->compute[0].range[0] = output_channels;
  dynamic_fully_connected_op->compute[0].tile[0] = nr;

  #if XNN_TEST_MODE
    const size_t nc = nr;
  #else
    size_t nc = output_channels;
    if (num_threads > 1) {
      const size_t num_other_tiles = divide_round_up(batch_size, mr);
      const size_t target_tiles_per_thread = 5;
      const size_t max_nc = divide_round_up(output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
      if (max_nc < nc) {
        nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
      }
    }
  #endif
  #if XNN_MAX_UARCH_TYPES > 1
    if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
      dynamic_fully_connected_op->compute[1].type = xnn_parallelization_type_2d_tile_2d_with_uarch;
      dynamic_fully_connected_op->compute[1].task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_gemm;
    } else {
      dynamic_fully_connected_op->compute[1].type = xnn_parallelization_type_2d_tile_2d;
      dynamic_fully_connected_op->compute[1].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
    }
  #else
    dynamic_fully_connected_op->compute[1].type = xnn_parallelization_type_2d_tile_2d;
    dynamic_fully_connected_op->compute[1].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
  #endif
  dynamic_fully_connected_op->compute[1].range[0] = batch_size;
  dynamic_fully_connected_op->compute[1].range[1] = output_channels;
  dynamic_fully_connected_op->compute[1].tile[0] = mr;
  dynamic_fully_connected_op->compute[1].tile[1] = nc;
  dynamic_fully_connected_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_dynamic_fully_connected_nc_f16(
    xnn_operator_t dynamic_fully_connected_op,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const void* input,
    const void* kernel,
    const void* bias,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_f16,
    batch_size, input_channels, output_channels, input_stride, output_stride,
    input, kernel, bias, output,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*bias_element_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    &dynamic_fully_connected_op->params.f16_minmax,
    sizeof(dynamic_fully_connected_op->params.f16_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_dynamic_fully_connected_nc_f32(
    xnn_operator_t dynamic_fully_connected_op,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_f32,
    batch_size, input_channels, output_channels, input_stride, output_stride,
    input, kernel, bias, output,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*bias_element_size=*/sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &dynamic_fully_connected_op->params.f32_minmax,
    sizeof(dynamic_fully_connected_op->params.f32_minmax),
    pthreadpool_get_threads_count(threadpool));
}
