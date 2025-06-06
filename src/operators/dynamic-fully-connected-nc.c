// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
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
#include "src/xnnpack/microkernel-type.h"
#include "src/xnnpack/microkernel-utils.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

static enum xnn_status create_dynamic_fully_connected_nc(
    uint32_t flags,
    uint32_t log2_input_element_size,
    const void* params,
    size_t params_size,
    const void* params2,
    size_t params2_size,
    const struct xnn_gemm_config* gemm_config,
    const struct gemm_fused_ukernels* gemm_ukernels,
    const struct xnn_gemm_config* gemm_nr2_config,
    const struct gemm_fused_ukernels* gemm_nr2_ukernels,
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
  const int num_compute_invocations = 3;
  dynamic_fully_connected_op->compute = xnn_allocate_zero_memory(num_compute_invocations * sizeof(struct compute_parameters));
  if (dynamic_fully_connected_op->compute == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct compute_parameters),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }
  dynamic_fully_connected_op->num_compute_invocations = num_compute_invocations;
  dynamic_fully_connected_op->params2 = xnn_allocate_zero_memory(sizeof(union xnn_params2));
  if (dynamic_fully_connected_op->params2 == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(union xnn_params2),
                  xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }

  dynamic_fully_connected_op->ukernel.gemm_ukernels = xnn_allocate_zero_simd_memory(sizeof(struct gemm_types));
  if (dynamic_fully_connected_op->ukernel.gemm_ukernels == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct gemm_types),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  dynamic_fully_connected_op->dynamic_context.gemm = xnn_allocate_zero_simd_memory(sizeof(struct gemm_op_context));
  if (dynamic_fully_connected_op->dynamic_context.gemm == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct gemm_op_context), xnn_operator_type_to_string(operator_type));
    goto error;
  }


  memcpy(&dynamic_fully_connected_op->params, params, params_size);
  memcpy(dynamic_fully_connected_op->params2, params2, params2_size);
  dynamic_fully_connected_op->type = operator_type;
  dynamic_fully_connected_op->flags = flags;

  const size_t nr = gemm_config->nr;
  const size_t mr = gemm_config->mr;
  const size_t mr_packed = gemm_config->mr_packed ? gemm_config->mr_packed : mr;
  dynamic_fully_connected_op->ukernel.type = xnn_microkernel_type_gemm;
  dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm = (struct xnn_ukernel_gemm) {
    .mr = mr,
    .nr = nr,
    .kr = UINT32_C(1) << gemm_config->log2_kr,
    .sr = UINT32_C(1) << gemm_config->log2_sr,
    .mr_packed = mr_packed,
  };
  dynamic_fully_connected_op->gemm_config = gemm_config;

  assert(mr <= XNN_MAX_MR);
  for (size_t i = 0; i < mr; i++) {
    dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm.gemm_cases[i] = gemm_ukernels->gemm[i];
  }
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm.packw_gemm_gio = gemm_config->pack_gemm_gio;
  } else {
    dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm.packw_gemm_goi = gemm_config->pack_gemm_goi;
  }

  if (gemm_nr2_config != NULL) {
    dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm_nr2 = (struct xnn_ukernel_gemm) {
      .mr = gemm_nr2_config->mr,
      .nr = gemm_nr2_config->nr,
      .kr = UINT32_C(1) << gemm_nr2_config->log2_kr,
      .sr = UINT32_C(1) << gemm_nr2_config->log2_sr,
    };
    assert(gemm_nr2_config->mr <= XNN_MAX_MR);
    for (size_t i = 0; i < gemm_nr2_config->mr; i++) {
      dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm_nr2.gemm_cases[i] = gemm_nr2_ukernels->gemm[i];
    }
    if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
      dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm_nr2.packw_gemm_gio = gemm_nr2_config->pack_gemm_gio;
    } else {
      dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm_nr2.packw_gemm_goi = gemm_nr2_config->pack_gemm_goi;
    }
  }

  dynamic_fully_connected_op->state = xnn_run_state_invalid;

  *dynamic_fully_connected_op_out = dynamic_fully_connected_op;
  return xnn_status_success;

error:
  xnn_delete_operator(dynamic_fully_connected_op);
  return status;
}

enum xnn_status create_dynamic_fully_connected_nc_f16(
  float output_min,
  float output_max,
  uint32_t flags,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
  xnn_operator_t* dynamic_fully_connected_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  const xnn_float16 fp16_output_min = xnn_float16_from_float(output_min);
  const xnn_float16 fp16_output_max = xnn_float16_from_float(output_max);
  const float rounded_output_min = xnn_float16_to_float(fp16_output_min);
  const float rounded_output_max = xnn_float16_to_float(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(expected_operator_type), rounded_output_min,
      rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  struct xnn_f16_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, fp16_output_min, fp16_output_max);
  }

  return create_dynamic_fully_connected_nc(
      flags,
      /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF, &params, sizeof(params),
      &params, sizeof(params), gemm_config, &gemm_config->minmax,
      /*gemm_nr2_config=*/NULL, /*gemm_nr2_ukernels=*/NULL,
      expected_operator_type, dynamic_fully_connected_op_out);
}

enum xnn_status xnn_create_dynamic_fully_connected_nc_f16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* dynamic_fully_connected_op_out)
{
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  return create_dynamic_fully_connected_nc_f16(
      output_min, output_max, flags, gemm_config,
      xnn_operator_type_dynamic_fully_connected_nc_f16,
      dynamic_fully_connected_op_out);
}

enum xnn_status xnn_create_dynamic_fully_connected_nc_pf16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* dynamic_fully_connected_op_out)
{
  const struct xnn_gemm_config* gemm_config = xnn_init_pf16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_dynamic_fully_connected_nc_pf16));
    return xnn_status_unsupported_hardware;
  }

  return create_dynamic_fully_connected_nc_f16(
      output_min, output_max, flags, gemm_config,
      xnn_operator_type_dynamic_fully_connected_nc_pf16,
      dynamic_fully_connected_op_out);
}

enum xnn_status create_dynamic_fully_connected_nc_f32(
    float output_min, float output_max, uint32_t flags,
    const struct xnn_gemm_config* gemm_config,
    const struct xnn_gemm_config* gemm_nr2_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* dynamic_fully_connected_op_out) {
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(expected_operator_type), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr-1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  const struct gemm_fused_ukernels* gemm_nr2_ukernels = NULL;
  struct xnn_f32_minmax_params params2;
  if (gemm_nr2_config != NULL) {
    gemm_nr2_ukernels = &gemm_nr2_config->minmax;
    if (linear_activation && gemm_nr2_config->linear.gemm[gemm_nr2_config->mr-1].function[XNN_UARCH_DEFAULT] != NULL) {
      gemm_nr2_ukernels = &gemm_nr2_config->linear;
    }

    if XNN_LIKELY(gemm_nr2_config->init.f32 != NULL) {
      gemm_nr2_config->init.f32(&params2, output_min, output_max);
    }
  }

  return create_dynamic_fully_connected_nc(
    flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &params, sizeof(params),
    &params2, sizeof(params2),
    gemm_config, gemm_ukernels,
    gemm_nr2_config, gemm_nr2_ukernels,
    expected_operator_type,
    dynamic_fully_connected_op_out);
}

enum xnn_status xnn_create_dynamic_fully_connected_nc_f32(
    float output_min, float output_max, uint32_t flags,
    xnn_operator_t* dynamic_fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_dynamic_fully_connected_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_gemm_config* gemm_nr2_config =
      xnn_init_f32_gemm_nr2_config();

  return create_dynamic_fully_connected_nc_f32(
      output_min, output_max, flags, gemm_config, gemm_nr2_config,
      xnn_operator_type_dynamic_fully_connected_nc_f32,
      dynamic_fully_connected_op_out);
}

enum xnn_status xnn_create_dynamic_fully_connected_nc_pf32(
    float output_min, float output_max, uint32_t flags,
    xnn_operator_t* dynamic_fully_connected_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_pf32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_dynamic_fully_connected_nc_pf32));
    return xnn_status_unsupported_hardware;
  }

  return create_dynamic_fully_connected_nc_f32(
      output_min, output_max, flags, gemm_config, /*gemm_nr2_config=*/NULL,
      xnn_operator_type_dynamic_fully_connected_nc_pf32,
      dynamic_fully_connected_op_out);
}

static enum xnn_status reshape_dynamic_fully_connected_nc(
    xnn_operator_t dynamic_fully_connected_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    uint32_t log2_output_element_size,
    const void* params,
    size_t params_size,
    const void* params2,
    size_t params2_size,
    pthreadpool_t threadpool)
{
  if (dynamic_fully_connected_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(dynamic_fully_connected_op));
    return xnn_status_invalid_parameter;
  }
  dynamic_fully_connected_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string_v2(dynamic_fully_connected_op));
    return xnn_status_uninitialized;
  }

  if (input_channels == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu input channels: number of "
        "channels must be non-zero",
        xnn_operator_type_to_string_v2(dynamic_fully_connected_op),
        input_channels);
    return xnn_status_invalid_parameter;
  }

  if (output_channels == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu output channels: number of "
        "channels must be non-zero",
        xnn_operator_type_to_string_v2(dynamic_fully_connected_op),
        output_channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < input_channels) {
    xnn_log_error(
        "failed to reshape %s operator with input element stride of %zu: "
        "stride must be at least as large as the number of input channels "
        "(%zu)",
        xnn_operator_type_to_string_v2(dynamic_fully_connected_op),
        input_stride, input_channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < output_channels) {
    xnn_log_error(
        "failed to reshape %s operator with output element stride of %zu: "
        "stride must be at least as large as the number of output channels "
        "(%zu)",
        xnn_operator_type_to_string_v2(dynamic_fully_connected_op),
        output_stride, output_channels);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    dynamic_fully_connected_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  struct xnn_ukernel_gemm* ukernel = &dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm;
  bool use_gemm_nr2 = false;
  if (ukernel->nr > output_channels) {
    uint32_t gemm_nr2_mr = dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm_nr2.mr;
    // Default microkernel is suboptimal, use a microkernel that better supports less output channels.
    if (gemm_nr2_mr != 0 && dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm_nr2.gemm_cases[gemm_nr2_mr-1].function[XNN_UARCH_DEFAULT] != NULL) {
      use_gemm_nr2 = true;
      ukernel = &dynamic_fully_connected_op->ukernel.gemm_ukernels->gemm_nr2;
    }
  }

  const uint32_t nr = ukernel->nr;
  uint32_t mr = ukernel->mr;
  uint32_t mr_packed = ukernel->mr_packed;

  if (batch_size == 1 && ukernel->gemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
    mr = 1;
    mr_packed = 1;
  }

  assert(mr != 0 && mr <= XNN_MAX_MR);
  struct xnn_hmp_gemm_ukernel gemm_ukernel = ukernel->gemm_cases[mr-1];

  const uint32_t kr = ukernel->kr;
  const uint32_t sr = ukernel->sr;
  const size_t n_stride = round_up(output_channels, nr);
  const size_t k_stride = round_up_po2(input_channels, kr * sr);
  const struct xnn_gemm_config* gemm_config =
      dynamic_fully_connected_op->gemm_config;
  const size_t weights_stride =
      gemm_config->packed_stride_weights_and_biases
          ? gemm_config->packed_stride_weights_and_biases(
                gemm_config, input_channels, /*block_size=*/k_stride, k_stride,
                /*extra_bytes=*/0)
          : (k_stride << log2_filter_element_size) + bias_element_size;
  const size_t num_threads = pthreadpool_get_threads_count(threadpool);

  // Clear the operator's compute data to avoid accidentally reusing values from
  // a previous reshape (this was an interesting bug to track down).
  memset(dynamic_fully_connected_op->compute, 0,
         3 * sizeof(struct compute_parameters));
  struct compute_parameters* packw_compute =
      &dynamic_fully_connected_op->compute[0];
  struct compute_parameters* gemm_compute =
      &dynamic_fully_connected_op->compute[1];
  dynamic_fully_connected_op->num_compute_invocations = 2;

  struct gemm_op_context* gemm_context =
      dynamic_fully_connected_op->dynamic_context.gemm;

  // TODO(zhin): fast path to query workspace size when workspace_size != NULL?
  *workspace_size = n_stride * weights_stride;

  if (dynamic_fully_connected_op->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    assert(ukernel->packw_gemm_gio || gemm_config->pack_weights_and_biases);
    gemm_context->packw_gemm_gio = (struct packw_gemm_gio_context){
        .kc = input_channels,
        .nr = nr,
        .kr = kr,
        .sr = sr,
        .k_stride_elements = output_channels,
        .n_stride = 1 << log2_filter_element_size,
        .b_stride = bias_element_size,
        .w_stride = weights_stride,
        .packw_gemm_gio = ukernel->packw_gemm_gio,
        .pack_weights_and_biases = gemm_config->pack_weights_and_biases,
        .gemm_config = gemm_config,
    };

    packw_compute->task_1d_tile_1d_dynamic =
        (pthreadpool_task_1d_tile_1d_dynamic_t)xnn_compute_packw_gemm_gio;
    packw_compute->context_offset =
        offsetof(struct gemm_op_context, packw_gemm_gio);
  } else {
    assert(ukernel->packw_gemm_goi || gemm_config->pack_weights_and_biases);
    gemm_context->packw_gemm_goi = (struct packw_gemm_goi_context){
        .kc = input_channels,
        .nr = nr,
        .kr = kr,
        .sr = sr,
        .k_stride = input_channels << log2_input_element_size,
        .b_stride = bias_element_size,
        .w_stride = weights_stride,
        .packw_gemm_goi = ukernel->packw_gemm_goi,
        .pack_weights_and_biases = gemm_config->pack_weights_and_biases,
        .gemm_config = gemm_config,
    };

    packw_compute->task_1d_tile_1d_dynamic =
        (pthreadpool_task_1d_tile_1d_dynamic_t)xnn_compute_packw_gemm_goi;
    packw_compute->context_offset =
        offsetof(struct gemm_op_context, packw_gemm_goi);
  }
  packw_compute->type = xnn_parallelization_type_1d_tile_1d_dynamic;
  packw_compute->range[0] = output_channels;
  packw_compute->tile[0] = nr;

  const struct xnn_pack_lh_config* packed_lh_config = NULL;
  bool inline_lhs_packing =
      dynamic_fully_connected_op->flags & XNN_FLAG_INLINE_LHS_PACKING;
  switch (dynamic_fully_connected_op->type) {
    case xnn_operator_type_dynamic_fully_connected_nc_pf16:
      packed_lh_config = xnn_init_x16_pack_lh_config();
      break;
    case xnn_operator_type_dynamic_fully_connected_nc_pf32:
      packed_lh_config = xnn_init_x32_pack_lh_config();
      break;
    default:
      break;
  }

  // Compute the optimal tile size for this GEMM.
  const size_t nc = xnn_gemm_best_tile_size(
      /*num_groups=*/1, /*m=*/batch_size, /*n=*/output_channels,
      /*m_stride=*/input_stride
          << (packed_lh_config ? packed_lh_config->log2_packed_element_size
                               : log2_input_element_size),
      /*n_stride=*/weights_stride,
      /*cn_stride=*/1 << log2_output_element_size, mr, nr,
      /*num_threads=*/num_threads);

  const size_t workspace_offset =
      round_up_po2(*workspace_size, XNN_ALLOCATION_ALIGNMENT);

  // If we are packing the LHS, provide a per-thread workspace to do so inline.
  memset(&gemm_context->pack_lh, 0, sizeof(struct pack_lh_context));
  if (packed_lh_config) {
    log2_input_element_size = packed_lh_config->log2_packed_element_size;
    if (inline_lhs_packing) {
      assert(workspace_size);
      const size_t per_thread_workspace_size = packed_lh_config->size_fn(
          mr, /*k=*/input_channels, mr_packed, kr, sr);

      // If `xnn_gemm_best_tile_size` suggests an `nc` that is smaller than `n`,
      // i.e. it suggests splitting along `output_channels`, then it's probably
      // not a good idea to inline the packing, which requires using `nc == n`.
      //
      // Similarly, inlining the packing also doesn't make sense if the number
      // of threads exceeds the number of tiles that we can parallelize over.
      //
      // In either case, we pack the entire LHS into the workspace in a separate
      // `compute`, just as if it were a separate op.
      const bool should_inline_lhs_packing = xnn_should_inline_lhs_packing(
          gemm_config,
          /*m_packed_stride=*/divide_round_up(per_thread_workspace_size, mr),
          /*n_stride=*/weights_stride,
          /*cn_stride=*/1 << log2_output_element_size, /*mc=*/batch_size,
          /*nc=*/output_channels);

      if (!should_inline_lhs_packing ||
          num_threads * mr > round_up(batch_size, mr)) {
        xnn_log_debug(
            "Pre-packing LHS of %s with m=%zu, n=%zu, and k=%zu despite "
            "request to inline because %s.",
            xnn_operator_type_to_string(dynamic_fully_connected_op->type),
            batch_size, output_channels, input_channels,
            !should_inline_lhs_packing
                ? "packed lhs will likely not stay in cache"
                : "batch size does not parallelize "
                  "well over the number of threads");

        // Allocate a workspace for the entire LHS.
        *workspace_size =
            workspace_offset + packed_lh_config->size_fn(batch_size,
                                                         /*k=*/input_channels,
                                                         mr_packed, kr, sr);

        // Set up the LHS packing as a separate compute.
        gemm_context->pack_lh = (struct pack_lh_context){
            .m = batch_size,
            .k = input_channels,
            .mr = mr_packed,
            .kr = kr,
            .sr = sr,
            .lhs_stride = input_channels
                          << packed_lh_config->log2_input_element_size,
            .packed_offset_fn = packed_lh_config->offset_fn,
            .pack_lh_ukernel = packed_lh_config->ukernel,
            .workspace_offset = workspace_offset,
        };

        struct compute_parameters* pack_lh_compute = gemm_compute++;
        dynamic_fully_connected_op->num_compute_invocations++;
        pack_lh_compute->context_offset =
            offsetof(struct gemm_op_context, pack_lh);
        pack_lh_compute->type = xnn_parallelization_type_2d_tile_1d_dynamic;
        pack_lh_compute->task_2d_tile_1d_dynamic =
            (pthreadpool_task_2d_tile_1d_dynamic_t)xnn_compute_pack_lh;
        pack_lh_compute->range[0] = 1;
        pack_lh_compute->range[1] = batch_size;
        pack_lh_compute->tile[0] = mr_packed;

        inline_lhs_packing = false;
      } else {
        xnn_log_debug(
            "Inlining LHS packing for %s with m=%zu, n=%zu, and k=%zu.",
            xnn_operator_type_to_string(dynamic_fully_connected_op->type),
            batch_size, output_channels, input_channels);
        const size_t per_thread_workspace_size = packed_lh_config->size_fn(
            mr, /*k=*/input_channels, mr_packed, kr, sr);
        *workspace_size =
            workspace_offset + num_threads * per_thread_workspace_size;
        xnn_log_debug(
            "Requesting workspace of %zu x %zu bytes for LHS packing.",
            pthreadpool_get_threads_count(threadpool),
            per_thread_workspace_size);
        log2_input_element_size = packed_lh_config->log2_input_element_size;
      }
    }
  }

  gemm_context->gemm = (struct gemm_context){
      .k_scaled = input_channels << log2_input_element_size,
      .w_stride = weights_stride,
      .a_stride = input_stride << log2_input_element_size,
      .cm_stride = output_stride << log2_output_element_size,
      .cn_stride = nr << log2_output_element_size,
      .log2_csize = log2_output_element_size,
      .ukernel = gemm_ukernel,
      .mr = mr,
      .kr = kr,
      .sr = sr,
      .kc = input_channels,
      .nc = output_channels,
      .packed_lh_config = packed_lh_config,
      .workspace_offset = workspace_offset,
      .mr_packed = mr_packed,
  };

  if (use_gemm_nr2) {
    memcpy(&gemm_context->gemm.params, params2, params2_size);
  } else {
    memcpy(&gemm_context->gemm.params, params, params_size);
  }
  gemm_context->gemm.fused_params = &gemm_context->gemm.params;

#if XNN_MAX_UARCH_TYPES > 1
  if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
    if (packed_lh_config) {
      if (inline_lhs_packing) {
        gemm_compute->type =
            xnn_parallelization_type_1d_tile_1d_dynamic_with_uarch_with_thread;
        dynamic_fully_connected_op->compute[1]
            .task_1d_tile_1d_dynamic_with_id_with_thread =
            (pthreadpool_task_1d_tile_1d_dynamic_with_id_with_thread_t)
                xnn_compute_hmp_inline_packed_qp8gemm;
      } else {
        gemm_compute->type =
            xnn_parallelization_type_2d_tile_2d_dynamic_with_uarch;
        gemm_compute->task_2d_tile_2d_dynamic_with_id =
            (pthreadpool_task_2d_tile_2d_with_id_t)xnn_compute_hmp_qp8gemm;
      }
    } else {
      gemm_compute->type =
          xnn_parallelization_type_2d_tile_2d_dynamic_with_uarch;
      gemm_compute->task_2d_tile_2d_dynamic_with_id =
          (pthreadpool_task_2d_tile_2d_dynamic_with_id_t)xnn_compute_hmp_gemm;
    }
  } else
#endif  // XNN_MAX_UARCH_TYPES > 1
    if (packed_lh_config) {
      if (inline_lhs_packing) {
        gemm_compute->type =
            xnn_parallelization_type_1d_tile_1d_dynamic_with_thread;
        gemm_compute->task_1d_tile_1d_dynamic_with_id =
            (pthreadpool_task_1d_tile_1d_dynamic_with_id_t)
                xnn_compute_inline_packed_qp8gemm;
      } else {
        gemm_compute->type = xnn_parallelization_type_2d_tile_2d_dynamic;
        gemm_compute->task_2d_tile_2d_dynamic =
            (pthreadpool_task_2d_tile_2d_dynamic_t)xnn_compute_qp8gemm;
      }
    } else {
      gemm_compute->type = xnn_parallelization_type_2d_tile_2d_dynamic;
      gemm_compute->task_2d_tile_2d_dynamic =
          (pthreadpool_task_2d_tile_2d_dynamic_t)xnn_compute_gemm;
    }

  if (packed_lh_config && inline_lhs_packing) {
    gemm_compute->range[0] = batch_size;
    gemm_compute->tile[0] = mr;
  } else {
    gemm_compute->range[1] = batch_size;
    gemm_compute->range[0] = output_channels;
    gemm_compute->tile[1] = mr;
    gemm_compute->tile[0] = nc;
  }

  dynamic_fully_connected_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_dynamic_fully_connected_nc_f16(
    xnn_operator_t dynamic_fully_connected_op,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    pthreadpool_t threadpool)
{
  return reshape_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_f16,
    batch_size, input_channels, output_channels, input_stride, output_stride,
    workspace_size,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*bias_element_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    &dynamic_fully_connected_op->params.f16_minmax,
    sizeof(dynamic_fully_connected_op->params.f16_minmax),
    &dynamic_fully_connected_op->params.f16_minmax,
    sizeof(dynamic_fully_connected_op->params.f16_minmax),
    threadpool);
}

enum xnn_status xnn_reshape_dynamic_fully_connected_nc_pf16(
    xnn_operator_t dynamic_fully_connected_op,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    pthreadpool_t threadpool)
{
  return reshape_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_pf16,
    batch_size, input_channels, output_channels, input_stride, output_stride,
    workspace_size,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*bias_element_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    &dynamic_fully_connected_op->params.f16_minmax,
    sizeof(dynamic_fully_connected_op->params.f16_minmax),
    &dynamic_fully_connected_op->params.f16_minmax,
    sizeof(dynamic_fully_connected_op->params.f16_minmax),
    threadpool);
}

enum xnn_status xnn_reshape_dynamic_fully_connected_nc_f32(
    xnn_operator_t dynamic_fully_connected_op,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    pthreadpool_t threadpool)
{
  return reshape_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_f32,
    batch_size, input_channels, output_channels, input_stride, output_stride,
    workspace_size,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*bias_element_size=*/sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &dynamic_fully_connected_op->params.f32_minmax,
    sizeof(dynamic_fully_connected_op->params.f32_minmax),
    &dynamic_fully_connected_op->params2->f32_minmax,
    sizeof(dynamic_fully_connected_op->params2->f32_minmax),
    threadpool);
}

enum xnn_status xnn_reshape_dynamic_fully_connected_nc_pf32(
    xnn_operator_t dynamic_fully_connected_op,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    size_t* workspace_size,
    pthreadpool_t threadpool)
{
  return reshape_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_pf32,
    batch_size, input_channels, output_channels, input_stride, output_stride,
    workspace_size,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*bias_element_size=*/sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &dynamic_fully_connected_op->params.f32_minmax,
    sizeof(dynamic_fully_connected_op->params.f32_minmax),
    &dynamic_fully_connected_op->params2->f32_minmax,
    sizeof(dynamic_fully_connected_op->params2->f32_minmax),
    threadpool);
}

static enum xnn_status setup_dynamic_fully_connected_nc(
  xnn_operator_t dynamic_fully_connected_op,
  enum xnn_operator_type expected_operator_type,
  void* workspace,
  const void* input,
  const void* kernel,
  const void* bias,
  void* output)
{
  if (dynamic_fully_connected_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(dynamic_fully_connected_op));
    return xnn_status_invalid_parameter;
  }

  switch (dynamic_fully_connected_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(dynamic_fully_connected_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  struct gemm_op_context* gemm_context =
      dynamic_fully_connected_op->dynamic_context.gemm;

  if (dynamic_fully_connected_op->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    gemm_context->packw_gemm_gio.kernel = kernel;
    gemm_context->packw_gemm_gio.bias = bias;
    gemm_context->packw_gemm_gio.packed_weights = workspace;
  } else {
    gemm_context->packw_gemm_goi.kernel = kernel;
    gemm_context->packw_gemm_goi.bias = bias;
    gemm_context->packw_gemm_goi.packed_weights = workspace;
  }

  if (gemm_context->pack_lh.m > 0) {
    gemm_context->pack_lh.lhs = input;
    void* pack_lh_workspace =
        (void*)((uintptr_t)workspace + gemm_context->pack_lh.workspace_offset);
    gemm_context->pack_lh.lhs_packed = pack_lh_workspace;
    gemm_context->gemm.a = pack_lh_workspace;
  } else {
    gemm_context->gemm.a = input;
    gemm_context->gemm.workspace = workspace;
  }
  gemm_context->gemm.packed_w = workspace;
  gemm_context->gemm.c = output;

  dynamic_fully_connected_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_dynamic_fully_connected_nc_f16(
  xnn_operator_t dynamic_fully_connected_op,
  void* workspace,
  const void* input,
  const void* kernel,
  const void* bias,
  void* output)
{
  return setup_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_f16,
    workspace, input, kernel, bias, output);
}

enum xnn_status xnn_setup_dynamic_fully_connected_nc_pf16(
  xnn_operator_t dynamic_fully_connected_op,
  void* workspace,
  const void* input,
  const void* kernel,
  const void* bias,
  void* output)
{
  return setup_dynamic_fully_connected_nc(
      dynamic_fully_connected_op,
      xnn_operator_type_dynamic_fully_connected_nc_pf16, workspace, input,
      kernel, bias, output);
}

enum xnn_status xnn_setup_dynamic_fully_connected_nc_f32(
  xnn_operator_t dynamic_fully_connected_op,
  void* workspace,
  const float* input,
  const float* kernel,
  const float* bias,
  float* output)
{
  return setup_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_f32,
    workspace, input, kernel, bias, output);
}

enum xnn_status xnn_setup_dynamic_fully_connected_nc_pf32(
  xnn_operator_t dynamic_fully_connected_op,
  void* workspace,
  const float* input,
  const float* kernel,
  const float* bias,
  float* output)
{
  return setup_dynamic_fully_connected_nc(
    dynamic_fully_connected_op, xnn_operator_type_dynamic_fully_connected_nc_pf32,
    workspace, input, kernel, bias, output);
}
