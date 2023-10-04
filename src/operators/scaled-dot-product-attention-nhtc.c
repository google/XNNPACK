// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/compute.h>
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microkernel-type.h>
#include <xnnpack/microparams.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/params.h>
#include <xnnpack/pack.h>


static enum xnn_status create_scaled_dot_product_attention_nhtc(
  enum xnn_attention_logits_cap_type cap_type,
  const void* cap_params,
  enum xnn_operator_type operator_type,
  const struct xnn_gemm_config* gemm_config,
  const struct xnn_raddstoreexpminusmax_config* raddstoreexpminusmax_config,
  const struct xnn_rmax_config* rmax_config,
  const struct xnn_binary_elementwise_config* vadd_config,
  const struct xnn_binary_elementwise_config* vmul_config,
  const struct xnn_unary_elementwise_config* vtanh_config,
  const void* minmax_params,
  size_t minmax_params_size,
  const void* expminus_params,
  size_t expminus_params_size,
  const void* rmax_params,
  size_t rmax_params_size,
  const void* tanh_params,
  size_t tanh_params_size,
  uint32_t flags,
  xnn_operator_t* attention_op_out)
{
  xnn_operator_t attention_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  attention_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (attention_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const size_t nr = gemm_config->nr;
  const size_t mr = gemm_config->mr;
  attention_op->ukernel.type = xnn_microkernel_type_gemm;
  attention_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
    .mr = mr,
    .nr = nr,
    .kr = UINT32_C(1) << gemm_config->log2_kr,
    .sr = UINT32_C(1) << gemm_config->log2_sr,
  };

  assert(XNN_MAX_MR >= mr);
  for (size_t i = 0; i < mr; i++) {
    attention_op->ukernel.gemm.gemm_cases[i] = gemm_config->minmax.gemm[i];
  }
  attention_op->ukernel.gemm.packw_gemm_goi = gemm_config->pack_gemm_goi;
  attention_op->ukernel.gemm.packw_gemm_gio = gemm_config->pack_gemm_gio;

  memcpy(&attention_op->params, minmax_params, minmax_params_size);
  memcpy(&attention_op->params2, expminus_params, expminus_params_size);
  memcpy(&attention_op->params3, rmax_params, rmax_params_size);
  memcpy(&attention_op->params4, tanh_params, tanh_params_size);

  if (cap_type == xnn_attention_logits_cap_type_tanh) {
    const struct xnn_attention_logits_cap_tanh_params* cap_tanh_params =
      (const struct xnn_attention_logits_cap_tanh_params*) cap_params;
    memcpy(&attention_op->attention.cap_params, cap_tanh_params, sizeof(struct xnn_attention_logits_cap_tanh_params));
  }

  attention_op->attention.raddstoreexpminusmax_config = raddstoreexpminusmax_config;
  attention_op->attention.rmax_config = rmax_config;
  attention_op->attention.vadd_config = vadd_config;
  attention_op->attention.vmul_config = vmul_config;
  attention_op->attention.vtanh_config = vtanh_config;
  attention_op->attention.cap_type = cap_type;

  attention_op->state = xnn_run_state_invalid;
  attention_op->type = operator_type;
  attention_op->flags = flags;

  *attention_op_out = attention_op;

  return xnn_status_success;

error:
  xnn_delete_operator(attention_op);
  return status;
}

enum xnn_status xnn_create_scaled_dot_product_attention_nhtc_f16(
  enum xnn_attention_logits_cap_type cap_type,
  const void* cap_params,
  uint32_t flags,
  xnn_operator_t* attention_op_out)
{
  const enum xnn_operator_type operator_type = xnn_operator_type_scaled_dot_product_attention_nhtc_f16;
  enum xnn_status status = xnn_status_unsupported_hardware;

  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f16_minmax_params minmax_params;
  if XNN_LIKELY(gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&minmax_params, fp16_ieee_from_fp32_value(-INFINITY), fp16_ieee_from_fp32_value(INFINITY));
  }

  const struct xnn_raddstoreexpminusmax_config* raddstoreexpminusmax_config =
    xnn_init_f16_raddstoreexpminusmax_config();
  if (raddstoreexpminusmax_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f16_expminus_params expminus_params;
  if (raddstoreexpminusmax_config->init.f16 != NULL) {
    raddstoreexpminusmax_config->init.f16(&expminus_params);
  }

  const struct xnn_rmax_config* rmax_config = xnn_init_f16_rmax_config();
  if (rmax_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f16_default_params rmax_params;
  if (rmax_config->init.f16 != NULL) {
    rmax_config->init.f16(&rmax_params);
  }

  const struct xnn_binary_elementwise_config* vadd_config = xnn_init_f16_vadd_config();
  if (vadd_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const struct xnn_binary_elementwise_config* vmul_config = xnn_init_f16_vmul_config();
  if (vmul_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const struct xnn_unary_elementwise_config* vtanh_config = xnn_init_f16_tanh_config();
  if (vtanh_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f16_tanh_params tanh_params;
  if XNN_LIKELY(vtanh_config->init.f16_tanh != NULL) {
    vtanh_config->init.f16_tanh(&tanh_params);
  }

  status = xnn_status_invalid_parameter;

  if (cap_type == xnn_attention_logits_cap_type_tanh) {
    const struct xnn_attention_logits_cap_tanh_params* cap_tanh_params =
      (const struct xnn_attention_logits_cap_tanh_params*) cap_params;
    const float cap = cap_tanh_params->cap;
    if (cap <= 0.0f || isnan(cap) || cap < 0x1.0p-14f || cap > 65504.0f) {
      xnn_log_error("failed to create %s operator with Cap TanH: cap value (%f) must be greater than 0, representable "
                    "in FP16, and not be NaN", xnn_operator_type_to_string(operator_type), cap_tanh_params->cap);
      goto error;
    }
  }

  return create_scaled_dot_product_attention_nhtc(
    cap_type, cap_params,
    operator_type,
    gemm_config,
    raddstoreexpminusmax_config,
    rmax_config,
    vadd_config,
    vmul_config,
    vtanh_config,
    &minmax_params, sizeof(minmax_params),
    &expminus_params, sizeof(expminus_params),
    &rmax_params, sizeof(rmax_params),
    &tanh_params, sizeof(tanh_params),
    flags,
    attention_op_out);

error:
  return status;
}

enum xnn_status xnn_create_scaled_dot_product_attention_nhtc_f32(
  enum xnn_attention_logits_cap_type cap_type,
  const void* cap_params,
  uint32_t flags,
  xnn_operator_t* attention_op_out)
{
  const enum xnn_operator_type operator_type = xnn_operator_type_scaled_dot_product_attention_nhtc_f32;
  enum xnn_status status = xnn_status_unsupported_hardware;

  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f32_minmax_params minmax_params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&minmax_params, -INFINITY , INFINITY);
  }

  const struct xnn_raddstoreexpminusmax_config* raddstoreexpminusmax_config =
    xnn_init_f32_raddstoreexpminusmax_config();
  if (raddstoreexpminusmax_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f32_expminus_params expminus_params;
  if (raddstoreexpminusmax_config->init.f32 != NULL) {
    raddstoreexpminusmax_config->init.f32(&expminus_params);
  }

  const struct xnn_rmax_config* rmax_config = xnn_init_f32_rmax_config();
  if (rmax_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f32_default_params rmax_params;
  if (rmax_config->init.f32 != NULL) {
    rmax_config->init.f32(&rmax_params);
  }

  const struct xnn_binary_elementwise_config* vadd_config = xnn_init_f32_vadd_config();
  if (vadd_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const struct xnn_binary_elementwise_config* vmul_config = xnn_init_f32_vmul_config();
  if (vmul_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const struct xnn_unary_elementwise_config* vtanh_config = xnn_init_f32_tanh_config();
  if (vtanh_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  union xnn_f32_tanh_params tanh_params;
  if XNN_LIKELY(vtanh_config->init.f32_tanh != NULL) {
    vtanh_config->init.f32_tanh(&tanh_params);
  }

  status = xnn_status_invalid_parameter;

  if (cap_type == xnn_attention_logits_cap_type_tanh) {
    const struct xnn_attention_logits_cap_tanh_params* cap_tanh_params =
      (const struct xnn_attention_logits_cap_tanh_params*) cap_params;
    float cap = cap_tanh_params->cap;
    if (cap <= 0.0f || !isnormal(cap)) {
      xnn_log_error("failed to create %s operator with Cap TanH: cap value (%f) must be finite and greater than 0",
                  xnn_operator_type_to_string(operator_type), cap_tanh_params->cap);
      goto error;
    }
  }

  return create_scaled_dot_product_attention_nhtc(
    cap_type, cap_params,
    operator_type,
    gemm_config,
    raddstoreexpminusmax_config,
    rmax_config,
    vadd_config,
    vmul_config,
    vtanh_config,
    &minmax_params, sizeof(minmax_params),
    &expminus_params, sizeof(expminus_params),
    &rmax_params, sizeof(rmax_params),
    &tanh_params, sizeof(tanh_params),
    flags,
    attention_op_out);

error:
  return status;
}

static void compute_reciprocal_f16(
    const uint16_t input[XNN_MIN_ELEMENTS(1)],
    uint16_t output[XNN_MIN_ELEMENTS(1)])
{
  *output = fp16_ieee_from_fp32_value(1.0f / fp16_ieee_to_fp32_value(*input));
}

static void compute_reciprocal_f32(
  const float input[XNN_MIN_ELEMENTS(1)],
  float output[XNN_MIN_ELEMENTS(1)])
{
  *output = 1.0f / *input;
}

static enum xnn_status reshape_scaled_dot_product_attention_nhtc(
  xnn_operator_t attention_op,
  enum xnn_operator_type expected_operator_type,
  size_t batch_size,
  size_t query_heads,
  size_t query_tokens,
  size_t key_value_heads,
  size_t key_value_tokens,
  size_t query_key_channels,
  size_t value_channels,
  size_t* workspace_size,
  size_t* workspace_alignment,
  size_t log2_element_size,
  size_t element_size,
  xnn_compute_reciprocal_fn compute_reciprocal,
  void* cap,
  void* cap_reciprocal,
  size_t cap_size,
  const void* minmax_params,
  size_t minmax_params_size,
  const void* expminus_params,
  size_t expminus_params_size,
  const void* rmax_params,
  size_t rmax_params_size,
  const void* tanh_params,
  size_t tanh_params_size,
  pthreadpool_t threadpool)
{
  if (attention_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(attention_op->type));
    return xnn_status_invalid_parameter;
  }
  attention_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(attention_op->type));
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    xnn_log_error(
      "failed to create %s operator with batch size of %zu: batch size must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), batch_size);
    return xnn_status_invalid_parameter;
  }

  if (query_heads == 0) {
    xnn_log_error(
      "failed to create %s operator with number of query heads %zu: number of query heads must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), query_heads);
    return xnn_status_invalid_parameter;
  }

  if (key_value_heads == 0) {
    xnn_log_error(
      "failed to create %s operator with number of key/value heads %zu: number of key/value heads must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), key_value_heads);
    return xnn_status_invalid_parameter;
  }

  if (key_value_heads != 1 && key_value_heads != query_heads) {
    xnn_log_error(
      "failed to create %s operator with number of key/value heads %zu: number of key/value heads must be either 1 or "
      "equal to number of query heads", xnn_operator_type_to_string(expected_operator_type), key_value_heads);
    return xnn_status_invalid_parameter;
  }

  if (query_tokens == 0) {
    xnn_log_error(
      "failed to create %s operator with query tokens of %zu: query tokens must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), query_tokens);
    return xnn_status_invalid_parameter;
  }

  if (key_value_tokens == 0) {
    xnn_log_error(
      "failed to create %s operator with key/value tokens of %zu: key/value tokens must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), query_tokens);
    return xnn_status_invalid_parameter;
  }

  if (query_key_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: query/key channels must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), query_key_channels);
    return xnn_status_invalid_parameter;
  }

  if (value_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: value channels must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), value_channels);
    return xnn_status_invalid_parameter;
  }

  const uint32_t mr = attention_op->ukernel.gemm.mr;
  const uint32_t nr = attention_op->ukernel.gemm.nr;
  const uint32_t kr = attention_op->ukernel.gemm.kr;
  const uint32_t sr = attention_op->ukernel.gemm.sr;

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  const size_t size_using_threads = num_threads * mr;
  const size_t size_using_batch = batch_size * query_heads * query_tokens;
  const bool use_threads_workspace_size = size_using_threads < size_using_batch;
  const size_t workspace_multiplier = use_threads_workspace_size ? size_using_threads : size_using_batch;
  // Calculate size required for workspace.
  // 1. Workspace for Q scaled, each thread computes a maximum of mr * query_key_channels.
  const size_t scaled_query_size =
    round_up_po2(workspace_multiplier * query_key_channels * element_size + XNN_EXTRA_BYTES, XNN_ALLOCATION_ALIGNMENT);

  // Key is [key_value_tokens (output channel), channels (input channel)].
  const size_t key_n_stride = round_up(key_value_tokens, nr);
  const size_t key_k_stride = round_up_po2(query_key_channels, kr * sr);
  const size_t key_head_stride = key_n_stride * (element_size + (key_k_stride << log2_element_size));
  // 2. Workspace for packed key.
  const size_t packed_key_size = round_up_po2(batch_size * key_value_heads * key_head_stride, XNN_ALLOCATION_ALIGNMENT);

  // Value is [key_value_tokens (input channel), channels (output channel)].
  const size_t value_n_stride = round_up(value_channels, nr);
  const size_t value_k_stride = round_up_po2(key_value_tokens, kr * sr);
  const size_t value_head_stride = value_n_stride * (element_size + (value_k_stride << log2_element_size));
  // 3. Workspace for packed key.
  const size_t packed_value_size = round_up_po2(batch_size * key_value_heads * value_head_stride, XNN_ALLOCATION_ALIGNMENT);

  // 4. Workspace for logits (Q*K), each thread computes mr * key_value_tokens.
  const size_t logits_size =
    round_up_po2(workspace_multiplier * key_value_tokens * element_size + XNN_EXTRA_BYTES, XNN_ALLOCATION_ALIGNMENT);

  const size_t total_workspace_size = scaled_query_size + packed_key_size + packed_value_size + logits_size;

  *workspace_size = total_workspace_size;
  *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;

  // Pack key.
  attention_op->context.packw_gemm_goi = (struct packw_gemm_goi_context) {
    .kc = query_key_channels,
    .nr = nr,
    .kr = kr,
    .sr = sr,
    .k_stride = query_key_channels << log2_element_size,
    // b_stride and gb_stride not needed because we do not have bias.
    .w_stride = element_size + (key_k_stride << log2_element_size),
    .packw_gemm_goi = attention_op->ukernel.gemm.packw_gemm_goi,
    .gk_stride = key_value_tokens * (query_key_channels << log2_element_size),
    .gc_stride = key_head_stride,
  };
  attention_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
  attention_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_batched_packw_gemm_goi;
  attention_op->compute[0].context_offset =
    offsetof(struct xnn_operator, context.packw_gemm_goi) - offsetof(struct xnn_operator, context);
  attention_op->compute[0].range[0] = batch_size * key_value_heads;
  attention_op->compute[0].range[1] = key_value_tokens;
  // We cannot tile key_value_tokens because we compute complete rows of Q*K,
  // rather than MRxNR (where NR < key_value_tokens) tiles of Q*K.
  attention_op->compute[0].tile[0] = key_value_tokens;

  // Pack value.
  attention_op->context.packw_gemm_gio = (struct packw_gemm_gio_context) {
    .kc = key_value_tokens,
    .nr = nr,
    .kr = kr,
    .sr = sr,
    .n_stride = 1 << log2_element_size,
    .k_stride_elements = value_channels,
    // b_stride and gb_stride not needed because we do not have bias.
    .w_stride = element_size + (value_k_stride << log2_element_size),
    .packw_gemm_gio = attention_op->ukernel.gemm.packw_gemm_gio,
    .gk_stride = key_value_tokens * (value_channels << log2_element_size),
    .gb_stride = value_channels * element_size,
    .gc_stride = value_head_stride,
  };
  attention_op->compute[1].type = xnn_parallelization_type_2d_tile_1d;
  attention_op->compute[1].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_batched_packw_gemm_gio;
  attention_op->compute[1].context_offset =
    offsetof(struct xnn_operator, context.packw_gemm_gio) - offsetof(struct xnn_operator, context);
  attention_op->compute[1].range[0] = batch_size * key_value_heads;
  attention_op->compute[1].range[1] = value_channels;
  attention_op->compute[1].tile[0] = value_channels;

  struct xnn_hmp_gemm_ukernel gemm_ukernel = attention_op->ukernel.gemm.gemm_cases[mr - 1];

  attention_op->context.attention = (struct scaled_dot_product_attention_context){
    .key_value_tokens = key_value_tokens,
    .key_value_tokens_scaled = key_value_tokens * element_size,
    .query_key_channels = query_key_channels,
    .query_key_scaled_channels = query_key_channels * element_size,
    .value_channels = value_channels,
    .value_scaled_channels = value_channels * element_size,
    .cn_stride = nr << log2_element_size,
    .query_batch_stride = query_heads * query_tokens * query_key_channels * element_size,
    .query_head_stride = query_tokens * query_key_channels * element_size,
    .key_batch_stride = key_value_heads * key_head_stride,
    .key_head_stride = key_value_heads == 1 ? 0 : key_head_stride,
    .value_batch_stride = key_value_heads * value_head_stride,
    .value_head_stride = key_value_heads == 1 ? 0 : value_head_stride,
    .logits_batch_stride = query_heads * query_tokens * key_value_tokens * element_size,
    .logits_head_stride = query_tokens * key_value_tokens * element_size,
    .output_batch_stride = query_heads * query_tokens * value_channels * element_size,
    .output_head_stride = query_tokens * value_channels * element_size,
    .scaled_query_thread_stride = mr * query_key_channels * element_size,
    .logits_thread_stride = mr * key_value_tokens * element_size,
    .gemm_ukernel = gemm_ukernel,
    .compute_reciprocal = compute_reciprocal,
    .raddstoreexpminusmax_ukernel = attention_op->attention.raddstoreexpminusmax_config->ukernel,
    .rmax_ukernel = attention_op->attention.rmax_config->ukernel,
    .vadd_ukernel = attention_op->attention.vadd_config->minmax.op_ukernel,
    .vmul_ukernel = attention_op->attention.vmul_config->minmax.op_ukernel,
    .vmulc_ukernel = attention_op->attention.vmul_config->minmax.opc_ukernel,
    .vtanh_ukernel = attention_op->attention.vtanh_config->ukernel,
  };

  if (attention_op->attention.cap_type == xnn_attention_logits_cap_type_tanh) {
    attention_op->context.attention.logits_cap.type = xnn_attention_logits_cap_type_tanh;
    memcpy(&attention_op->context.attention.logits_cap.cap, cap, cap_size);
    memcpy(&attention_op->context.attention.logits_cap.cap_reciprocal, cap_reciprocal, cap_size);
  }

  #if XNN_MAX_UARCH_TYPES > 1
    if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
      if (use_threads_workspace_size) {
        attention_op->compute[2].type = xnn_parallelization_type_3d_tile_1d_with_uarch_with_thread;
        attention_op->compute[2].task_3d_tile_1d_with_id_with_thread =
          (pthreadpool_task_3d_tile_1d_with_id_with_thread_t) xnn_compute_hmp_scaled_dot_product_attention_with_thread;
      } else {
        attention_op->compute[2].type = xnn_parallelization_type_3d_tile_1d_with_uarch;
        attention_op->compute[2].task_3d_tile_1d_with_id =
          (pthreadpool_task_3d_tile_1d_with_id_t) xnn_compute_hmp_scaled_dot_product_attention;
      }
    } else {
      if (use_threads_workspace_size) {
        attention_op->compute[2].type = xnn_parallelization_type_3d_tile_1d_with_thread;
        attention_op->compute[2].task_3d_tile_1d_with_thread =
          (pthreadpool_task_3d_tile_1d_with_thread_t) xnn_compute_scaled_dot_product_attention_with_thread;
      } else {
        attention_op->compute[2].type = xnn_parallelization_type_3d_tile_1d;
        attention_op->compute[2].task_3d_tile_1d =
          (pthreadpool_task_3d_tile_1d_t) xnn_compute_scaled_dot_product_attention;
      }
    }
  #else
    if (use_threads_workspace_size) {
      attention_op->compute[2].type = xnn_parallelization_type_3d_tile_1d_with_thread;
      attention_op->compute[2].task_3d_tile_1d_with_thread =
        (pthreadpool_task_3d_tile_1d_with_thread_t) xnn_compute_scaled_dot_product_attention_with_thread;
    } else {
      attention_op->compute[2].type = xnn_parallelization_type_3d_tile_1d;
      attention_op->compute[2].task_3d_tile_1d =
        (pthreadpool_task_3d_tile_1d_t) xnn_compute_scaled_dot_product_attention;
    }
  #endif  // XNN_MAX_UARCH_TYPES > 1

  attention_op->compute[2].range[0] = batch_size;
  attention_op->compute[2].range[1] = query_heads;
  attention_op->compute[2].range[2] = query_tokens;
  attention_op->compute[2].tile[0] = mr;

  attention_op->context.attention.scaled_query_offset = 0;
  attention_op->context.attention.packed_k_offset = scaled_query_size;
  attention_op->context.attention.packed_v_offset = scaled_query_size + packed_key_size;
  attention_op->context.attention.logits_offset = scaled_query_size + packed_key_size + packed_value_size;

  memcpy(&attention_op->context.attention.minmax_params, minmax_params, minmax_params_size);
  memcpy(&attention_op->context.attention.expminus_params, expminus_params, expminus_params_size);
  memcpy(&attention_op->context.attention.rmax_params, rmax_params, rmax_params_size);
  memcpy(&attention_op->context.attention.tanh_params, tanh_params, tanh_params_size);

  attention_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;

}

enum xnn_status xnn_reshape_scaled_dot_product_attention_nhtc_f16(
  xnn_operator_t attention_op,
  size_t batch_size,
  size_t heads,
  size_t query_tokens,
  size_t key_value_heads,
  size_t key_value_tokens,
  size_t query_key_channels,
  size_t value_channels,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool)
{
  uint16_t cap = fp16_ieee_from_fp32_value(attention_op->attention.cap_params.cap);
  uint16_t cap_reciprocal = fp16_ieee_from_fp32_value(1.0f / attention_op->attention.cap_params.cap);

  return reshape_scaled_dot_product_attention_nhtc(
    attention_op,
    xnn_operator_type_scaled_dot_product_attention_nhtc_f16,
    batch_size,
    heads,
    query_tokens,
    key_value_heads,
    key_value_tokens,
    query_key_channels,
    value_channels,
    workspace_size, workspace_alignment,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_UINT16_T,
    /*element_size=*/sizeof(uint16_t),
    (xnn_compute_reciprocal_fn) compute_reciprocal_f16,
    &cap, &cap_reciprocal, sizeof(uint16_t),
    &attention_op->params.f16_minmax, sizeof(attention_op->params.f16_minmax),
    &attention_op->params2.f16_expminus_params, sizeof(attention_op->params2.f16_expminus_params),
    &attention_op->params3.f16_rmax, sizeof(attention_op->params3.f16_rmax),
    &attention_op->params4.f16_tanh, sizeof(attention_op->params4.f16_tanh),
    threadpool);
}

enum xnn_status xnn_reshape_scaled_dot_product_attention_nhtc_f32(
  xnn_operator_t attention_op,
  size_t batch_size,
  size_t query_heads,
  size_t query_tokens,
  size_t key_value_heads,
  size_t key_value_tokens,
  size_t query_key_channels,
  size_t value_channels,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool)
{
  float cap = attention_op->attention.cap_params.cap;
  float cap_reciprocal = 1 / attention_op->attention.cap_params.cap;

  return reshape_scaled_dot_product_attention_nhtc(
    attention_op,
    xnn_operator_type_scaled_dot_product_attention_nhtc_f32,
    batch_size,
    query_heads,
    query_tokens,
    key_value_heads,
    key_value_tokens,
    query_key_channels,
    value_channels,
    workspace_size, workspace_alignment,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*element_size=*/sizeof(float),
    (xnn_compute_reciprocal_fn) compute_reciprocal_f32,
    &cap, &cap_reciprocal, sizeof(float),
    &attention_op->params.f32_minmax, sizeof(attention_op->params.f32_minmax),
    &attention_op->params2.f32_expminus_params, sizeof(attention_op->params2.f32_expminus_params),
    &attention_op->params3.f32_rmax, sizeof(attention_op->params3.f32_rmax),
    &attention_op->params4.f32_tanh, sizeof(attention_op->params4.f32_tanh),
    threadpool);
}

static enum xnn_status setup_scaled_dot_product_attention_nhtc(
  xnn_operator_t attention_op,
  enum xnn_operator_type expected_operator_type,
  void* workspace,
  const float* query,
  const float* key,
  const float* value,
  const float* scale,
  const float* mask,
  float* output)
{
  if (attention_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got %s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string(attention_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (attention_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(attention_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  attention_op->context.packw_gemm_goi.kernel = key;
  attention_op->context.packw_gemm_goi.packed_weights =
    (void*) ((uintptr_t) workspace + attention_op->context.attention.packed_k_offset);
  attention_op->context.packw_gemm_goi.bias = NULL;

  attention_op->context.packw_gemm_gio.kernel = value;
  attention_op->context.packw_gemm_gio.packed_weights =
    (void*) ((uintptr_t) workspace + attention_op->context.attention.packed_v_offset);
  attention_op->context.packw_gemm_gio.bias = NULL;

  attention_op->context.attention.scaled_query =
    (void*) ((uintptr_t) workspace + attention_op->context.attention.scaled_query_offset);
  attention_op->context.attention.logits_buffer =
    (void*) ((uintptr_t) workspace + attention_op->context.attention.logits_offset);
  attention_op->context.attention.query = query;
  attention_op->context.attention.key = attention_op->context.packw_gemm_goi.packed_weights;
  attention_op->context.attention.value = attention_op->context.packw_gemm_gio.packed_weights;
  attention_op->context.attention.scale = scale;
  attention_op->context.attention.mask = mask;
  attention_op->context.attention.output = output;
  attention_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_scaled_dot_product_attention_nhtc_f16(
  xnn_operator_t attention_op,
  void* workspace,
  const void* query,
  const void* key,
  const void* value,
  const void* scale,
  const void* mask,
  void* output)
{
  return setup_scaled_dot_product_attention_nhtc(
    attention_op, xnn_operator_type_scaled_dot_product_attention_nhtc_f16,
    workspace,
    query, key, value,
    scale, mask,
    output);
}

enum xnn_status xnn_setup_scaled_dot_product_attention_nhtc_f32(
  xnn_operator_t attention_op,
  void* workspace,
  const float* query,
  const float* key,
  const float* value,
  const float* scale,
  const float* mask,
  float* output)
{
  return setup_scaled_dot_product_attention_nhtc(
    attention_op, xnn_operator_type_scaled_dot_product_attention_nhtc_f32,
    workspace,
    query, key, value,
    scale, mask,
    output);
}
