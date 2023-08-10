// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

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


enum xnn_status xnn_create_scaled_dot_attention_ntc_f32(
  enum xnn_attention_logits_cap_type cap_type,
  const void* cap_params,
  uint32_t flags,
  xnn_operator_t* attention_op_out)
{
  const enum xnn_operator_type operator_type = xnn_operator_type_scaled_dot_attention_ntc_f32;

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

  status = xnn_status_unsupported_hardware;

  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(operator_type));
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

  union xnn_f32_minmax_params minmax_params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&minmax_params, -INFINITY , INFINITY);
  }

  memcpy(&attention_op->params, &minmax_params, sizeof(minmax_params));

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
  memcpy(&attention_op->params2, &expminus_params, sizeof(expminus_params));

  const struct xnn_rmax_config* rmax_config = xnn_init_f32_rmax_config();
  if (rmax_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
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
  memcpy(&attention_op->params3, &tanh_params, sizeof(tanh_params));

  status = xnn_status_invalid_parameter;

  if (cap_type == xnn_attention_logits_cap_type_tanh) {
    const struct xnn_attention_logits_cap_tanh_params* cap_tanh_params =
      (const struct xnn_attention_logits_cap_tanh_params*) cap_params;
    if (cap_tanh_params->cap <= 0.0f) {
      xnn_log_error("failed to create %s operator: logits cap tanh specified but cap value (%f) is <= 0.0f",
                  xnn_operator_type_to_string(operator_type), cap_tanh_params->cap);
      goto error;
    }
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

static void compute_reciprocal_f32(
  const float input[XNN_MIN_ELEMENTS(1)],
  float output[XNN_MIN_ELEMENTS(1)])
{
  *output = 1.0f / *input;
}

enum xnn_status xnn_reshape_scaled_dot_attention_ntc_f32(
  xnn_operator_t attention_op,
  size_t batch_size,
  size_t query_tokens,
  size_t key_value_tokens,
  size_t channels,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool)
{
  const enum xnn_operator_type expected_operator_type = xnn_operator_type_scaled_dot_attention_ntc_f32;
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

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  const size_t log2_element_size = XNN_LOG2_SIZEOF_FLOAT;
  const size_t element_size = sizeof(float);

  const uint32_t mr = attention_op->ukernel.gemm.mr;
  const uint32_t nr = attention_op->ukernel.gemm.nr;
  const uint32_t kr = attention_op->ukernel.gemm.kr;
  const uint32_t sr = attention_op->ukernel.gemm.sr;

  // Calculate size required for workspace.
  // 1. Workspace for Q scaled, same size as Q.
  // TODO(zhin): change this to num_threads * channels when pthreadpool can pass thread id.
  const size_t q_scaled_size =
      round_up_po2(
        (batch_size * query_tokens * channels) << log2_element_size,
        XNN_ALLOCATION_ALIGNMENT);

  // Key is [key_value_tokens (output channel), channels (input channel)].
  const size_t key_n_stride = round_up(key_value_tokens, nr);
  const size_t key_k_stride = round_up_po2(channels, kr * sr);
  const size_t key_batch_stride = key_n_stride * (element_size + (key_k_stride << log2_element_size));
  // 2. Workspace for packed key.
  const size_t packed_key_size = round_up_po2(batch_size * key_batch_stride, XNN_ALLOCATION_ALIGNMENT);

  // Value is [key_value_tokens (input channel), channels (output channel)].
  const size_t value_n_stride = round_up(channels, nr);
  const size_t value_k_stride = round_up_po2(key_value_tokens, kr * sr);
  const size_t value_batch_stride = value_n_stride * (element_size + (value_k_stride << log2_element_size));
  // 3. Workspace for packed key.
  const size_t packed_value_size = round_up_po2(batch_size * value_batch_stride, XNN_ALLOCATION_ALIGNMENT);

  // 4. Workspace for logits (Q*K), of dimension query_tokens * key_value_tokens.
  // TODO(zhin): change this to be num_threads * key_value_tokens when pthreadpool can pass thread id.
  // BxNxN buffer for temporary storage of QK.
  const size_t logits_size =
      round_up_po2(XNN_EXTRA_BYTES + batch_size * query_tokens * key_value_tokens * element_size,
                   XNN_ALLOCATION_ALIGNMENT);

  const size_t total_workspace_size = q_scaled_size + packed_key_size + packed_value_size + logits_size;

  *workspace_size = total_workspace_size;
  *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;

  // Pack key.
  attention_op->context.packw_gemm_goi = (struct packw_gemm_goi_context) {
    .kc = channels,
    .nr = nr,
    .kr = kr,
    .sr = sr,
    .k_stride = channels << log2_element_size,
    // b_stride and gb_stride not needed because we do not have bias.
    .w_stride = element_size + (key_k_stride << log2_element_size),
    .packw_gemm_goi = attention_op->ukernel.gemm.packw_gemm_goi,
    .gk_stride = key_value_tokens * (channels << log2_element_size),
    .gc_stride = key_batch_stride,
  };
  attention_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
  attention_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_batched_packw_gemm_goi;
  attention_op->compute[0].context_offset =
    offsetof(struct xnn_operator, context.packw_gemm_goi) - offsetof(struct xnn_operator, context);
  attention_op->compute[0].range[0] = batch_size;
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
    .k_stride_elements = channels,
    // b_stride and gb_stride not needed because we do not have bias.
    .w_stride = element_size + (value_k_stride << log2_element_size),
    .packw_gemm_gio = attention_op->ukernel.gemm.packw_gemm_gio,
    .gk_stride = key_value_tokens * (channels << log2_element_size),
    .gb_stride = channels * element_size,
    .gc_stride = value_batch_stride,
  };
  attention_op->compute[1].type = xnn_parallelization_type_2d_tile_1d;
  attention_op->compute[1].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_batched_packw_gemm_gio;
  attention_op->compute[1].context_offset =
    offsetof(struct xnn_operator, context.packw_gemm_gio) - offsetof(struct xnn_operator, context);
  attention_op->compute[1].range[0] = batch_size;
  attention_op->compute[1].range[1] = channels;
  attention_op->compute[1].tile[0] = channels;

  struct xnn_hmp_gemm_ukernel gemm_ukernel = attention_op->ukernel.gemm.gemm_cases[mr - 1];

  attention_op->context.attention = (struct scaled_dot_attention_context){
    .channels = channels,
    .scaled_channels = channels * element_size,
    .key_value_tokens = key_value_tokens,
    .key_value_tokens_scaled = key_value_tokens * element_size,
    .cn_stride = nr << log2_element_size,
    .query_batch_stride = query_tokens * channels * element_size,
    .key_batch_stride = key_batch_stride,
    .value_batch_stride = value_batch_stride,
    .logits_batch_stride = query_tokens * key_value_tokens * element_size,
    .gemm_ukernel = gemm_ukernel,
    .compute_reciprocal = (xnn_compute_reciprocal_fn) compute_reciprocal_f32,
    .raddstoreexpminusmax_ukernel = attention_op->attention.raddstoreexpminusmax_config->ukernel,
    .rmax_ukernel = attention_op->attention.rmax_config->rmax.f32,
    .vadd_ukernel = attention_op->attention.vadd_config->minmax.op_ukernel,
    .vmul_ukernel = attention_op->attention.vmul_config->minmax.op_ukernel,
    .vmulc_ukernel = attention_op->attention.vmul_config->minmax.opc_ukernel,
    .vtanh_ukernel = attention_op->attention.vtanh_config->ukernel,
  };

  if (attention_op->attention.cap_type == xnn_attention_logits_cap_type_tanh) {
    attention_op->context.attention.logits_cap.type = xnn_attention_logits_cap_type_tanh;
    attention_op->context.attention.logits_cap.cap.f32 = attention_op->attention.cap_params.cap;
    attention_op->context.attention.logits_cap.cap_reciprocal.f32 = 1 / attention_op->attention.cap_params.cap;
  }

  #if XNN_MAX_UARCH_TYPES > 1
  if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
    attention_op->compute[2].type = xnn_parallelization_type_2d_tile_1d_with_uarch;
    attention_op->compute[2].task_2d_tile_1d_with_id = (pthreadpool_task_2d_tile_1d_with_id_t) xnn_compute_hmp_scaled_dot_attention;
  } else {
    attention_op->compute[2].type = xnn_parallelization_type_2d_tile_1d;
    attention_op->compute[2].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_scaled_dot_attention;
  }
  #else
    attention_op->compute[2].type = xnn_parallelization_type_2d_tile_1d;
    attention_op->compute[2].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_scaled_dot_attention;
  #endif  // XNN_MAX_UARCH_TYPES > 1

  attention_op->compute[2].range[0] = batch_size;
  attention_op->compute[2].range[1] = query_tokens;
  attention_op->compute[2].tile[0] = mr;

  attention_op->context.attention.scaled_query_offset = 0;
  attention_op->context.attention.packed_k_offset = q_scaled_size;
  attention_op->context.attention.packed_v_offset =
      q_scaled_size + packed_key_size;
  attention_op->context.attention.logits_offset =
      q_scaled_size + packed_key_size + packed_value_size;

  memcpy(&attention_op->context.attention.minmax_params,
         &attention_op->params.f32_minmax,
         sizeof(attention_op->params.f32_minmax));

  memcpy(&attention_op->context.attention.expminus_params,
         &attention_op->params2.f32_expminus_params,
         sizeof(attention_op->params2.f32_expminus_params));

  memcpy(&attention_op->context.attention.tanh_params,
         &attention_op->params3.f32_tanh,
         sizeof(attention_op->params3.f32_tanh));

  attention_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_setup_scaled_dot_attention_ntc_f32(
  xnn_operator_t attention_op,
  void* workspace,
  const float* query,
  const float* key,
  const float* value,
  const float* scale,
  const float* mask,
  float* output)
{
  enum xnn_operator_type expected_operator_type = xnn_operator_type_scaled_dot_attention_ntc_f32;
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
