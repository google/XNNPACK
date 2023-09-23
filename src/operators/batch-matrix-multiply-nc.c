// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params.h>

enum xnn_status create_batch_matrix_multiply_nc(
  uint32_t flags,
  const void* params,
  size_t params_size,
  const struct xnn_gemm_config* gemm_config,
  const struct gemm_fused_ukernels* gemm_ukernels,
  xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio,
  enum xnn_operator_type operator_type,
  xnn_operator_t* batch_matrix_multiply_op_out)
{
  xnn_operator_t batch_matrix_multiply_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to create %s operator: XNNPACK is not initialized", xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;
  batch_matrix_multiply_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (batch_matrix_multiply_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  memcpy(&batch_matrix_multiply_op->params, params, params_size);
  batch_matrix_multiply_op->type = operator_type;
  batch_matrix_multiply_op->flags = flags;

  const size_t mr = gemm_config->mr;
  batch_matrix_multiply_op->ukernel.type = xnn_microkernel_type_gemm;
  batch_matrix_multiply_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
    .mr = mr,
    .nr = gemm_config->nr,
    .kr = UINT32_C(1) << gemm_config->log2_kr,
    .sr = UINT32_C(1) << gemm_config->log2_sr,
  };

  assert(XNN_MAX_MR >= mr);
  for (size_t i = 0; i < mr; i++) {
    batch_matrix_multiply_op->ukernel.gemm.gemm_cases[i] = gemm_ukernels->gemm[i];
  }
  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
    batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi = gemm_config->pack_gemm_goi;
  } else {
    batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio = pack_gemm_gio;
  }

  batch_matrix_multiply_op->state = xnn_run_state_invalid;

  *batch_matrix_multiply_op_out = batch_matrix_multiply_op;
  return xnn_status_success;

error:
  xnn_delete_operator(batch_matrix_multiply_op);
  return status;
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_f32(
  uint32_t flags,
  xnn_operator_t* batch_matrix_multiply_op_out)
{
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_batch_matrix_multiply_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr-1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  union xnn_f32_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, -INFINITY, INFINITY);
  }

  return create_batch_matrix_multiply_nc(
    flags,
    &params, sizeof(params),
    gemm_config, gemm_ukernels,
    (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w,
    xnn_operator_type_batch_matrix_multiply_nc_f32,
    batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_f16(
  uint32_t flags,
  xnn_operator_t* batch_matrix_multiply_op_out)
{
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_batch_matrix_multiply_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr-1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  union xnn_f16_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, UINT16_C(0xFC00), UINT16_C(0x7C00));
  }

  return create_batch_matrix_multiply_nc(
    flags,
    &params, sizeof(params),
    gemm_config, gemm_ukernels,
    (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w,
    xnn_operator_type_batch_matrix_multiply_nc_f16,
    batch_matrix_multiply_op_out);
}

static enum xnn_status reshape_batch_matrix_multiply_nc(
  xnn_operator_t batch_matrix_multiply_op,
  enum xnn_operator_type expected_operator_type,
  size_t batch_size,
  size_t m,
  size_t k,
  size_t n,
  size_t* workspace_size,
  size_t* workspace_alignment,
  uint32_t log2_lhs_input_element_size,
  uint32_t log2_rhs_input_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  size_t params_size,
  size_t num_threads)
{
  if (batch_matrix_multiply_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(batch_matrix_multiply_op->type));
    return xnn_status_invalid_parameter;
  }
  batch_matrix_multiply_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(batch_matrix_multiply_op->type));
    return xnn_status_uninitialized;
  }

  if (m == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu rows: number of rows must be non-zero",
      xnn_operator_type_to_string(batch_matrix_multiply_op->type), m);
    return xnn_status_invalid_parameter;
  }

  if (k == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu columns: number of columns must be non-zero",
      xnn_operator_type_to_string(batch_matrix_multiply_op->type), k);
    return xnn_status_invalid_parameter;
  }

  if (n == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu columns: number of columns must be non-zero",
      xnn_operator_type_to_string(batch_matrix_multiply_op->type), n);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    batch_matrix_multiply_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const uint32_t nr = batch_matrix_multiply_op->ukernel.gemm.nr;
  const uint32_t kr = batch_matrix_multiply_op->ukernel.gemm.kr;
  const uint32_t sr = batch_matrix_multiply_op->ukernel.gemm.sr;

  const size_t n_stride = round_up(n, nr);
  const size_t k_stride = round_up_po2(k, kr * sr);
  const size_t rhs_input_batch_stride = (n_stride * bias_element_size + ((n_stride * k_stride) << log2_rhs_input_element_size));

  *workspace_size = batch_size * rhs_input_batch_stride;
  *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;

  uint32_t mr = batch_matrix_multiply_op->ukernel.gemm.mr;
  struct xnn_hmp_gemm_ukernel *gemm_cases = batch_matrix_multiply_op->ukernel.gemm.gemm_cases;

  if (m == 1 && batch_matrix_multiply_op->ukernel.gemm.gemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
    mr = 1;
  }

  assert(mr != 0 && mr <= XNN_MAX_MR);
  struct xnn_hmp_gemm_ukernel gemm_ukernel = gemm_cases[mr-1];

  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
    assert(batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi != NULL);
    batch_matrix_multiply_op->context.packw_gemm_goi = (struct packw_gemm_goi_context) {
      .kc = k,
      .nr = nr,
      .kr = kr,
      .sr = sr,
      .k_stride = k << log2_rhs_input_element_size,
      .bias = NULL,
      .b_stride = bias_element_size,
      .w_stride = bias_element_size + (k_stride << log2_rhs_input_element_size),
      .packw_gemm_goi = batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi,
      .gk_stride = n * (k << log2_rhs_input_element_size),
      .gb_stride = n * bias_element_size,
      .gc_stride = rhs_input_batch_stride,
    };
    batch_matrix_multiply_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
    batch_matrix_multiply_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_batched_packw_gemm_goi;
    batch_matrix_multiply_op->compute[0].context_offset =
      offsetof(struct xnn_operator, context.packw_gemm_goi) - offsetof(struct xnn_operator, context);
    batch_matrix_multiply_op->compute[0].range[0] = batch_size;
    batch_matrix_multiply_op->compute[0].range[1] = n;
    batch_matrix_multiply_op->compute[0].tile[0] = nr;
  } else {
    assert(batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio != NULL);
    batch_matrix_multiply_op->context.packw_gemm_gio = (struct packw_gemm_gio_context) {
      .n_stride = 1 << log2_rhs_input_element_size,
      .k_stride_elements = n,
      .kc = k,
      .nr = nr,
      .kr = kr,
      .sr = sr,
      .bias = NULL,
      .b_stride = bias_element_size,
      .w_stride = bias_element_size + (k_stride << log2_lhs_input_element_size),
      .packw_gemm_gio = batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio,
      .gk_stride = k * (n << log2_rhs_input_element_size),
      .gb_stride = n * bias_element_size,
      .gc_stride = rhs_input_batch_stride,
    };

    batch_matrix_multiply_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
    batch_matrix_multiply_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_batched_packw_gemm_gio;
    batch_matrix_multiply_op->compute[0].context_offset =
      offsetof(struct xnn_operator, context.packw_gemm_gio) - offsetof(struct xnn_operator, context);
    batch_matrix_multiply_op->compute[0].range[0] = batch_size;
    batch_matrix_multiply_op->compute[0].range[1] = n;
    batch_matrix_multiply_op->compute[0].tile[0] = nr;

  }

  size_t w_stride = bias_element_size + (round_up_po2(k, kr * sr) << log2_lhs_input_element_size);
  batch_matrix_multiply_op->context.gemm = (struct gemm_context) {
    .k_scaled = k << log2_lhs_input_element_size,
    .a_stride = k << log2_lhs_input_element_size,
    .ga_stride = m * (k << log2_lhs_input_element_size),
    .w_stride = w_stride,
    .gw_stride = w_stride * round_up(n, nr),
    .cm_stride = n << log2_output_element_size,
    .cn_stride = nr << log2_output_element_size,
    .gc_stride = (m * n) << log2_output_element_size,
    .log2_csize = log2_output_element_size,
    .ukernel = gemm_ukernel,
  };
  memcpy(&batch_matrix_multiply_op->context.gemm.params, params, params_size);
  batch_matrix_multiply_op->context.gemm.fused_params = &batch_matrix_multiply_op->context.gemm.params;

  #if XNN_TEST_MODE
    const size_t nc = nr;
  #else
    size_t nc = n;
    if (num_threads > 1) {
      const size_t num_other_tiles = divide_round_up(m, mr);
      const size_t target_tiles_per_thread = 5;
      const size_t max_nc = divide_round_up(n * num_other_tiles, num_threads * target_tiles_per_thread);
      if (max_nc < nc) {
        nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
      }
    }
  #endif
  #if XNN_MAX_UARCH_TYPES > 1
    if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
      batch_matrix_multiply_op->compute[1].type = xnn_parallelization_type_3d_tile_2d_with_uarch;
      batch_matrix_multiply_op->compute[1].task_3d_tile_2d_with_id =
        (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_gemm;
    } else {
      batch_matrix_multiply_op->compute[1].type = xnn_parallelization_type_3d_tile_2d;
      batch_matrix_multiply_op->compute[1].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
    }
  #else
    batch_matrix_multiply_op->compute[1].type = xnn_parallelization_type_3d_tile_2d;
    batch_matrix_multiply_op->compute[1].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
  #endif
  batch_matrix_multiply_op->compute[1].range[0] = batch_size;
  batch_matrix_multiply_op->compute[1].range[1] = m;
  batch_matrix_multiply_op->compute[1].range[2] = n;
  batch_matrix_multiply_op->compute[1].tile[0] = mr;
  batch_matrix_multiply_op->compute[1].tile[1] = nc;
  batch_matrix_multiply_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f16(
  xnn_operator_t batch_matrix_multiply_op,
  size_t batch_size,
  size_t m,
  size_t k,
  size_t n,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool)
{
  return reshape_batch_matrix_multiply_nc(
    batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f16,
    batch_size, m, k, n,
    workspace_size, workspace_alignment,
    /*log2_lhs_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_rhs_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*bias_element_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    &batch_matrix_multiply_op->params.f16_minmax,
    sizeof(batch_matrix_multiply_op->params.f16_minmax),
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f32(
  xnn_operator_t batch_matrix_multiply_op,
  size_t batch_size,
  size_t m,
  size_t k,
  size_t n,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool)
{
  return reshape_batch_matrix_multiply_nc(
    batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f32,
    batch_size, m, k, n,
    workspace_size, workspace_alignment,
    /*log2_lhs_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_rhs_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*bias_element_size=*/sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    &batch_matrix_multiply_op->params.f32_minmax,
    sizeof(batch_matrix_multiply_op->params.f32_minmax),
    pthreadpool_get_threads_count(threadpool));
}

static enum xnn_status setup_batch_matrix_multiply_nc(
  xnn_operator_t batch_matrix_multiply_op,
  enum xnn_operator_type expected_operator_type,
  void* workspace,
  const void* lhs_input,
  const void* rhs_input,
  void* output)
{
  if (batch_matrix_multiply_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(batch_matrix_multiply_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (batch_matrix_multiply_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(batch_matrix_multiply_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
    batch_matrix_multiply_op->context.packw_gemm_goi.kernel = rhs_input;
    batch_matrix_multiply_op->context.packw_gemm_goi.bias = NULL;
    batch_matrix_multiply_op->context.packw_gemm_goi.packed_weights = workspace;
  } else {
    batch_matrix_multiply_op->context.packw_gemm_gio.kernel = rhs_input;
    batch_matrix_multiply_op->context.packw_gemm_gio.bias = NULL;
    batch_matrix_multiply_op->context.packw_gemm_gio.packed_weights = workspace;
  }

  batch_matrix_multiply_op->context.gemm.a = lhs_input;
  batch_matrix_multiply_op->context.gemm.packed_w = workspace;
  batch_matrix_multiply_op->context.gemm.c = output;

  batch_matrix_multiply_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_f16(
  xnn_operator_t batch_matrix_multiply_op,
  void* workspace,
  const void* lhs_input,
  const void* rhs_input,
  void* output)
{
  return setup_batch_matrix_multiply_nc(
    batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f16,
    workspace, lhs_input, rhs_input, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_f32(
  xnn_operator_t batch_matrix_multiply_op,
  void* workspace,
  const float* lhs_input,
  const float* rhs_input,
  float* output)
{
  return setup_batch_matrix_multiply_nc(
    batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f32,
    workspace, lhs_input, rhs_input, output);
}
