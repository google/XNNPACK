// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/config-types.h"
#include "xnnpack/config.h"
#include "xnnpack/log.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"

enum xnn_status xnn_create_pack_lh_x32(
    uint32_t flags,
    xnn_operator_t* pack_lh_op_out)
{
  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x32_pack_lh_config();
  xnn_operator_t pack_lh_op = NULL;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_pack_lh_x32));
    return xnn_status_uninitialized;
  }

  if (pack_lh_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_pack_lh_x32));
    return xnn_status_unsupported_hardware;
  }

  pack_lh_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (pack_lh_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_pack_lh_x32));
    return xnn_status_out_of_memory;
  }

  pack_lh_op->pack_lh_config = pack_lh_config;
  pack_lh_op->type = xnn_operator_type_pack_lh_x32;
  pack_lh_op->flags = flags;
  pack_lh_op->state = xnn_run_state_invalid;

  *pack_lh_op_out = pack_lh_op;
  return xnn_status_success;
}

enum xnn_status xnn_reshape_pack_lh_x32(
  xnn_operator_t pack_lh_op,
  size_t batch_size,
  size_t channels,
  size_t *output_size_bytes,
  pthreadpool_t threadpool)
{
  if (pack_lh_op->type != xnn_operator_type_pack_lh_x32) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(xnn_operator_type_pack_lh_x32),
        xnn_operator_type_to_string(pack_lh_op->type));
    return xnn_status_invalid_parameter;
  }
  pack_lh_op->state = xnn_run_state_invalid;

  if (batch_size == 0) {
    pack_lh_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  pack_lh_op->batch_size = batch_size;

  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x32_pack_lh_config();
  const struct xnn_gemm_config* gemm_config =
      xnn_init_pf32_gemm_config();
  const uint32_t mr_packed = batch_size == 1 ? 1 : gemm_config->mr_packed;
  const size_t mr = gemm_config->mr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;

  pack_lh_op->context.x32_pack_lh = (struct x32_pack_lh_context) {
      .m = batch_size,
      .k = channels,
      .mr = mr,//mr_packed,
      .kr = kr,
      .sr = sr,
      .lhs_stride = channels * sizeof(float),
      .pack_lh_ukernel = (xnn_x32_pack_lh_ukernel_fn)
                           pack_lh_op->pack_lh_config->ukernel,
  };

  *output_size_bytes = pack_lh_config->size_fn(batch_size, channels, mr, kr, sr);
  pack_lh_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
  pack_lh_op->compute[0].task_1d =
      (pthreadpool_task_1d_t) xnn_compute_x32_pack_lh;
  pack_lh_op->compute[0].range[0] = batch_size;
  pack_lh_op->compute[0].tile[0] = mr_packed;

  pack_lh_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_setup_pack_lh_x32(
  xnn_operator_t pack_lh_op,
  const void* input,
  void* output)
{
  if (pack_lh_op->type != xnn_operator_type_pack_lh_x32) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(xnn_operator_type_pack_lh_x32),
        xnn_operator_type_to_string(pack_lh_op->type));
    return xnn_status_invalid_parameter;
  }
  switch (pack_lh_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string(pack_lh_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different
      // pointers.
      break;
  }

  pack_lh_op->context.x32_pack_lh.lhs = input;
  pack_lh_op->context.x32_pack_lh.lhs_packed = output;
  pack_lh_op->state = xnn_run_state_ready;
  return xnn_status_success;
}
