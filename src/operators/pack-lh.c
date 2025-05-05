// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

enum xnn_status create_pack_lh(
    uint32_t flags,
    const struct xnn_pack_lh_config *pack_lh_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* pack_lh_op_out)
{
  xnn_operator_t pack_lh_op = NULL;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (pack_lh_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  pack_lh_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (pack_lh_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_out_of_memory;
  }

  pack_lh_op->pack_lh_config = pack_lh_config;
  pack_lh_op->type = expected_operator_type;
  pack_lh_op->flags = flags;
  pack_lh_op->state = xnn_run_state_invalid;

  *pack_lh_op_out = pack_lh_op;
  return xnn_status_success;
}

enum xnn_status xnn_create_pack_lh_x32(
    uint32_t flags,
    xnn_operator_t* pack_lh_op_out) {
  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x32_pack_lh_config();
  return create_pack_lh(flags, pack_lh_config,
          xnn_operator_type_pack_lh_x32, pack_lh_op_out);
}

enum xnn_status xnn_create_pack_lh_x16(
    uint32_t flags,
    xnn_operator_t* pack_lh_op_out) {
  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x16_pack_lh_config();
  return create_pack_lh(flags, pack_lh_config,
          xnn_operator_type_pack_lh_x16, pack_lh_op_out);
}

enum xnn_status xnn_create_pack_lh_x8(
    uint32_t flags,
    xnn_operator_t* pack_lh_op_out) {
  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x8_pack_lh_config();
  return create_pack_lh(flags, pack_lh_config,
          xnn_operator_type_pack_lh_x8, pack_lh_op_out);
}

enum xnn_status reshape_pack_lh(xnn_operator_t pack_lh_op, size_t num_groups,
                                size_t batch_size, size_t channels,
                                size_t* output_size_bytes,
                                enum xnn_operator_type expected_operator_type,
                                size_t element_size,
                                const struct xnn_gemm_config* gemm_config,
                                const struct xnn_pack_lh_config* pack_lh_config,
                                pthreadpool_t threadpool) {
  if (pack_lh_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(pack_lh_op));
    return xnn_status_invalid_parameter;
  }
  pack_lh_op->state = xnn_run_state_invalid;

  if (num_groups == 0 || batch_size == 0) {
    pack_lh_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  pack_lh_op->batch_size = batch_size;

  const uint32_t mr_packed = batch_size == 1          ? 1
                             : gemm_config->mr_packed ? gemm_config->mr_packed
                                                      : gemm_config->mr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;

  const size_t group_size =
      pack_lh_config->size_fn(batch_size, channels, mr_packed, kr, sr);

  pack_lh_op->context.pack_lh = (struct pack_lh_context){
      .m = batch_size,
      .k = channels,
      .mr = mr_packed,
      .kr = kr,
      .sr = sr,
      .lhs_stride = channels * element_size,
      .gi_stride = batch_size *channels * element_size ,
      .gp_stride = group_size,
      .packed_offset_fn = (xnn_pack_lh_offset_fn)pack_lh_config->offset_fn,
      .pack_lh_ukernel = (xnn_pack_lh_ukernel_fn)pack_lh_config->ukernel,
  };

  *output_size_bytes = num_groups * group_size;
  pack_lh_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
  pack_lh_op->compute[0].task_2d_tile_1d =
      (pthreadpool_task_2d_tile_1d_t) xnn_compute_pack_lh;
  pack_lh_op->compute[0].range[0] = num_groups;
  pack_lh_op->compute[0].range[1] = batch_size;
  pack_lh_op->compute[0].tile[0] = mr_packed;

  pack_lh_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_pack_lh_x16(xnn_operator_t pack_lh_op,
                                        size_t num_groups, size_t batch_size,
                                        size_t channels,
                                        size_t* output_size_bytes,
                                        pthreadpool_t threadpool) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_pf16_gemm_config();
  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x16_pack_lh_config();
  return reshape_pack_lh(pack_lh_op, num_groups, batch_size, channels,
                         output_size_bytes, xnn_operator_type_pack_lh_x16,
                         /*element_size=*/sizeof(xnn_float16), gemm_config,
                         pack_lh_config, threadpool);
}

enum xnn_status xnn_reshape_pack_lh_x8(xnn_operator_t pack_lh_op,
                                       size_t num_groups, size_t batch_size,
                                       size_t channels,
                                       size_t* output_size_bytes,
                                       pthreadpool_t threadpool) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_pqs8_qc8w_gemm_config();
  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x8_pack_lh_config();
  return reshape_pack_lh(pack_lh_op, num_groups, batch_size, channels,
                         output_size_bytes, xnn_operator_type_pack_lh_x8,
                         /*element_size=*/sizeof(int8_t), gemm_config,
                         pack_lh_config, threadpool);
}

enum xnn_status xnn_reshape_pack_lh_x32(xnn_operator_t pack_lh_op,
                                        size_t num_groups, size_t batch_size,
                                        size_t channels,
                                        size_t* output_size_bytes,
                                        pthreadpool_t threadpool) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_pf32_gemm_config();
  const struct xnn_pack_lh_config *pack_lh_config = xnn_init_x32_pack_lh_config();
  return reshape_pack_lh(pack_lh_op, num_groups, batch_size, channels,
                         output_size_bytes, xnn_operator_type_pack_lh_x32,
                         /*element_size=*/sizeof(float), gemm_config,
                         pack_lh_config, threadpool);
}

enum xnn_status setup_pack_lh(
  xnn_operator_t pack_lh_op,
  const void* input,
  void* output,
  enum xnn_operator_type expected_operator_type)
{
  if (pack_lh_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(pack_lh_op));
    return xnn_status_invalid_parameter;
  }
  switch (pack_lh_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(pack_lh_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different
      // pointers.
      break;
  }

  pack_lh_op->context.pack_lh.lhs = input;
  pack_lh_op->context.pack_lh.lhs_packed = output;
  pack_lh_op->state = xnn_run_state_ready;
  return xnn_status_success;
}

enum xnn_status xnn_setup_pack_lh_x16(
  xnn_operator_t pack_lh_op,
  const void* input,
  void* output)
{
  return setup_pack_lh(pack_lh_op, input, output,
          xnn_operator_type_pack_lh_x16);
}

enum xnn_status xnn_setup_pack_lh_x8(
  xnn_operator_t pack_lh_op,
  const void* input,
  void* output)
{
  return setup_pack_lh(pack_lh_op, input, output,
          xnn_operator_type_pack_lh_x8);
}

enum xnn_status xnn_setup_pack_lh_x32(
  xnn_operator_t pack_lh_op,
  const void* input,
  void* output)
{
  return setup_pack_lh(pack_lh_op, input, output, xnn_operator_type_pack_lh_x32);
}
