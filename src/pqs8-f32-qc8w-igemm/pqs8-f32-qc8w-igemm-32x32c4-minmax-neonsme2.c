// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr(void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
#else
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr(void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
#else
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
#if XNN_ENABLE_KLEIDIAI
  const size_t kai_kr = 4;
  const size_t k = ks * round_up(kc, kai_kr);

  // Repackage the params.
  struct kai_matmul_requantize32_params kai_params;
  kai_params.output_zero_point = params->fp32_scalar.output_zero_point;
  kai_params.min_value = (int8_t)params->fp32_scalar.output_min;
  kai_params.max_value = (int8_t)params->fp32_scalar.output_max;

  kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
      mr, nc, k, packed_lhs, w, c, cm_stride, sizeof(int8_t), &kai_params);
#else
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}