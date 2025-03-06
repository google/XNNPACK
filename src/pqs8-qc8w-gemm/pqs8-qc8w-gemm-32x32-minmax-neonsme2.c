// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
  // Keep this line indented to avoid it being pulled out of the #ifdef when the
  // sources are amalgamated.
  #include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#endif  // XNN_ENABLE_KLEIDIAI


size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32__neonsme2_get_mr() {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
#else
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32__neonsme2_get_nr() {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
#else
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

// Wraps the `kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params
        minmax_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]) {
#if XNN_ENABLE_KLEIDIAI
  struct kai_matmul_requantize32_params kai_params;
  kai_params.output_zero_point = minmax_params->fp32_scalar.output_zero_point;
  kai_params.min_value = minmax_params->fp32_scalar.output_min;
  kai_params.max_value = minmax_params->fp32_scalar.output_max;

  kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
      m, n, k / sizeof(int8_t), lhs_packed, rhs_packed, dst, dst_stride_row, /*dst_stride_col=*/sizeof(int8_t),
      &kai_params);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}
