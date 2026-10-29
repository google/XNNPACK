// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr(void) {
  return kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr(void) {
  return kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
}

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
  const size_t kai_kr = 4;
  const size_t k = ks * round_up(kc, kai_kr);

  // Repackage the params.
  struct kai_matmul_requantize32_params kai_params;
  kai_params.output_zero_point = params->fp32_scalar.output_zero_point;
  kai_params.min_value = (int8_t)params->fp32_scalar.output_min;
  kai_params.max_value = (int8_t)params->fp32_scalar.output_max;

  kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
      mr, nc, k, packed_lhs, w, c, cm_stride, sizeof(int8_t), &kai_params);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

XNN_INTERNAL void xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* w,
    void* dst, size_t dst_stride_row, size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* params);

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr(void) {
  return 2 * svcntsw();
}

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
  xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2(
      mr, nc, ks * round_up_po2(kc, 4) * sizeof(int8_t), packed_lhs, w, c,
      cm_stride, sizeof(int8_t), params);
}
#else
#include <assert.h>

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
