// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme_get_mr(void) {
  return kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa();
}

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme_get_nr(void) {
  return kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa();
}

void xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, xnn_float16* restrict c, size_t cm_stride,
    const struct xnn_f16_minmax_params* params) {
  const size_t kai_kr = 2;
  const size_t k = ks * round_up(kc, kai_kr);

  kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa(
      mr, nc, k, packed_lhs, w, c, cm_stride * sizeof(xnn_float16),
      sizeof(xnn_float16), xnn_float16_to_float(params->scalar.min),
      xnn_float16_to_float(params->scalar.max));
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme_get_nr(void) {
  return 2 * svcntsw();
}

void xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, xnn_float16* restrict c, size_t cm_stride,
    const struct xnn_f16_minmax_params* params) {
  xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme(
      mr, nc, ks * round_up_po2(kc, 2) * sizeof(xnn_float16), packed_lhs, w, c,
      cm_stride * sizeof(xnn_float16), sizeof(xnn_float16), params);
}
#else
#include <assert.h>

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme_get_mr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme_get_nr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, xnn_float16* restrict c, size_t cm_stride,
    const struct xnn_f16_minmax_params* params) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
