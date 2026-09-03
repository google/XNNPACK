// Copyright 2026 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot.h"

// XNNPACK QC2 weights use signed two's-complement crumbs. This fast path is
// selected only when every per-channel weight zero point is zero.
static const int32_t xnn_qc2w_signed_lut[4] = {0, 1, -2, -1};

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr(void) {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();
}

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr(void) {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();
}
#endif  // XNN_ENABLE_KLEIDIAI

void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  assert(m == 1);
  kai_run_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max, xnn_qc2w_signed_lut);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
