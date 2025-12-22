// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot.h"

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme_get_mr() {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot();
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot();
}
#endif  // XNN_ENABLE_KLEIDIAI

// Wraps the `kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    struct xnn_f32_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row, /*dst_stride_col=*/sizeof(float),
      minmax_params->scalar.min, minmax_params->scalar.max);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
