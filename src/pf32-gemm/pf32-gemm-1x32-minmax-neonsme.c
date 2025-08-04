// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_mr() {
  return 1;
}

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla();
}
#endif  // XNN_ENABLE_KLEIDIAI

// Wraps the `kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const struct xnn_f32_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla(
      m, n, k / sizeof(float), lhs_packed, /*unused_lhs_stride=*/0, rhs_packed,
      dst, dst_stride_row, /*dst_stride_col=*/sizeof(float),
      minmax_params->scalar.min, minmax_params->scalar.max);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}
