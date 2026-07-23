// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55.h"
#endif  // XNN_ENABLE_KLEIDIAI

void xnn_pf16_gemm_minmax_ukernel_6x32__kai_aarch64_neonfp16arith(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const struct xnn_f16_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla(
      m, n, k / sizeof(xnn_float16), lhs_packed,
      /*lhs_stride=*/k, rhs_packed, dst, dst_stride_row, dst_stride_col,
      xnn_float16_to_float(minmax_params->scalar.min),
      xnn_float16_to_float(minmax_params->scalar.max));
#else
  (void)m;
  (void)n;
  (void)k;
  (void)lhs_packed;
  (void)rhs_packed;
  (void)dst;
  (void)dst_stride_row;
  (void)dst_stride_col;
  (void)minmax_params;
  assert(
      "Calling KleidiAI Adv SIMD F16 6x32 wrapper, but XNNPACK was compiled "
      "without `XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}

void xnn_pf16_gemm_minmax_ukernel_6x32__kai_aarch64_neonfp16arith_cortex_a55(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const struct xnn_f16_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55(
      m, n, k / sizeof(xnn_float16), lhs_packed,
      /*lhs_stride=*/k, rhs_packed, dst, dst_stride_row, dst_stride_col,
      xnn_float16_to_float(minmax_params->scalar.min),
      xnn_float16_to_float(minmax_params->scalar.max));
#else
  (void)m;
  (void)n;
  (void)k;
  (void)lhs_packed;
  (void)rhs_packed;
  (void)dst;
  (void)dst_stride_row;
  (void)dst_stride_col;
  (void)minmax_params;
  assert(
      "Calling KleidiAI Adv SIMD F16 6x32 Arm(R) Cortex(TM)-A55 wrapper, but XNNPACK was "
      "compiled without `XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
