// Copyright 2024 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_mr() {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
#else
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_nr() {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
#else
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

// Wraps the `kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const struct xnn_f32_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
    assert(k % sizeof(float) == 0);
    assert(dst_stride_col == sizeof(float));
    (void) dst_stride_col;
    const size_t k_elements = k / sizeof(float);

    const struct kai_matmul_uker_config config = {0};
    const struct kai_matmul_uker_api api = kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
    const struct kai_matmul_uker_lhs_dim_args lhs_shape = {m, k_elements};
    const struct kai_matmul_uker_rhs_dim_args rhs_shape = {n, k_elements};
    struct kai_matmul_uker_args args = {
        .flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
        .shape.m = m,
        .shape.n = n,
        .shape.k = k_elements,

        .operand.dst.ptr = dst,
        .operand.dst.stride.m = dst_stride_row,
        .operand.lhs.ptr = lhs_packed,
        .operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape),
        .operand.rhs.ptr = rhs_packed,
        .operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape),

        .activation.clamp.min_ptr = &minmax_params->scalar.min,
        .activation.clamp.max_ptr = &minmax_params->scalar.max,
    };
    api.run(&config, &args);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
