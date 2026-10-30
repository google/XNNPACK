// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_mr() { return 1; }

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla();
}

// Wraps the `kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla(
      m, n, k / sizeof(float), lhs_packed, /*unused_lhs_stride=*/0, rhs_packed,
      dst, dst_stride_row, /*dst_stride_col=*/sizeof(float),
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_mr() { return 1; }

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pf32_gemm_1x8vl_sme_mla_impl(size_t m, size_t n,
                                                       size_t k_elements,
                                                       const float* lhs,
                                                       const float* rhs_packed,
                                                       void* dst,
                                                       size_t dst_stride_row,
                                                       float clamp_min,
                                                       float clamp_max) {
  const size_t vl = svcntw();
  const size_t n_step = 2 * vl;
  const size_t panel_floats = n_step * (1 + k_elements);
  const svbool_t pg32 = svptrue_b32();
  const svbool_t p_row0 = svwhilelt_b32_u64(0, 1);
  const svbool_t p_row1 = sveor_b_z(pg32, svwhilelt_b32_u64(0, 2), p_row0);
  const svbool_t p_row2 =
      sveor_b_z(pg32, svwhilelt_b32_u64(0, 3), svwhilelt_b32_u64(0, 2));
  const svbool_t p_row3 =
      sveor_b_z(pg32, svwhilelt_b32_u64(0, 4), svwhilelt_b32_u64(0, 3));
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const float* lhs_row = lhs + m_idx * k_elements;
    float* dst_row = (float*)((uintptr_t)dst + m_idx * dst_stride_row);

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
      const float* p0 = rhs_packed + (n_idx / n_step) * panel_floats;
      const float* w0 = p0 + n_step;

      svzero_za();

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        __builtin_prefetch(w0 + 16 * n_step, 0, 0);
        const svfloat32_t va0 = svld1rq_f32(pg32, lhs_row + k_idx);
        const svfloat32_t va1 = svld1rq_f32(pg32, lhs_row + k_idx + 4);

        svmopa_za32_f32_m(0, p_row0, pg32, va0, svld1_vnum_f32(pg32, w0, 0));
        svmopa_za32_f32_m(1, p_row0, pg32, va0, svld1_vnum_f32(pg32, w0, 1));
        svmopa_za32_f32_m(0, p_row1, pg32, va0, svld1_vnum_f32(pg32, w0, 2));
        svmopa_za32_f32_m(1, p_row1, pg32, va0, svld1_vnum_f32(pg32, w0, 3));
        svmopa_za32_f32_m(0, p_row2, pg32, va0, svld1_vnum_f32(pg32, w0, 4));
        svmopa_za32_f32_m(1, p_row2, pg32, va0, svld1_vnum_f32(pg32, w0, 5));
        svmopa_za32_f32_m(0, p_row3, pg32, va0, svld1_vnum_f32(pg32, w0, 6));
        svmopa_za32_f32_m(1, p_row3, pg32, va0, svld1_vnum_f32(pg32, w0, 7));
        w0 += 4 * n_step;

        svmopa_za32_f32_m(2, p_row0, pg32, va1, svld1_vnum_f32(pg32, w0, 0));
        svmopa_za32_f32_m(3, p_row0, pg32, va1, svld1_vnum_f32(pg32, w0, 1));
        svmopa_za32_f32_m(2, p_row1, pg32, va1, svld1_vnum_f32(pg32, w0, 2));
        svmopa_za32_f32_m(3, p_row1, pg32, va1, svld1_vnum_f32(pg32, w0, 3));
        svmopa_za32_f32_m(2, p_row2, pg32, va1, svld1_vnum_f32(pg32, w0, 4));
        svmopa_za32_f32_m(3, p_row2, pg32, va1, svld1_vnum_f32(pg32, w0, 5));
        svmopa_za32_f32_m(2, p_row3, pg32, va1, svld1_vnum_f32(pg32, w0, 6));
        svmopa_za32_f32_m(3, p_row3, pg32, va1, svld1_vnum_f32(pg32, w0, 7));
        w0 += 4 * n_step;
      }

      for (; k_idx < k_elements; ++k_idx) {
        const svfloat32_t va = svdup_n_f32(lhs_row[k_idx]);
        svmopa_za32_f32_m(0, p_row0, pg32, va, svld1_vnum_f32(pg32, w0, 0));
        svmopa_za32_f32_m(1, p_row0, pg32, va, svld1_vnum_f32(pg32, w0, 1));
        w0 += n_step;
      }

      const svfloat32_t vsum02 = svadd_f32_x(
          pg32,
          svadd_f32_x(
              pg32,
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 0),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 1)),
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 2),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 3))),
          svadd_f32_x(
              pg32,
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 0),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 1)),
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 2),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 3))));
      const svfloat32_t vsum13 = svadd_f32_x(
          pg32,
          svadd_f32_x(
              pg32,
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 0),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 1)),
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 2),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 3))),
          svadd_f32_x(
              pg32,
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 0),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 1)),
              svadd_f32_x(pg32,
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 2),
                          svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 3))));

      svfloat32_t vacc0 =
          svadd_f32_x(pg32, svld1_vnum_f32(pg32, p0, 0), vsum02);
      svfloat32_t vacc1 =
          svadd_f32_x(pg32, svld1_vnum_f32(pg32, p0, 1), vsum13);
      vacc0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vacc0, vmin), vmax);
      vacc1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vacc1, vmin), vmax);

      const size_t n_rem = n - n_idx;
      svst1_vnum_f32(svwhilelt_b32_u64(0 * vl, n_rem), dst_row + n_idx, 0,
                     vacc0);
      svst1_vnum_f32(svwhilelt_b32_u64(1 * vl, n_rem), dst_row + n_idx, 1,
                     vacc1);
    }
  }
}

void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  (void)dst_stride_col;
  xnn_pf32_gemm_1x8vl_sme_mla_impl(
      m, n, k / sizeof(float), (const float*)lhs_packed,
      (const float*)rhs_packed, dst, dst_stride_row, minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#else
#include <assert.h>

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_mr() { return 0; }
size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_nr() { return 0; }
void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
