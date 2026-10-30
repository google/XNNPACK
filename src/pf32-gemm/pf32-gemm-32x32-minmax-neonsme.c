// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_mr() {
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
}

// Wraps the `kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
      m, n, k / sizeof(float), lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pf32_gemm_2vlx2vl_sme_mopa_impl(size_t m, size_t n,
                                                          size_t k_elements,
                                                          const float*
                                                              lhs_packed,
                                                          const float*
                                                              rhs_packed,
                                                          float* dst,
                                                          size_t dst_stride_row,
                                                          float clamp_min,
                                                          float clamp_max) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t n_step = 2 * vl;
  const size_t lhs_panel_floats = m_step * k_elements;
  const size_t rhs_panel_floats = n_step * (1 + k_elements);
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vones = svdup_n_f32(1.0f);
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
    const size_t m_rem = m - m_idx < m_step ? m - m_idx : m_step;
    const float* lhs_panel = lhs_packed + (m_idx / m_step) * lhs_panel_floats;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
      if (m_rem <= vl && n_idx + n_step < n) {
        const float* rhs_panel0 =
            rhs_packed + (n_idx / n_step) * rhs_panel_floats;
        const float* rhs_panel1 = rhs_panel0 + rhs_panel_floats;
        const float* rhs_w0 = rhs_panel0 + n_step;
        const float* rhs_w1 = rhs_panel1 + n_step;
        const svfloat32_t vbias0 = svld1_vnum_f32(pg32, rhs_panel0, 0);
        const svfloat32_t vbias1 = svld1_vnum_f32(pg32, rhs_panel0, 1);
        const svfloat32_t vbias2 = svld1_vnum_f32(pg32, rhs_panel1, 0);
        const svfloat32_t vbias3 = svld1_vnum_f32(pg32, rhs_panel1, 1);
        svzero_za();
        svmopa_za32_f32_m(0, pg32, pg32, vones, vbias0);
        svmopa_za32_f32_m(1, pg32, pg32, vones, vbias1);
        svmopa_za32_f32_m(2, pg32, pg32, vones, vbias2);
        svmopa_za32_f32_m(3, pg32, pg32, vones, vbias3);

        const float* a_ptr = lhs_panel;
        const float* b0_ptr = rhs_w0;
        const float* b1_ptr = rhs_w1;
        size_t k_idx = 0;
        for (; k_idx + 4 <= k_elements; k_idx += 4) {
#pragma clang loop unroll(full)
          for (int u = 0; u < 4; ++u) {
            const svfloat32_t va0 = svld1_vnum_f32(pg32, a_ptr, 2 * u);
            const svfloat32_t vb0 = svld1_vnum_f32(pg32, b0_ptr, 2 * u + 0);
            const svfloat32_t vb1 = svld1_vnum_f32(pg32, b0_ptr, 2 * u + 1);
            const svfloat32_t vb2 = svld1_vnum_f32(pg32, b1_ptr, 2 * u + 0);
            const svfloat32_t vb3 = svld1_vnum_f32(pg32, b1_ptr, 2 * u + 1);

            svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
            svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
            svmopa_za32_f32_m(2, pg32, pg32, va0, vb2);
            svmopa_za32_f32_m(3, pg32, pg32, va0, vb3);
          }
          a_ptr += 4 * m_step;
          b0_ptr += 4 * n_step;
          b1_ptr += 4 * n_step;
        }
        for (; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_vnum_f32(pg32, a_ptr, 0);
          const svfloat32_t vb0 = svld1_vnum_f32(pg32, b0_ptr, 0);
          const svfloat32_t vb1 = svld1_vnum_f32(pg32, b0_ptr, 1);
          const svfloat32_t vb2 = svld1_vnum_f32(pg32, b1_ptr, 0);
          const svfloat32_t vb3 = svld1_vnum_f32(pg32, b1_ptr, 1);
          a_ptr += m_step;
          b0_ptr += n_step;
          b1_ptr += n_step;

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
          svmopa_za32_f32_m(2, pg32, pg32, va0, vb2);
          svmopa_za32_f32_m(3, pg32, pg32, va0, vb3);
        }

        const size_t n_rem = n - n_idx;
        const svbool_t p_out2 = svwhilelt_b32_u64(2 * vl, n_rem);
        const svbool_t p_out3 = svwhilelt_b32_u64(3 * vl, n_rem);

        for (size_t row = 0; row < m_rem; ++row) {
          svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
          svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
          svfloat32_t vf2 = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          svfloat32_t vf3 = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
          vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
          vf2 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf2, vmin), vmax);
          vf3 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf3, vmin), vmax);

          float* dst_ptr =
              (float*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
          svst1_vnum_f32(pg32, dst_ptr, 0, vf0);
          svst1_vnum_f32(pg32, dst_ptr, 1, vf1);
          svst1_vnum_f32(p_out2, dst_ptr, 2, vf2);
          svst1_vnum_f32(p_out3, dst_ptr, 3, vf3);
        }
        n_idx += n_step;
        continue;
      }

      const float* rhs_panel = rhs_packed + (n_idx / n_step) * rhs_panel_floats;
      const float* rhs_w = rhs_panel + n_step;
      const svfloat32_t vbias0 = svld1_vnum_f32(pg32, rhs_panel, 0);
      const svfloat32_t vbias1 = svld1_vnum_f32(pg32, rhs_panel, 1);
      svzero_za();
      svmopa_za32_f32_m(0, pg32, pg32, vones, vbias0);
      svmopa_za32_f32_m(1, pg32, pg32, vones, vbias1);
      svmopa_za32_f32_m(2, pg32, pg32, vones, vbias0);
      svmopa_za32_f32_m(3, pg32, pg32, vones, vbias1);

      const float* a_ptr = lhs_panel;
      const float* b_ptr = rhs_w;
      size_t k_idx = 0;
      for (; k_idx + 4 <= k_elements; k_idx += 4) {
#pragma clang loop unroll(full)
        for (int u = 0; u < 4; ++u) {
          const svfloat32_t va0 = svld1_vnum_f32(pg32, a_ptr, 2 * u + 0);
          const svfloat32_t va1 = svld1_vnum_f32(pg32, a_ptr, 2 * u + 1);
          const svfloat32_t vb0 = svld1_vnum_f32(pg32, b_ptr, 2 * u + 0);
          const svfloat32_t vb1 = svld1_vnum_f32(pg32, b_ptr, 2 * u + 1);

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
          svmopa_za32_f32_m(2, pg32, pg32, va1, vb0);
          svmopa_za32_f32_m(3, pg32, pg32, va1, vb1);
        }
        a_ptr += 4 * m_step;
        b_ptr += 4 * n_step;
      }
      for (; k_idx < k_elements; ++k_idx) {
        const svfloat32_t va0 = svld1_vnum_f32(pg32, a_ptr, 0);
        const svfloat32_t va1 = svld1_vnum_f32(pg32, a_ptr, 1);
        const svfloat32_t vb0 = svld1_vnum_f32(pg32, b_ptr, 0);
        const svfloat32_t vb1 = svld1_vnum_f32(pg32, b_ptr, 1);
        a_ptr += m_step;
        b_ptr += n_step;

        svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
        svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
        svmopa_za32_f32_m(2, pg32, pg32, va1, vb0);
        svmopa_za32_f32_m(3, pg32, pg32, va1, vb1);
      }

      const size_t n_rem = n - n_idx;
      const svbool_t p_out0 = svwhilelt_b32_u64(0, n_rem);
      const svbool_t p_out1 = svwhilelt_b32_u64(vl, n_rem);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
        svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
        svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
        vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
        vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);

        float* dst_ptr =
            (float*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
        svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
        svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        for (size_t row = 0; row < rows1; ++row) {
          svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
          vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);

          float* dst_ptr =
              (float*)((uintptr_t)dst + (m_idx + vl + row) * dst_stride_row) +
              n_idx;
          svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
          svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
        }
      }
    }
  }
}

void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  (void)dst_stride_col;
  xnn_pf32_gemm_2vlx2vl_sme_mopa_impl(
      m, n, k / sizeof(float), (const float*)lhs_packed,
      (const float*)rhs_packed, dst, dst_stride_row, minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#else
#include <assert.h>

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_mr() {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_nr() {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
