// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_mr(void) {
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
}

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_nr(void) {
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
}

void xnn_pf32_igemm_minmax_ukernel_32x32__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    const struct xnn_f32_minmax_params* minmax_params) {
  const size_t kai_kr = 1;
  const size_t k = ks * round_up(kc, kai_kr);

  kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
      mr, nc, k, lhs_packed, rhs_packed, dst, dst_stride_row * sizeof(float),
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pf32_igemm_2vlx2vl_sme_mopa_impl(size_t m, size_t n,
                                                           size_t k_elements,
                                                           const float*
                                                               lhs_packed,
                                                           const float*
                                                               rhs_packed,
                                                           float* dst,
                                                           size_t
                                                               dst_stride_elems,
                                                           float clamp_min,
                                                           float clamp_max) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t n_step = 2 * vl;
  const size_t lhs_panel_floats = m_step * k_elements;
  const size_t rhs_panel_floats = n_step * (1 + k_elements);
  const svbool_t pg32 = svptrue_b32();
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
        svzero_za();

        for (size_t k_idx = 0; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 =
              svld1_vnum_f32(pg32, lhs_panel + k_idx * m_step, 0);
          const float* b_step0 = rhs_w0 + k_idx * n_step;
          const float* b_step1 = rhs_w1 + k_idx * n_step;
          const svfloat32_t vb0 = svld1_vnum_f32(pg32, b_step0, 0);
          const svfloat32_t vb1 = svld1_vnum_f32(pg32, b_step0, 1);
          const svfloat32_t vb2 = svld1_vnum_f32(pg32, b_step1, 0);
          const svfloat32_t vb3 = svld1_vnum_f32(pg32, b_step1, 1);

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
          svmopa_za32_f32_m(2, pg32, pg32, va0, vb2);
          svmopa_za32_f32_m(3, pg32, pg32, va0, vb3);
        }

        const svfloat32_t vbias0 = svld1_vnum_f32(pg32, rhs_panel0, 0);
        const svfloat32_t vbias1 = svld1_vnum_f32(pg32, rhs_panel0, 1);
        const svfloat32_t vbias2 = svld1_vnum_f32(pg32, rhs_panel1, 0);
        const svfloat32_t vbias3 = svld1_vnum_f32(pg32, rhs_panel1, 1);
        const size_t n_rem = n - n_idx;
        const svbool_t p_out2 = svwhilelt_b32_u64(2 * vl, n_rem);
        const svbool_t p_out3 = svwhilelt_b32_u64(3 * vl, n_rem);

        for (size_t row = 0; row < m_rem; ++row) {
          svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
          svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
          svfloat32_t vf2 = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          svfloat32_t vf3 = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          vf0 = svadd_f32_m(pg32, vf0, vbias0);
          vf1 = svadd_f32_m(pg32, vf1, vbias1);
          vf2 = svadd_f32_m(pg32, vf2, vbias2);
          vf3 = svadd_f32_m(pg32, vf3, vbias3);
          vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
          vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
          vf2 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf2, vmin), vmax);
          vf3 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf3, vmin), vmax);

          float* dst_ptr = dst + (m_idx + row) * dst_stride_elems + n_idx;
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
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_elements; ++k_idx) {
        const float* a_step = lhs_panel + k_idx * m_step;
        const float* b_step = rhs_w + k_idx * n_step;
        const svfloat32_t va0 = svld1_vnum_f32(pg32, a_step, 0);
        const svfloat32_t va1 = svld1_vnum_f32(pg32, a_step, 1);
        const svfloat32_t vb0 = svld1_vnum_f32(pg32, b_step, 0);
        const svfloat32_t vb1 = svld1_vnum_f32(pg32, b_step, 1);

        svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
        svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
        svmopa_za32_f32_m(2, pg32, pg32, va1, vb0);
        svmopa_za32_f32_m(3, pg32, pg32, va1, vb1);
      }

      const svfloat32_t vbias0 = svld1_vnum_f32(pg32, rhs_panel, 0);
      const svfloat32_t vbias1 = svld1_vnum_f32(pg32, rhs_panel, 1);
      const size_t n_rem = n - n_idx;
      const svbool_t p_out0 = svwhilelt_b32_u64(0, n_rem);
      const svbool_t p_out1 = svwhilelt_b32_u64(vl, n_rem);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
        svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
        svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
        vf0 = svadd_f32_m(pg32, vf0, vbias0);
        vf1 = svadd_f32_m(pg32, vf1, vbias1);
        vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
        vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);

        float* dst_ptr = dst + (m_idx + row) * dst_stride_elems + n_idx;
        svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
        svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        for (size_t row = 0; row < rows1; ++row) {
          svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          vf0 = svadd_f32_m(pg32, vf0, vbias0);
          vf1 = svadd_f32_m(pg32, vf1, vbias1);
          vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
          vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);

          float* dst_ptr = dst + (m_idx + vl + row) * dst_stride_elems + n_idx;
          svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
          svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
        }
      }
    }
  }
}

void xnn_pf32_igemm_minmax_ukernel_32x32__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    const struct xnn_f32_minmax_params* minmax_params) {
  xnn_pf32_igemm_2vlx2vl_sme_mopa_impl(
      mr, nc, ks * kc, (const float*)lhs_packed, (const float*)rhs_packed, dst,
      dst_stride_row, minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_mr(void) { return 0; }
size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_nr(void) { return 0; }
void xnn_pf32_igemm_minmax_ukernel_32x32__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
