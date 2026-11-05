// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2_get_mr(void) {
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
}

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2_get_nr(void) {
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
}

void xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    const struct xnn_f32_minmax_params* minmax_params) {
  const size_t kai_kr = 1;
  const size_t k = ks * round_up(kc, kai_kr);

  kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
      mr, nc, k, lhs_packed, rhs_packed, dst, dst_stride_row * sizeof(float),
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_pf32_igemm_2vlx2vl_sme2_mopa_impl(size_t m, size_t n, size_t k_elements,
                                      const float* lhs_packed,
                                      const float* rhs_packed, float* dst,
                                      size_t dst_stride_elems, float clamp_min,
                                      float clamp_max) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t n_step = 2 * vl;
  const size_t lhs_panel_floats = m_step * k_elements;
  const size_t rhs_panel_floats = n_step * (1 + k_elements);
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc32 = svptrue_c32();
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
        const float* a_ptr = lhs_panel;
        const float* b0_ptr = rhs_panel0 + n_step;
        const float* b1_ptr = rhs_panel1 + n_step;

        svzero_za();
        svmopa_za32_f32_m(0, pg32, pg32, vones,
                          svld1_vnum_f32(pg32, rhs_panel0, 0));
        svmopa_za32_f32_m(1, pg32, pg32, vones,
                          svld1_vnum_f32(pg32, rhs_panel0, 1));
        svmopa_za32_f32_m(2, pg32, pg32, vones,
                          svld1_vnum_f32(pg32, rhs_panel1, 0));
        svmopa_za32_f32_m(3, pg32, pg32, vones,
                          svld1_vnum_f32(pg32, rhs_panel1, 1));

        size_t k_idx = 0;
        for (; k_idx + 4 <= k_elements; k_idx += 4) {
          const svfloat32x4_t va_01 = svld1_vnum_f32_x4(pc32, a_ptr, 0);
          const svfloat32x4_t va_23 = svld1_vnum_f32_x4(pc32, a_ptr, 4);
          const svfloat32x4_t vb0_01 = svld1_vnum_f32_x4(pc32, b0_ptr, 0);
          const svfloat32x4_t vb0_23 = svld1_vnum_f32_x4(pc32, b0_ptr, 4);
          const svfloat32x4_t vb1_01 = svld1_vnum_f32_x4(pc32, b1_ptr, 0);
          const svfloat32x4_t vb1_23 = svld1_vnum_f32_x4(pc32, b1_ptr, 4);

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_01, 0),
                            svget4_f32(vb0_01, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_01, 0),
                            svget4_f32(vb0_01, 1));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_01, 0),
                            svget4_f32(vb1_01, 0));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_01, 0),
                            svget4_f32(vb1_01, 1));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_01, 2),
                            svget4_f32(vb0_01, 2));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_01, 2),
                            svget4_f32(vb0_01, 3));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_01, 2),
                            svget4_f32(vb1_01, 2));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_01, 2),
                            svget4_f32(vb1_01, 3));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_23, 0),
                            svget4_f32(vb0_23, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_23, 0),
                            svget4_f32(vb0_23, 1));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_23, 0),
                            svget4_f32(vb1_23, 0));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_23, 0),
                            svget4_f32(vb1_23, 1));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_23, 2),
                            svget4_f32(vb0_23, 2));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_23, 2),
                            svget4_f32(vb0_23, 3));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_23, 2),
                            svget4_f32(vb1_23, 2));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_23, 2),
                            svget4_f32(vb1_23, 3));

          a_ptr += 4 * m_step;
          b0_ptr += 4 * n_step;
          b1_ptr += 4 * n_step;
        }
        for (; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_vnum_f32(pg32, a_ptr, 0);
          const svfloat32x2_t vb0 = svld1_f32_x2(pc32, b0_ptr);
          const svfloat32x2_t vb1 = svld1_f32_x2(pc32, b1_ptr);
          a_ptr += m_step;
          b0_ptr += n_step;
          b1_ptr += n_step;

          svmopa_za32_f32_m(0, pg32, pg32, va0, svget2_f32(vb0, 0));
          svmopa_za32_f32_m(1, pg32, pg32, va0, svget2_f32(vb0, 1));
          svmopa_za32_f32_m(2, pg32, pg32, va0, svget2_f32(vb1, 0));
          svmopa_za32_f32_m(3, pg32, pg32, va0, svget2_f32(vb1, 1));
        }

        const svcount_t p_out4 = svwhilelt_c32_u64(n_idx, n, 4);
        float* dst_row = dst + m_idx * dst_stride_elems + n_idx;
        for (size_t row = 0; row < m_rem; ++row) {
          const svfloat32x4_t vza = svreinterpret_f32_u8_x4(
              svread_hor_za8_u8_vg4(0, (uint32_t)(row * 4)));
          const svfloat32x4_t vout = svclamp_single_f32_x4(vza, vmin, vmax);
          svst1_f32_x4(p_out4, dst_row, vout);
          dst_row += dst_stride_elems;
        }
        n_idx += n_step;
        continue;
      }

      const float* rhs_panel = rhs_packed + (n_idx / n_step) * rhs_panel_floats;
      const svfloat32_t vbias0 = svld1_vnum_f32(pg32, rhs_panel, 0);
      const svfloat32_t vbias1 = svld1_vnum_f32(pg32, rhs_panel, 1);
      svzero_za();
      svmopa_za32_f32_m(0, pg32, pg32, vones, vbias0);
      svmopa_za32_f32_m(1, pg32, pg32, vones, vbias1);
      if (m_rem > vl) {
        svmopa_za32_f32_m(2, pg32, pg32, vones, vbias0);
        svmopa_za32_f32_m(3, pg32, pg32, vones, vbias1);
      }

      const float* a_ptr = lhs_panel;
      const float* b_ptr = rhs_panel + n_step;
      size_t k_idx = 0;
      if (m_rem <= vl) {
        for (; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_vnum_f32(pg32, a_ptr, 0);
          const svfloat32x2_t vb = svld1_f32_x2(pc32, b_ptr);
          a_ptr += m_step;
          b_ptr += n_step;
          svmopa_za32_f32_m(0, pg32, pg32, va0, svget2_f32(vb, 0));
          svmopa_za32_f32_m(1, pg32, pg32, va0, svget2_f32(vb, 1));
        }
      } else {
        if (k_elements >= 4) {
          svfloat32x4_t va_x4_0 = svld1_f32_x4(pc32, a_ptr);
          svfloat32x4_t vb_x4_0 = svld1_f32_x4(pc32, b_ptr);
          svfloat32x4_t va_x4_1 = svld1_f32_x4(pc32, a_ptr + 2 * m_step);
          svfloat32x4_t vb_x4_1 = svld1_f32_x4(pc32, b_ptr + 2 * n_step);
          a_ptr += 4 * m_step;
          b_ptr += 4 * n_step;

          for (; k_idx + 8 <= k_elements; k_idx += 4) {
            svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_0, 0),
                              svget4_f32(vb_x4_0, 0));
            svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_0, 0),
                              svget4_f32(vb_x4_0, 1));
            svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_0, 1),
                              svget4_f32(vb_x4_0, 0));
            svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_0, 1),
                              svget4_f32(vb_x4_0, 1));
            svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_0, 2),
                              svget4_f32(vb_x4_0, 2));
            svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_0, 2),
                              svget4_f32(vb_x4_0, 3));
            svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_0, 3),
                              svget4_f32(vb_x4_0, 2));
            svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_0, 3),
                              svget4_f32(vb_x4_0, 3));

            va_x4_0 = svld1_f32_x4(pc32, a_ptr);
            vb_x4_0 = svld1_f32_x4(pc32, b_ptr);

            svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_1, 0),
                              svget4_f32(vb_x4_1, 0));
            svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_1, 0),
                              svget4_f32(vb_x4_1, 1));
            svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_1, 1),
                              svget4_f32(vb_x4_1, 0));
            svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_1, 1),
                              svget4_f32(vb_x4_1, 1));
            svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_1, 2),
                              svget4_f32(vb_x4_1, 2));
            svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_1, 2),
                              svget4_f32(vb_x4_1, 3));
            svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_1, 3),
                              svget4_f32(vb_x4_1, 2));
            svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_1, 3),
                              svget4_f32(vb_x4_1, 3));

            va_x4_1 = svld1_f32_x4(pc32, a_ptr + 2 * m_step);
            vb_x4_1 = svld1_f32_x4(pc32, b_ptr + 2 * n_step);
            a_ptr += 4 * m_step;
            b_ptr += 4 * n_step;
          }

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_0, 0),
                            svget4_f32(vb_x4_0, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_0, 0),
                            svget4_f32(vb_x4_0, 1));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_0, 1),
                            svget4_f32(vb_x4_0, 0));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_0, 1),
                            svget4_f32(vb_x4_0, 1));
          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_0, 2),
                            svget4_f32(vb_x4_0, 2));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_0, 2),
                            svget4_f32(vb_x4_0, 3));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_0, 3),
                            svget4_f32(vb_x4_0, 2));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_0, 3),
                            svget4_f32(vb_x4_0, 3));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_1, 0),
                            svget4_f32(vb_x4_1, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_1, 0),
                            svget4_f32(vb_x4_1, 1));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_1, 1),
                            svget4_f32(vb_x4_1, 0));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_1, 1),
                            svget4_f32(vb_x4_1, 1));
          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va_x4_1, 2),
                            svget4_f32(vb_x4_1, 2));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va_x4_1, 2),
                            svget4_f32(vb_x4_1, 3));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va_x4_1, 3),
                            svget4_f32(vb_x4_1, 2));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va_x4_1, 3),
                            svget4_f32(vb_x4_1, 3));
          k_idx += 4;
        }
        for (; k_idx < k_elements; ++k_idx) {
          const svfloat32x2_t va = svld1_f32_x2(pc32, a_ptr);
          const svfloat32x2_t vb = svld1_f32_x2(pc32, b_ptr);
          a_ptr += m_step;
          b_ptr += n_step;

          svmopa_za32_f32_m(0, pg32, pg32, svget2_f32(va, 0),
                            svget2_f32(vb, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget2_f32(va, 0),
                            svget2_f32(vb, 1));
          svmopa_za32_f32_m(2, pg32, pg32, svget2_f32(va, 1),
                            svget2_f32(vb, 0));
          svmopa_za32_f32_m(3, pg32, pg32, svget2_f32(va, 1),
                            svget2_f32(vb, 1));
        }
      }

      const svcount_t p_out2 = svwhilelt_c32_u64(n_idx, n, 2);
      const size_t rows0 = m_rem < vl ? m_rem : vl;
      const size_t paired_rows = m_rem > vl ? (m_rem - vl) : 0;
      float* dst_top = dst + m_idx * dst_stride_elems + n_idx;
      float* dst_bot = dst_top + vl * dst_stride_elems;

      size_t row = 0;
      for (; row < paired_rows; ++row) {
        const svfloat32x4_t vza = svreinterpret_f32_u8_x4(
            svread_hor_za8_u8_vg4(0, (uint32_t)(row * 4)));
        const svfloat32x4_t vout = svclamp_single_f32_x4(vza, vmin, vmax);
        svst1_f32_x2(p_out2, dst_top,
                     svcreate2_f32(svget4_f32(vout, 0), svget4_f32(vout, 1)));
        svst1_f32_x2(p_out2, dst_bot,
                     svcreate2_f32(svget4_f32(vout, 2), svget4_f32(vout, 3)));
        dst_top += dst_stride_elems;
        dst_bot += dst_stride_elems;
      }
      for (; row < rows0; ++row) {
        const svfloat32x2_t vza = svreinterpret_f32_u8_x2(
            svread_hor_za8_u8_vg2(0, (uint32_t)(row * 4)));
        const svfloat32x2_t vout = svclamp_single_f32_x2(vza, vmin, vmax);
        svst1_f32_x2(p_out2, dst_top, vout);
        dst_top += dst_stride_elems;
      }
    }
  }
}

void xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    const struct xnn_f32_minmax_params* minmax_params) {
  xnn_pf32_igemm_2vlx2vl_sme2_mopa_impl(
      mr, nc, ks * kc, (const float*)lhs_packed, (const float*)rhs_packed, dst,
      dst_stride_row, minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2_get_mr(void) { return 0; }
size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2_get_nr(void) { return 0; }
void xnn_pf32_igemm_minmax_ukernel_32x32__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
