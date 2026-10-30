// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__sme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__sme2_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pf16_gemm_2vlx2vl_sme_mopa_impl(size_t m, size_t n,
                                                          size_t k_elements,
                                                          const float16_t*
                                                              lhs_packed,
                                                          const float16_t*
                                                              rhs_packed,
                                                          void* dst,
                                                          size_t dst_stride_row,
                                                          uint16_t min_bits,
                                                          uint16_t max_bits) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t n_step = 2 * vl;
  const size_t k_rounded = (k_elements + 1) & ~(size_t)1;
  const size_t k_pairs = k_rounded / 2;
  const size_t lhs_panel_elems = m_step * k_rounded;
  const size_t rhs_panel_elems = n_step * (1 + k_rounded);
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const svfloat16_t vzero16 = svdup_n_f16((float16_t)0.0f);
  const svfloat16_t vones16 = svdup_n_f16((float16_t)1.0f);
  const svfloat16_t vmin = svreinterpret_f16_u16(svdup_n_u16(min_bits));
  const svfloat16_t vmax = svreinterpret_f16_u16(svdup_n_u16(max_bits));

  for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
    const size_t m_rem = m - m_idx < m_step ? m - m_idx : m_step;
    const float16_t* lhs_panel =
        lhs_packed + (m_idx / m_step) * lhs_panel_elems;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
      if (m_rem <= vl && n_idx + n_step < n) {
        const float16_t* rhs_panel0 =
            rhs_packed + (n_idx / n_step) * rhs_panel_elems;
        const float16_t* rhs_panel1 = rhs_panel0 + rhs_panel_elems;
        const svfloat16_t vbias_a = svld1_f16(pg16, rhs_panel0);
        const svfloat16_t vbias_b = svld1_f16(pg16, rhs_panel1);
        const svfloat16_t vbias0 = svzip1_f16(vbias_a, vzero16);
        const svfloat16_t vbias1 = svzip2_f16(vbias_a, vzero16);
        const svfloat16_t vbias2 = svzip1_f16(vbias_b, vzero16);
        const svfloat16_t vbias3 = svzip2_f16(vbias_b, vzero16);

        svzero_za();
        svmopa_za32_f16_m(0, pg16, pg16, vones16, vbias0);
        svmopa_za32_f16_m(1, pg16, pg16, vones16, vbias1);
        svmopa_za32_f16_m(2, pg16, pg16, vones16, vbias2);
        svmopa_za32_f16_m(3, pg16, pg16, vones16, vbias3);

        const float16_t* a_ptr = lhs_panel;
        const float16_t* b0_ptr = rhs_panel0 + n_step;
        const float16_t* b1_ptr = rhs_panel1 + n_step;
        size_t kp = 0;
        for (; kp + 4 <= k_pairs; kp += 4) {
#pragma clang loop unroll(full)
          for (int u = 0; u < 4; ++u) {
            const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 2 * u);
            const svfloat16_t vb0 = svld1_vnum_f16(pg16, b0_ptr, 2 * u + 0);
            const svfloat16_t vb1 = svld1_vnum_f16(pg16, b0_ptr, 2 * u + 1);
            const svfloat16_t vb2 = svld1_vnum_f16(pg16, b1_ptr, 2 * u + 0);
            const svfloat16_t vb3 = svld1_vnum_f16(pg16, b1_ptr, 2 * u + 1);

            svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
            svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
            svmopa_za32_f16_m(2, pg16, pg16, va0, vb2);
            svmopa_za32_f16_m(3, pg16, pg16, va0, vb3);
          }
          a_ptr += 8 * m_step;
          b0_ptr += 8 * n_step;
          b1_ptr += 8 * n_step;
        }
        for (; kp < k_pairs; ++kp) {
          const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
          const svfloat16_t vb0 = svld1_vnum_f16(pg16, b0_ptr, 0);
          const svfloat16_t vb1 = svld1_vnum_f16(pg16, b0_ptr, 1);
          const svfloat16_t vb2 = svld1_vnum_f16(pg16, b1_ptr, 0);
          const svfloat16_t vb3 = svld1_vnum_f16(pg16, b1_ptr, 1);
          a_ptr += 2 * m_step;
          b0_ptr += 2 * n_step;
          b1_ptr += 2 * n_step;

          svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
          svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
          svmopa_za32_f16_m(2, pg16, pg16, va0, vb2);
          svmopa_za32_f16_m(3, pg16, pg16, va0, vb3);
        }

        const svbool_t p_out1 = svwhilelt_b16_u64(n_step, n - n_idx);
        for (size_t row = 0; row < m_rem; ++row) {
          const svfloat32_t vf0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
          const svfloat32_t vf1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
          const svfloat32_t vf2 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          const svfloat32_t vf3 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          svfloat16_t vout0 = svuzp1_f16(svcvt_f16_f32_x(pg32, vf0),
                                         svcvt_f16_f32_x(pg32, vf1));
          svfloat16_t vout1 = svuzp1_f16(svcvt_f16_f32_x(pg32, vf2),
                                         svcvt_f16_f32_x(pg32, vf3));
          vout0 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout0, vmin), vmax);
          vout1 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout1, vmin), vmax);
          float16_t* dst_ptr =
              (float16_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) +
              n_idx;
          svst1_f16(pg16, dst_ptr, vout0);
          svst1_f16(p_out1, dst_ptr + n_step, vout1);
        }
        n_idx += n_step;
        continue;
      }

      const float16_t* rhs_panel =
          rhs_packed + (n_idx / n_step) * rhs_panel_elems;
      const svfloat16_t vbias = svld1_f16(pg16, rhs_panel);
      const svfloat16_t vbias0 = svzip1_f16(vbias, vzero16);
      const svfloat16_t vbias1 = svzip2_f16(vbias, vzero16);

      svzero_za();
      svmopa_za32_f16_m(0, pg16, pg16, vones16, vbias0);
      svmopa_za32_f16_m(1, pg16, pg16, vones16, vbias1);
      svmopa_za32_f16_m(2, pg16, pg16, vones16, vbias0);
      svmopa_za32_f16_m(3, pg16, pg16, vones16, vbias1);

      const float16_t* a_ptr = lhs_panel;
      const float16_t* b_ptr = rhs_panel + n_step;
      size_t kp = 0;
      for (; kp + 4 <= k_pairs; kp += 4) {
#pragma clang loop unroll(full)
        for (int u = 0; u < 4; ++u) {
          const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 2 * u + 0);
          const svfloat16_t va1 = svld1_vnum_f16(pg16, a_ptr, 2 * u + 1);
          const svfloat16_t vb0 = svld1_vnum_f16(pg16, b_ptr, 2 * u + 0);
          const svfloat16_t vb1 = svld1_vnum_f16(pg16, b_ptr, 2 * u + 1);

          svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
          svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
          svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
          svmopa_za32_f16_m(3, pg16, pg16, va1, vb1);
        }
        a_ptr += 8 * m_step;
        b_ptr += 8 * n_step;
      }
      for (; kp < k_pairs; ++kp) {
        const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
        const svfloat16_t va1 = svld1_vnum_f16(pg16, a_ptr, 1);
        const svfloat16_t vb0 = svld1_vnum_f16(pg16, b_ptr, 0);
        const svfloat16_t vb1 = svld1_vnum_f16(pg16, b_ptr, 1);
        a_ptr += 2 * m_step;
        b_ptr += 2 * n_step;

        svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
        svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
        svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
        svmopa_za32_f16_m(3, pg16, pg16, va1, vb1);
      }

      const svbool_t p_out = svwhilelt_b16_u64(0, n - n_idx);
      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
        const svfloat32_t vf0 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
        const svfloat32_t vf1 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
        svfloat16_t vout =
            svuzp1_f16(svcvt_f16_f32_x(pg32, vf0), svcvt_f16_f32_x(pg32, vf1));
        vout = svmin_f16_m(pg16, svmax_f16_m(pg16, vout, vmin), vmax);
        float16_t* dst_ptr =
            (float16_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) +
            n_idx;
        svst1_f16(p_out, dst_ptr, vout);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        for (size_t row = 0; row < rows1; ++row) {
          const svfloat32_t vf0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          const svfloat32_t vf1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          svfloat16_t vout = svuzp1_f16(svcvt_f16_f32_x(pg32, vf0),
                                        svcvt_f16_f32_x(pg32, vf1));
          vout = svmin_f16_m(pg16, svmax_f16_m(pg16, vout, vmin), vmax);
          float16_t* dst_ptr =
              (float16_t*)((uintptr_t)dst +
                           (m_idx + vl + row) * dst_stride_row) +
              n_idx;
          svst1_f16(p_out, dst_ptr, vout);
        }
      }
    }
  }
}

void xnn_pf16_gemm_minmax_ukernel_32x32c2__sme2(
    size_t mr, size_t nc, size_t kc, const void* a, const void* w_fp16, void* c,
    size_t cm_stride, size_t cn_stride,
    const struct xnn_f16_minmax_params* params) {
  (void)cn_stride;
  if XNN_UNLIKELY (mr == 0 || nc == 0) {
    return;
  }
  uint16_t min_bits;
  uint16_t max_bits;
  memcpy(&min_bits, &params->scalar.min, sizeof(uint16_t));
  memcpy(&max_bits, &params->scalar.max, sizeof(uint16_t));
  xnn_pf16_gemm_2vlx2vl_sme_mopa_impl(
      mr, nc, kc / sizeof(xnn_float16), (const float16_t*)a,
      (const float16_t*)w_fp16, c, cm_stride, min_bits, max_bits);
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
