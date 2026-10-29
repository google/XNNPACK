// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme_get_mr() {
  return kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa();
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa();
}

// Wraps the `kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa(
      m, n, k / sizeof(xnn_float16), lhs_packed, rhs_packed, dst,
      dst_stride_row, /*dst_stride_col=*/sizeof(xnn_float16),
      xnn_float16_to_float(minmax_params->scalar.min),
      xnn_float16_to_float(minmax_params->scalar.max));
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>
#include <string.h>

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_pf16_gemm_2vlx2vl_sme_mopa_impl(size_t m, size_t n, size_t k_elements,
                                    const float16_t* lhs_packed,
                                    const float16_t* rhs_packed, void* dst,
                                    size_t dst_stride_row, uint16_t min_bits,
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
        const svfloat16_t vbias0 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)rhs_panel0));
        const svfloat16_t vbias1 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)(rhs_panel0 + vl)));
        const svfloat16_t vbias2 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)rhs_panel1));
        const svfloat16_t vbias3 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)(rhs_panel1 + vl)));

        svzero_za();
        svmopa_za32_f16_m(0, pg16, pg16, vones16, vbias0);
        svmopa_za32_f16_m(1, pg16, pg16, vones16, vbias1);
        svmopa_za32_f16_m(2, pg16, pg16, vones16, vbias2);
        svmopa_za32_f16_m(3, pg16, pg16, vones16, vbias3);

        const float16_t* a_ptr = lhs_panel;
        const float16_t* b0_ptr = rhs_panel0 + n_step;
        const float16_t* b1_ptr = rhs_panel1 + n_step;
        size_t kp = 0;
        if (k_pairs >= 4) {
          svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
          svfloat16_t va1 = svld1_vnum_f16(pg16, a_ptr, 2);
          svfloat16_t va2 = svld1_vnum_f16(pg16, a_ptr, 4);
          svfloat16_t va3 = svld1_vnum_f16(pg16, a_ptr, 6);
          a_ptr += 8 * m_step;

          svfloat16_t vb00 = svld1_vnum_f16(pg16, b0_ptr, 0);
          svfloat16_t vb01 = svld1_vnum_f16(pg16, b0_ptr, 1);
          svfloat16_t vb10 = svld1_vnum_f16(pg16, b1_ptr, 0);
          svfloat16_t vb11 = svld1_vnum_f16(pg16, b1_ptr, 1);
          svfloat16_t vb20 = svld1_vnum_f16(pg16, b0_ptr, 2);
          svfloat16_t vb21 = svld1_vnum_f16(pg16, b0_ptr, 3);
          svfloat16_t vb30 = svld1_vnum_f16(pg16, b1_ptr, 2);
          svfloat16_t vb31 = svld1_vnum_f16(pg16, b1_ptr, 3);

          for (; kp + 8 <= k_pairs; kp += 4) {
            svmopa_za32_f16_m(0, pg16, pg16, va0, vb00);
            svmopa_za32_f16_m(1, pg16, pg16, va0, vb01);
            vb00 = svld1_vnum_f16(pg16, b0_ptr, 4);
            svmopa_za32_f16_m(2, pg16, pg16, va0, vb10);
            vb01 = svld1_vnum_f16(pg16, b0_ptr, 5);
            svmopa_za32_f16_m(3, pg16, pg16, va0, vb11);
            vb10 = svld1_vnum_f16(pg16, b1_ptr, 4);
            vb11 = svld1_vnum_f16(pg16, b1_ptr, 5);
            va0 = svld1_vnum_f16(pg16, a_ptr, 0);

            svmopa_za32_f16_m(0, pg16, pg16, va1, vb20);
            svmopa_za32_f16_m(1, pg16, pg16, va1, vb21);
            vb20 = svld1_vnum_f16(pg16, b0_ptr, 6);
            svmopa_za32_f16_m(2, pg16, pg16, va1, vb30);
            vb21 = svld1_vnum_f16(pg16, b0_ptr, 7);
            svmopa_za32_f16_m(3, pg16, pg16, va1, vb31);
            vb30 = svld1_vnum_f16(pg16, b1_ptr, 6);
            vb31 = svld1_vnum_f16(pg16, b1_ptr, 7);
            va1 = svld1_vnum_f16(pg16, a_ptr, 2);
            b0_ptr += 8 * n_step;
            b1_ptr += 8 * n_step;

            svmopa_za32_f16_m(0, pg16, pg16, va2, vb00);
            svmopa_za32_f16_m(1, pg16, pg16, va2, vb01);
            vb00 = svld1_vnum_f16(pg16, b0_ptr, 0);
            svmopa_za32_f16_m(2, pg16, pg16, va2, vb10);
            vb01 = svld1_vnum_f16(pg16, b0_ptr, 1);
            svmopa_za32_f16_m(3, pg16, pg16, va2, vb11);
            vb10 = svld1_vnum_f16(pg16, b1_ptr, 0);
            vb11 = svld1_vnum_f16(pg16, b1_ptr, 1);
            va2 = svld1_vnum_f16(pg16, a_ptr, 4);

            svmopa_za32_f16_m(0, pg16, pg16, va3, vb20);
            svmopa_za32_f16_m(1, pg16, pg16, va3, vb21);
            vb20 = svld1_vnum_f16(pg16, b0_ptr, 2);
            svmopa_za32_f16_m(2, pg16, pg16, va3, vb30);
            vb21 = svld1_vnum_f16(pg16, b0_ptr, 3);
            svmopa_za32_f16_m(3, pg16, pg16, va3, vb31);
            vb30 = svld1_vnum_f16(pg16, b1_ptr, 2);
            vb31 = svld1_vnum_f16(pg16, b1_ptr, 3);
            va3 = svld1_vnum_f16(pg16, a_ptr, 6);
            a_ptr += 8 * m_step;
          }

          svmopa_za32_f16_m(0, pg16, pg16, va0, vb00);
          svmopa_za32_f16_m(1, pg16, pg16, va0, vb01);
          vb00 = svld1_vnum_f16(pg16, b0_ptr, 4);
          svmopa_za32_f16_m(2, pg16, pg16, va0, vb10);
          vb01 = svld1_vnum_f16(pg16, b0_ptr, 5);
          svmopa_za32_f16_m(3, pg16, pg16, va0, vb11);
          vb10 = svld1_vnum_f16(pg16, b1_ptr, 4);
          vb11 = svld1_vnum_f16(pg16, b1_ptr, 5);

          svmopa_za32_f16_m(0, pg16, pg16, va1, vb20);
          svmopa_za32_f16_m(1, pg16, pg16, va1, vb21);
          vb20 = svld1_vnum_f16(pg16, b0_ptr, 6);
          svmopa_za32_f16_m(2, pg16, pg16, va1, vb30);
          vb21 = svld1_vnum_f16(pg16, b0_ptr, 7);
          svmopa_za32_f16_m(3, pg16, pg16, va1, vb31);
          vb30 = svld1_vnum_f16(pg16, b1_ptr, 6);
          vb31 = svld1_vnum_f16(pg16, b1_ptr, 7);
          b0_ptr += 8 * n_step;
          b1_ptr += 8 * n_step;

          svmopa_za32_f16_m(0, pg16, pg16, va2, vb00);
          svmopa_za32_f16_m(1, pg16, pg16, va2, vb01);
          svmopa_za32_f16_m(2, pg16, pg16, va2, vb10);
          svmopa_za32_f16_m(3, pg16, pg16, va2, vb11);

          svmopa_za32_f16_m(0, pg16, pg16, va3, vb20);
          svmopa_za32_f16_m(1, pg16, pg16, va3, vb21);
          svmopa_za32_f16_m(2, pg16, pg16, va3, vb30);
          svmopa_za32_f16_m(3, pg16, pg16, va3, vb31);
          kp += 4;
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
        uint8_t* dst_ptr =
            (uint8_t*)dst + m_idx * dst_stride_row + n_idx * sizeof(float16_t);
        size_t row = 0;
        for (; row + 2 <= m_rem; row += 2) {
          const svfloat32_t r0_0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row + 0);
          const svfloat32_t r0_1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row + 0);
          const svfloat32_t r0_2 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row + 0);
          const svfloat32_t r0_3 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row + 0);
          const svfloat32_t r1_0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row + 1);
          const svfloat32_t r1_1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row + 1);
          const svfloat32_t r1_2 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row + 1);
          const svfloat32_t r1_3 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row + 1);

          svfloat16_t vout0_0 = svuzp1_f16(svcvt_f16_f32_x(pg32, r0_0),
                                           svcvt_f16_f32_x(pg32, r0_1));
          svfloat16_t vout0_1 = svuzp1_f16(svcvt_f16_f32_x(pg32, r0_2),
                                           svcvt_f16_f32_x(pg32, r0_3));
          svfloat16_t vout1_0 = svuzp1_f16(svcvt_f16_f32_x(pg32, r1_0),
                                           svcvt_f16_f32_x(pg32, r1_1));
          svfloat16_t vout1_1 = svuzp1_f16(svcvt_f16_f32_x(pg32, r1_2),
                                           svcvt_f16_f32_x(pg32, r1_3));
          vout0_0 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout0_0, vmin), vmax);
          vout0_1 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout0_1, vmin), vmax);
          vout1_0 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout1_0, vmin), vmax);
          vout1_1 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout1_1, vmin), vmax);

          svst1_f16(pg16, (float16_t*)dst_ptr, vout0_0);
          svst1_f16(p_out1, (float16_t*)dst_ptr + n_step, vout0_1);
          dst_ptr += dst_stride_row;
          svst1_f16(pg16, (float16_t*)dst_ptr, vout1_0);
          svst1_f16(p_out1, (float16_t*)dst_ptr + n_step, vout1_1);
          dst_ptr += dst_stride_row;
        }
        for (; row < m_rem; ++row) {
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
          svst1_f16(pg16, (float16_t*)dst_ptr, vout0);
          svst1_f16(p_out1, (float16_t*)dst_ptr + n_step, vout1);
          dst_ptr += dst_stride_row;
        }
        n_idx += n_step;
        continue;
      }

      const float16_t* rhs_panel =
          rhs_packed + (n_idx / n_step) * rhs_panel_elems;
      const svfloat16_t vbias0 =
          svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)rhs_panel));
      const svfloat16_t vbias1 = svreinterpret_f16_u32(
          svld1uh_u32(pg32, (const uint16_t*)(rhs_panel + vl)));

      svzero_za();
      svmopa_za32_f16_m(0, pg16, pg16, vones16, vbias0);
      svmopa_za32_f16_m(1, pg16, pg16, vones16, vbias1);

      const float16_t* a_ptr = lhs_panel;
      const float16_t* b_ptr = rhs_panel + n_step;
      size_t kp = 0;

      if (m_rem <= vl) {
        for (; kp + 4 <= k_pairs; kp += 4) {
          const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
          const svfloat16_t va2 = svld1_vnum_f16(pg16, a_ptr, 2);
          const svfloat16_t va4 = svld1_vnum_f16(pg16, a_ptr, 4);
          const svfloat16_t va6 = svld1_vnum_f16(pg16, a_ptr, 6);
          const svfloat16_t vb0 = svld1_vnum_f16(pg16, b_ptr, 0);
          const svfloat16_t vb1 = svld1_vnum_f16(pg16, b_ptr, 1);
          const svfloat16_t vb2 = svld1_vnum_f16(pg16, b_ptr, 2);
          const svfloat16_t vb3 = svld1_vnum_f16(pg16, b_ptr, 3);
          const svfloat16_t vb4 = svld1_vnum_f16(pg16, b_ptr, 4);
          const svfloat16_t vb5 = svld1_vnum_f16(pg16, b_ptr, 5);
          const svfloat16_t vb6 = svld1_vnum_f16(pg16, b_ptr, 6);
          const svfloat16_t vb7 = svld1_vnum_f16(pg16, b_ptr, 7);
          a_ptr += 8 * m_step;
          b_ptr += 8 * n_step;

          svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
          svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
          svmopa_za32_f16_m(0, pg16, pg16, va2, vb2);
          svmopa_za32_f16_m(1, pg16, pg16, va2, vb3);
          svmopa_za32_f16_m(0, pg16, pg16, va4, vb4);
          svmopa_za32_f16_m(1, pg16, pg16, va4, vb5);
          svmopa_za32_f16_m(0, pg16, pg16, va6, vb6);
          svmopa_za32_f16_m(1, pg16, pg16, va6, vb7);
        }
        for (; kp < k_pairs; ++kp) {
          const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
          const svfloat16_t vb0 = svld1_vnum_f16(pg16, b_ptr, 0);
          const svfloat16_t vb1 = svld1_vnum_f16(pg16, b_ptr, 1);
          a_ptr += 2 * m_step;
          b_ptr += 2 * n_step;
          svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
          svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
        }
      } else {
        svmopa_za32_f16_m(2, pg16, pg16, vones16, vbias0);
        svmopa_za32_f16_m(3, pg16, pg16, vones16, vbias1);
        if (k_pairs >= 4) {
          svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
          svfloat16_t va1 = svld1_vnum_f16(pg16, a_ptr, 1);
          svfloat16_t va2 = svld1_vnum_f16(pg16, a_ptr, 2);
          svfloat16_t va3 = svld1_vnum_f16(pg16, a_ptr, 3);
          svfloat16_t va4 = svld1_vnum_f16(pg16, a_ptr, 4);
          svfloat16_t va5 = svld1_vnum_f16(pg16, a_ptr, 5);
          svfloat16_t va6 = svld1_vnum_f16(pg16, a_ptr, 6);
          svfloat16_t va7 = svld1_vnum_f16(pg16, a_ptr, 7);
          a_ptr += 8 * m_step;

          svfloat16_t vb0 = svld1_vnum_f16(pg16, b_ptr, 0);
          svfloat16_t vb1 = svld1_vnum_f16(pg16, b_ptr, 1);
          svfloat16_t vb2 = svld1_vnum_f16(pg16, b_ptr, 2);
          svfloat16_t vb3 = svld1_vnum_f16(pg16, b_ptr, 3);
          svfloat16_t vb4 = svld1_vnum_f16(pg16, b_ptr, 4);
          svfloat16_t vb5 = svld1_vnum_f16(pg16, b_ptr, 5);
          svfloat16_t vb6 = svld1_vnum_f16(pg16, b_ptr, 6);
          svfloat16_t vb7 = svld1_vnum_f16(pg16, b_ptr, 7);
          b_ptr += 8 * n_step;

          for (; kp + 8 <= k_pairs; kp += 4) {
            svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
            svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
            va0 = svld1_vnum_f16(pg16, a_ptr, 0);
            svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
            vb0 = svld1_vnum_f16(pg16, b_ptr, 0);
            svmopa_za32_f16_m(3, pg16, pg16, va1, vb1);
            va1 = svld1_vnum_f16(pg16, a_ptr, 1);
            vb1 = svld1_vnum_f16(pg16, b_ptr, 1);

            svmopa_za32_f16_m(0, pg16, pg16, va2, vb2);
            svmopa_za32_f16_m(1, pg16, pg16, va2, vb3);
            va2 = svld1_vnum_f16(pg16, a_ptr, 2);
            svmopa_za32_f16_m(2, pg16, pg16, va3, vb2);
            vb2 = svld1_vnum_f16(pg16, b_ptr, 2);
            svmopa_za32_f16_m(3, pg16, pg16, va3, vb3);
            va3 = svld1_vnum_f16(pg16, a_ptr, 3);
            vb3 = svld1_vnum_f16(pg16, b_ptr, 3);

            svmopa_za32_f16_m(0, pg16, pg16, va4, vb4);
            svmopa_za32_f16_m(1, pg16, pg16, va4, vb5);
            va4 = svld1_vnum_f16(pg16, a_ptr, 4);
            svmopa_za32_f16_m(2, pg16, pg16, va5, vb4);
            vb4 = svld1_vnum_f16(pg16, b_ptr, 4);
            svmopa_za32_f16_m(3, pg16, pg16, va5, vb5);
            va5 = svld1_vnum_f16(pg16, a_ptr, 5);
            vb5 = svld1_vnum_f16(pg16, b_ptr, 5);

            svmopa_za32_f16_m(0, pg16, pg16, va6, vb6);
            svmopa_za32_f16_m(1, pg16, pg16, va6, vb7);
            va6 = svld1_vnum_f16(pg16, a_ptr, 6);
            svmopa_za32_f16_m(2, pg16, pg16, va7, vb6);
            vb6 = svld1_vnum_f16(pg16, b_ptr, 6);
            svmopa_za32_f16_m(3, pg16, pg16, va7, vb7);
            va7 = svld1_vnum_f16(pg16, a_ptr, 7);
            vb7 = svld1_vnum_f16(pg16, b_ptr, 7);

            a_ptr += 8 * m_step;
            b_ptr += 8 * n_step;
          }

          svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
          svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
          svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
          svmopa_za32_f16_m(3, pg16, pg16, va1, vb1);

          svmopa_za32_f16_m(0, pg16, pg16, va2, vb2);
          svmopa_za32_f16_m(1, pg16, pg16, va2, vb3);
          svmopa_za32_f16_m(2, pg16, pg16, va3, vb2);
          svmopa_za32_f16_m(3, pg16, pg16, va3, vb3);

          svmopa_za32_f16_m(0, pg16, pg16, va4, vb4);
          svmopa_za32_f16_m(1, pg16, pg16, va4, vb5);
          svmopa_za32_f16_m(2, pg16, pg16, va5, vb4);
          svmopa_za32_f16_m(3, pg16, pg16, va5, vb5);

          svmopa_za32_f16_m(0, pg16, pg16, va6, vb6);
          svmopa_za32_f16_m(1, pg16, pg16, va6, vb7);
          svmopa_za32_f16_m(2, pg16, pg16, va7, vb6);
          svmopa_za32_f16_m(3, pg16, pg16, va7, vb7);
          kp += 4;
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
      }

      const svbool_t p_out = svwhilelt_b16_u64(0, n - n_idx);
      const size_t rows0 = m_rem < vl ? m_rem : vl;
      uint8_t* dst_ptr =
          (uint8_t*)dst + m_idx * dst_stride_row + n_idx * sizeof(float16_t);
      size_t row = 0;
      for (; row + 4 <= rows0; row += 4) {
        const svfloat32_t r0_0 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row + 0);
        const svfloat32_t r0_1 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row + 0);
        const svfloat32_t r1_0 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row + 1);
        const svfloat32_t r1_1 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row + 1);
        const svfloat32_t r2_0 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row + 2);
        const svfloat32_t r2_1 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row + 2);
        const svfloat32_t r3_0 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row + 3);
        const svfloat32_t r3_1 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row + 3);

        svfloat16_t vout0 = svuzp1_f16(svcvt_f16_f32_x(pg32, r0_0),
                                       svcvt_f16_f32_x(pg32, r0_1));
        svfloat16_t vout1 = svuzp1_f16(svcvt_f16_f32_x(pg32, r1_0),
                                       svcvt_f16_f32_x(pg32, r1_1));
        svfloat16_t vout2 = svuzp1_f16(svcvt_f16_f32_x(pg32, r2_0),
                                       svcvt_f16_f32_x(pg32, r2_1));
        svfloat16_t vout3 = svuzp1_f16(svcvt_f16_f32_x(pg32, r3_0),
                                       svcvt_f16_f32_x(pg32, r3_1));
        vout0 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout0, vmin), vmax);
        vout1 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout1, vmin), vmax);
        vout2 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout2, vmin), vmax);
        vout3 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout3, vmin), vmax);

        svst1_f16(p_out, (float16_t*)dst_ptr, vout0);
        dst_ptr += dst_stride_row;
        svst1_f16(p_out, (float16_t*)dst_ptr, vout1);
        dst_ptr += dst_stride_row;
        svst1_f16(p_out, (float16_t*)dst_ptr, vout2);
        dst_ptr += dst_stride_row;
        svst1_f16(p_out, (float16_t*)dst_ptr, vout3);
        dst_ptr += dst_stride_row;
      }
      for (; row < rows0; ++row) {
        const svfloat32_t vf0 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
        const svfloat32_t vf1 =
            svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
        svfloat16_t vout =
            svuzp1_f16(svcvt_f16_f32_x(pg32, vf0), svcvt_f16_f32_x(pg32, vf1));
        vout = svmin_f16_m(pg16, svmax_f16_m(pg16, vout, vmin), vmax);
        svst1_f16(p_out, (float16_t*)dst_ptr, vout);
        dst_ptr += dst_stride_row;
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        dst_ptr = (uint8_t*)dst + (m_idx + vl) * dst_stride_row +
                  n_idx * sizeof(float16_t);
        row = 0;
        for (; row + 4 <= rows1; row += 4) {
          const svfloat32_t r0_0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row + 0);
          const svfloat32_t r0_1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row + 0);
          const svfloat32_t r1_0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row + 1);
          const svfloat32_t r1_1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row + 1);
          const svfloat32_t r2_0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row + 2);
          const svfloat32_t r2_1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row + 2);
          const svfloat32_t r3_0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row + 3);
          const svfloat32_t r3_1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row + 3);

          svfloat16_t vout0 = svuzp1_f16(svcvt_f16_f32_x(pg32, r0_0),
                                         svcvt_f16_f32_x(pg32, r0_1));
          svfloat16_t vout1 = svuzp1_f16(svcvt_f16_f32_x(pg32, r1_0),
                                         svcvt_f16_f32_x(pg32, r1_1));
          svfloat16_t vout2 = svuzp1_f16(svcvt_f16_f32_x(pg32, r2_0),
                                         svcvt_f16_f32_x(pg32, r2_1));
          svfloat16_t vout3 = svuzp1_f16(svcvt_f16_f32_x(pg32, r3_0),
                                         svcvt_f16_f32_x(pg32, r3_1));
          vout0 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout0, vmin), vmax);
          vout1 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout1, vmin), vmax);
          vout2 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout2, vmin), vmax);
          vout3 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout3, vmin), vmax);

          svst1_f16(p_out, (float16_t*)dst_ptr, vout0);
          dst_ptr += dst_stride_row;
          svst1_f16(p_out, (float16_t*)dst_ptr, vout1);
          dst_ptr += dst_stride_row;
          svst1_f16(p_out, (float16_t*)dst_ptr, vout2);
          dst_ptr += dst_stride_row;
          svst1_f16(p_out, (float16_t*)dst_ptr, vout3);
          dst_ptr += dst_stride_row;
        }
        for (; row < rows1; ++row) {
          const svfloat32_t vf0 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          const svfloat32_t vf1 =
              svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          svfloat16_t vout = svuzp1_f16(svcvt_f16_f32_x(pg32, vf0),
                                        svcvt_f16_f32_x(pg32, vf1));
          vout = svmin_f16_m(pg16, svmax_f16_m(pg16, vout, vmin), vmax);
          svst1_f16(p_out, (float16_t*)dst_ptr, vout);
          dst_ptr += dst_stride_row;
        }
      }
    }
  }
}

void xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  (void)dst_stride_col;
  uint16_t min_bits;
  uint16_t max_bits;
  memcpy(&min_bits, &minmax_params->scalar.min, sizeof(uint16_t));
  memcpy(&max_bits, &minmax_params->scalar.max, sizeof(uint16_t));
  xnn_pf16_gemm_2vlx2vl_sme_mopa_impl(
      m, n, k / sizeof(xnn_float16), (const float16_t*)lhs_packed,
      (const float16_t*)rhs_packed, dst, dst_stride_row, min_bits, max_bits);
}
#else
#include <assert.h>

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme_get_mr() {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme_get_nr() {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
