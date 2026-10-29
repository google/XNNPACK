// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdio.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_mr(void) {
  return kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
}

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_nr(void) {
  return kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
}

void xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, xnn_float16* restrict c, size_t cm_stride,
    const struct xnn_f16_minmax_params* params) {
  const size_t kai_kr = 2;
  const size_t k = ks * round_up(kc, kai_kr);

  kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(
      mr, nc, k, packed_lhs, w, c, cm_stride * sizeof(xnn_float16),
      sizeof(xnn_float16), xnn_float16_to_float(params->scalar.min),
      xnn_float16_to_float(params->scalar.max));
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/math.h"

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pf16_igemm_2vlx2vl_sme2_mopa_impl(size_t m, size_t n,
                                                            size_t k_rounded,
                                                            const float16_t*
                                                                lhs_packed,
                                                            const float16_t*
                                                                rhs_packed,
                                                            float16_t* dst,
                                                            size_t cm_stride,
                                                            uint16_t min_bits,
                                                            uint16_t max_bits) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t n_step = 2 * vl;
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

        const float16_t* rhs_w0 = rhs_panel0 + n_step;
        const float16_t* rhs_w1 = rhs_panel1 + n_step;
        for (size_t kp = 0; kp < k_pairs; ++kp) {
          const float16_t* a_step = lhs_panel + kp * (2 * m_step);
          const float16_t* b_step0 = rhs_w0 + kp * (2 * n_step);
          const float16_t* b_step1 = rhs_w1 + kp * (2 * n_step);
          const svfloat16_t va0 = svld1_vnum_f16(pg16, a_step, 0);
          const svfloat16_t vb0 = svld1_vnum_f16(pg16, b_step0, 0);
          const svfloat16_t vb1 = svld1_vnum_f16(pg16, b_step0, 1);
          const svfloat16_t vb2 = svld1_vnum_f16(pg16, b_step1, 0);
          const svfloat16_t vb3 = svld1_vnum_f16(pg16, b_step1, 1);

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
          float16_t* dst_ptr = dst + (m_idx + row) * cm_stride + n_idx;
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

      const float16_t* rhs_w = rhs_panel + n_step;
      for (size_t kp = 0; kp < k_pairs; ++kp) {
        const float16_t* a_step = lhs_panel + kp * (2 * m_step);
        const float16_t* b_step = rhs_w + kp * (2 * n_step);
        const svfloat16_t va0 = svld1_vnum_f16(pg16, a_step, 0);
        const svfloat16_t va1 = svld1_vnum_f16(pg16, a_step, 1);
        const svfloat16_t vb0 = svld1_vnum_f16(pg16, b_step, 0);
        const svfloat16_t vb1 = svld1_vnum_f16(pg16, b_step, 1);

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
        svst1_f16(p_out, dst + (m_idx + row) * cm_stride + n_idx, vout);
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
          svst1_f16(p_out, dst + (m_idx + vl + row) * cm_stride + n_idx, vout);
        }
      }
    }
  }
}

void xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, xnn_float16* restrict c, size_t cm_stride,
    const struct xnn_f16_minmax_params* params) {
  uint16_t min_bits;
  uint16_t max_bits;
  memcpy(&min_bits, &params->scalar.min, sizeof(uint16_t));
  memcpy(&max_bits, &params->scalar.max, sizeof(uint16_t));
  const size_t k_rounded = ks * round_up_po2(kc, 2);
  xnn_pf16_igemm_2vlx2vl_sme2_mopa_impl(
      mr, nc, k_rounded, (const float16_t*)packed_lhs, (const float16_t*)w,
      (float16_t*)c, cm_stride, min_bits, max_bits);
}
#else
#include <assert.h>

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_mr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_nr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, xnn_float16* restrict c, size_t cm_stride,
    const struct xnn_f16_minmax_params* params) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
