// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"

#if XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <assert.h>
#include <stdint.h>

#include "src/xnnpack/igemm.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pqd8_f32_qc8w_igemm_minmax_ukernel_32x32c4__sme_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pqd8_f32_qc8w_igemm_minmax_ukernel_32x32c4__sme_get_nr(void) {
  return 2 * svcntsw();
}

static XNN_INLINE svfloat32_t xnn_pqd8_sme_dequant_vec(
    svint32_t vacc, svfloat32_t vscale, svfloat32_t vbias, svfloat32_t vmin,
    svfloat32_t vmax) __arm_streaming __arm_preserves("za") {
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vo =
      svmla_f32_x(pg32, vbias, svcvt_f32_s32_x(pg32, vacc), vscale);
  return svmin_f32_x(pg32, svmax_f32_x(pg32, vo, vmin), vmax);
}

__arm_locally_streaming __arm_new("za") static __attribute__((noinline)) void
xnn_pqd8_f32_qc8w_igemm_2vlx2vl_sme_impl(
    size_t mr, size_t nc, size_t k_total, const int8_t* restrict lhs_packed,
    const void* restrict rhs_packed, float* restrict c, size_t cm_stride,
    int32_t input_zero_point, float input_inv_scale, float output_min,
    float output_max) {
  const size_t vl = svcntw();
  const size_t m_panel = 2 * vl;
  const size_t n_panel = 2 * vl;
  const size_t lhs_panel_bytes = m_panel * k_total;
  const size_t rhs_panel_bytes =
      n_panel * (sizeof(int32_t) + k_total * sizeof(int8_t) + 2 * sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vinput_scale = svdup_n_f32(input_inv_scale);
  const svfloat32_t vmin = svdup_n_f32(output_min);
  const svfloat32_t vmax = svdup_n_f32(output_max);

  for (size_t m_idx = 0; m_idx < mr; m_idx += m_panel) {
    const size_t m_rem = mr - m_idx < m_panel ? mr - m_idx : m_panel;
    const int8_t* lhs_p = lhs_packed + (m_idx / m_panel) * lhs_panel_bytes;

    for (size_t n_idx = 0; n_idx < nc; n_idx += n_panel) {
      if (m_rem <= vl && n_idx + n_panel < nc) {
        const size_t n_rem4 =
            nc - n_idx < 2 * n_panel ? nc - n_idx : 2 * n_panel;
        const size_t n_rem_p1 = n_rem4 - n_panel;
        const uint8_t* rhs_p0 =
            (const uint8_t*)rhs_packed + (n_idx / n_panel) * rhs_panel_bytes;
        const uint8_t* rhs_p1 = rhs_p0 + rhs_panel_bytes;
        const int32_t* ksum_ptr0 = (const int32_t*)rhs_p0;
        const int32_t* ksum_ptr1 = (const int32_t*)rhs_p1;
        const int8_t* b0_ptr =
            (const int8_t*)(rhs_p0 + n_panel * sizeof(int32_t));
        const int8_t* b1_ptr =
            (const int8_t*)(rhs_p1 + n_panel * sizeof(int32_t));
        const float* scale_ptr0 = (const float*)(b0_ptr + n_panel * k_total);
        const float* scale_ptr1 = (const float*)(b1_ptr + n_panel * k_total);
        const float* bias_ptr0 = scale_ptr0 + n_panel;
        const float* bias_ptr1 = scale_ptr1 + n_panel;

        const svbool_t p_n2 = svwhilelt_b32_u64(0, n_rem_p1);
        const svbool_t p_n3 = svwhilelt_b32_u64(vl, n_rem_p1);
        const svint32_t vinit0 = svmul_n_s32_x(
            pg32, svld1_vnum_s32(pg32, ksum_ptr0, 0), input_zero_point);
        const svint32_t vinit1 = svmul_n_s32_x(
            pg32, svld1_vnum_s32(pg32, ksum_ptr0, 1), input_zero_point);
        const svint32_t vinit2 =
            svmul_n_s32_z(p_n2, svld1_s32(p_n2, ksum_ptr1), input_zero_point);
        const svint32_t vinit3 = svmul_n_s32_z(
            p_n3, svld1_s32(p_n3, ksum_ptr1 + vl), input_zero_point);
        svzero_za();
        svaddha_za32_s32_m(0, pg32, pg32, vinit0);
        svaddha_za32_s32_m(1, pg32, pg32, vinit1);
        svaddha_za32_s32_m(2, pg32, pg32, vinit2);
        svaddha_za32_s32_m(3, pg32, pg32, vinit3);

        const int8_t* a_ptr = lhs_p;
        size_t k_idx = 0;
        if (k_total >= 16) {
          svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          svint8_t vb00 = svld1_vnum_s8(pg8, b0_ptr, 0);
          svint8_t vb01 = svld1_vnum_s8(pg8, b0_ptr, 1);
          svint8_t vb02 = svld1_vnum_s8(pg8, b1_ptr, 0);
          svint8_t vb03 = svld1_vnum_s8(pg8, b1_ptr, 1);
          svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 2);
          svint8_t vb10 = svld1_vnum_s8(pg8, b0_ptr, 2);
          svint8_t vb11 = svld1_vnum_s8(pg8, b0_ptr, 3);
          svint8_t vb12 = svld1_vnum_s8(pg8, b1_ptr, 2);
          svint8_t vb13 = svld1_vnum_s8(pg8, b1_ptr, 3);
          svint8_t va2 = svld1_vnum_s8(pg8, a_ptr, 4);
          svint8_t vb20 = svld1_vnum_s8(pg8, b0_ptr, 4);
          svint8_t vb21 = svld1_vnum_s8(pg8, b0_ptr, 5);
          svint8_t vb22 = svld1_vnum_s8(pg8, b1_ptr, 4);
          svint8_t vb23 = svld1_vnum_s8(pg8, b1_ptr, 5);
          svint8_t va3 = svld1_vnum_s8(pg8, a_ptr, 6);
          svint8_t vb30 = svld1_vnum_s8(pg8, b0_ptr, 6);
          svint8_t vb31 = svld1_vnum_s8(pg8, b0_ptr, 7);
          svint8_t vb32 = svld1_vnum_s8(pg8, b1_ptr, 6);
          svint8_t vb33 = svld1_vnum_s8(pg8, b1_ptr, 7);
          a_ptr += 16 * m_panel;
          b0_ptr += 16 * n_panel;
          b1_ptr += 16 * n_panel;
          k_idx = 16;

          for (; k_idx + 16 <= k_total; k_idx += 16) {
            svmopa_za32_s8_m(0, pg8, pg8, va0, vb00);
            vb00 = svld1_vnum_s8(pg8, b0_ptr, 0);
            svmopa_za32_s8_m(1, pg8, pg8, va0, vb01);
            vb01 = svld1_vnum_s8(pg8, b0_ptr, 1);
            svmopa_za32_s8_m(2, pg8, pg8, va0, vb02);
            vb02 = svld1_vnum_s8(pg8, b1_ptr, 0);
            svmopa_za32_s8_m(3, pg8, pg8, va0, vb03);
            va0 = svld1_vnum_s8(pg8, a_ptr, 0);
            vb03 = svld1_vnum_s8(pg8, b1_ptr, 1);

            svmopa_za32_s8_m(0, pg8, pg8, va1, vb10);
            vb10 = svld1_vnum_s8(pg8, b0_ptr, 2);
            svmopa_za32_s8_m(1, pg8, pg8, va1, vb11);
            vb11 = svld1_vnum_s8(pg8, b0_ptr, 3);
            svmopa_za32_s8_m(2, pg8, pg8, va1, vb12);
            vb12 = svld1_vnum_s8(pg8, b1_ptr, 2);
            svmopa_za32_s8_m(3, pg8, pg8, va1, vb13);
            va1 = svld1_vnum_s8(pg8, a_ptr, 2);
            vb13 = svld1_vnum_s8(pg8, b1_ptr, 3);

            svmopa_za32_s8_m(0, pg8, pg8, va2, vb20);
            vb20 = svld1_vnum_s8(pg8, b0_ptr, 4);
            svmopa_za32_s8_m(1, pg8, pg8, va2, vb21);
            vb21 = svld1_vnum_s8(pg8, b0_ptr, 5);
            svmopa_za32_s8_m(2, pg8, pg8, va2, vb22);
            vb22 = svld1_vnum_s8(pg8, b1_ptr, 4);
            svmopa_za32_s8_m(3, pg8, pg8, va2, vb23);
            va2 = svld1_vnum_s8(pg8, a_ptr, 4);
            vb23 = svld1_vnum_s8(pg8, b1_ptr, 5);

            svmopa_za32_s8_m(0, pg8, pg8, va3, vb30);
            vb30 = svld1_vnum_s8(pg8, b0_ptr, 6);
            svmopa_za32_s8_m(1, pg8, pg8, va3, vb31);
            vb31 = svld1_vnum_s8(pg8, b0_ptr, 7);
            svmopa_za32_s8_m(2, pg8, pg8, va3, vb32);
            vb32 = svld1_vnum_s8(pg8, b1_ptr, 6);
            svmopa_za32_s8_m(3, pg8, pg8, va3, vb33);
            va3 = svld1_vnum_s8(pg8, a_ptr, 6);
            vb33 = svld1_vnum_s8(pg8, b1_ptr, 7);

            a_ptr += 16 * m_panel;
            b0_ptr += 16 * n_panel;
            b1_ptr += 16 * n_panel;
          }

          svmopa_za32_s8_m(0, pg8, pg8, va0, vb00);
          svmopa_za32_s8_m(1, pg8, pg8, va0, vb01);
          svmopa_za32_s8_m(2, pg8, pg8, va0, vb02);
          svmopa_za32_s8_m(3, pg8, pg8, va0, vb03);
          svmopa_za32_s8_m(0, pg8, pg8, va1, vb10);
          svmopa_za32_s8_m(1, pg8, pg8, va1, vb11);
          svmopa_za32_s8_m(2, pg8, pg8, va1, vb12);
          svmopa_za32_s8_m(3, pg8, pg8, va1, vb13);
          svmopa_za32_s8_m(0, pg8, pg8, va2, vb20);
          svmopa_za32_s8_m(1, pg8, pg8, va2, vb21);
          svmopa_za32_s8_m(2, pg8, pg8, va2, vb22);
          svmopa_za32_s8_m(3, pg8, pg8, va2, vb23);
          svmopa_za32_s8_m(0, pg8, pg8, va3, vb30);
          svmopa_za32_s8_m(1, pg8, pg8, va3, vb31);
          svmopa_za32_s8_m(2, pg8, pg8, va3, vb32);
          svmopa_za32_s8_m(3, pg8, pg8, va3, vb33);
        }
        for (; k_idx < k_total; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svld1_vnum_s8(pg8, b0_ptr, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svld1_vnum_s8(pg8, b0_ptr, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va0, svld1_vnum_s8(pg8, b1_ptr, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va0, svld1_vnum_s8(pg8, b1_ptr, 1));
          a_ptr += 4 * m_panel;
          b0_ptr += 4 * n_panel;
          b1_ptr += 4 * n_panel;
        }

        const svfloat32_t vscale0 = svmul_f32_x(
            pg32, svld1_vnum_f32(pg32, scale_ptr0, 0), vinput_scale);
        const svfloat32_t vscale1 = svmul_f32_x(
            pg32, svld1_vnum_f32(pg32, scale_ptr0, 1), vinput_scale);
        const svfloat32_t vscale2 =
            svmul_f32_z(p_n2, svld1_f32(p_n2, scale_ptr1), vinput_scale);
        const svfloat32_t vscale3 =
            svmul_f32_z(p_n3, svld1_f32(p_n3, scale_ptr1 + vl), vinput_scale);
        const svfloat32_t vbias0 = svld1_vnum_f32(pg32, bias_ptr0, 0);
        const svfloat32_t vbias1 = svld1_vnum_f32(pg32, bias_ptr0, 1);
        const svfloat32_t vbias2 = svld1_f32(p_n2, bias_ptr1);
        const svfloat32_t vbias3 = svld1_f32(p_n3, bias_ptr1 + vl);

        for (size_t row = 0; row < m_rem; ++row) {
          float* dst_ptr =
              (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
          const svfloat32_t vo0 = xnn_pqd8_sme_dequant_vec(
              svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row), vscale0,
              vbias0, vmin, vmax);
          const svfloat32_t vo1 = xnn_pqd8_sme_dequant_vec(
              svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row), vscale1,
              vbias1, vmin, vmax);
          const svfloat32_t vo2 = xnn_pqd8_sme_dequant_vec(
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row), vscale2,
              vbias2, vmin, vmax);
          const svfloat32_t vo3 = xnn_pqd8_sme_dequant_vec(
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row), vscale3,
              vbias3, vmin, vmax);
          svst1_vnum_f32(pg32, dst_ptr, 0, vo0);
          svst1_vnum_f32(pg32, dst_ptr, 1, vo1);
          svst1_f32(p_n2, dst_ptr + 2 * vl, vo2);
          svst1_f32(p_n3, dst_ptr + 3 * vl, vo3);
        }
        n_idx += n_panel;
        continue;
      }

      const size_t n_rem = nc - n_idx < n_panel ? nc - n_idx : n_panel;
      const uint8_t* rhs_p =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * rhs_panel_bytes;
      const int32_t* ksum_ptr = (const int32_t*)rhs_p;
      const int8_t* b_ptr = (const int8_t*)(rhs_p + n_panel * sizeof(int32_t));
      const float* scale_ptr = (const float*)(b_ptr + n_panel * k_total);
      const float* bias_ptr = scale_ptr + n_panel;

      const svbool_t p_n0 = svwhilelt_b32_u64(0, n_rem);
      const svbool_t p_n1 = svwhilelt_b32_u64(vl, n_rem);
      const svint32_t vinit0 =
          svmul_n_s32_z(p_n0, svld1_s32(p_n0, ksum_ptr), input_zero_point);
      const svint32_t vinit1 =
          svmul_n_s32_z(p_n1, svld1_s32(p_n1, ksum_ptr + vl), input_zero_point);
      svzero_za();
      svaddha_za32_s32_m(0, pg32, pg32, vinit0);
      svaddha_za32_s32_m(1, pg32, pg32, vinit1);

      const int8_t* a_ptr = lhs_p;
      size_t k_idx = 0;
      if (m_rem <= vl) {
        for (; k_idx + 16 <= k_total; k_idx += 16) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 2);
          const svint8_t va2 = svld1_vnum_s8(pg8, a_ptr, 4);
          const svint8_t va3 = svld1_vnum_s8(pg8, a_ptr, 6);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 1));
          svmopa_za32_s8_m(0, pg8, pg8, va1, svld1_vnum_s8(pg8, b_ptr, 2));
          svmopa_za32_s8_m(1, pg8, pg8, va1, svld1_vnum_s8(pg8, b_ptr, 3));
          svmopa_za32_s8_m(0, pg8, pg8, va2, svld1_vnum_s8(pg8, b_ptr, 4));
          svmopa_za32_s8_m(1, pg8, pg8, va2, svld1_vnum_s8(pg8, b_ptr, 5));
          svmopa_za32_s8_m(0, pg8, pg8, va3, svld1_vnum_s8(pg8, b_ptr, 6));
          svmopa_za32_s8_m(1, pg8, pg8, va3, svld1_vnum_s8(pg8, b_ptr, 7));
          a_ptr += 16 * m_panel;
          b_ptr += 16 * n_panel;
        }
        for (; k_idx < k_total; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 1));
          a_ptr += 4 * m_panel;
          b_ptr += 4 * n_panel;
        }
      } else {
        svaddha_za32_s32_m(2, pg32, pg32, vinit0);
        svaddha_za32_s32_m(3, pg32, pg32, vinit1);
        if (k_total >= 16) {
          svint8_t va00 = svld1_vnum_s8(pg8, a_ptr, 0);
          svint8_t va01 = svld1_vnum_s8(pg8, a_ptr, 1);
          svint8_t vb00 = svld1_vnum_s8(pg8, b_ptr, 0);
          svint8_t vb01 = svld1_vnum_s8(pg8, b_ptr, 1);
          svint8_t va10 = svld1_vnum_s8(pg8, a_ptr, 2);
          svint8_t va11 = svld1_vnum_s8(pg8, a_ptr, 3);
          svint8_t vb10 = svld1_vnum_s8(pg8, b_ptr, 2);
          svint8_t vb11 = svld1_vnum_s8(pg8, b_ptr, 3);
          svint8_t va20 = svld1_vnum_s8(pg8, a_ptr, 4);
          svint8_t va21 = svld1_vnum_s8(pg8, a_ptr, 5);
          svint8_t vb20 = svld1_vnum_s8(pg8, b_ptr, 4);
          svint8_t vb21 = svld1_vnum_s8(pg8, b_ptr, 5);
          svint8_t va30 = svld1_vnum_s8(pg8, a_ptr, 6);
          svint8_t va31 = svld1_vnum_s8(pg8, a_ptr, 7);
          svint8_t vb30 = svld1_vnum_s8(pg8, b_ptr, 6);
          svint8_t vb31 = svld1_vnum_s8(pg8, b_ptr, 7);
          a_ptr += 16 * m_panel;
          b_ptr += 16 * n_panel;
          k_idx = 16;

          for (; k_idx + 16 <= k_total; k_idx += 16) {
            // Each LHS/RHS vector feeds two tiles; reload it only after its
            // second use.
            svmopa_za32_s8_m(0, pg8, pg8, va00, vb00);
            svmopa_za32_s8_m(1, pg8, pg8, va00, vb01);
            va00 = svld1_vnum_s8(pg8, a_ptr, 0);
            svmopa_za32_s8_m(2, pg8, pg8, va01, vb00);
            vb00 = svld1_vnum_s8(pg8, b_ptr, 0);
            svmopa_za32_s8_m(3, pg8, pg8, va01, vb01);
            va01 = svld1_vnum_s8(pg8, a_ptr, 1);
            vb01 = svld1_vnum_s8(pg8, b_ptr, 1);

            svmopa_za32_s8_m(0, pg8, pg8, va10, vb10);
            svmopa_za32_s8_m(1, pg8, pg8, va10, vb11);
            va10 = svld1_vnum_s8(pg8, a_ptr, 2);
            svmopa_za32_s8_m(2, pg8, pg8, va11, vb10);
            vb10 = svld1_vnum_s8(pg8, b_ptr, 2);
            svmopa_za32_s8_m(3, pg8, pg8, va11, vb11);
            va11 = svld1_vnum_s8(pg8, a_ptr, 3);
            vb11 = svld1_vnum_s8(pg8, b_ptr, 3);

            svmopa_za32_s8_m(0, pg8, pg8, va20, vb20);
            svmopa_za32_s8_m(1, pg8, pg8, va20, vb21);
            va20 = svld1_vnum_s8(pg8, a_ptr, 4);
            svmopa_za32_s8_m(2, pg8, pg8, va21, vb20);
            vb20 = svld1_vnum_s8(pg8, b_ptr, 4);
            svmopa_za32_s8_m(3, pg8, pg8, va21, vb21);
            va21 = svld1_vnum_s8(pg8, a_ptr, 5);
            vb21 = svld1_vnum_s8(pg8, b_ptr, 5);

            svmopa_za32_s8_m(0, pg8, pg8, va30, vb30);
            svmopa_za32_s8_m(1, pg8, pg8, va30, vb31);
            va30 = svld1_vnum_s8(pg8, a_ptr, 6);
            svmopa_za32_s8_m(2, pg8, pg8, va31, vb30);
            vb30 = svld1_vnum_s8(pg8, b_ptr, 6);
            svmopa_za32_s8_m(3, pg8, pg8, va31, vb31);
            va31 = svld1_vnum_s8(pg8, a_ptr, 7);
            vb31 = svld1_vnum_s8(pg8, b_ptr, 7);

            a_ptr += 16 * m_panel;
            b_ptr += 16 * n_panel;
          }

          svmopa_za32_s8_m(0, pg8, pg8, va00, vb00);
          svmopa_za32_s8_m(1, pg8, pg8, va00, vb01);
          svmopa_za32_s8_m(2, pg8, pg8, va01, vb00);
          svmopa_za32_s8_m(3, pg8, pg8, va01, vb01);
          svmopa_za32_s8_m(0, pg8, pg8, va10, vb10);
          svmopa_za32_s8_m(1, pg8, pg8, va10, vb11);
          svmopa_za32_s8_m(2, pg8, pg8, va11, vb10);
          svmopa_za32_s8_m(3, pg8, pg8, va11, vb11);
          svmopa_za32_s8_m(0, pg8, pg8, va20, vb20);
          svmopa_za32_s8_m(1, pg8, pg8, va20, vb21);
          svmopa_za32_s8_m(2, pg8, pg8, va21, vb20);
          svmopa_za32_s8_m(3, pg8, pg8, va21, vb21);
          svmopa_za32_s8_m(0, pg8, pg8, va30, vb30);
          svmopa_za32_s8_m(1, pg8, pg8, va30, vb31);
          svmopa_za32_s8_m(2, pg8, pg8, va31, vb30);
          svmopa_za32_s8_m(3, pg8, pg8, va31, vb31);
        }
        for (; k_idx < k_total; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 1);
          const svint8_t vb0 = svld1_vnum_s8(pg8, b_ptr, 0);
          const svint8_t vb1 = svld1_vnum_s8(pg8, b_ptr, 1);
          svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
          svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
          svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
          svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
          a_ptr += 4 * m_panel;
          b_ptr += 4 * n_panel;
        }
      }

      const svfloat32_t vscale0 =
          svmul_f32_z(p_n0, svld1_f32(p_n0, scale_ptr), vinput_scale);
      const svfloat32_t vscale1 =
          svmul_f32_z(p_n1, svld1_f32(p_n1, scale_ptr + vl), vinput_scale);
      const svfloat32_t vbias0 = svld1_f32(p_n0, bias_ptr);
      const svfloat32_t vbias1 = svld1_f32(p_n1, bias_ptr + vl);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
        float* dst_ptr =
            (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
        const svfloat32_t vo0 = xnn_pqd8_sme_dequant_vec(
            svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row), vscale0, vbias0,
            vmin, vmax);
        const svfloat32_t vo1 = xnn_pqd8_sme_dequant_vec(
            svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row), vscale1, vbias1,
            vmin, vmax);
        svst1_f32(p_n0, dst_ptr, vo0);
        svst1_f32(p_n1, dst_ptr + vl, vo1);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        for (size_t row1 = 0; row1 < rows1; ++row1) {
          float* dst_ptr =
              (float*)((uintptr_t)c + (m_idx + vl + row1) * cm_stride) + n_idx;
          const svfloat32_t vo0 = xnn_pqd8_sme_dequant_vec(
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row1), vscale0,
              vbias0, vmin, vmax);
          const svfloat32_t vo1 = xnn_pqd8_sme_dequant_vec(
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row1), vscale1,
              vbias1, vmin, vmax);
          svst1_f32(p_n0, dst_ptr, vo0);
          svst1_f32(p_n1, dst_ptr + vl, vo1);
        }
      }
    }
  }
}

void xnn_pqd8_f32_qc8w_igemm_minmax_ukernel_32x32c4__sme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* restrict packed_lhs,
    const void* restrict w, float* restrict c, size_t cm_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) {
  assert(mr != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(packed_lhs != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const size_t k_total = ks * ((kc + 3) & ~(size_t)3);
  xnn_pqd8_f32_qc8w_igemm_2vlx2vl_sme_impl(
      mr, nc, k_total, (const int8_t*)packed_lhs, w, c, cm_stride,
      quantization_params->zero_point, quantization_params->inv_scale,
      params->scalar.min, params->scalar.max);
}
#endif  // XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
