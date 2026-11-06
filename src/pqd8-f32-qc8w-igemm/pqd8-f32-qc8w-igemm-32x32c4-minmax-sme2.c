// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <assert.h>
#include <stdint.h>

#include "src/xnnpack/igemm.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pqd8_f32_qc8w_igemm_minmax_ukernel_32x32c4__sme2_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pqd8_f32_qc8w_igemm_minmax_ukernel_32x32c4__sme2_get_nr(void) {
  return 2 * svcntsw();
}

static XNN_INLINE svfloat32x4_t xnn_pqd8_dequant_tile_4rows(
    svint32x4_t vacc, svfloat32_t vscale, svfloat32_t vbias, svfloat32_t vmin,
    svfloat32_t vmax) __arm_streaming __arm_preserves("za") {
  const svbool_t pg32 = svptrue_b32();
  const svfloat32x4_t vf = svcvt_f32_s32_x4(vacc);
  const svfloat32x4_t vout =
      svcreate4_f32(svmla_f32_x(pg32, vbias, svget4_f32(vf, 0), vscale),
                    svmla_f32_x(pg32, vbias, svget4_f32(vf, 1), vscale),
                    svmla_f32_x(pg32, vbias, svget4_f32(vf, 2), vscale),
                    svmla_f32_x(pg32, vbias, svget4_f32(vf, 3), vscale));
  return svclamp_single_f32_x4(vout, vmin, vmax);
}

static XNN_INLINE void xnn_pqd8_store_2vl_4rows(
    svint32x4_t za_rows0, svint32x4_t za_rows1, svfloat32_t vscale0,
    svfloat32_t vscale1, svfloat32_t vbias0, svfloat32_t vbias1,
    svfloat32_t vmin, svfloat32_t vmax, svcount_t pc32_out, float* dst_base,
    size_t cm_stride) __arm_streaming __arm_preserves("za") {
  const svfloat32x4_t vo0 =
      xnn_pqd8_dequant_tile_4rows(za_rows0, vscale0, vbias0, vmin, vmax);
  const svfloat32x4_t vo1 =
      xnn_pqd8_dequant_tile_4rows(za_rows1, vscale1, vbias1, vmin, vmax);

  svst1_f32_x2(pc32_out, dst_base,
               svcreate2_f32(svget4_f32(vo0, 0), svget4_f32(vo1, 0)));
  svst1_f32_x2(pc32_out, (float*)((uintptr_t)dst_base + cm_stride),
               svcreate2_f32(svget4_f32(vo0, 1), svget4_f32(vo1, 1)));
  svst1_f32_x2(pc32_out, (float*)((uintptr_t)dst_base + 2 * cm_stride),
               svcreate2_f32(svget4_f32(vo0, 2), svget4_f32(vo1, 2)));
  svst1_f32_x2(pc32_out, (float*)((uintptr_t)dst_base + 3 * cm_stride),
               svcreate2_f32(svget4_f32(vo0, 3), svget4_f32(vo1, 3)));
}

__arm_locally_streaming __arm_new("za") static __attribute__((noinline)) void
xnn_pqd8_f32_qc8w_igemm_2vlx2vl_sme2_impl(
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
  const svcount_t pc8 = svptrue_c8();
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
        for (; k_idx + 16 <= k_total; k_idx += 16) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 2);
          const svint8x4_t vb0a = svld1_vnum_s8_x4(pc8, b0_ptr, 0);
          const svint8x4_t vb1a = svld1_vnum_s8_x4(pc8, b1_ptr, 0);

          svmopa_za32_s8_m(0, pg8, pg8, va0, svget4_s8(vb0a, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svget4_s8(vb0a, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va0, svget4_s8(vb1a, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va0, svget4_s8(vb1a, 1));

          svmopa_za32_s8_m(0, pg8, pg8, va1, svget4_s8(vb0a, 2));
          svmopa_za32_s8_m(1, pg8, pg8, va1, svget4_s8(vb0a, 3));
          svmopa_za32_s8_m(2, pg8, pg8, va1, svget4_s8(vb1a, 2));
          svmopa_za32_s8_m(3, pg8, pg8, va1, svget4_s8(vb1a, 3));

          const svint8_t va2 = svld1_vnum_s8(pg8, a_ptr, 4);
          const svint8_t va3 = svld1_vnum_s8(pg8, a_ptr, 6);
          const svint8x4_t vb0b = svld1_vnum_s8_x4(pc8, b0_ptr, 4);
          const svint8x4_t vb1b = svld1_vnum_s8_x4(pc8, b1_ptr, 4);

          svmopa_za32_s8_m(0, pg8, pg8, va2, svget4_s8(vb0b, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va2, svget4_s8(vb0b, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va2, svget4_s8(vb1b, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va2, svget4_s8(vb1b, 1));

          svmopa_za32_s8_m(0, pg8, pg8, va3, svget4_s8(vb0b, 2));
          svmopa_za32_s8_m(1, pg8, pg8, va3, svget4_s8(vb0b, 3));
          svmopa_za32_s8_m(2, pg8, pg8, va3, svget4_s8(vb1b, 2));
          svmopa_za32_s8_m(3, pg8, pg8, va3, svget4_s8(vb1b, 3));

          a_ptr += 16 * m_panel;
          b0_ptr += 16 * n_panel;
          b1_ptr += 16 * n_panel;
        }
        for (; k_idx < k_total; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8x2_t vb0 = svld1_s8_x2(pc8, b0_ptr);
          const svint8x2_t vb1 = svld1_s8_x2(pc8, b1_ptr);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb0, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb0, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va0, svget2_s8(vb1, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va0, svget2_s8(vb1, 1));
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
        const svcount_t pc32_out4 = svwhilelt_c32_u64(0, n_rem4, 4);

        size_t row = 0;
        for (; row + 4 <= m_rem; row += 4) {
          const svfloat32x4_t vo0 = xnn_pqd8_dequant_tile_4rows(
              svread_hor_za32_s32_vg4(0, (uint32_t)row), vscale0, vbias0, vmin,
              vmax);
          const svfloat32x4_t vo1 = xnn_pqd8_dequant_tile_4rows(
              svread_hor_za32_s32_vg4(1, (uint32_t)row), vscale1, vbias1, vmin,
              vmax);
          const svfloat32x4_t vo2 = xnn_pqd8_dequant_tile_4rows(
              svread_hor_za32_s32_vg4(2, (uint32_t)row), vscale2, vbias2, vmin,
              vmax);
          const svfloat32x4_t vo3 = xnn_pqd8_dequant_tile_4rows(
              svread_hor_za32_s32_vg4(3, (uint32_t)row), vscale3, vbias3, vmin,
              vmax);
          float* dst_ptr =
              (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
          svst1_f32_x4(pc32_out4, dst_ptr,
                       svcreate4_f32(svget4_f32(vo0, 0), svget4_f32(vo1, 0),
                                     svget4_f32(vo2, 0), svget4_f32(vo3, 0)));
          svst1_f32_x4(pc32_out4, (float*)((uintptr_t)dst_ptr + cm_stride),
                       svcreate4_f32(svget4_f32(vo0, 1), svget4_f32(vo1, 1),
                                     svget4_f32(vo2, 1), svget4_f32(vo3, 1)));
          svst1_f32_x4(pc32_out4, (float*)((uintptr_t)dst_ptr + 2 * cm_stride),
                       svcreate4_f32(svget4_f32(vo0, 2), svget4_f32(vo1, 2),
                                     svget4_f32(vo2, 2), svget4_f32(vo3, 2)));
          svst1_f32_x4(pc32_out4, (float*)((uintptr_t)dst_ptr + 3 * cm_stride),
                       svcreate4_f32(svget4_f32(vo0, 3), svget4_f32(vo1, 3),
                                     svget4_f32(vo2, 3), svget4_f32(vo3, 3)));
        }
        for (; row < m_rem; ++row) {
          float* dst_ptr =
              (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
          const svint32_t vacc0 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
          const svint32_t vacc1 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row);
          const svint32_t vacc2 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row);
          const svint32_t vacc3 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row);
          const svfloat32x4_t vf =
              svcvt_f32_s32_x4(svcreate4_s32(vacc0, vacc1, vacc2, vacc3));
          const svfloat32x4_t vout = svclamp_single_f32_x4(
              svcreate4_f32(
                  svmla_f32_x(pg32, vbias0, svget4_f32(vf, 0), vscale0),
                  svmla_f32_x(pg32, vbias1, svget4_f32(vf, 1), vscale1),
                  svmla_f32_x(pg32, vbias2, svget4_f32(vf, 2), vscale2),
                  svmla_f32_x(pg32, vbias3, svget4_f32(vf, 3), vscale3)),
              vmin, vmax);
          svst1_f32_x4(pc32_out4, dst_ptr, vout);
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
          const svint8x4_t vb01 = svld1_vnum_s8_x4(pc8, b_ptr, 0);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svget4_s8(vb01, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svget4_s8(vb01, 1));
          svmopa_za32_s8_m(0, pg8, pg8, va1, svget4_s8(vb01, 2));
          svmopa_za32_s8_m(1, pg8, pg8, va1, svget4_s8(vb01, 3));

          const svint8_t va2 = svld1_vnum_s8(pg8, a_ptr, 4);
          const svint8_t va3 = svld1_vnum_s8(pg8, a_ptr, 6);
          const svint8x4_t vb23 = svld1_vnum_s8_x4(pc8, b_ptr, 4);
          svmopa_za32_s8_m(0, pg8, pg8, va2, svget4_s8(vb23, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va2, svget4_s8(vb23, 1));
          svmopa_za32_s8_m(0, pg8, pg8, va3, svget4_s8(vb23, 2));
          svmopa_za32_s8_m(1, pg8, pg8, va3, svget4_s8(vb23, 3));

          a_ptr += 16 * m_panel;
          b_ptr += 16 * n_panel;
        }
        for (; k_idx < k_total; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8x2_t vb = svld1_s8_x2(pc8, b_ptr);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb, 1));
          a_ptr += 4 * m_panel;
          b_ptr += 4 * n_panel;
        }
      } else {
        svaddha_za32_s32_m(2, pg32, pg32, vinit0);
        svaddha_za32_s32_m(3, pg32, pg32, vinit1);
        if (k_total >= 16) {
          svint8x4_t va01 = svld1_vnum_s8_x4(pc8, a_ptr, 0);
          svint8x4_t vb01 = svld1_vnum_s8_x4(pc8, b_ptr, 0);
          k_idx = 16;
          for (; k_idx + 16 <= k_total; k_idx += 16) {
            const svint8x4_t va23 = svld1_vnum_s8_x4(pc8, a_ptr, 4);
            const svint8x4_t vb23 = svld1_vnum_s8_x4(pc8, b_ptr, 4);
            svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va01, 0),
                             svget4_s8(vb01, 0));
            svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va01, 0),
                             svget4_s8(vb01, 1));
            svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va01, 1),
                             svget4_s8(vb01, 0));
            svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va01, 1),
                             svget4_s8(vb01, 1));

            svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va01, 2),
                             svget4_s8(vb01, 2));
            svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va01, 2),
                             svget4_s8(vb01, 3));
            svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va01, 3),
                             svget4_s8(vb01, 2));
            svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va01, 3),
                             svget4_s8(vb01, 3));

            a_ptr += 16 * m_panel;
            b_ptr += 16 * n_panel;
            va01 = svld1_vnum_s8_x4(pc8, a_ptr, 0);
            vb01 = svld1_vnum_s8_x4(pc8, b_ptr, 0);

            svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va23, 0),
                             svget4_s8(vb23, 0));
            svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va23, 0),
                             svget4_s8(vb23, 1));
            svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va23, 1),
                             svget4_s8(vb23, 0));
            svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va23, 1),
                             svget4_s8(vb23, 1));

            svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va23, 2),
                             svget4_s8(vb23, 2));
            svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va23, 2),
                             svget4_s8(vb23, 3));
            svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va23, 3),
                             svget4_s8(vb23, 2));
            svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va23, 3),
                             svget4_s8(vb23, 3));
          }

          const svint8x4_t va23 = svld1_vnum_s8_x4(pc8, a_ptr, 4);
          const svint8x4_t vb23 = svld1_vnum_s8_x4(pc8, b_ptr, 4);
          svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va01, 0), svget4_s8(vb01, 0));
          svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va01, 0), svget4_s8(vb01, 1));
          svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va01, 1), svget4_s8(vb01, 0));
          svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va01, 1), svget4_s8(vb01, 1));

          svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va01, 2), svget4_s8(vb01, 2));
          svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va01, 2), svget4_s8(vb01, 3));
          svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va01, 3), svget4_s8(vb01, 2));
          svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va01, 3), svget4_s8(vb01, 3));

          svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va23, 0), svget4_s8(vb23, 0));
          svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va23, 0), svget4_s8(vb23, 1));
          svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va23, 1), svget4_s8(vb23, 0));
          svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va23, 1), svget4_s8(vb23, 1));

          svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va23, 2), svget4_s8(vb23, 2));
          svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va23, 2), svget4_s8(vb23, 3));
          svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va23, 3), svget4_s8(vb23, 2));
          svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va23, 3), svget4_s8(vb23, 3));

          a_ptr += 16 * m_panel;
          b_ptr += 16 * n_panel;
        }
        for (; k_idx < k_total; k_idx += 4) {
          const svint8x2_t va = svld1_s8_x2(pc8, a_ptr);
          const svint8x2_t vb = svld1_s8_x2(pc8, b_ptr);
          svmopa_za32_s8_m(0, pg8, pg8, svget2_s8(va, 0), svget2_s8(vb, 0));
          svmopa_za32_s8_m(1, pg8, pg8, svget2_s8(va, 0), svget2_s8(vb, 1));
          svmopa_za32_s8_m(2, pg8, pg8, svget2_s8(va, 1), svget2_s8(vb, 0));
          svmopa_za32_s8_m(3, pg8, pg8, svget2_s8(va, 1), svget2_s8(vb, 1));
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
      const svcount_t pc32_out = svwhilelt_c32_u64(0, n_rem, 2);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      size_t row = 0;
      for (; row + 4 <= rows0; row += 4) {
        float* dst_ptr =
            (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
        xnn_pqd8_store_2vl_4rows(
            svread_hor_za32_s32_vg4(0, (uint32_t)row),
            svread_hor_za32_s32_vg4(1, (uint32_t)row), vscale0, vscale1, vbias0,
            vbias1, vmin, vmax, pc32_out, dst_ptr, cm_stride);
      }
      for (; row < rows0; ++row) {
        float* dst_ptr =
            (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
        const svint32_t vacc0 =
            svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
        const svint32_t vacc1 =
            svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row);
        svfloat32_t vo0 = svmla_f32_x(
            pg32, vbias0, svcvt_f32_s32_x(pg32, vacc0), vscale0);
        svfloat32_t vo1 = svmla_f32_x(
            pg32, vbias1, svcvt_f32_s32_x(pg32, vacc1), vscale1);
        vo0 = svmin_f32_x(pg32, svmax_f32_x(pg32, vo0, vmin), vmax);
        vo1 = svmin_f32_x(pg32, svmax_f32_x(pg32, vo1, vmin), vmax);
        svst1_f32_x2(pc32_out, dst_ptr, svcreate2_f32(vo0, vo1));
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        size_t row1 = 0;
        for (; row1 + 4 <= rows1; row1 += 4) {
          float* dst_ptr =
              (float*)((uintptr_t)c + (m_idx + vl + row1) * cm_stride) + n_idx;
          xnn_pqd8_store_2vl_4rows(
              svread_hor_za32_s32_vg4(2, (uint32_t)row1),
              svread_hor_za32_s32_vg4(3, (uint32_t)row1), vscale0, vscale1,
              vbias0, vbias1, vmin, vmax, pc32_out, dst_ptr, cm_stride);
        }
        for (; row1 < rows1; ++row1) {
          float* dst_ptr =
              (float*)((uintptr_t)c + (m_idx + vl + row1) * cm_stride) + n_idx;
          const svint32_t vacc0 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row1);
          const svint32_t vacc1 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row1);
          svfloat32_t vo0 = svmla_f32_x(
              pg32, vbias0, svcvt_f32_s32_x(pg32, vacc0), vscale0);
          svfloat32_t vo1 = svmla_f32_x(
              pg32, vbias1, svcvt_f32_s32_x(pg32, vacc1), vscale1);
          vo0 = svmin_f32_x(pg32, svmax_f32_x(pg32, vo0, vmin), vmax);
          vo1 = svmin_f32_x(pg32, svmax_f32_x(pg32, vo1, vmin), vmax);
          svst1_f32_x2(pc32_out, dst_ptr, svcreate2_f32(vo0, vo1));
        }
      }
    }
  }
}

void xnn_pqd8_f32_qc8w_igemm_minmax_ukernel_32x32c4__sme2(
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
  xnn_pqd8_f32_qc8w_igemm_2vlx2vl_sme2_impl(
      mr, nc, k_total, (const int8_t*)packed_lhs, w, c, cm_stride,
      quantization_params->zero_point, quantization_params->inv_scale,
      params->scalar.min, params->scalar.max);
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
