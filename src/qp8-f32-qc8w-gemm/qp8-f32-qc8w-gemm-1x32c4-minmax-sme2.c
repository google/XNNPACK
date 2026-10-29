// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

__arm_locally_streaming __arm_new(
    "za") void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme2(size_t mr,
                                                                 size_t nc,
                                                                 size_t kc,
                                                                 const void* a,
                                                                 const void* w,
                                                                 float* c,
                                                                 size_t
                                                                     cm_stride,
                                                                 size_t
                                                                     cn_stride,
                                                                 const void*
                                                                     params) {
  (void)cn_stride;
  const size_t vl = svcntw();
  const size_t nr = 2 * vl;
  const size_t k_internal = (kc + 31) & ~(size_t)31;
  const size_t lhs_row_stride =
      k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float);
  const size_t rhs_row_stride =
      nr * (k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float) +
            sizeof(float));
  const struct xnn_f32_minmax_params* minmax_params =
      (const struct xnn_f32_minmax_params*)params;
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svbool_t p_row0 = svwhilelt_b8_u64(0, 4);
  const svbool_t p_row1 = sveor_b_z(pg8, svwhilelt_b8_u64(0, 8), p_row0);
  const svbool_t p_row2 =
      sveor_b_z(pg8, svwhilelt_b8_u64(0, 12), svwhilelt_b8_u64(0, 8));
  const svbool_t p_row3 =
      sveor_b_z(pg8, svwhilelt_b8_u64(0, 16), svwhilelt_b8_u64(0, 12));
  const svfloat32_t vmin = svdup_n_f32(minmax_params->scalar.min);
  const svfloat32_t vmax = svdup_n_f32(minmax_params->scalar.max);

  for (size_t m_idx = 0; m_idx < mr; ++m_idx) {
    const int8_t* lhs_row =
        (const int8_t*)((uintptr_t)a + m_idx * lhs_row_stride);
    const int32_t lhs_neg_zp = *(const int32_t*)(lhs_row + k_internal);
    const float lhs_scale =
        *(const float*)(lhs_row + k_internal + sizeof(int32_t));
    const svint32_t vlhs_neg_zp = svdup_n_s32(lhs_neg_zp);
    const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scale);
    float* dst_row = (float*)((uintptr_t)c + m_idx * cm_stride);

    size_t n_idx = 0;
    for (; n_idx + 2 * nr <= nc; n_idx += 2 * nr) {
      const uint8_t* rhs_block0 =
          (const uint8_t*)w + (n_idx / nr) * rhs_row_stride;
      const uint8_t* rhs_block1 = rhs_block0 + rhs_row_stride;

      svzero_za();

      const int8_t* w0 = (const int8_t*)rhs_block0;
      const int8_t* w1 = (const int8_t*)rhs_block1;

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);

        svmopa_za32_s8_m(0, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 0));
        svmopa_za32_s8_m(1, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 1));
        svmopa_za32_s8_m(2, p_row0, pg8, va, svld1_vnum_s8(pg8, w1, 0));
        svmopa_za32_s8_m(3, p_row0, pg8, va, svld1_vnum_s8(pg8, w1, 1));

        svmopa_za32_s8_m(0, p_row1, pg8, va, svld1_vnum_s8(pg8, w0, 2));
        svmopa_za32_s8_m(1, p_row1, pg8, va, svld1_vnum_s8(pg8, w0, 3));
        svmopa_za32_s8_m(2, p_row1, pg8, va, svld1_vnum_s8(pg8, w1, 2));
        svmopa_za32_s8_m(3, p_row1, pg8, va, svld1_vnum_s8(pg8, w1, 3));

        svmopa_za32_s8_m(0, p_row2, pg8, va, svld1_vnum_s8(pg8, w0, 4));
        svmopa_za32_s8_m(1, p_row2, pg8, va, svld1_vnum_s8(pg8, w0, 5));
        svmopa_za32_s8_m(2, p_row2, pg8, va, svld1_vnum_s8(pg8, w1, 4));
        svmopa_za32_s8_m(3, p_row2, pg8, va, svld1_vnum_s8(pg8, w1, 5));

        svmopa_za32_s8_m(0, p_row3, pg8, va, svld1_vnum_s8(pg8, w0, 6));
        svmopa_za32_s8_m(1, p_row3, pg8, va, svld1_vnum_s8(pg8, w0, 7));
        svmopa_za32_s8_m(2, p_row3, pg8, va, svld1_vnum_s8(pg8, w1, 6));
        svmopa_za32_s8_m(3, p_row3, pg8, va, svld1_vnum_s8(pg8, w1, 7));

        w0 += 16 * nr;
        w1 += 16 * nr;
      }

      svint32_t vacc0 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 3)));
      svint32_t vacc1 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 3)));
      svint32_t vacc2 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 3)));
      svint32_t vacc3 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 3)));

      const int32_t* rhs_sums0 = (const int32_t*)(rhs_block0 + nr * k_internal);
      const float* rhs_scales0 = (const float*)(rhs_sums0 + nr);
      const float* rhs_biases0 = rhs_scales0 + nr;
      const int32_t* rhs_sums1 = (const int32_t*)(rhs_block1 + nr * k_internal);
      const float* rhs_scales1 = (const float*)(rhs_sums1 + nr);
      const float* rhs_biases1 = rhs_scales1 + nr;

      vacc0 = svmla_s32_m(pg32, vacc0, svld1_vnum_s32(pg32, rhs_sums0, 0),
                          vlhs_neg_zp);
      vacc1 = svmla_s32_m(pg32, vacc1, svld1_vnum_s32(pg32, rhs_sums0, 1),
                          vlhs_neg_zp);
      vacc2 = svmla_s32_m(pg32, vacc2, svld1_vnum_s32(pg32, rhs_sums1, 0),
                          vlhs_neg_zp);
      vacc3 = svmla_s32_m(pg32, vacc3, svld1_vnum_s32(pg32, rhs_sums1, 1),
                          vlhs_neg_zp);

      svfloat32_t vf0 = svcvt_f32_s32_x(pg32, vacc0);
      svfloat32_t vf1 = svcvt_f32_s32_x(pg32, vacc1);
      svfloat32_t vf2 = svcvt_f32_s32_x(pg32, vacc2);
      svfloat32_t vf3 = svcvt_f32_s32_x(pg32, vacc3);

      vf0 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases0, 0), vf0,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales0, 0)));
      vf1 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases0, 1), vf1,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales0, 1)));
      vf2 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases1, 0), vf2,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales1, 0)));
      vf3 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases1, 1), vf3,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales1, 1)));

      vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
      vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
      vf2 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf2, vmin), vmax);
      vf3 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf3, vmin), vmax);

      svst1_vnum_f32(pg32, dst_row + n_idx, 0, vf0);
      svst1_vnum_f32(pg32, dst_row + n_idx, 1, vf1);
      svst1_vnum_f32(pg32, dst_row + n_idx, 2, vf2);
      svst1_vnum_f32(pg32, dst_row + n_idx, 3, vf3);
    }

    for (; n_idx < nc; n_idx += nr) {
      const uint8_t* rhs_block =
          (const uint8_t*)w + (n_idx / nr) * rhs_row_stride;
      svzero_za();

      const int8_t* w0 = (const int8_t*)rhs_block;
      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svmopa_za32_s8_m(0, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 0));
        svmopa_za32_s8_m(1, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 1));
        svmopa_za32_s8_m(0, p_row1, pg8, va, svld1_vnum_s8(pg8, w0, 2));
        svmopa_za32_s8_m(1, p_row1, pg8, va, svld1_vnum_s8(pg8, w0, 3));
        svmopa_za32_s8_m(0, p_row2, pg8, va, svld1_vnum_s8(pg8, w0, 4));
        svmopa_za32_s8_m(1, p_row2, pg8, va, svld1_vnum_s8(pg8, w0, 5));
        svmopa_za32_s8_m(0, p_row3, pg8, va, svld1_vnum_s8(pg8, w0, 6));
        svmopa_za32_s8_m(1, p_row3, pg8, va, svld1_vnum_s8(pg8, w0, 7));
        w0 += 16 * nr;
      }

      svint32_t vacc0 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 3)));
      svint32_t vacc1 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 3)));

      const int32_t* rhs_sums = (const int32_t*)(rhs_block + nr * k_internal);
      const float* rhs_scales = (const float*)(rhs_sums + nr);
      const float* rhs_biases = rhs_scales + nr;

      vacc0 = svmla_s32_m(pg32, vacc0, svld1_vnum_s32(pg32, rhs_sums, 0),
                          vlhs_neg_zp);
      vacc1 = svmla_s32_m(pg32, vacc1, svld1_vnum_s32(pg32, rhs_sums, 1),
                          vlhs_neg_zp);

      svfloat32_t vf0 = svcvt_f32_s32_x(pg32, vacc0);
      svfloat32_t vf1 = svcvt_f32_s32_x(pg32, vacc1);

      vf0 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases, 0), vf0,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales, 0)));
      vf1 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases, 1), vf1,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales, 1)));

      vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
      vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);

      const size_t n_rem = nc - n_idx;
      svst1_vnum_f32(svwhilelt_b32_u64(0 * vl, n_rem), dst_row + n_idx, 0, vf0);
      svst1_vnum_f32(svwhilelt_b32_u64(1 * vl, n_rem), dst_row + n_idx, 1, vf1);
    }
  }
}

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme2_get_mr() { return 1; }

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme2_get_nr() {
  return 2 * svcntsw();
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
