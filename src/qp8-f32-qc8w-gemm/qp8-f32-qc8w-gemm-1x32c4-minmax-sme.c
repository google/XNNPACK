// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ENABLE_ARM_SME && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme_get_mr(void) {
  return 1;
}

__arm_locally_streaming size_t
xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme(
    size_t mr, size_t nc, size_t kc, const void* a, const void* w, float* c,
    size_t cm_stride, size_t cn_stride, const void* minmax_params) {
  (void)mr;
  (void)cm_stride;
  const struct xnn_f32_minmax_params* params =
      (const struct xnn_f32_minmax_params*)minmax_params;

  const size_t svl_w = svcntsw();
  const size_t svl_b = svcntb();
  const size_t nr = 2 * svl_w;
  const size_t k_rup = (kc + 31) & ~(size_t)31;
  const size_t cn_step = nr * cn_stride;
  const size_t w_offset_nr = nr * (k_rup + 3 * sizeof(int32_t));

  const int8_t* a0 = (const int8_t*)a;
  const int32_t* a_zp = (const int32_t*)(a0 + k_rup);
  const float* a_inv_scale = (const float*)(a_zp + 1);

  const svbool_t pg_8_all = svptrue_b8();
  const svbool_t pg_32_all = svptrue_b32();

  const svint32_t vzp = svdup_n_s32(*a_zp);
  const svfloat32_t vinvs = svdup_n_f32(*a_inv_scale);
  const svfloat32_t voutput_min = svdup_n_f32(params->scalar.min);
  const svfloat32_t voutput_max = svdup_n_f32(params->scalar.max);

  float* c0 = c;
  const int8_t* w0 = (const int8_t*)w;

  while (nc > nr) {
    const int8_t* w1 = w0 + w_offset_nr;
    const int8_t* a_inc = a0;

    svint32_t vacc0_0 = svdup_n_s32(0);
    svint32_t vacc0_1 = svdup_n_s32(0);
    svint32_t vacc0_2 = svdup_n_s32(0);
    svint32_t vacc0_3 = svdup_n_s32(0);
    svint32_t vacc1_0 = svdup_n_s32(0);
    svint32_t vacc1_1 = svdup_n_s32(0);
    svint32_t vacc1_2 = svdup_n_s32(0);
    svint32_t vacc1_3 = svdup_n_s32(0);
    svint32_t vacc2_0 = svdup_n_s32(0);
    svint32_t vacc2_1 = svdup_n_s32(0);
    svint32_t vacc2_2 = svdup_n_s32(0);
    svint32_t vacc2_3 = svdup_n_s32(0);
    svint32_t vacc3_0 = svdup_n_s32(0);
    svint32_t vacc3_1 = svdup_n_s32(0);
    svint32_t vacc3_2 = svdup_n_s32(0);
    svint32_t vacc3_3 = svdup_n_s32(0);

    for (size_t k = k_rup; k != 0; k -= 16) {
      const svint8_t va0 = svld1rq_s8(pg_8_all, a_inc);
      a_inc += 16;

      const svint8_t vb0_0 = svld1_vnum_s8(pg_8_all, w0, 0);
      const svint8_t vb0_1 = svld1_vnum_s8(pg_8_all, w0, 1);
      const svint8_t vb1_0 = svld1_vnum_s8(pg_8_all, w0, 2);
      const svint8_t vb1_1 = svld1_vnum_s8(pg_8_all, w0, 3);
      const svint8_t vb2_0 = svld1_vnum_s8(pg_8_all, w0, 4);
      const svint8_t vb2_1 = svld1_vnum_s8(pg_8_all, w0, 5);
      const svint8_t vb3_0 = svld1_vnum_s8(pg_8_all, w0, 6);
      const svint8_t vb3_1 = svld1_vnum_s8(pg_8_all, w0, 7);
      w0 += 8 * svl_b;

      const svint8_t vb0_2 = svld1_vnum_s8(pg_8_all, w1, 0);
      const svint8_t vb0_3 = svld1_vnum_s8(pg_8_all, w1, 1);
      const svint8_t vb1_2 = svld1_vnum_s8(pg_8_all, w1, 2);
      const svint8_t vb1_3 = svld1_vnum_s8(pg_8_all, w1, 3);
      const svint8_t vb2_2 = svld1_vnum_s8(pg_8_all, w1, 4);
      const svint8_t vb2_3 = svld1_vnum_s8(pg_8_all, w1, 5);
      const svint8_t vb3_2 = svld1_vnum_s8(pg_8_all, w1, 6);
      const svint8_t vb3_3 = svld1_vnum_s8(pg_8_all, w1, 7);
      w1 += 8 * svl_b;

      vacc0_0 = svdot_lane_s32(vacc0_0, vb0_0, va0, 0);
      vacc0_1 = svdot_lane_s32(vacc0_1, vb0_1, va0, 0);
      vacc0_2 = svdot_lane_s32(vacc0_2, vb0_2, va0, 0);
      vacc0_3 = svdot_lane_s32(vacc0_3, vb0_3, va0, 0);

      vacc1_0 = svdot_lane_s32(vacc1_0, vb1_0, va0, 1);
      vacc1_1 = svdot_lane_s32(vacc1_1, vb1_1, va0, 1);
      vacc1_2 = svdot_lane_s32(vacc1_2, vb1_2, va0, 1);
      vacc1_3 = svdot_lane_s32(vacc1_3, vb1_3, va0, 1);

      vacc2_0 = svdot_lane_s32(vacc2_0, vb2_0, va0, 2);
      vacc2_1 = svdot_lane_s32(vacc2_1, vb2_1, va0, 2);
      vacc2_2 = svdot_lane_s32(vacc2_2, vb2_2, va0, 2);
      vacc2_3 = svdot_lane_s32(vacc2_3, vb2_3, va0, 2);

      vacc3_0 = svdot_lane_s32(vacc3_0, vb3_0, va0, 3);
      vacc3_1 = svdot_lane_s32(vacc3_1, vb3_1, va0, 3);
      vacc3_2 = svdot_lane_s32(vacc3_2, vb3_2, va0, 3);
      vacc3_3 = svdot_lane_s32(vacc3_3, vb3_3, va0, 3);
    }

    svint32_t vacco_0 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_0, vacc1_0),
                    svadd_s32_x(pg_32_all, vacc2_0, vacc3_0));
    svint32_t vacco_1 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_1, vacc1_1),
                    svadd_s32_x(pg_32_all, vacc2_1, vacc3_1));
    svint32_t vacco_2 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_2, vacc1_2),
                    svadd_s32_x(pg_32_all, vacc2_2, vacc3_2));
    svint32_t vacco_3 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_3, vacc1_3),
                    svadd_s32_x(pg_32_all, vacc2_3, vacc3_3));

    const int32_t* w0_i32 = (const int32_t*)w0;
    const int32_t* w1_i32 = (const int32_t*)w1;
    const svint32_t vksum_0 = svld1_s32(pg_32_all, w0_i32);
    const svint32_t vksum_1 = svld1_s32(pg_32_all, w0_i32 + svl_w);
    const svint32_t vksum_2 = svld1_s32(pg_32_all, w1_i32);
    const svint32_t vksum_3 = svld1_s32(pg_32_all, w1_i32 + svl_w);

    const float* w0_f32 = (const float*)(w0_i32 + nr);
    const float* w1_f32 = (const float*)(w1_i32 + nr);
    const svfloat32_t vfilter_0 = svld1_f32(pg_32_all, w0_f32);
    const svfloat32_t vfilter_1 = svld1_f32(pg_32_all, w0_f32 + svl_w);
    const svfloat32_t vfilter_2 = svld1_f32(pg_32_all, w1_f32);
    const svfloat32_t vfilter_3 = svld1_f32(pg_32_all, w1_f32 + svl_w);

    const svfloat32_t vbias_0 = svld1_f32(pg_32_all, w0_f32 + nr);
    const svfloat32_t vbias_1 = svld1_f32(pg_32_all, w0_f32 + nr + svl_w);
    const svfloat32_t vbias_2 = svld1_f32(pg_32_all, w1_f32 + nr);
    const svfloat32_t vbias_3 = svld1_f32(pg_32_all, w1_f32 + nr + svl_w);

    vacco_0 = svmla_s32_x(pg_32_all, vacco_0, vksum_0, vzp);
    vacco_1 = svmla_s32_x(pg_32_all, vacco_1, vksum_1, vzp);
    vacco_2 = svmla_s32_x(pg_32_all, vacco_2, vksum_2, vzp);
    vacco_3 = svmla_s32_x(pg_32_all, vacco_3, vksum_3, vzp);

    svfloat32_t vout_0 = svcvt_f32_s32_x(pg_32_all, vacco_0);
    svfloat32_t vout_1 = svcvt_f32_s32_x(pg_32_all, vacco_1);
    svfloat32_t vout_2 = svcvt_f32_s32_x(pg_32_all, vacco_2);
    svfloat32_t vout_3 = svcvt_f32_s32_x(pg_32_all, vacco_3);

    vout_0 = svmad_f32_x(pg_32_all, vout_0,
                         svmul_f32_x(pg_32_all, vfilter_0, vinvs), vbias_0);
    vout_1 = svmad_f32_x(pg_32_all, vout_1,
                         svmul_f32_x(pg_32_all, vfilter_1, vinvs), vbias_1);
    vout_2 = svmad_f32_x(pg_32_all, vout_2,
                         svmul_f32_x(pg_32_all, vfilter_2, vinvs), vbias_2);
    vout_3 = svmad_f32_x(pg_32_all, vout_3,
                         svmul_f32_x(pg_32_all, vfilter_3, vinvs), vbias_3);

    vout_0 = svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_0, voutput_min),
                         voutput_max);
    vout_1 = svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_1, voutput_min),
                         voutput_max);
    vout_2 = svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_2, voutput_min),
                         voutput_max);
    vout_3 = svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_3, voutput_min),
                         voutput_max);

    svst1_f32(pg_32_all, c0, vout_0);
    svst1_f32(pg_32_all, c0 + svl_w, vout_1);
    const svbool_t pg_n2 = svwhilelt_b32_u64(2 * svl_w, nc);
    const svbool_t pg_n3 = svwhilelt_b32_u64(3 * svl_w, nc);
    svst1_f32(pg_n2, c0 + 2 * svl_w, vout_2);
    svst1_f32(pg_n3, c0 + 3 * svl_w, vout_3);

    if (nc > 2 * nr) {
      w0 = (const int8_t*)(w1_f32 + 2 * nr);
      c0 = (float*)((uintptr_t)c0 + 2 * cn_step);
      nc -= 2 * nr;
    } else {
      return;
    }
  }

  if (nc != 0) {
    const int8_t* a_inc = a0;

    svint32_t vacc0_0 = svdup_n_s32(0);
    svint32_t vacc0_1 = svdup_n_s32(0);
    svint32_t vacc1_0 = svdup_n_s32(0);
    svint32_t vacc1_1 = svdup_n_s32(0);
    svint32_t vacc2_0 = svdup_n_s32(0);
    svint32_t vacc2_1 = svdup_n_s32(0);
    svint32_t vacc3_0 = svdup_n_s32(0);
    svint32_t vacc3_1 = svdup_n_s32(0);

    for (size_t k = k_rup; k != 0; k -= 16) {
      const svint8_t va0 = svld1rq_s8(pg_8_all, a_inc);
      a_inc += 16;

      const svint8_t vb0_0 = svld1_vnum_s8(pg_8_all, w0, 0);
      const svint8_t vb0_1 = svld1_vnum_s8(pg_8_all, w0, 1);
      const svint8_t vb1_0 = svld1_vnum_s8(pg_8_all, w0, 2);
      const svint8_t vb1_1 = svld1_vnum_s8(pg_8_all, w0, 3);
      const svint8_t vb2_0 = svld1_vnum_s8(pg_8_all, w0, 4);
      const svint8_t vb2_1 = svld1_vnum_s8(pg_8_all, w0, 5);
      const svint8_t vb3_0 = svld1_vnum_s8(pg_8_all, w0, 6);
      const svint8_t vb3_1 = svld1_vnum_s8(pg_8_all, w0, 7);
      w0 += 8 * svl_b;

      vacc0_0 = svdot_lane_s32(vacc0_0, vb0_0, va0, 0);
      vacc0_1 = svdot_lane_s32(vacc0_1, vb0_1, va0, 0);
      vacc1_0 = svdot_lane_s32(vacc1_0, vb1_0, va0, 1);
      vacc1_1 = svdot_lane_s32(vacc1_1, vb1_1, va0, 1);
      vacc2_0 = svdot_lane_s32(vacc2_0, vb2_0, va0, 2);
      vacc2_1 = svdot_lane_s32(vacc2_1, vb2_1, va0, 2);
      vacc3_0 = svdot_lane_s32(vacc3_0, vb3_0, va0, 3);
      vacc3_1 = svdot_lane_s32(vacc3_1, vb3_1, va0, 3);
    }

    svint32_t vacco_0 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_0, vacc1_0),
                    svadd_s32_x(pg_32_all, vacc2_0, vacc3_0));
    svint32_t vacco_1 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_1, vacc1_1),
                    svadd_s32_x(pg_32_all, vacc2_1, vacc3_1));

    const int32_t* w0_i32 = (const int32_t*)w0;
    const svint32_t vksum_0 = svld1_s32(pg_32_all, w0_i32);
    const svint32_t vksum_1 = svld1_s32(pg_32_all, w0_i32 + svl_w);

    const float* w0_f32 = (const float*)(w0_i32 + nr);
    const svfloat32_t vfilter_0 = svld1_f32(pg_32_all, w0_f32);
    const svfloat32_t vfilter_1 = svld1_f32(pg_32_all, w0_f32 + svl_w);

    const svfloat32_t vbias_0 = svld1_f32(pg_32_all, w0_f32 + nr);
    const svfloat32_t vbias_1 = svld1_f32(pg_32_all, w0_f32 + nr + svl_w);

    vacco_0 = svmla_s32_x(pg_32_all, vacco_0, vksum_0, vzp);
    vacco_1 = svmla_s32_x(pg_32_all, vacco_1, vksum_1, vzp);

    svfloat32_t vout_0 = svcvt_f32_s32_x(pg_32_all, vacco_0);
    svfloat32_t vout_1 = svcvt_f32_s32_x(pg_32_all, vacco_1);

    vout_0 = svmad_f32_x(pg_32_all, vout_0,
                         svmul_f32_x(pg_32_all, vfilter_0, vinvs), vbias_0);
    vout_1 = svmad_f32_x(pg_32_all, vout_1,
                         svmul_f32_x(pg_32_all, vfilter_1, vinvs), vbias_1);

    vout_0 = svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_0, voutput_min),
                         voutput_max);
    vout_1 = svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_1, voutput_min),
                         voutput_max);

    const svbool_t pg_n0 = svwhilelt_b32_u64(0, nc);
    const svbool_t pg_n1 = svwhilelt_b32_u64(svl_w, nc);
    svst1_f32(pg_n0, c0, vout_0);
    svst1_f32(pg_n1, c0 + svl_w, vout_1);
  }
}
#endif  // XNN_ENABLE_ARM_SME && XNN_ARCH_ARM64
