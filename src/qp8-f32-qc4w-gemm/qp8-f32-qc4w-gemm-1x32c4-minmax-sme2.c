// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x32c4__sme2_get_mr(void) {
  return 1;
}

__arm_locally_streaming size_t
xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x32c4__sme2_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za",
    "zt0") void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x32c4__sme2(size_t mr,
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
                                                                      minmax_params) {
  (void)mr;
  (void)cm_stride;
  const struct xnn_f32_qc4w_minmax_params* params =
      (const struct xnn_f32_qc4w_minmax_params*)minmax_params;

  const size_t svl_w = svcntsw();
  const size_t svl_b = svcntb();
  const size_t nr = 2 * svl_w;
  const size_t k_rup = (kc + 31) & ~(size_t)31;
  const size_t cn_step = nr * cn_stride;
  const size_t w_offset_nr = nr * (k_rup / 2 + 3 * sizeof(int32_t));

  static const int8_t lut[64] = {
      0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
      0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
      -5, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};
  svldr_zt(0, &lut[0]);

  const int8_t* a0 = (const int8_t*)a;
  const int32_t* a_zp = (const int32_t*)(a0 + k_rup);
  const float* a_inv_scale = (const float*)(a_zp + 1);

  const svbool_t pg_8_all = svptrue_b8();
  const svbool_t pg_32_all = svptrue_b32();
  const svcount_t pcnt_8_all = svptrue_c8();
  const svcount_t pcnt_32_all = svptrue_c32();

  const svint32_t vzp = svdup_n_s32(*a_zp);
  const svfloat32_t vinvs = svdup_n_f32(*a_inv_scale);
  const svfloat32_t voutput_min = svdup_n_f32(params->scalar.min);
  const svfloat32_t voutput_max = svdup_n_f32(params->scalar.max);

  float* c0 = c;
  const int8_t* w0 = (const int8_t*)w;

  while (nc > nr) {
    const int8_t* w1 = w0 + w_offset_nr;
    const int8_t* a_inc = a0;

    svzero_za();

    for (size_t k = k_rup; k != 0; k -= 16) {
      const svint8_t va0 = svld1rq_s8(pg_8_all, a_inc);
      a_inc += 16;

      const svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w0);
      w0 += 4 * svl_b;
      const svint8x4_t vb_x4_1 = svld1_s8_x4(pcnt_8_all, w1);
      w1 += 4 * svl_b;

      const svint8x2_t luti_0 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 0)), 0);
      const svint8x2_t luti_1 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 1)), 0);
      const svint8x2_t luti_2 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 2)), 0);
      const svint8x2_t luti_3 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 3)), 0);

      const svint8x2_t luti_4 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 0)), 0);
      const svint8x2_t luti_5 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 1)), 0);
      const svint8x2_t luti_6 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 2)), 0);
      const svint8x2_t luti_7 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 3)), 0);

      const svint8x4_t luti_k0 =
          svcreate4(svget2(luti_0, 0), svget2(luti_0, 1), svget2(luti_4, 0),
                    svget2(luti_4, 1));
      const svint8x4_t luti_k1 =
          svcreate4(svget2(luti_1, 0), svget2(luti_1, 1), svget2(luti_5, 0),
                    svget2(luti_5, 1));
      const svint8x4_t luti_k2 =
          svcreate4(svget2(luti_2, 0), svget2(luti_2, 1), svget2(luti_6, 0),
                    svget2(luti_6, 1));
      const svint8x4_t luti_k3 =
          svcreate4(svget2(luti_3, 0), svget2(luti_3, 1), svget2(luti_7, 0),
                    svget2(luti_7, 1));

      svdot_lane_za32_s8_vg1x4(0, luti_k0, va0, 0);
      svdot_lane_za32_s8_vg1x4(1, luti_k1, va0, 1);
      svdot_lane_za32_s8_vg1x4(2, luti_k2, va0, 2);
      svdot_lane_za32_s8_vg1x4(3, luti_k3, va0, 3);
    }

    const svint32x2_t vksum_01 = svld1_s32_x2(pcnt_32_all, (const int32_t*)w0);
    w0 += nr * sizeof(int32_t);
    const svint32x2_t vksum_23 = svld1_s32_x2(pcnt_32_all, (const int32_t*)w1);
    w1 += nr * sizeof(int32_t);

    const svfloat32x2_t vfilter_01 =
        svld1_f32_x2(pcnt_32_all, (const float*)w0);
    w0 += nr * sizeof(float);
    const svfloat32x2_t vfilter_23 =
        svld1_f32_x2(pcnt_32_all, (const float*)w1);
    w1 += nr * sizeof(float);

    const svfloat32x2_t vbias_01 = svld1_f32_x2(pcnt_32_all, (const float*)w0);
    w0 += nr * sizeof(float);
    const svfloat32x2_t vbias_23 = svld1_f32_x2(pcnt_32_all, (const float*)w1);
    w1 += nr * sizeof(float);

    const svint32x4_t za0_x4 = svread_za32_s32_vg1x4(0);
    const svint32x4_t za1_x4 = svread_za32_s32_vg1x4(1);
    const svint32x4_t za2_x4 = svread_za32_s32_vg1x4(2);
    const svint32x4_t za3_x4 = svread_za32_s32_vg1x4(3);

    svint32_t vacco_0 = svadd_s32_x(
        pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_x4, 0), svget4(za1_x4, 0)),
        svadd_s32_x(pg_32_all, svget4(za2_x4, 0), svget4(za3_x4, 0)));
    svint32_t vacco_1 = svadd_s32_x(
        pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_x4, 1), svget4(za1_x4, 1)),
        svadd_s32_x(pg_32_all, svget4(za2_x4, 1), svget4(za3_x4, 1)));
    svint32_t vacco_2 = svadd_s32_x(
        pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_x4, 2), svget4(za1_x4, 2)),
        svadd_s32_x(pg_32_all, svget4(za2_x4, 2), svget4(za3_x4, 2)));
    svint32_t vacco_3 = svadd_s32_x(
        pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_x4, 3), svget4(za1_x4, 3)),
        svadd_s32_x(pg_32_all, svget4(za2_x4, 3), svget4(za3_x4, 3)));

    vacco_0 = svmla_s32_x(pg_32_all, vacco_0, svget2(vksum_01, 0), vzp);
    vacco_1 = svmla_s32_x(pg_32_all, vacco_1, svget2(vksum_01, 1), vzp);
    vacco_2 = svmla_s32_x(pg_32_all, vacco_2, svget2(vksum_23, 0), vzp);
    vacco_3 = svmla_s32_x(pg_32_all, vacco_3, svget2(vksum_23, 1), vzp);

    const svfloat32x4_t vout_x4 =
        svcvt_f32(svcreate4(vacco_0, vacco_1, vacco_2, vacco_3));
    const svfloat32_t vout_0 =
        svmad_f32_m(pg_32_all, svget4(vout_x4, 0),
                    svmul_f32_x(pg_32_all, svget2(vfilter_01, 0), vinvs),
                    svget2(vbias_01, 0));
    const svfloat32_t vout_1 =
        svmad_f32_m(pg_32_all, svget4(vout_x4, 1),
                    svmul_f32_x(pg_32_all, svget2(vfilter_01, 1), vinvs),
                    svget2(vbias_01, 1));
    const svfloat32_t vout_2 =
        svmad_f32_m(pg_32_all, svget4(vout_x4, 2),
                    svmul_f32_x(pg_32_all, svget2(vfilter_23, 0), vinvs),
                    svget2(vbias_23, 0));
    const svfloat32_t vout_3 =
        svmad_f32_m(pg_32_all, svget4(vout_x4, 3),
                    svmul_f32_x(pg_32_all, svget2(vfilter_23, 1), vinvs),
                    svget2(vbias_23, 1));

    const svfloat32x4_t vout0123 = svclamp(
        svcreate4(vout_0, vout_1, vout_2, vout_3), voutput_min, voutput_max);
    const svcount_t pcnt_32_nc = svwhilelt_c32_u64(0, nc, 4);
    svst1_f32_x4(pcnt_32_nc, c0, vout0123);

    if (nc > 2 * nr) {
      w0 = w1;
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

      const svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w0);
      w0 += 4 * svl_b;

      const svint8x2_t luti_0 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 0)), 0);
      const svint8x2_t luti_1 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 1)), 0);
      const svint8x2_t luti_2 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 2)), 0);
      const svint8x2_t luti_3 =
          svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 3)), 0);

      vacc0_0 = svdot_lane_s32(vacc0_0, svget2(luti_0, 0), va0, 0);
      vacc0_1 = svdot_lane_s32(vacc0_1, svget2(luti_0, 1), va0, 0);
      vacc1_0 = svdot_lane_s32(vacc1_0, svget2(luti_1, 0), va0, 1);
      vacc1_1 = svdot_lane_s32(vacc1_1, svget2(luti_1, 1), va0, 1);
      vacc2_0 = svdot_lane_s32(vacc2_0, svget2(luti_2, 0), va0, 2);
      vacc2_1 = svdot_lane_s32(vacc2_1, svget2(luti_2, 1), va0, 2);
      vacc3_0 = svdot_lane_s32(vacc3_0, svget2(luti_3, 0), va0, 3);
      vacc3_1 = svdot_lane_s32(vacc3_1, svget2(luti_3, 1), va0, 3);
    }

    svint32_t vacco_0 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_0, vacc1_0),
                    svadd_s32_x(pg_32_all, vacc2_0, vacc3_0));
    svint32_t vacco_1 =
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, vacc0_1, vacc1_1),
                    svadd_s32_x(pg_32_all, vacc2_1, vacc3_1));

    const svint32x2_t vksum_01 = svld1_s32_x2(pcnt_32_all, (const int32_t*)w0);
    w0 += nr * sizeof(int32_t);
    const svfloat32x2_t vfilter_01 =
        svld1_f32_x2(pcnt_32_all, (const float*)w0);
    w0 += nr * sizeof(float);
    const svfloat32x2_t vbias_01 = svld1_f32_x2(pcnt_32_all, (const float*)w0);

    vacco_0 = svmla_s32_x(pg_32_all, vacco_0, svget2(vksum_01, 0), vzp);
    vacco_1 = svmla_s32_x(pg_32_all, vacco_1, svget2(vksum_01, 1), vzp);

    const svfloat32x2_t vout_x2 = svcvt_f32(svcreate2(vacco_0, vacco_1));
    const svfloat32_t vout_0 =
        svmad_f32_m(pg_32_all, svget2(vout_x2, 0),
                    svmul_f32_x(pg_32_all, svget2(vfilter_01, 0), vinvs),
                    svget2(vbias_01, 0));
    const svfloat32_t vout_1 =
        svmad_f32_m(pg_32_all, svget2(vout_x2, 1),
                    svmul_f32_x(pg_32_all, svget2(vfilter_01, 1), vinvs),
                    svget2(vbias_01, 1));

    const svfloat32x2_t vout01 =
        svclamp(svcreate2(vout_0, vout_1), voutput_min, voutput_max);
    const svcount_t pcnt_32_nc = svwhilelt_c32_u64(0, nc, 2);
    svst1_f32_x2(pcnt_32_nc, c0, vout01);
  }
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
