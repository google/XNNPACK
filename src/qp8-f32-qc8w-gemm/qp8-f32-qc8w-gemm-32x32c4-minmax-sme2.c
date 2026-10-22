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

__arm_locally_streaming size_t
xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme2_get_mr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming size_t
xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme2_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme2(size_t mr,
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
  const struct xnn_f32_minmax_params* params =
      (const struct xnn_f32_minmax_params*)minmax_params;
  const size_t svl_w = svcntsw();
  const size_t svl_b = svcntb();
  const size_t nr = 2 * svl_w;
  const size_t mr_max = 2 * svl_w;
  const size_t k_rup = (kc + 31) & ~(size_t)31;
  const size_t cn_step = nr * cn_stride;
  const size_t w_offset_nr = nr * (k_rup + 3 * sizeof(int32_t));

  const int8_t* a0 = (const int8_t*)a;
  const int32_t* a_zp = (const int32_t*)(a0 + mr_max * k_rup);
  const float* a_inv_scale = (const float*)(a_zp + mr_max);
  float* c0 = c;

  const svbool_t pg_8_all = svptrue_b8();
  const svbool_t pg_32_all = svptrue_b32();
  const svcount_t pcnt_8_all = svptrue_c8();
  const svcount_t pcnt_32_all = svptrue_c32();
  const svfloat32_t voutput_min = svdup_n_f32(params->scalar.min);
  const svfloat32_t voutput_max = svdup_n_f32(params->scalar.max);

  if (mr <= svl_w) {
    const int8_t* w0 = (const int8_t*)w;

    while (nc > nr) {
      const int8_t* w1 = w0 + w_offset_nr;
      const int8_t* a_inc = a0;

      svzero_za();

      for (size_t k = k_rup; k != 0; k -= 16) {
        const svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w0);
        const svint8x4_t vb_x4_1 = svld1_s8_x4(pcnt_8_all, w0 + 4 * svl_b);
        w0 += 8 * svl_b;

        const svint8x4_t vb_x4_2 = svld1_s8_x4(pcnt_8_all, w1);
        const svint8x4_t vb_x4_3 = svld1_s8_x4(pcnt_8_all, w1 + 4 * svl_b);
        w1 += 8 * svl_b;

        const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
        const svint8_t va1_0 = svld1_vnum_s8(pg_8_all, a_inc, 2);
        const svint8_t va2_0 = svld1_vnum_s8(pg_8_all, a_inc, 4);
        const svint8_t va3_0 = svld1_vnum_s8(pg_8_all, a_inc, 6);
        a_inc += 8 * svl_b;

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, svget4(vb_x4_0, 0));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0_0, svget4(vb_x4_0, 1));
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0_0, svget4(vb_x4_2, 0));
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va0_0, svget4(vb_x4_2, 1));

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va1_0, svget4(vb_x4_0, 2));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1_0, svget4(vb_x4_0, 3));
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va1_0, svget4(vb_x4_2, 2));
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va1_0, svget4(vb_x4_2, 3));

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2_0, svget4(vb_x4_1, 0));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va2_0, svget4(vb_x4_1, 1));
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va2_0, svget4(vb_x4_3, 0));
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va2_0, svget4(vb_x4_3, 1));

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va3_0, svget4(vb_x4_1, 2));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3_0, svget4(vb_x4_1, 3));
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va3_0, svget4(vb_x4_3, 2));
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va3_0, svget4(vb_x4_3, 3));
      }

      const svint32x2_t vksum_01 =
          svld1_s32_x2(pcnt_32_all, (const int32_t*)w0);
      w0 += nr * sizeof(int32_t);
      const svint32x2_t vksum_23 =
          svld1_s32_x2(pcnt_32_all, (const int32_t*)w1);
      w1 += nr * sizeof(int32_t);

      const svfloat32x2_t vfilter_01 =
          svld1_f32_x2(pcnt_32_all, (const float*)w0);
      w0 += nr * sizeof(float);
      const svfloat32x2_t vfilter_23 =
          svld1_f32_x2(pcnt_32_all, (const float*)w1);
      w1 += nr * sizeof(float);

      const svfloat32x2_t vbias_01 =
          svld1_f32_x2(pcnt_32_all, (const float*)w0);
      w0 += nr * sizeof(float);
      const svfloat32x2_t vbias_23 =
          svld1_f32_x2(pcnt_32_all, (const float*)w1);
      w1 += nr * sizeof(float);

      const svint32_t vksum_0 = svget2(vksum_01, 0);
      const svint32_t vksum_1 = svget2(vksum_01, 1);
      const svint32_t vksum_2 = svget2(vksum_23, 0);
      const svint32_t vksum_3 = svget2(vksum_23, 1);

      const svfloat32_t vfilter_0 = svget2(vfilter_01, 0);
      const svfloat32_t vfilter_1 = svget2(vfilter_01, 1);
      const svfloat32_t vfilter_2 = svget2(vfilter_23, 0);
      const svfloat32_t vfilter_3 = svget2(vfilter_23, 1);

      const svfloat32_t vbias_0 = svget2(vbias_01, 0);
      const svfloat32_t vbias_1 = svget2(vbias_01, 1);
      const svfloat32_t vbias_2 = svget2(vbias_23, 0);
      const svfloat32_t vbias_3 = svget2(vbias_23, 1);

      const svcount_t pcnt_32_nc = svwhilelt_c32_u64(0, nc, 4);
      float* c_inc = c0;

      for (size_t r = 0; r < mr; ++r) {
        const int32_t zp_r = a_zp[r];
        const float inv_scale_r = a_inv_scale[r];

        const svint32x4_t vacc_x4 =
            svreinterpret_s32(svread_hor_za8_u8_vg4(0, r * 4));
        const svint32_t vacc_0 =
            svmla_n_s32_x(pg_32_all, svget4(vacc_x4, 0), vksum_0, zp_r);
        const svint32_t vacc_1 =
            svmla_n_s32_x(pg_32_all, svget4(vacc_x4, 1), vksum_1, zp_r);
        const svint32_t vacc_2 =
            svmla_n_s32_x(pg_32_all, svget4(vacc_x4, 2), vksum_2, zp_r);
        const svint32_t vacc_3 =
            svmla_n_s32_x(pg_32_all, svget4(vacc_x4, 3), vksum_3, zp_r);

        const svfloat32x4_t vout_x4 =
            svcvt_f32(svcreate4(vacc_0, vacc_1, vacc_2, vacc_3));
        const svfloat32_t vout_0 = svmad_f32_m(
            pg_32_all, svget4(vout_x4, 0),
            svmul_n_f32_x(pg_32_all, vfilter_0, inv_scale_r), vbias_0);
        const svfloat32_t vout_1 = svmad_f32_m(
            pg_32_all, svget4(vout_x4, 1),
            svmul_n_f32_x(pg_32_all, vfilter_1, inv_scale_r), vbias_1);
        const svfloat32_t vout_2 = svmad_f32_m(
            pg_32_all, svget4(vout_x4, 2),
            svmul_n_f32_x(pg_32_all, vfilter_2, inv_scale_r), vbias_2);
        const svfloat32_t vout_3 = svmad_f32_m(
            pg_32_all, svget4(vout_x4, 3),
            svmul_n_f32_x(pg_32_all, vfilter_3, inv_scale_r), vbias_3);

        const svfloat32x4_t vout0123 =
            svclamp(svcreate4(vout_0, vout_1, vout_2, vout_3), voutput_min,
                    voutput_max);
        svst1_f32_x4(pcnt_32_nc, c_inc, vout0123);
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
      }

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

      svzero_za();

      for (size_t k = k_rup; k != 0; k -= 16) {
        const svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w0);
        const svint8x4_t vb_x4_1 = svld1_s8_x4(pcnt_8_all, w0 + 4 * svl_b);
        w0 += 8 * svl_b;

        const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
        const svint8_t va1_0 = svld1_vnum_s8(pg_8_all, a_inc, 2);
        const svint8_t va2_0 = svld1_vnum_s8(pg_8_all, a_inc, 4);
        const svint8_t va3_0 = svld1_vnum_s8(pg_8_all, a_inc, 6);
        a_inc += 8 * svl_b;

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, svget4(vb_x4_0, 0));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0_0, svget4(vb_x4_0, 1));
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va1_0, svget4(vb_x4_0, 2));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1_0, svget4(vb_x4_0, 3));
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2_0, svget4(vb_x4_1, 0));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va2_0, svget4(vb_x4_1, 1));
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va3_0, svget4(vb_x4_1, 2));
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3_0, svget4(vb_x4_1, 3));
      }

      const svint32x2_t vksum_01 =
          svld1_s32_x2(pcnt_32_all, (const int32_t*)w0);
      w0 += nr * sizeof(int32_t);
      const svfloat32x2_t vfilter_01 =
          svld1_f32_x2(pcnt_32_all, (const float*)w0);
      w0 += nr * sizeof(float);
      const svfloat32x2_t vbias_01 =
          svld1_f32_x2(pcnt_32_all, (const float*)w0);

      const svint32_t vksum_0 = svget2(vksum_01, 0);
      const svint32_t vksum_1 = svget2(vksum_01, 1);
      const svfloat32_t vfilter_0 = svget2(vfilter_01, 0);
      const svfloat32_t vfilter_1 = svget2(vfilter_01, 1);
      const svfloat32_t vbias_0 = svget2(vbias_01, 0);
      const svfloat32_t vbias_1 = svget2(vbias_01, 1);

      const svcount_t pcnt_32_nc = svwhilelt_c32_u64(0, nc, 2);
      float* c_inc = c0;

      for (size_t r = 0; r < mr; ++r) {
        const int32_t zp_r = a_zp[r];
        const float inv_scale_r = a_inv_scale[r];

        const svint32x4_t vacc_x4 =
            svreinterpret_s32(svread_hor_za8_u8_vg4(0, r * 4));
        const svint32_t vacc_0 =
            svmla_n_s32_x(pg_32_all, svget4(vacc_x4, 0), vksum_0, zp_r);
        const svint32_t vacc_1 =
            svmla_n_s32_x(pg_32_all, svget4(vacc_x4, 1), vksum_1, zp_r);

        const svfloat32x2_t vout_x2 = svcvt_f32(svcreate2(vacc_0, vacc_1));
        const svfloat32_t vout_0 = svmad_f32_m(
            pg_32_all, svget2(vout_x2, 0),
            svmul_n_f32_x(pg_32_all, vfilter_0, inv_scale_r), vbias_0);
        const svfloat32_t vout_1 = svmad_f32_m(
            pg_32_all, svget2(vout_x2, 1),
            svmul_n_f32_x(pg_32_all, vfilter_1, inv_scale_r), vbias_1);

        const svfloat32x2_t vout01 =
            svclamp(svcreate2(vout_0, vout_1), voutput_min, voutput_max);
        svst1_f32_x2(pcnt_32_nc, c_inc, vout01);
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
      }
    }
  } else {
    const int8_t* w_ptr = (const int8_t*)w;
    const svbool_t pg_4 = svwhilelt_b32_u64(0, 4);

    do {
      const int8_t* a_inc = a0;

      svzero_za();

      for (size_t k = k_rup; k != 0; k -= 16) {
        const svint8x4_t va_x4_0 = svld1_s8_x4(pcnt_8_all, a_inc);
        a_inc += 4 * svl_b;
        const svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w_ptr);
        w_ptr += 4 * svl_b;

        const svint8x4_t va_x4_1 = svld1_s8_x4(pcnt_8_all, a_inc);
        a_inc += 4 * svl_b;
        const svint8x4_t vb_x4_1 = svld1_s8_x4(pcnt_8_all, w_ptr);
        w_ptr += 4 * svl_b;

        const svint8_t va0_0 = svget4(va_x4_0, 0);
        const svint8_t va0_1 = svget4(va_x4_0, 1);
        const svint8_t va1_0 = svget4(va_x4_0, 2);
        const svint8_t va1_1 = svget4(va_x4_0, 3);
        const svint8_t va2_0 = svget4(va_x4_1, 0);
        const svint8_t va2_1 = svget4(va_x4_1, 1);
        const svint8_t va3_0 = svget4(va_x4_1, 2);
        const svint8_t va3_1 = svget4(va_x4_1, 3);

        const svint8_t vb0_0 = svget4(vb_x4_0, 0);
        const svint8_t vb0_1 = svget4(vb_x4_0, 1);
        const svint8_t vb1_0 = svget4(vb_x4_0, 2);
        const svint8_t vb1_1 = svget4(vb_x4_0, 3);
        const svint8_t vb2_0 = svget4(vb_x4_1, 0);
        const svint8_t vb2_1 = svget4(vb_x4_1, 1);
        const svint8_t vb3_0 = svget4(vb_x4_1, 2);
        const svint8_t vb3_1 = svget4(vb_x4_1, 3);

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, vb0_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0_0, vb0_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0_1, vb0_0);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va0_1, vb0_1);
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va1_0, vb1_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1_0, vb1_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va1_1, vb1_0);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va1_1, vb1_1);
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2_0, vb2_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va2_0, vb2_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va2_1, vb2_0);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va2_1, vb2_1);
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va3_0, vb3_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3_0, vb3_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va3_1, vb3_0);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va3_1, vb3_1);
      }

      const svint32x2_t vksum_lr =
          svld1_s32_x2(pcnt_32_all, (const int32_t*)w_ptr);
      w_ptr += nr * sizeof(int32_t);
      const svfloat32x2_t vfilter_lr =
          svld1_f32_x2(pcnt_32_all, (const float*)w_ptr);
      w_ptr += nr * sizeof(float);
      const svfloat32x2_t vbias_lr =
          svld1_f32_x2(pcnt_32_all, (const float*)w_ptr);
      w_ptr += nr * sizeof(float);

      const svint32_t vksum_l = svget2(vksum_lr, 0);
      const svint32_t vksum_r = svget2(vksum_lr, 1);
      const svfloat32_t vfilter_l = svget2(vfilter_lr, 0);
      const svfloat32_t vfilter_r = svget2(vfilter_lr, 1);
      const svfloat32_t vbias_l = svget2(vbias_lr, 0);
      const svfloat32_t vbias_r = svget2(vbias_lr, 1);

      const svcount_t pn = svwhilelt_c32_u64(0, nc, 2);
      float* c_inc = c0;

      for (size_t i = 0; i < svl_w; i += 4) {
        const svint32_t vzp_0123 =
            svdupq_lane_s32(svld1_s32(pg_4, a_zp + i), 0);
        const svfloat32_t vinvs_0123 =
            svdupq_lane_f32(svld1_f32(pg_4, a_inv_scale + i), 0);

        const svint32x4_t za0_x4 = svread_hor_za32_s32_vg4(0, i);
        const svint32x4_t za1_x4 = svread_hor_za32_s32_vg4(1, i);

        const svint32_t za0_0 =
            svmla_lane_s32(svget4(za0_x4, 0), vksum_l, vzp_0123, 0);
        const svint32_t za0_1 =
            svmla_lane_s32(svget4(za0_x4, 1), vksum_l, vzp_0123, 1);
        const svint32_t za0_2 =
            svmla_lane_s32(svget4(za0_x4, 2), vksum_l, vzp_0123, 2);
        const svint32_t za0_3 =
            svmla_lane_s32(svget4(za0_x4, 3), vksum_l, vzp_0123, 3);

        const svint32_t za1_0 =
            svmla_lane_s32(svget4(za1_x4, 0), vksum_r, vzp_0123, 0);
        const svint32_t za1_1 =
            svmla_lane_s32(svget4(za1_x4, 1), vksum_r, vzp_0123, 1);
        const svint32_t za1_2 =
            svmla_lane_s32(svget4(za1_x4, 2), vksum_r, vzp_0123, 2);
        const svint32_t za1_3 =
            svmla_lane_s32(svget4(za1_x4, 3), vksum_r, vzp_0123, 3);

        svfloat32x4_t vout0_x4 =
            svcvt_f32(svcreate4(za0_0, za0_1, za0_2, za0_3));
        svfloat32x4_t vout1_x4 =
            svcvt_f32(svcreate4(za1_0, za1_1, za1_2, za1_3));

        const svfloat32_t vout0_0 =
            svmad_f32_m(pg_32_all, svget4(vout0_x4, 0),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 0), vbias_l);
        const svfloat32_t vout0_1 =
            svmad_f32_m(pg_32_all, svget4(vout0_x4, 1),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 1), vbias_l);
        const svfloat32_t vout0_2 =
            svmad_f32_m(pg_32_all, svget4(vout0_x4, 2),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 2), vbias_l);
        const svfloat32_t vout0_3 =
            svmad_f32_m(pg_32_all, svget4(vout0_x4, 3),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 3), vbias_l);

        const svfloat32_t vout1_0 =
            svmad_f32_m(pg_32_all, svget4(vout1_x4, 0),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 0), vbias_r);
        const svfloat32_t vout1_1 =
            svmad_f32_m(pg_32_all, svget4(vout1_x4, 1),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 1), vbias_r);
        const svfloat32_t vout1_2 =
            svmad_f32_m(pg_32_all, svget4(vout1_x4, 2),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 2), vbias_r);
        const svfloat32_t vout1_3 =
            svmad_f32_m(pg_32_all, svget4(vout1_x4, 3),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 3), vbias_r);

        vout0_x4 = svclamp(svcreate4(vout0_0, vout0_1, vout0_2, vout0_3),
                           voutput_min, voutput_max);
        vout1_x4 = svclamp(svcreate4(vout1_0, vout1_1, vout1_2, vout1_3),
                           voutput_min, voutput_max);

        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout0_x4, 0), svget4(vout1_x4, 0)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout0_x4, 1), svget4(vout1_x4, 1)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout0_x4, 2), svget4(vout1_x4, 2)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout0_x4, 3), svget4(vout1_x4, 3)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
      }

      const size_t m_rem = mr - svl_w;
      const size_t m_rem_vec4 = m_rem & ~3;
      for (size_t i = 0; i < m_rem_vec4; i += 4) {
        const svint32_t vzp_0123 =
            svdupq_lane_s32(svld1_s32(pg_4, a_zp + svl_w + i), 0);
        const svfloat32_t vinvs_0123 =
            svdupq_lane_f32(svld1_f32(pg_4, a_inv_scale + svl_w + i), 0);

        const svint32x4_t za2_x4 = svread_hor_za32_s32_vg4(2, i);
        const svint32x4_t za3_x4 = svread_hor_za32_s32_vg4(3, i);

        const svint32_t za2_0 =
            svmla_lane_s32(svget4(za2_x4, 0), vksum_l, vzp_0123, 0);
        const svint32_t za2_1 =
            svmla_lane_s32(svget4(za2_x4, 1), vksum_l, vzp_0123, 1);
        const svint32_t za2_2 =
            svmla_lane_s32(svget4(za2_x4, 2), vksum_l, vzp_0123, 2);
        const svint32_t za2_3 =
            svmla_lane_s32(svget4(za2_x4, 3), vksum_l, vzp_0123, 3);

        const svint32_t za3_0 =
            svmla_lane_s32(svget4(za3_x4, 0), vksum_r, vzp_0123, 0);
        const svint32_t za3_1 =
            svmla_lane_s32(svget4(za3_x4, 1), vksum_r, vzp_0123, 1);
        const svint32_t za3_2 =
            svmla_lane_s32(svget4(za3_x4, 2), vksum_r, vzp_0123, 2);
        const svint32_t za3_3 =
            svmla_lane_s32(svget4(za3_x4, 3), vksum_r, vzp_0123, 3);

        svfloat32x4_t vout2_x4 =
            svcvt_f32(svcreate4(za2_0, za2_1, za2_2, za2_3));
        svfloat32x4_t vout3_x4 =
            svcvt_f32(svcreate4(za3_0, za3_1, za3_2, za3_3));

        const svfloat32_t vout2_0 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 0),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 0), vbias_l);
        const svfloat32_t vout2_1 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 1),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 1), vbias_l);
        const svfloat32_t vout2_2 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 2),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 2), vbias_l);
        const svfloat32_t vout2_3 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 3),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 3), vbias_l);

        const svfloat32_t vout3_0 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 0),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 0), vbias_r);
        const svfloat32_t vout3_1 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 1),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 1), vbias_r);
        const svfloat32_t vout3_2 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 2),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 2), vbias_r);
        const svfloat32_t vout3_3 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 3),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 3), vbias_r);

        vout2_x4 = svclamp(svcreate4(vout2_0, vout2_1, vout2_2, vout2_3),
                           voutput_min, voutput_max);
        vout3_x4 = svclamp(svcreate4(vout3_0, vout3_1, vout3_2, vout3_3),
                           voutput_min, voutput_max);

        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout2_x4, 0), svget4(vout3_x4, 0)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout2_x4, 1), svget4(vout3_x4, 1)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout2_x4, 2), svget4(vout3_x4, 2)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout2_x4, 3), svget4(vout3_x4, 3)));
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
      }

      if (m_rem_vec4 < m_rem) {
        const size_t i = m_rem_vec4;
        const svint32_t vzp_0123 =
            svdupq_lane_s32(svld1_s32(pg_4, a_zp + svl_w + i), 0);
        const svfloat32_t vinvs_0123 =
            svdupq_lane_f32(svld1_f32(pg_4, a_inv_scale + svl_w + i), 0);

        const svint32x4_t za2_x4 = svread_hor_za32_s32_vg4(2, i);
        const svint32x4_t za3_x4 = svread_hor_za32_s32_vg4(3, i);

        const svint32_t za2_0 =
            svmla_lane_s32(svget4(za2_x4, 0), vksum_l, vzp_0123, 0);
        const svint32_t za2_1 =
            svmla_lane_s32(svget4(za2_x4, 1), vksum_l, vzp_0123, 1);
        const svint32_t za2_2 =
            svmla_lane_s32(svget4(za2_x4, 2), vksum_l, vzp_0123, 2);
        const svint32_t za2_3 =
            svmla_lane_s32(svget4(za2_x4, 3), vksum_l, vzp_0123, 3);

        const svint32_t za3_0 =
            svmla_lane_s32(svget4(za3_x4, 0), vksum_r, vzp_0123, 0);
        const svint32_t za3_1 =
            svmla_lane_s32(svget4(za3_x4, 1), vksum_r, vzp_0123, 1);
        const svint32_t za3_2 =
            svmla_lane_s32(svget4(za3_x4, 2), vksum_r, vzp_0123, 2);
        const svint32_t za3_3 =
            svmla_lane_s32(svget4(za3_x4, 3), vksum_r, vzp_0123, 3);

        svfloat32x4_t vout2_x4 =
            svcvt_f32(svcreate4(za2_0, za2_1, za2_2, za2_3));
        svfloat32x4_t vout3_x4 =
            svcvt_f32(svcreate4(za3_0, za3_1, za3_2, za3_3));

        const svfloat32_t vout2_0 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 0),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 0), vbias_l);
        const svfloat32_t vout2_1 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 1),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 1), vbias_l);
        const svfloat32_t vout2_2 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 2),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 2), vbias_l);
        const svfloat32_t vout2_3 =
            svmad_f32_m(pg_32_all, svget4(vout2_x4, 3),
                        svmul_lane_f32(vfilter_l, vinvs_0123, 3), vbias_l);

        const svfloat32_t vout3_0 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 0),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 0), vbias_r);
        const svfloat32_t vout3_1 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 1),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 1), vbias_r);
        const svfloat32_t vout3_2 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 2),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 2), vbias_r);
        const svfloat32_t vout3_3 =
            svmad_f32_m(pg_32_all, svget4(vout3_x4, 3),
                        svmul_lane_f32(vfilter_r, vinvs_0123, 3), vbias_r);

        vout2_x4 = svclamp(svcreate4(vout2_0, vout2_1, vout2_2, vout2_3),
                           voutput_min, voutput_max);
        vout3_x4 = svclamp(svcreate4(vout3_0, vout3_1, vout3_2, vout3_3),
                           voutput_min, voutput_max);

        svst1_f32_x2(pn, c_inc,
                     svcreate2(svget4(vout2_x4, 0), svget4(vout3_x4, 0)));
        if (i + 1 < m_rem) {
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride),
                       svcreate2(svget4(vout2_x4, 1), svget4(vout3_x4, 1)));
        }
        if (i + 2 < m_rem) {
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride),
                       svcreate2(svget4(vout2_x4, 2), svget4(vout3_x4, 2)));
        }
      }

      if (nc > nr) {
        c0 = (float*)((uintptr_t)c0 + cn_step);
        nc -= nr;
      } else {
        nc = 0;
      }
    } while (nc != 0);
  }
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
