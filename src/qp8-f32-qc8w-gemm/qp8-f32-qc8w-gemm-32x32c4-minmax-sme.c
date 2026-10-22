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

__arm_locally_streaming size_t
xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme_get_mr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming size_t
xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme(size_t mr,
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
  const svint32_t vzero = svdup_n_s32(0);
  const svfloat32_t voutput_min = svdup_n_f32(params->scalar.min);
  const svfloat32_t voutput_max = svdup_n_f32(params->scalar.max);

  if (mr <= svl_w) {
    const int8_t* w0 = (const int8_t*)w;

    while (nc > nr) {
      const int8_t* w1 = w0 + w_offset_nr;
      const int8_t* a_inc = a0;

      svzero_za();

      for (size_t k = k_rup; k != 0; k -= 16) {
        const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
        const svint8_t va1_0 = svld1_vnum_s8(pg_8_all, a_inc, 2);
        const svint8_t va2_0 = svld1_vnum_s8(pg_8_all, a_inc, 4);
        const svint8_t va3_0 = svld1_vnum_s8(pg_8_all, a_inc, 6);
        a_inc += 8 * svl_b;

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

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, vb0_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0_0, vb0_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0_0, vb0_2);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va0_0, vb0_3);

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va1_0, vb1_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1_0, vb1_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va1_0, vb1_2);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va1_0, vb1_3);

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2_0, vb2_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va2_0, vb2_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va2_0, vb2_2);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va2_0, vb2_3);

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va3_0, vb3_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3_0, vb3_1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va3_0, vb3_2);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va3_0, vb3_3);
      }

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

      const svbool_t pg_n2 = svwhilelt_b32_u64(2 * svl_w, nc);
      const svbool_t pg_n3 = svwhilelt_b32_u64(3 * svl_w, nc);
      float* c_inc = c0;

      for (size_t r = 0; r < mr; ++r) {
        const int32_t zp_r = a_zp[r];
        const float inv_scale_r = a_inv_scale[r];

        svint32_t vacc_0 = svread_hor_za32_s32_m(vzero, pg_32_all, 0, r);
        svint32_t vacc_1 = svread_hor_za32_s32_m(vzero, pg_32_all, 1, r);
        svint32_t vacc_2 = svread_hor_za32_s32_m(vzero, pg_32_all, 2, r);
        svint32_t vacc_3 = svread_hor_za32_s32_m(vzero, pg_32_all, 3, r);

        vacc_0 = svmla_n_s32_x(pg_32_all, vacc_0, vksum_0, zp_r);
        vacc_1 = svmla_n_s32_x(pg_32_all, vacc_1, vksum_1, zp_r);
        vacc_2 = svmla_n_s32_x(pg_32_all, vacc_2, vksum_2, zp_r);
        vacc_3 = svmla_n_s32_x(pg_32_all, vacc_3, vksum_3, zp_r);

        svfloat32_t vout_0 = svcvt_f32_s32_x(pg_32_all, vacc_0);
        svfloat32_t vout_1 = svcvt_f32_s32_x(pg_32_all, vacc_1);
        svfloat32_t vout_2 = svcvt_f32_s32_x(pg_32_all, vacc_2);
        svfloat32_t vout_3 = svcvt_f32_s32_x(pg_32_all, vacc_3);

        vout_0 = svmad_f32_x(pg_32_all, vout_0,
                             svmul_n_f32_x(pg_32_all, vfilter_0, inv_scale_r),
                             vbias_0);
        vout_1 = svmad_f32_x(pg_32_all, vout_1,
                             svmul_n_f32_x(pg_32_all, vfilter_1, inv_scale_r),
                             vbias_1);
        vout_2 = svmad_f32_x(pg_32_all, vout_2,
                             svmul_n_f32_x(pg_32_all, vfilter_2, inv_scale_r),
                             vbias_2);
        vout_3 = svmad_f32_x(pg_32_all, vout_3,
                             svmul_n_f32_x(pg_32_all, vfilter_3, inv_scale_r),
                             vbias_3);

        vout_0 =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_0, voutput_min),
                        voutput_max);
        vout_1 =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_1, voutput_min),
                        voutput_max);
        vout_2 =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_2, voutput_min),
                        voutput_max);
        vout_3 =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_3, voutput_min),
                        voutput_max);

        svst1_f32(pg_32_all, c_inc, vout_0);
        svst1_f32(pg_32_all, c_inc + svl_w, vout_1);
        svst1_f32(pg_n2, c_inc + 2 * svl_w, vout_2);
        svst1_f32(pg_n3, c_inc + 3 * svl_w, vout_3);
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
      }

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

      svzero_za();

      for (size_t k = k_rup; k != 0; k -= 16) {
        const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
        const svint8_t va1_0 = svld1_vnum_s8(pg_8_all, a_inc, 2);
        const svint8_t va2_0 = svld1_vnum_s8(pg_8_all, a_inc, 4);
        const svint8_t va3_0 = svld1_vnum_s8(pg_8_all, a_inc, 6);
        a_inc += 8 * svl_b;

        const svint8_t vb0_0 = svld1_vnum_s8(pg_8_all, w0, 0);
        const svint8_t vb0_1 = svld1_vnum_s8(pg_8_all, w0, 1);
        const svint8_t vb1_0 = svld1_vnum_s8(pg_8_all, w0, 2);
        const svint8_t vb1_1 = svld1_vnum_s8(pg_8_all, w0, 3);
        const svint8_t vb2_0 = svld1_vnum_s8(pg_8_all, w0, 4);
        const svint8_t vb2_1 = svld1_vnum_s8(pg_8_all, w0, 5);
        const svint8_t vb3_0 = svld1_vnum_s8(pg_8_all, w0, 6);
        const svint8_t vb3_1 = svld1_vnum_s8(pg_8_all, w0, 7);
        w0 += 8 * svl_b;

        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, vb0_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0_0, vb0_1);
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va1_0, vb1_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1_0, vb1_1);
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2_0, vb2_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va2_0, vb2_1);
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va3_0, vb3_0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3_0, vb3_1);
      }

      const int32_t* w0_i32 = (const int32_t*)w0;
      const svint32_t vksum_0 = svld1_s32(pg_32_all, w0_i32);
      const svint32_t vksum_1 = svld1_s32(pg_32_all, w0_i32 + svl_w);

      const float* w0_f32 = (const float*)(w0_i32 + nr);
      const svfloat32_t vfilter_0 = svld1_f32(pg_32_all, w0_f32);
      const svfloat32_t vfilter_1 = svld1_f32(pg_32_all, w0_f32 + svl_w);

      const svfloat32_t vbias_0 = svld1_f32(pg_32_all, w0_f32 + nr);
      const svfloat32_t vbias_1 = svld1_f32(pg_32_all, w0_f32 + nr + svl_w);

      const svbool_t pg_n0 = svwhilelt_b32_u64(0, nc);
      const svbool_t pg_n1 = svwhilelt_b32_u64(svl_w, nc);
      float* c_inc = c0;

      for (size_t r = 0; r < mr; ++r) {
        const int32_t zp_r = a_zp[r];
        const float inv_scale_r = a_inv_scale[r];

        svint32_t vacc_0 = svread_hor_za32_s32_m(vzero, pg_32_all, 0, r);
        svint32_t vacc_1 = svread_hor_za32_s32_m(vzero, pg_32_all, 1, r);

        vacc_0 = svmla_n_s32_x(pg_32_all, vacc_0, vksum_0, zp_r);
        vacc_1 = svmla_n_s32_x(pg_32_all, vacc_1, vksum_1, zp_r);

        svfloat32_t vout_0 = svcvt_f32_s32_x(pg_32_all, vacc_0);
        svfloat32_t vout_1 = svcvt_f32_s32_x(pg_32_all, vacc_1);

        vout_0 = svmad_f32_x(pg_32_all, vout_0,
                             svmul_n_f32_x(pg_32_all, vfilter_0, inv_scale_r),
                             vbias_0);
        vout_1 = svmad_f32_x(pg_32_all, vout_1,
                             svmul_n_f32_x(pg_32_all, vfilter_1, inv_scale_r),
                             vbias_1);

        vout_0 =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_0, voutput_min),
                        voutput_max);
        vout_1 =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_1, voutput_min),
                        voutput_max);

        svst1_f32(pg_n0, c_inc, vout_0);
        svst1_f32(pg_n1, c_inc + svl_w, vout_1);
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
      }
    }
  } else {
    const int8_t* w_ptr = (const int8_t*)w;

    do {
      const int8_t* a_inc = a0;

      svzero_za();

      for (size_t k = k_rup; k != 0; k -= 16) {
        const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
        const svint8_t va0_1 = svld1_vnum_s8(pg_8_all, a_inc, 1);
        const svint8_t va1_0 = svld1_vnum_s8(pg_8_all, a_inc, 2);
        const svint8_t va1_1 = svld1_vnum_s8(pg_8_all, a_inc, 3);
        const svint8_t va2_0 = svld1_vnum_s8(pg_8_all, a_inc, 4);
        const svint8_t va2_1 = svld1_vnum_s8(pg_8_all, a_inc, 5);
        const svint8_t va3_0 = svld1_vnum_s8(pg_8_all, a_inc, 6);
        const svint8_t va3_1 = svld1_vnum_s8(pg_8_all, a_inc, 7);
        a_inc += 8 * svl_b;

        const svint8_t vb0_0 = svld1_vnum_s8(pg_8_all, w_ptr, 0);
        const svint8_t vb0_1 = svld1_vnum_s8(pg_8_all, w_ptr, 1);
        const svint8_t vb1_0 = svld1_vnum_s8(pg_8_all, w_ptr, 2);
        const svint8_t vb1_1 = svld1_vnum_s8(pg_8_all, w_ptr, 3);
        const svint8_t vb2_0 = svld1_vnum_s8(pg_8_all, w_ptr, 4);
        const svint8_t vb2_1 = svld1_vnum_s8(pg_8_all, w_ptr, 5);
        const svint8_t vb3_0 = svld1_vnum_s8(pg_8_all, w_ptr, 6);
        const svint8_t vb3_1 = svld1_vnum_s8(pg_8_all, w_ptr, 7);
        w_ptr += 8 * svl_b;

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

      const int32_t* w_i32 = (const int32_t*)w_ptr;
      const svint32_t vksum_l = svld1_s32(pg_32_all, w_i32);
      const svint32_t vksum_r = svld1_s32(pg_32_all, w_i32 + svl_w);

      const float* w_f32 = (const float*)(w_i32 + nr);
      const svfloat32_t vfilter_l = svld1_f32(pg_32_all, w_f32);
      const svfloat32_t vfilter_r = svld1_f32(pg_32_all, w_f32 + svl_w);

      const svfloat32_t vbias_l = svld1_f32(pg_32_all, w_f32 + nr);
      const svfloat32_t vbias_r = svld1_f32(pg_32_all, w_f32 + nr + svl_w);
      w_ptr = (const int8_t*)(w_f32 + 2 * nr);

      const svbool_t pg_n0 = svwhilelt_b32_u64(0, nc);
      const svbool_t pg_n1 = svwhilelt_b32_u64(svl_w, nc);
      float* c_inc = c0;

      for (size_t r = 0; r < svl_w; ++r) {
        const int32_t zp_r = a_zp[r];
        const float inv_scale_r = a_inv_scale[r];

        svint32_t vacc_l = svread_hor_za32_s32_m(vzero, pg_32_all, 0, r);
        svint32_t vacc_r = svread_hor_za32_s32_m(vzero, pg_32_all, 1, r);

        vacc_l = svmla_n_s32_x(pg_32_all, vacc_l, vksum_l, zp_r);
        vacc_r = svmla_n_s32_x(pg_32_all, vacc_r, vksum_r, zp_r);

        svfloat32_t vout_l = svcvt_f32_s32_x(pg_32_all, vacc_l);
        svfloat32_t vout_r = svcvt_f32_s32_x(pg_32_all, vacc_r);

        vout_l = svmad_f32_x(pg_32_all, vout_l,
                             svmul_n_f32_x(pg_32_all, vfilter_l, inv_scale_r),
                             vbias_l);
        vout_r = svmad_f32_x(pg_32_all, vout_r,
                             svmul_n_f32_x(pg_32_all, vfilter_r, inv_scale_r),
                             vbias_r);

        vout_l =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_l, voutput_min),
                        voutput_max);
        vout_r =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_r, voutput_min),
                        voutput_max);

        svst1_f32(pg_n0, c_inc, vout_l);
        svst1_f32(pg_n1, c_inc + svl_w, vout_r);
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
      }

      const size_t m_rem = mr - svl_w;
      for (size_t r = 0; r < m_rem; ++r) {
        const int32_t zp_r = a_zp[svl_w + r];
        const float inv_scale_r = a_inv_scale[svl_w + r];

        svint32_t vacc_l = svread_hor_za32_s32_m(vzero, pg_32_all, 2, r);
        svint32_t vacc_r = svread_hor_za32_s32_m(vzero, pg_32_all, 3, r);

        vacc_l = svmla_n_s32_x(pg_32_all, vacc_l, vksum_l, zp_r);
        vacc_r = svmla_n_s32_x(pg_32_all, vacc_r, vksum_r, zp_r);

        svfloat32_t vout_l = svcvt_f32_s32_x(pg_32_all, vacc_l);
        svfloat32_t vout_r = svcvt_f32_s32_x(pg_32_all, vacc_r);

        vout_l = svmad_f32_x(pg_32_all, vout_l,
                             svmul_n_f32_x(pg_32_all, vfilter_l, inv_scale_r),
                             vbias_l);
        vout_r = svmad_f32_x(pg_32_all, vout_r,
                             svmul_n_f32_x(pg_32_all, vfilter_r, inv_scale_r),
                             vbias_r);

        vout_l =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_l, voutput_min),
                        voutput_max);
        vout_r =
            svmin_f32_x(pg_32_all, svmax_f32_x(pg_32_all, vout_r, voutput_min),
                        voutput_max);

        svst1_f32(pg_n0, c_inc, vout_l);
        svst1_f32(pg_n1, c_inc + svl_w, vout_r);
        c_inc = (float*)((uintptr_t)c_inc + cm_stride);
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
#endif  // XNN_ENABLE_ARM_SME && XNN_ARCH_ARM64
