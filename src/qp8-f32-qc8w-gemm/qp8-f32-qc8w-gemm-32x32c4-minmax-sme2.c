// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2_ACLE
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme2(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params)
{
  const size_t vl = svcntw();
  const size_t nr = 2 * vl;
  const size_t mr_packed = 2 * vl;
  const size_t kloop_bytes = 16 * vl;
  const struct xnn_f32_minmax_params* minmax_params =
      (const struct xnn_f32_minmax_params*) params;

  kc = ((kc+31)/32) * 32;
  cn_stride = nr*cn_stride;

  const int8_t* a0 = (const int8_t*) a;
  float* c0 = c;

  svbool_t pg_8_all    = svptrue_b8();
  svbool_t pg_32_all   = svptrue_b32();

  if (mr<=vl) {

      size_t k_rup_32 = ((kc+31)/32)*32;
      size_t w_offset_nr = nr*(k_rup_32/1 + 12);
      const void* w0 = w;
      const void* w1 = (const int8_t*) w + w_offset_nr;

  #ifdef PREFETCH_W_MATRIX
      const size_t kPrefetchDistance = 24;
      const size_t kBytesPerKLoop = 2 * kloop_bytes;
      const size_t kCacheLineSize = 64;
      const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
      const size_t kPrfmStride = kBytesPerKLoop / 8;
      const size_t kTrailerPrfmStride = 2 * nr;
      size_t block_size = w_offset_nr;
      size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
      size_t total_w_size = (kc * nc_rup_nr) + nc_rup_nr * (4*3);
      bool need_prfm_0 = (block_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > (128*1024));
  #endif

      do {
        svzero_za();

  #ifdef PREFETCH_W_MATRIX
        bool need_prfm_1 = (nc > nr) && need_prfm_0;
        size_t prfm_acc_w_size_0 = 0;
        size_t prfm_acc_w_size_1 = 0;
        const int8_t* prfm_ptr_0 = (const int8_t*)w0;
        const int8_t* prfm_ptr_1 = (const int8_t*)w1;

        if (need_prfm_0) {
          prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
          int i = 0;
          for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
            __builtin_prefetch((const int8_t*)w0 + 64 * i, 0, 2);
          }
          prfm_ptr_0 = (const int8_t*)w0 + 64 * i;
        }
        if (need_prfm_1) {
          prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
          int i = 0;
          for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
            __builtin_prefetch((const int8_t*)w1 + 64 * i, 0, 2);
          }
          prfm_ptr_1 = (const int8_t*)w1 + 64 * i;
        }
  #endif
        size_t k = kc; //always roundup(kc, 32);
        const int8_t* a_inc = a0;

        while (k >= 16 * sizeof(int8_t)) {

          svcount_t pcnt_8_all = svptrue_c8();
          svcount_t pcnt_8_next_nr32 = (nc > nr) ? svptrue_c8() : svpfalse_c();
          //load w matrix
          svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w0);  w0 = (const int8_t*) w0 + kloop_bytes;
          svint8x4_t vb_x4_1 = svld1_s8_x4(pcnt_8_all, w0);  w0 = (const int8_t*) w0 + kloop_bytes;
          svint8x4_t vb_x4_2 = svld1_s8_x4(pcnt_8_next_nr32, w1);  w1 = (const int8_t*) w1 + kloop_bytes;
          svint8x4_t vb_x4_3 = svld1_s8_x4(pcnt_8_next_nr32, w1);  w1 = (const int8_t*) w1 + kloop_bytes;

          //load a matrix
          svint8x4_t va_x4_0 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += kloop_bytes;
          svint8x4_t va_x4_1 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += kloop_bytes;

  #ifdef PREFETCH_W_MATRIX
          if (need_prfm_0 && (prfm_acc_w_size_0 < (block_size - kBytesPerKLoop))) {
            prfm_acc_w_size_0 += kBytesPerKLoop;
            __builtin_prefetch(prfm_ptr_0 + 0 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_0 + 1 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_0 + 2 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_0 + 3 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_0 + 4 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_0 + 5 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_0 + 6 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_0 + 7 * kPrfmStride, 0, 2);
            prfm_ptr_0 += kBytesPerKLoop;
          }
          if (need_prfm_1 && (prfm_acc_w_size_1 < (block_size - kBytesPerKLoop))) {
            prfm_acc_w_size_1 += kBytesPerKLoop;
            __builtin_prefetch(prfm_ptr_1 + 0 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_1 + 1 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_1 + 2 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_1 + 3 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_1 + 4 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_1 + 5 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_1 + 6 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_1 + 7 * kPrfmStride, 0, 2);
            prfm_ptr_1 += kBytesPerKLoop;
          }
  #endif

          svint8_t va0_0 = svget4(va_x4_0, 0);
          svint8_t va1_0 = svget4(va_x4_0, 2);
          svint8_t va2_0 = svget4(va_x4_1, 0);
          svint8_t va3_0 = svget4(va_x4_1, 2);

          svint8_t vb0_0 = svget4(vb_x4_0, 0);
          svint8_t vb0_1 = svget4(vb_x4_0, 1);
          svint8_t vb1_0 = svget4(vb_x4_0, 2);
          svint8_t vb1_1 = svget4(vb_x4_0, 3);

          svint8_t vb2_0 = svget4(vb_x4_1, 0);
          svint8_t vb2_1 = svget4(vb_x4_1, 1);
          svint8_t vb3_0 = svget4(vb_x4_1, 2);
          svint8_t vb3_1 = svget4(vb_x4_1, 3);

          svint8_t vb0_2 = svget4(vb_x4_2, 0);
          svint8_t vb0_3 = svget4(vb_x4_2, 1);
          svint8_t vb1_2 = svget4(vb_x4_2, 2);
          svint8_t vb1_3 = svget4(vb_x4_2, 3);

          svint8_t vb2_2 = svget4(vb_x4_3, 0);
          svint8_t vb2_3 = svget4(vb_x4_3, 1);
          svint8_t vb3_2 = svget4(vb_x4_3, 2);
          svint8_t vb3_3 = svget4(vb_x4_3, 3);

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

          k -= 16 * sizeof(int8_t);
        }

        //Load 1. zero_point  2. inv_scale from A matrix
        //Load 3. ksum  4. filter/scale  5. bias from W matrix
        svbool_t pg_ksum = svptrue_b32(); //nr, int32_t
        svbool_t pg_filt = svptrue_b32(); //nr, float
        svbool_t pg_bias = svptrue_b32(); //nr, float

  #ifdef PREFETCH_W_MATRIX
        if (need_prfm_0 && (prfm_acc_w_size_0 <= (block_size - 12 * nr))) {
          prfm_acc_w_size_0 += 12 * nr;
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 4 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 5 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 12 * nr;
        }
        if (need_prfm_1 && (prfm_acc_w_size_1 <= (block_size - 12 * nr))) {
          prfm_acc_w_size_1 += 12 * nr;
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 4 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 5 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 12 * nr;
        }
  #endif

        // Load ksum nr=0~63
        svint32_t vksum_0 = svld1_s32(pg_ksum, w0); w0 = (const int32_t*) w0 + vl;
        svint32_t vksum_1 = svld1_s32(pg_ksum, w0); w0 = (const int32_t*) w0 + vl;
        svint32_t vksum_2 = svdup_s32(0);
        svint32_t vksum_3 = svdup_s32(0);
        if (nc > nr) {
          vksum_2 = svld1_s32(pg_ksum, w1); w1 = (const int32_t*) w1 + vl;
          vksum_3 = svld1_s32(pg_ksum, w1); w1 = (const int32_t*) w1 + vl;
        }
        // Load kernel/filter nr=0~63
        svfloat32_t vfilter_0 = svreinterpret_f32_s32(svld1_s32(pg_filt, w0)); w0 = (const int32_t*) w0 + vl;
        svfloat32_t vfilter_1 = svreinterpret_f32_s32(svld1_s32(pg_filt, w0)); w0 = (const int32_t*) w0 + vl;
        svfloat32_t vfilter_2 = svdup_f32(0.0f);
        svfloat32_t vfilter_3 = svdup_f32(0.0f);
        if (nc > nr) {
          vfilter_2 = svreinterpret_f32_s32(svld1_s32(pg_filt, w1)); w1 = (const int32_t*) w1 + vl;
          vfilter_3 = svreinterpret_f32_s32(svld1_s32(pg_filt, w1)); w1 = (const int32_t*) w1 + vl;
        }
        // Load bias nr=0~63
        svfloat32_t vbias_0 = svreinterpret_f32_s32(svld1_s32(pg_bias, w0)); w0 = (const int32_t*) w0 + vl;
        svfloat32_t vbias_1 = svreinterpret_f32_s32(svld1_s32(pg_bias, w0)); w0 = (const int32_t*) w0 + vl;

        svfloat32_t vbias_2 = svdup_f32(0.0f);
        svfloat32_t vbias_3 = svdup_f32(0.0f);
        if (nc > nr) {
          vbias_2 = svreinterpret_f32_s32(svld1_s32(pg_bias, w1)); w1 = (const int32_t*) w1 + vl;
          vbias_3 = svreinterpret_f32_s32(svld1_s32(pg_bias, w1)); w1 = (const int32_t*) w1 + vl;
        }

        const int8_t* a_inc_zp  = a0 + mr_packed * kc;
        const int8_t* a_inc_inv = a_inc_zp + mr_packed * sizeof(int32_t);
        const int8_t* a_inc_zp0;
        const int8_t* a_inc_inv0;

        svfloat32_t voutput_min = svdup_n_f32(minmax_params->scalar.min);
        svfloat32_t voutput_max = svdup_n_f32(minmax_params->scalar.max);

        size_t si_rup_2 = (mr + 1) / 2;
        for (size_t si=0; si < si_rup_2; si++) { // process 2 rows per iteration

          // mr0~mr1, mr2~mr3, ...
          a_inc_zp0 = a_inc_zp + (si * 2 * sizeof(int32_t));
          a_inc_inv0 = a_inc_inv + (si * 2 * sizeof(int32_t));

          svbool_t pg_row = svwhilelt_b32_s32(0, mr - si * 2);

          svint32_t vinput_zero_point_0 = svld1_s32(pg_row, (const int32_t*)a_inc_zp0);
          svint32_t vzp_01_0 = svdupq_lane_s32(vinput_zero_point_0, 0); // [zp0, zp1, X, X, ...]

          svfloat32_t vinv_scale_0 = svreinterpret_f32_s32( svld1_s32(pg_row, (const int32_t*)a_inc_inv0) );
          svfloat32_t vinvs_01_0 = svdupq_lane_f32(vinv_scale_0, 0); // [inv_s0, inv_s1, X, X, ...]

          // za0
          svint32x2_t za0_vacco_x2 = svread_hor_za32_s32_vg2(0, si * 2);
          svint32_t za0_vout0 = svget2(za0_vacco_x2, 0);
          svint32_t za0_vout1 = svget2(za0_vacco_x2, 1);

          svfloat32_t vscale0 = svmul_lane_f32(vfilter_0, vinvs_01_0, 0);
          za0_vout0 = svmla_lane_s32(za0_vout0, vksum_0, vzp_01_0, 0);

          svfloat32_t vscale1 = svmul_lane_f32(vfilter_0, vinvs_01_0, 1);
          za0_vout1 = svmla_lane_s32(za0_vout1, vksum_0, vzp_01_0, 1);

          svfloat32x2_t vout_x2 = svcvt_f32(svcreate2(za0_vout0, za0_vout1));

          svfloat32_t vout0_0 = svget2(vout_x2, 0);
          svfloat32_t vout1_0 = svget2(vout_x2, 1);

          vout0_0 = svmad_f32_m(pg_32_all, vout0_0, vscale0, vbias_0);
          vout1_0 = svmad_f32_m(pg_32_all, vout1_0, vscale1, vbias_0);

          vout_x2 = svcreate2(vout0_0, vout1_0);
          svwrite_hor_za32_f32_vg2(0, si * 2, vout_x2);

          // za1
          svint32x2_t za1_vacco_x2 = svread_hor_za32_s32_vg2(1, si * 2);
          svint32_t za1_vout0 = svget2(za1_vacco_x2, 0);
          svint32_t za1_vout1 = svget2(za1_vacco_x2, 1);

          vscale0 = svmul_lane_f32(vfilter_1, vinvs_01_0, 0);
          za1_vout0 = svmla_lane_s32(za1_vout0, vksum_1, vzp_01_0, 0);

          vscale1 = svmul_lane_f32(vfilter_1, vinvs_01_0, 1);
          za1_vout1 = svmla_lane_s32(za1_vout1, vksum_1, vzp_01_0, 1);

          vout_x2 = svcvt_f32(svcreate2(za1_vout0, za1_vout1));

          svfloat32_t vout0_1 = svget2(vout_x2, 0);
          svfloat32_t vout1_1 = svget2(vout_x2, 1);

          vout0_1 = svmad_f32_m(pg_32_all, vout0_1, vscale0, vbias_1);
          vout1_1 = svmad_f32_m(pg_32_all, vout1_1, vscale1, vbias_1);

          vout_x2 = svcreate2(vout0_1, vout1_1);
          svwrite_hor_za32_f32_vg2(1, si * 2, vout_x2);

          // za2,3
          if (nc > nr) {
            // za2
            svint32x2_t za2_vacco_x2 = svread_hor_za32_s32_vg2(2, si * 2);
            svint32_t za2_vout0 = svget2(za2_vacco_x2, 0);
            svint32_t za2_vout1 = svget2(za2_vacco_x2, 1);

            vscale0 = svmul_lane_f32(vfilter_2, vinvs_01_0, 0);
            za2_vout0 = svmla_lane_s32(za2_vout0, vksum_2, vzp_01_0, 0);

            vscale1 = svmul_lane_f32(vfilter_2, vinvs_01_0, 1);
            za2_vout1 = svmla_lane_s32(za2_vout1, vksum_2, vzp_01_0, 1);

            vout_x2 = svcvt_f32(svcreate2(za2_vout0, za2_vout1));

            svfloat32_t vout0_2 = svget2(vout_x2, 0);
            svfloat32_t vout1_2 = svget2(vout_x2, 1);

            vout0_2 = svmad_f32_m(pg_32_all, vout0_2, vscale0, vbias_2);
            vout1_2 = svmad_f32_m(pg_32_all, vout1_2, vscale1, vbias_2);

            vout_x2 = svcreate2(vout0_2, vout1_2);
            svwrite_hor_za32_f32_vg2(2, si * 2, vout_x2);

            // za3
            svint32x2_t za3_vacco_x2 = svread_hor_za32_s32_vg2(3, si * 2);
            svint32_t za3_vout0 = svget2(za3_vacco_x2, 0);
            svint32_t za3_vout1 = svget2(za3_vacco_x2, 1);

            vscale0 = svmul_lane_f32(vfilter_3, vinvs_01_0, 0);
            za3_vout0 = svmla_lane_s32(za3_vout0, vksum_3, vzp_01_0, 0);

            vscale1 = svmul_lane_f32(vfilter_3, vinvs_01_0, 1);
            za3_vout1 = svmla_lane_s32(za3_vout1, vksum_3, vzp_01_0, 1);

            vout_x2 = svcvt_f32(svcreate2(za3_vout0, za3_vout1));

            svfloat32_t vout0_3 = svget2(vout_x2, 0);
            svfloat32_t vout1_3 = svget2(vout_x2, 1);

            vout0_3 = svmad_f32_m(pg_32_all, vout0_3, vscale0, vbias_3);
            vout1_3 = svmad_f32_m(pg_32_all, vout1_3, vscale1, vbias_3);

            vout_x2 = svcreate2(vout0_3, vout1_3);
            svwrite_hor_za32_f32_vg2(3, si * 2, vout_x2);
          }
        }

        float* c_inc = c0;

        size_t n_len = (nc >= 2 * nr) ? 2 * nr : nc;

        if (XNN_LIKELY(nc != 0)) {

          svcount_t pcnt_32_len_x4 = svwhilelt_c32_u64(0, n_len, 4);

          size_t mr_idx = 0;
          for (size_t i = 0; i < mr; i++) {
            svfloat32x4_t za0x4 = svreinterpret_f32( svread_hor_za8_u8_vg4(0, mr_idx) );
            za0x4 = svclamp(za0x4, voutput_min, voutput_max);
            svst1_f32_x4(pcnt_32_len_x4,  c_inc, za0x4);
            c_inc = (float*) ((uintptr_t)c_inc + cm_stride);
            mr_idx += 4;
          }

          if (nc >= 2 * nr) {
              a_inc = a0;
              //Adjust pointers - set next output pointers
              c0 = (float*)((uintptr_t)c0 + cn_stride*2);
              w0 = (const int8_t*) w0 + w_offset_nr;
              w1 = (const int8_t*) w1 + w_offset_nr;
              nc -= 2 * nr; // start next Nr loop

          } else {
              nc = 0; //finish kernel
          }

        } //store


      } while (nc != 0);


  } else { //mr>vl

  #ifdef PREFETCH_W_MATRIX
    size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
    size_t total_w_size = (kc * nc_rup_nr) + nc_rup_nr * (4*3);
    bool need_prfm = (total_w_size > (128*1024));
    const size_t kPrfmStride = (2 * kloop_bytes) / 8;
    const size_t kTrailerPrfmStride = 2 * nr;
    size_t prfm_acc_w_size = 0;
    const int8_t* prfm_ptr = (const int8_t*)w;

    if (need_prfm) {
      prfm_acc_w_size = (24*8)*kPrfmStride;
      int i = 0;
      for (; i < (24*8); i=i+8) { // 24 means prefetch 24 k-loops w, 8 means 8*64=512 bytes w per k-loop
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+0), 0, 2);
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+1), 0, 2);
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+2), 0, 2);
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+3), 0, 2);
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+4), 0, 2);
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+5), 0, 2);
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+6), 0, 2);
        __builtin_prefetch((const int8_t*)w+kPrfmStride*(i+7), 0, 2);
      }
      prfm_ptr = (const int8_t*)w+kPrfmStride*i;
    }
  #endif

    do {

      svzero_za();

      size_t k = kc; //always roundup(kc, 32);
      const int8_t* a_inc = a0;

      while (k >= 16 * sizeof(int8_t)) {

        svcount_t pcnt_8_all = svptrue_c8();

        //load a matrix
        svint8x4_t va_x4_0 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += kloop_bytes;
        svint8_t va0_0 = svget4(va_x4_0, 0);
        svint8_t va0_1 = svget4(va_x4_0, 1);
        svint8_t va1_0 = svget4(va_x4_0, 2);
        svint8_t va1_1 = svget4(va_x4_0, 3);

        //load w matrix
        svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w);  w = (const int8_t*) w + kloop_bytes;
        svint8_t vb0_0 = svget4(vb_x4_0, 0);
        svint8_t vb0_1 = svget4(vb_x4_0, 1);
        svint8_t vb1_0 = svget4(vb_x4_0, 2);
        svint8_t vb1_1 = svget4(vb_x4_0, 3);

        //load a matrix
        svint8x4_t va_x4_1 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += kloop_bytes;
        svint8_t va2_0 = svget4(va_x4_1, 0);
        svint8_t va2_1 = svget4(va_x4_1, 1);
        svint8_t va3_0 = svget4(va_x4_1, 2);
        svint8_t va3_1 = svget4(va_x4_1, 3);

        //load w matrix
        svint8x4_t vb_x4_1 = svld1_s8_x4(pcnt_8_all, w);  w = (const int8_t*) w + kloop_bytes;
        svint8_t vb2_0 = svget4(vb_x4_1, 0);
        svint8_t vb2_1 = svget4(vb_x4_1, 1);
        svint8_t vb3_0 = svget4(vb_x4_1, 2);
        svint8_t vb3_1 = svget4(vb_x4_1, 3);

  #ifdef PREFETCH_W_MATRIX
        if (need_prfm && (prfm_acc_w_size <= (total_w_size - 2 * kloop_bytes))) {
          prfm_acc_w_size += 2 * kloop_bytes;
          __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
          prfm_ptr += 2 * kloop_bytes;
        }
  #endif

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

        k -= 16 * sizeof(int8_t);
      }

      //Load 1. zero_point  2. inv_scale from A matrix
      //Load 3. ksum  4. filter/scale  5. bias from W matrix
      svbool_t pg_zp   = svwhilelt_b32_s32(0, 4); //mr, int32_t
      svbool_t pg_inv  = svwhilelt_b32_s32(0, 4); //mr, float
      svcount_t pcnt_x4 = svptrue_c32();
      svcount_t pcnt_x2 = svptrue_c32();

      svint32x2_t   vec_tmp_x2 = svld1_s32_x2(pcnt_x2, w); w = (const int32_t*) w + 2 * vl;
      svfloat32x4_t vec_tmp_x4 = svld1_f32_x4(pcnt_x4, w); w = (const int32_t*) w + 4 * vl;

      // Load ksum nr=0~31
      svint32_t vksum_l = svget2(vec_tmp_x2, 0);
      svint32_t vksum_r = svget2(vec_tmp_x2, 1);
      // Load kernel/filter nr=0~31
      svfloat32_t vfilter_l = svget4(vec_tmp_x4, 0);
      svfloat32_t vfilter_r = svget4(vec_tmp_x4, 1);
      // Load bias nr=0~31
      svfloat32_t vbias_l = svget4(vec_tmp_x4, 2);
      svfloat32_t vbias_r = svget4(vec_tmp_x4, 3);

  #ifdef PREFETCH_W_MATRIX
        if (need_prfm) {
          if (prfm_acc_w_size <= (total_w_size - 12 * nr)) {
            prfm_acc_w_size += 12 * nr;
            __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 2 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 3 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 4 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 5 * kTrailerPrfmStride, 0, 2);
            prfm_ptr += 12 * nr;
          } else if (prfm_acc_w_size <= (total_w_size - 8 * nr)) {
            prfm_acc_w_size += 8 * nr;
            __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 2 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 3 * kTrailerPrfmStride, 0, 2);
            prfm_ptr += 8 * nr;
          } else if (prfm_acc_w_size <= (total_w_size - 4 * nr)) {
            prfm_acc_w_size += 4 * nr;
            __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
            prfm_ptr += 4 * nr;
          }
        }
  #endif

      const int8_t* a_inc_zp  = a_inc;
      const int8_t* a_inc_inv = a_inc + mr_packed*sizeof(int32_t);
      const int8_t* a_inc_zp0;
      const int8_t* a_inc_zp1;
      const int8_t* a_inc_inv0;
      const int8_t* a_inc_inv1;

      size_t si_rup_4 = vl / 4;
      for (size_t si=0; si < si_rup_4; si++) {

        //za0,1
        a_inc_zp0 = a_inc_zp + ((0  + si * 4) * sizeof(int32_t));
        a_inc_inv0 = a_inc_inv + ((0  + si * 4) * sizeof(int32_t));

        svint32_t vinput_zero_point_0 = svld1_s32(pg_zp, (const int32_t*)a_inc_zp0);
        svint32_t vzp_0123_0 = svdupq_lane_s32(vinput_zero_point_0, 0);

        svfloat32_t vinv_scale_0 = svreinterpret_f32_s32( svld1_s32(pg_inv, (const int32_t*)a_inc_inv0) );
        svfloat32_t vinvs_0123_0 = svdupq_lane_f32(vinv_scale_0, 0);


        //za0
        svint32x4_t za0_vacco_x4_0 = svread_hor_za32_s32_vg4(0, 4*si);

        svfloat32_t vinv_filter_scale0 = svmul_lane_f32(vfilter_l, vinvs_0123_0, 0);
        svfloat32_t vinv_filter_scale1 = svmul_lane_f32(vfilter_l, vinvs_0123_0, 1);
        svfloat32_t vinv_filter_scale2 = svmul_lane_f32(vfilter_l, vinvs_0123_0, 2);
        svfloat32_t vinv_filter_scale3 = svmul_lane_f32(vfilter_l, vinvs_0123_0, 3);

        svint32_t za0_vout0 = svmla_lane_s32( svget4(za0_vacco_x4_0, 0), vksum_l, vzp_0123_0, 0);
        svint32_t za0_vout1 = svmla_lane_s32( svget4(za0_vacco_x4_0, 1), vksum_l, vzp_0123_0, 1);
        svint32_t za0_vout2 = svmla_lane_s32( svget4(za0_vacco_x4_0, 2), vksum_l, vzp_0123_0, 2);
        svint32_t za0_vout3 = svmla_lane_s32( svget4(za0_vacco_x4_0, 3), vksum_l, vzp_0123_0, 3);

        svfloat32x4_t vout0_x4_0 = svcvt_f32(svcreate4(za0_vout0, za0_vout1, za0_vout2, za0_vout3));
        svfloat32_t vout0_0 = svget4(vout0_x4_0, 0);
        svfloat32_t vout1_0 = svget4(vout0_x4_0, 1);
        svfloat32_t vout2_0 = svget4(vout0_x4_0, 2);
        svfloat32_t vout3_0 = svget4(vout0_x4_0, 3);

        vout0_0 = svmad_f32_m(pg_32_all, vout0_0, vinv_filter_scale0, vbias_l);
        vout1_0 = svmad_f32_m(pg_32_all, vout1_0, vinv_filter_scale1, vbias_l);
        vout2_0 = svmad_f32_m(pg_32_all, vout2_0, vinv_filter_scale2, vbias_l);
        vout3_0 = svmad_f32_m(pg_32_all, vout3_0, vinv_filter_scale3, vbias_l);

        svfloat32x4_t vout0123_0 = svcreate4(vout0_0, vout1_0, vout2_0, vout3_0);
        svwrite_hor_za32_f32_vg4(0, 4*si, vout0123_0);


        //za1
        svint32x4_t za1_vacco_x4_0 = svread_hor_za32_s32_vg4(1, 4*si);

        vinv_filter_scale0 = svmul_lane_f32(vfilter_r, vinvs_0123_0, 0);
        vinv_filter_scale1 = svmul_lane_f32(vfilter_r, vinvs_0123_0, 1);
        vinv_filter_scale2 = svmul_lane_f32(vfilter_r, vinvs_0123_0, 2);
        vinv_filter_scale3 = svmul_lane_f32(vfilter_r, vinvs_0123_0, 3);

        svint32_t za1_vout0 = svmla_lane_s32( svget4(za1_vacco_x4_0, 0), vksum_r, vzp_0123_0, 0);
        svint32_t za1_vout1 = svmla_lane_s32( svget4(za1_vacco_x4_0, 1), vksum_r, vzp_0123_0, 1);
        svint32_t za1_vout2 = svmla_lane_s32( svget4(za1_vacco_x4_0, 2), vksum_r, vzp_0123_0, 2);
        svint32_t za1_vout3 = svmla_lane_s32( svget4(za1_vacco_x4_0, 3), vksum_r, vzp_0123_0, 3);

        svfloat32x4_t vout1_x4_0 = svcvt_f32(svcreate4(za1_vout0, za1_vout1, za1_vout2, za1_vout3));
        svfloat32_t vout0_1 = svget4(vout1_x4_0, 0);
        svfloat32_t vout1_1 = svget4(vout1_x4_0, 1);
        svfloat32_t vout2_1 = svget4(vout1_x4_0, 2);
        svfloat32_t vout3_1 = svget4(vout1_x4_0, 3);

        vout0_1 = svmad_f32_m(pg_32_all, vout0_1, vinv_filter_scale0, vbias_r);
        vout1_1 = svmad_f32_m(pg_32_all, vout1_1, vinv_filter_scale1, vbias_r);
        vout2_1 = svmad_f32_m(pg_32_all, vout2_1, vinv_filter_scale2, vbias_r);
        vout3_1 = svmad_f32_m(pg_32_all, vout3_1, vinv_filter_scale3, vbias_r);

        svfloat32x4_t vout0123_1 = svcreate4(vout0_1, vout1_1, vout2_1, vout3_1);
        svwrite_hor_za32_f32_vg4(1, 4*si, vout0123_1);


        //za2,3
        a_inc_zp1 = a_inc_zp + ((vl + si * 4) * sizeof(int32_t));
        a_inc_inv1 = a_inc_inv + ((vl + si * 4) * sizeof(int32_t));

        svint32_t vinput_zero_point_1 = svld1_s32(pg_zp, (const int32_t*)a_inc_zp1);
        svint32_t vzp_0123_1 = svdupq_lane_s32(vinput_zero_point_1, 0);

        svfloat32_t vinv_scale_1 = svreinterpret_f32_s32( svld1_s32(pg_inv, (const int32_t*)a_inc_inv1) );
        svfloat32_t vinvs_0123_1 = svdupq_lane_f32(vinv_scale_1, 0);


        //za2
        za0_vacco_x4_0 = svread_hor_za32_s32_vg4(2, 4*si);

        vinv_filter_scale0 = svmul_lane_f32(vfilter_l, vinvs_0123_1, 0);
        vinv_filter_scale1 = svmul_lane_f32(vfilter_l, vinvs_0123_1, 1);
        vinv_filter_scale2 = svmul_lane_f32(vfilter_l, vinvs_0123_1, 2);
        vinv_filter_scale3 = svmul_lane_f32(vfilter_l, vinvs_0123_1, 3);

        za0_vout0 = svmla_lane_s32( svget4(za0_vacco_x4_0, 0), vksum_l, vzp_0123_1, 0);
        za0_vout1 = svmla_lane_s32( svget4(za0_vacco_x4_0, 1), vksum_l, vzp_0123_1, 1);
        za0_vout2 = svmla_lane_s32( svget4(za0_vacco_x4_0, 2), vksum_l, vzp_0123_1, 2);
        za0_vout3 = svmla_lane_s32( svget4(za0_vacco_x4_0, 3), vksum_l, vzp_0123_1, 3);

        vout0_x4_0 = svcvt_f32(svcreate4(za0_vout0, za0_vout1, za0_vout2, za0_vout3));
        vout0_0 = svget4(vout0_x4_0, 0);
        vout1_0 = svget4(vout0_x4_0, 1);
        vout2_0 = svget4(vout0_x4_0, 2);
        vout3_0 = svget4(vout0_x4_0, 3);

        vout0_0 = svmad_f32_m(pg_32_all, vout0_0, vinv_filter_scale0, vbias_l);
        vout1_0 = svmad_f32_m(pg_32_all, vout1_0, vinv_filter_scale1, vbias_l);
        vout2_0 = svmad_f32_m(pg_32_all, vout2_0, vinv_filter_scale2, vbias_l);
        vout3_0 = svmad_f32_m(pg_32_all, vout3_0, vinv_filter_scale3, vbias_l);

        vout0123_0 = svcreate4(vout0_0, vout1_0, vout2_0, vout3_0);
        svwrite_hor_za32_f32_vg4(2, 4*si, vout0123_0);


        //za3
        za1_vacco_x4_0 = svread_hor_za32_s32_vg4(3, 4*si);

        vinv_filter_scale0 = svmul_lane_f32(vfilter_r, vinvs_0123_1, 0);
        vinv_filter_scale1 = svmul_lane_f32(vfilter_r, vinvs_0123_1, 1);
        vinv_filter_scale2 = svmul_lane_f32(vfilter_r, vinvs_0123_1, 2);
        vinv_filter_scale3 = svmul_lane_f32(vfilter_r, vinvs_0123_1, 3);

        za1_vout0 = svmla_lane_s32( svget4(za1_vacco_x4_0, 0), vksum_r, vzp_0123_1, 0);
        za1_vout1 = svmla_lane_s32( svget4(za1_vacco_x4_0, 1), vksum_r, vzp_0123_1, 1);
        za1_vout2 = svmla_lane_s32( svget4(za1_vacco_x4_0, 2), vksum_r, vzp_0123_1, 2);
        za1_vout3 = svmla_lane_s32( svget4(za1_vacco_x4_0, 3), vksum_r, vzp_0123_1, 3);

        vout1_x4_0 = svcvt_f32(svcreate4(za1_vout0, za1_vout1, za1_vout2, za1_vout3));
        vout0_1 = svget4(vout1_x4_0, 0);
        vout1_1 = svget4(vout1_x4_0, 1);
        vout2_1 = svget4(vout1_x4_0, 2);
        vout3_1 = svget4(vout1_x4_0, 3);

        vout0_1 = svmad_f32_m(pg_32_all, vout0_1, vinv_filter_scale0, vbias_r);
        vout1_1 = svmad_f32_m(pg_32_all, vout1_1, vinv_filter_scale1, vbias_r);
        vout2_1 = svmad_f32_m(pg_32_all, vout2_1, vinv_filter_scale2, vbias_r);
        vout3_1 = svmad_f32_m(pg_32_all, vout3_1, vinv_filter_scale3, vbias_r);

        vout0123_1 = svcreate4(vout0_1, vout1_1, vout2_1, vout3_1);
        svwrite_hor_za32_f32_vg4(3, 4*si, vout0123_1);

      }


      svfloat32_t voutput_min = svdup_n_f32(minmax_params->scalar.min);
      svfloat32_t voutput_max = svdup_n_f32(minmax_params->scalar.max);
      float* c_inc = c0;
      svcount_t pn = svwhilelt_c32_u64(0, nc, 2);
      size_t m_chunk = mr;

      size_t m_chunk_0 = (m_chunk < vl) ? m_chunk : vl;
      size_t m_chunk_0_vec4 = m_chunk_0 & ~3;
      #pragma clang loop unroll(disable)
      for (size_t i = 0; i < m_chunk_0_vec4; i += 4) {
          svfloat32x4_t z0x4 = svread_hor_za32_f32_vg4(0, i);
          svfloat32x4_t z1x4 = svread_hor_za32_f32_vg4(1, i);
          z0x4 = svclamp(z0x4, voutput_min, voutput_max);
          z1x4 = svclamp(z1x4, voutput_min, voutput_max);

          svfloat32x2_t z01 = svundef2_f32();
          z01 = svcreate2(svget4_f32(z0x4, 0), svget4_f32(z1x4, 0));
          svst1_f32_x2(pn, c_inc + 0, z01);

          z01 = svcreate2(svget4_f32(z0x4, 1), svget4_f32(z1x4, 1));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z01);

          z01 = svcreate2(svget4_f32(z0x4, 2), svget4_f32(z1x4, 2));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z01);

          z01 = svcreate2(svget4_f32(z0x4, 3), svget4_f32(z1x4, 3));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 3 * cm_stride), z01);

          c_inc = (float*) ((uintptr_t)c_inc + 4 * cm_stride);
      }

      if (m_chunk_0_vec4 < m_chunk_0) {
          size_t i = m_chunk_0_vec4;
          svfloat32x4_t z0x4 = svread_hor_za32_f32_vg4(0, i);
          svfloat32x4_t z1x4 = svread_hor_za32_f32_vg4(1, i);
          z0x4 = svclamp(z0x4, voutput_min, voutput_max);
          z1x4 = svclamp(z1x4, voutput_min, voutput_max);

          svfloat32x2_t z01 = svundef2_f32();
          z01 = svcreate2(svget4_f32(z0x4, 0), svget4_f32(z1x4, 0));
          svst1_f32_x2(pn, c_inc + 0, z01);

          if (i + 1 < m_chunk_0) {
              z01 = svcreate2(svget4_f32(z0x4, 1), svget4_f32(z1x4, 1));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z01);
          }
          if (i + 2 < m_chunk_0) {
              z01 = svcreate2(svget4_f32(z0x4, 2), svget4_f32(z1x4, 2));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z01);
          }
          c_inc = (float*) ((uintptr_t)c_inc + (m_chunk_0 - m_chunk_0_vec4) * cm_stride);
      }

      c_inc = (float*)((uintptr_t)c0 + vl * cm_stride);

      size_t m_chunk_1_rem = m_chunk - vl;
      size_t m_chunk_1_vec4 = m_chunk_1_rem & ~3;
      #pragma clang loop unroll(disable)
      for (size_t i = 0; i < m_chunk_1_vec4; i += 4) {
          svfloat32x4_t z2x4 = svread_hor_za32_f32_vg4(2, i);
          svfloat32x4_t z3x4 = svread_hor_za32_f32_vg4(3, i);
          z2x4 = svclamp(z2x4, voutput_min, voutput_max);
          z3x4 = svclamp(z3x4, voutput_min, voutput_max);

          svfloat32x2_t z23 = svundef2_f32();
          z23 = svcreate2(svget4_f32(z2x4, 0), svget4_f32(z3x4, 0));
          svst1_f32_x2(pn, c_inc + 0, z23);

          z23 = svcreate2(svget4_f32(z2x4, 1), svget4_f32(z3x4, 1));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z23);

          z23 = svcreate2(svget4_f32(z2x4, 2), svget4_f32(z3x4, 2));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z23);

          z23 = svcreate2(svget4_f32(z2x4, 3), svget4_f32(z3x4, 3));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 3 * cm_stride), z23);

          c_inc = (float*) ((uintptr_t)c_inc + 4 * cm_stride);
      }

      if (m_chunk_1_vec4 < m_chunk_1_rem) {
          size_t i = m_chunk_1_vec4;
          svfloat32x4_t z2x4 = svread_hor_za32_f32_vg4(2, i);
          svfloat32x4_t z3x4 = svread_hor_za32_f32_vg4(3, i);
          z2x4 = svclamp(z2x4, voutput_min, voutput_max);
          z3x4 = svclamp(z3x4, voutput_min, voutput_max);

          svfloat32x2_t z23 = svundef2_f32();
          z23 = svcreate2(svget4_f32(z2x4, 0), svget4_f32(z3x4, 0));
          svst1_f32_x2(pn, c_inc + 0, z23);

          if (i + 1 < m_chunk_1_rem) {
              z23 = svcreate2(svget4_f32(z2x4, 1), svget4_f32(z3x4, 1));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z23);
          }
          if (i + 2 < m_chunk_1_rem) {
              z23 = svcreate2(svget4_f32(z2x4, 2), svget4_f32(z3x4, 2));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z23);
          }
          c_inc = (float*) ((uintptr_t)c_inc + (m_chunk_1_rem - m_chunk_1_vec4) * cm_stride);
      }


      if (nc >= nr) {
          a_inc = a0;
          //Adjust pointers - set next output pointers
          c0 = (float*)((uintptr_t)c0 + cn_stride);
          nc -= nr; // start next Nr loop

      } else {
          nc = 0; //finish kernel
      }

    } while (nc != 0);

  } //mr>vl

}

#undef PREFETCH_W_MATRIX

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_32x32c4__sme2_get_nr() {
  return 2 * svcntsw();
}
#endif  // XNN_ENABLE_ARM_SME2_ACLE
