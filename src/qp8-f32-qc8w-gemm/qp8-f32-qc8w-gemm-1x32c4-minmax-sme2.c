// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme2(
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
  const size_t kloop_bytes = 32 * vl;
  const struct xnn_f32_minmax_params* minmax_params =
      (const struct xnn_f32_minmax_params*) params;

  cn_stride = nr * cn_stride;

  const int8_t* a0 = (const int8_t*) a;
  float* c0 = c;

  svbool_t pg_32_all   = svptrue_b32();

  size_t k_rup_32 = ((kc+31)/32)*32;
  size_t w_offset_nr = nr * (k_rup_32/1 + 12);
  const void* w0 = w;
  const void* w1 = (nc > nr) ? (const void*)((const int8_t*) w + w_offset_nr) : w;

#ifdef PREFETCH_W_MATRIX
  const size_t kPrefetchDistance = 16;
  const size_t kBytesPerKLoop = kloop_bytes;
  const size_t kCacheLineSize = 64;
  const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
  const size_t kPrfmStride = kBytesPerKLoop / 8;
  const size_t kTrailerPrfmStride = 2 * nr;

  size_t block_size = w_offset_nr;
  size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  size_t total_w_size = (k_rup_32 * nc_rup_nr) + nc_rup_nr * (4*3);
  bool need_prfm_0 = (block_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > (1024*1024));
  bool need_prfm_1 = (nc > nr) && need_prfm_0;

  size_t prfm_acc_w_size_0 = 0;
  size_t prfm_acc_w_size_1 = 0;

  const int8_t* prfm_ptr_0 = (const int8_t*)w0;
  const int8_t* prfm_ptr_1 = (const int8_t*)w1;

  if (need_prfm_0) {
    prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
    int i = 0;
    for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
      __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * i, 0, 2);
    }
    prfm_ptr_0 += kCacheLineSize * i;
  }

  if (need_prfm_1) {
    prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
    int i = 0;
    for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
      __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * i, 0, 2);
    }
    prfm_ptr_1 += kCacheLineSize * i;
  }
#endif

  do {
    svzero_za();
    size_t krup = ((kc+31)/32) * 32;  //always roundup(kc, 32);
    int32_t k_left = kc;

    const int8_t* a_inc = a0;

    while (krup >= 16 * sizeof(int8_t)) {

      //load one A matrix row
      size_t k_len = (k_left >= 16) ? 16 : (k_left > 0) ? k_left : 0;
      svbool_t pg_8_msk_n = svwhilelt_b8_u64(0, k_len);
      svint8_t va0_16x4 = svld1rq_s8(pg_8_msk_n, a_inc); a_inc += 16;

      svcount_t pcnt_8_all = svptrue_c8();
      svcount_t pcnt_8_next_nr32 = (nc > nr) ? svptrue_c8() : svpfalse_c();
      //load w matrix
      svint8x2_t vb_lo_0 = svld1_vnum_s8_x2(pcnt_8_all, w0, 0);
      svint8x2_t vb_lo_1 = svld1_vnum_s8_x2(pcnt_8_all, w0, 2);
      svint8x2_t vb_lo_2 = svld1_vnum_s8_x2(pcnt_8_all, w0, 4);
      svint8x2_t vb_lo_3 = svld1_vnum_s8_x2(pcnt_8_all, w0, 6);
      w0 = (const int8_t*) w0 + kloop_bytes;
      svint8x2_t vb_hi_0 = svld1_vnum_s8_x2(pcnt_8_next_nr32, w1, 0);
      svint8x2_t vb_hi_1 = svld1_vnum_s8_x2(pcnt_8_next_nr32, w1, 2);
      svint8x2_t vb_hi_2 = svld1_vnum_s8_x2(pcnt_8_next_nr32, w1, 4);
      svint8x2_t vb_hi_3 = svld1_vnum_s8_x2(pcnt_8_next_nr32, w1, 6);
      if (nc > nr) {
        w1 = (const int8_t*) w1 + kloop_bytes;
      }

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

      svint8x4_t vb_k4_0 = svcreate4(svget2(vb_lo_0,0), svget2(vb_lo_0,1), svget2(vb_hi_0,0), svget2(vb_hi_0,1));
      svint8x4_t vb_k4_1 = svcreate4(svget2(vb_lo_1,0), svget2(vb_lo_1,1), svget2(vb_hi_1,0), svget2(vb_hi_1,1));
      svint8x4_t vb_k4_2 = svcreate4(svget2(vb_lo_2,0), svget2(vb_lo_2,1), svget2(vb_hi_2,0), svget2(vb_hi_2,1));
      svint8x4_t vb_k4_3 = svcreate4(svget2(vb_lo_3,0), svget2(vb_lo_3,1), svget2(vb_hi_3,0), svget2(vb_hi_3,1));

      svdot_lane_za32_s8_vg1x4(0, vb_k4_0, va0_16x4, 0);
      svdot_lane_za32_s8_vg1x4(1, vb_k4_1, va0_16x4, 1);
      svdot_lane_za32_s8_vg1x4(2, vb_k4_2, va0_16x4, 2);
      svdot_lane_za32_s8_vg1x4(3, vb_k4_3, va0_16x4, 3);

      k_left -= 16 * sizeof(int8_t);
      krup -= 16 * sizeof(int8_t);
    }

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

    // Load metadata for 2 * nr columns
    svcount_t pcnt_32_01 = svwhilelt_c32_u64(0, nr, 2);
    svcount_t pcnt_32_23 = (nc > nr) ? svwhilelt_c32_u64(0, nr, 2) : svpfalse_c();

    svint32x2_t vksum_01 = svld1_s32_x2(pcnt_32_01, w0); w0 = (const int32_t*) w0 + nr;
    svint32x2_t vksum_23 = svld1_s32_x2(pcnt_32_23, w1);
    if (nc > nr) {
      w1 = (const int32_t*) w1 + nr;
    }

    svint32x2_t vfilter_01_s32 = svld1_s32_x2(pcnt_32_01, w0); w0 = (const int32_t*) w0 + nr;
    svint32x2_t vfilter_23_s32 = svld1_s32_x2(pcnt_32_23, w1);
    if (nc > nr) {
      w1 = (const int32_t*) w1 + nr;
    }

    svint32x2_t vbias_01_s32 = svld1_s32_x2(pcnt_32_01, w0); w0 = (const int32_t*) w0 + nr;
    svint32x2_t vbias_23_s32 = svld1_s32_x2(pcnt_32_23, w1);
    if (nc > nr) {
      w1 = (const int32_t*) w1 + nr;
    }

    svint32_t vksum_0 = svget2(vksum_01, 0);
    svint32_t vksum_1 = svget2(vksum_01, 1);
    svint32_t vksum_2 = svget2(vksum_23, 0);
    svint32_t vksum_3 = svget2(vksum_23, 1);

    svfloat32_t vfilter_0 = svreinterpret_f32_s32(svget2(vfilter_01_s32, 0));
    svfloat32_t vfilter_1 = svreinterpret_f32_s32(svget2(vfilter_01_s32, 1));
    svfloat32_t vfilter_2 = svreinterpret_f32_s32(svget2(vfilter_23_s32, 0));
    svfloat32_t vfilter_3 = svreinterpret_f32_s32(svget2(vfilter_23_s32, 1));

    svfloat32_t vbias_0 = svreinterpret_f32_s32(svget2(vbias_01_s32, 0));
    svfloat32_t vbias_1 = svreinterpret_f32_s32(svget2(vbias_01_s32, 1));
    svfloat32_t vbias_2 = svreinterpret_f32_s32(svget2(vbias_23_s32, 0));
    svfloat32_t vbias_3 = svreinterpret_f32_s32(svget2(vbias_23_s32, 1));

    const int32_t* a_inc_zp = (const int32_t*) a_inc;
    const int32_t* a_inc_inv = a_inc_zp + 1;

    svint32_t vzp_mr0 = svdup_s32((const int32_t)*a_inc_zp);
    svfloat32_t vinvs_mr0 = svreinterpret_f32_s32(svdup_s32((const int32_t)*a_inc_inv));

    // Read out ZA for 2 * nr columns across ZA0..ZA3 and combine partial K sums
    svint32x4_t za0_vacco_x4 = svread_za32_s32_vg1x4(0);
    svint32x4_t za1_vacco_x4 = svread_za32_s32_vg1x4(1);
    svint32x4_t za2_vacco_x4 = svread_za32_s32_vg1x4(2);
    svint32x4_t za3_vacco_x4 = svread_za32_s32_vg1x4(3);

    svint32_t vacco_0 = svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_vacco_x4, 0), svget4(za1_vacco_x4, 0)), svadd_s32_x(pg_32_all, svget4(za2_vacco_x4, 0), svget4(za3_vacco_x4, 0)));
    svint32_t vacco_1 = svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_vacco_x4, 1), svget4(za1_vacco_x4, 1)), svadd_s32_x(pg_32_all, svget4(za2_vacco_x4, 1), svget4(za3_vacco_x4, 1)));
    svint32_t vacco_2 = svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_vacco_x4, 2), svget4(za1_vacco_x4, 2)), svadd_s32_x(pg_32_all, svget4(za2_vacco_x4, 2), svget4(za3_vacco_x4, 2)));
    svint32_t vacco_3 = svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, svget4(za0_vacco_x4, 3), svget4(za1_vacco_x4, 3)), svadd_s32_x(pg_32_all, svget4(za2_vacco_x4, 3), svget4(za3_vacco_x4, 3)));

    svfloat32_t vinv_filter_scale0 = svmul_f32_x(pg_32_all, vfilter_0, vinvs_mr0);
    svfloat32_t vinv_filter_scale1 = svmul_f32_x(pg_32_all, vfilter_1, vinvs_mr0);
    svfloat32_t vinv_filter_scale2 = svmul_f32_x(pg_32_all, vfilter_2, vinvs_mr0);
    svfloat32_t vinv_filter_scale3 = svmul_f32_x(pg_32_all, vfilter_3, vinvs_mr0);

    svint32_t za0_vout0 = svmla_s32_x(pg_32_all, vacco_0, vksum_0, vzp_mr0);
    svint32_t za0_vout1 = svmla_s32_x(pg_32_all, vacco_1, vksum_1, vzp_mr0);
    svint32_t za0_vout2 = svmla_s32_x(pg_32_all, vacco_2, vksum_2, vzp_mr0);
    svint32_t za0_vout3 = svmla_s32_x(pg_32_all, vacco_3, vksum_3, vzp_mr0);

    svfloat32x4_t vout0_x4 = svcvt_f32(svcreate4(za0_vout0, za0_vout1, za0_vout2, za0_vout3));
    svfloat32_t vout0_0 = svget4(vout0_x4, 0);
    svfloat32_t vout1_0 = svget4(vout0_x4, 1);
    svfloat32_t vout2_0 = svget4(vout0_x4, 2);
    svfloat32_t vout3_0 = svget4(vout0_x4, 3);

    vout0_0 = svmad_f32_m(pg_32_all, vout0_0, vinv_filter_scale0, vbias_0);
    vout1_0 = svmad_f32_m(pg_32_all, vout1_0, vinv_filter_scale1, vbias_1);
    vout2_0 = svmad_f32_m(pg_32_all, vout2_0, vinv_filter_scale2, vbias_2);
    vout3_0 = svmad_f32_m(pg_32_all, vout3_0, vinv_filter_scale3, vbias_3);

    svfloat32_t voutput_min = svdup_n_f32(minmax_params->scalar.min);
    svfloat32_t voutput_max = svdup_n_f32(minmax_params->scalar.max);

    svfloat32x4_t vout0123 = svclamp(svcreate4(vout0_0, vout1_0, vout2_0, vout3_0), voutput_min, voutput_max);
    svcount_t pcnt_32_nc = svwhilelt_c32_u64(0, nc, 4);
    svst1_f32_x4(pcnt_32_nc, c0, vout0123);

    if (nc > 2 * nr) {
      a_inc = a0;
      c0 = (float*)((uintptr_t)c0 + cn_stride * 2); // Advance by 2 * nr columns
      w0 = (const int8_t*) w0 + w_offset_nr;
      nc -= 2 * nr;
      w1 = (nc > nr) ? (const void*)((const int8_t*) w0 + w_offset_nr) : w0;
#ifdef PREFETCH_W_MATRIX
      if (nc != 0) {
        need_prfm_1 = (nc > nr) && need_prfm_0;
        if (need_prfm_0) {
          prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
          prfm_ptr_0 = (const int8_t*) w0;
          int i = 0;
          for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
            __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * i, 0, 2);
          }
          prfm_ptr_0 += kCacheLineSize * i;
        }
        if (need_prfm_1) {
          prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
          prfm_ptr_1 = (const int8_t*) w1;
          int i = 0;
          for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
            __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * i, 0, 2);
          }
          prfm_ptr_1 += kCacheLineSize * i;
        }
      }
#endif
    } else {
      nc = 0;
    }
 } while (nc != 0);

}

#undef PREFETCH_W_MATRIX

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme2_get_mr() {
  return 1;
}

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__sme2_get_nr() {
  return 2 * svcntsw();
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
