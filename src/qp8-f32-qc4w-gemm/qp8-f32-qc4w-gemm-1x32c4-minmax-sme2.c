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

__arm_locally_streaming __arm_new("za", "zt0")
void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x32c4__sme2(
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
  const struct xnn_f32_qc4w_minmax_params* minmax_params =
      (const struct xnn_f32_qc4w_minmax_params*) params;

  kc = ((kc+31)/32) * 32;
  cn_stride = 32*cn_stride; //32:nr

  static const int8_t lut[64] = {0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
                                 0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
                                -5,  0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};
  svldr_zt(0, &lut[0]);


  const int8_t* a0 = (const int8_t*) a;
  float* c0 = c;

  svbool_t pg_32_all   = svptrue_b32();
  svcount_t pcnt_8_all = svptrue_c8();

  size_t k_rup_32 = ((kc+31)/32)*32;
  size_t w_offset_nr = 32*(k_rup_32/2 + 12);
  const void* w0 = w;
  const void* w1 = (const int8_t*) w + w_offset_nr;

#ifdef PREFETCH_W_MATRIX
  const size_t kPrefetchDistance = 16;
  const size_t kBytesPerKLoop = 256;
  const size_t kCacheLineSize = 64;
  const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;

  size_t block_size = w_offset_nr;
  size_t nc_rup_nr = ((nc+31)/32) * 32;
  size_t total_w_size = (kc * nc_rup_nr) / 2 + nc_rup_nr * (4*3);

  bool need_prfm_0 = (block_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > (1024*1024));
  bool need_prfm_1 = (nc > 32) && need_prfm_0;

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
    size_t krup = k_rup_32;
    int32_t k_left = kc;

    const int8_t* a_inc = a0;

    while (krup >= 16 * sizeof(int8_t)) {
      size_t k_len = (k_left >= 16) ? 16 : (k_left > 0) ? k_left : 0;
      svbool_t pg_8_msk_n = svwhilelt_b8_u64(0, k_len);
      svint8_t va0 = svld1_s8(pg_8_msk_n, a_inc); a_inc += 16;
      svint8_t va0_16x4 = svdupq_lane_s8(va0, 0);

      // Load W for 64 columns
      svint8x4_t vb_x4_0 = svld1_s8_x4(pcnt_8_all, w0); w0 = (const int8_t*) w0 + 256;
      svcount_t pcnt_8_next_nr32 = (nc > 32) ? svwhilelt_c8_u64(0, 4*64, 4) : svpfalse_c();
      svint8x4_t vb_x4_1 = svld1_s8_x4(pcnt_8_next_nr32, w1); w1 = (const int8_t*) w1 + 256;

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0 && (prfm_acc_w_size_0 < (block_size - kBytesPerKLoop))) {
        prfm_acc_w_size_0 += kBytesPerKLoop;
        __builtin_prefetch(prfm_ptr_0 + 0 * kCacheLineSize, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 1 * kCacheLineSize, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 2 * kCacheLineSize, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 3 * kCacheLineSize, 0, 2);
        prfm_ptr_0 += kBytesPerKLoop;
      }
      if (need_prfm_1 && (prfm_acc_w_size_1 < (block_size - kBytesPerKLoop))) {
        prfm_acc_w_size_1 += kBytesPerKLoop;
        __builtin_prefetch(prfm_ptr_1 + 0 * kCacheLineSize, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 1 * kCacheLineSize, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 2 * kCacheLineSize, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 3 * kCacheLineSize, 0, 2);
        prfm_ptr_1 += kBytesPerKLoop;
      }
#endif

      // Expand using LUT
      svint8x2_t luti_0 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 0)), 0);
      svint8x2_t luti_1 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 1)), 0);
      svint8x2_t luti_2 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 2)), 0);
      svint8x2_t luti_3 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_0, 3)), 0);

      svint8x2_t luti_4 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 0)), 0);
      svint8x2_t luti_5 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 1)), 0);
      svint8x2_t luti_6 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 2)), 0);
      svint8x2_t luti_7 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8(svget4(vb_x4_1, 3)), 0);

      // Combine into x4 vectors
      svint8x4_t luti_k0 = svcreate4(svget2(luti_0, 0), svget2(luti_0, 1), svget2(luti_4, 0), svget2(luti_4, 1));
      svint8x4_t luti_k1 = svcreate4(svget2(luti_1, 0), svget2(luti_1, 1), svget2(luti_5, 0), svget2(luti_5, 1));
      svint8x4_t luti_k2 = svcreate4(svget2(luti_2, 0), svget2(luti_2, 1), svget2(luti_6, 0), svget2(luti_6, 1));
      svint8x4_t luti_k3 = svcreate4(svget2(luti_3, 0), svget2(luti_3, 1), svget2(luti_7, 0), svget2(luti_7, 1));

      // Accumulate using vg1x4 across ZA0..ZA3
      svdot_lane_za32_s8_vg1x4(0, luti_k0, va0_16x4, 0);
      svdot_lane_za32_s8_vg1x4(1, luti_k1, va0_16x4, 1);
      svdot_lane_za32_s8_vg1x4(2, luti_k2, va0_16x4, 2);
      svdot_lane_za32_s8_vg1x4(3, luti_k3, va0_16x4, 3);

      k_left -= 16 * sizeof(int8_t);
      krup -= 16 * sizeof(int8_t);
    }

#ifdef PREFETCH_W_MATRIX
    if (need_prfm_0) {
      if (prfm_acc_w_size_0 <= (block_size - 384)) {
        prfm_acc_w_size_0 += 384;
        __builtin_prefetch(prfm_ptr_0 + 0 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 1 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 2 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 3 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 4 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 5 * 64, 0, 2);
        prfm_ptr_0 += 384;
      } else if (prfm_acc_w_size_0 <= (block_size - 256)) {
        prfm_acc_w_size_0 += 256;
        __builtin_prefetch(prfm_ptr_0 + 0 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 1 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 2 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 3 * 64, 0, 2);
        prfm_ptr_0 += 256;
      } else if (prfm_acc_w_size_0 <= (block_size - 128)) {
        prfm_acc_w_size_0 += 128;
        __builtin_prefetch(prfm_ptr_0 + 0 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 1 * 64, 0, 2);
        prfm_ptr_0 += 128;
      }
    }
    if (need_prfm_1) {
      if (prfm_acc_w_size_1 <= (block_size - 384)) {
        prfm_acc_w_size_1 += 384;
        __builtin_prefetch(prfm_ptr_1 + 0 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 1 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 2 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 3 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 4 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 5 * 64, 0, 2);
        prfm_ptr_1 += 384;
      } else if (prfm_acc_w_size_1 <= (block_size - 256)) {
        prfm_acc_w_size_1 += 256;
        __builtin_prefetch(prfm_ptr_1 + 0 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 1 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 2 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 3 * 64, 0, 2);
        prfm_ptr_1 += 256;
      } else if (prfm_acc_w_size_1 <= (block_size - 128)) {
        prfm_acc_w_size_1 += 128;
        __builtin_prefetch(prfm_ptr_1 + 0 * 64, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 1 * 64, 0, 2);
        prfm_ptr_1 += 128;
      }
    }
#endif

    // Load metadata for 64 columns
    svcount_t pcnt_32_01 = svwhilelt_c32_u64(0, 32, 2);
    svcount_t pcnt_32_23 = (nc > 32) ? svwhilelt_c32_u64(0, 32, 2) : svpfalse_c();

    svint32x2_t vksum_01 = svld1_s32_x2(pcnt_32_01, w0); w0 = (const int32_t*) w0 + 32;
    svint32x2_t vksum_23 = svld1_s32_x2(pcnt_32_23, w1); w1 = (const int32_t*) w1 + 32;

    svint32x2_t vfilter_01_s32 = svld1_s32_x2(pcnt_32_01, w0); w0 = (const int32_t*) w0 + 32;
    svint32x2_t vfilter_23_s32 = svld1_s32_x2(pcnt_32_23, w1); w1 = (const int32_t*) w1 + 32;

    svint32x2_t vbias_01_s32 = svld1_s32_x2(pcnt_32_01, w0); w0 = (const int32_t*) w0 + 32;
    svint32x2_t vbias_23_s32 = svld1_s32_x2(pcnt_32_23, w1); w1 = (const int32_t*) w1 + 32;

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

    // Read out ZA for 64 columns across ZA0..ZA3 and combine partial K sums
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

    if (nc >= 64) {
      a_inc = a0;
      c0 = (float*)((uintptr_t)c0 + cn_stride*(64/32)); // Adjust for 64 columns
      w0 = (const int8_t*) w0 + w_offset_nr;
      w1 = (const int8_t*) w1 + w_offset_nr;
      nc -= 64;
#ifdef PREFETCH_W_MATRIX
      if (nc != 0) {
        need_prfm_1 = (nc > 32) && need_prfm_0;
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
#endif  // XNN_ENABLE_ARM_SME2_ACLE
