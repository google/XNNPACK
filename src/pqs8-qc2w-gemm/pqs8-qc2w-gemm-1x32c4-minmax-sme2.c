// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_neon.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_1x32c4__sme2_get_mr() {
  return 1;
}

size_t xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_1x32c4__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za", "zt0")
void xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_1x32c4__sme2(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* lhs_packed,
    const void* w,
    void* dst,
    size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params)
{
#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_line_size = 128;
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
#else
  const size_t prfm_line_size = 64;
  const size_t prfm_min_w_size = (1024 * 1024);
#endif
#endif
  (void)mr;
  (void)dst_stride_row;
  (void)dst_stride_col;
  assert(mr == 1);
  assert(nc != 0);
  assert(kc != 0);

  static const int8_t lut[64] = {
      0,  0, 0, 0, 1,  0, 0, 0, -2, 0, 0,  0, -1, 0, 0,  0,
      0,  0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,
      0,  0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,
      0,  0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0};
  svldr_zt(0, &lut[0]);

  const int32_t output_zero_point = minmax_params->fp32_scalar.output_zero_point;
  const int32_t output_min = minmax_params->fp32_scalar.output_min;
  const int32_t output_max = minmax_params->fp32_scalar.output_max;

  const svint32_t voutput_zp_vec = svdup_n_s32(output_zero_point);
  const svint8_t voutput_min_vec = svdup_n_s8((int8_t) output_min);
  const svint8_t voutput_max_vec = svdup_n_s8((int8_t) output_max);

  const svbool_t pg_8_all = svptrue_b8();
  const svbool_t pg_32_all = svptrue_b32();
  const svcount_t pcnt_8_all = svptrue_c8();
  const svcount_t pcnt_32_all = svptrue_c32();
  const size_t nr = 2 * svcntw();

  const size_t k_rounded_32 = ((kc + 31) / 32) * 32;
  const size_t k_block4_rounded = k_rounded_32 / 4;
  const size_t k_iters = k_rounded_32 / 16;
  const uint32_t o1 = (k_iters <= 16) ? 0 : 1;
  const uint32_t o3 = (k_iters <= 16) ? 2 : 3;

  const size_t w_block_size = 4 * nr + k_block4_rounded * nr + 4 * nr;

  int8_t* c_inc = (int8_t*) dst;

#ifdef PREFETCH_W_MATRIX
  const int8_t* w_base = (const int8_t*) w;
  const size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  const size_t total_w_size = (nc_rup_nr / nr) * w_block_size;
  const size_t prfm_dist_size = 16 * 4 * nr;
  const size_t upfront_prefetch_w_block = (prfm_dist_size < w_block_size) ? prfm_dist_size : w_block_size;
  const bool need_prfm = total_w_size > prfm_min_w_size;
#endif

  if (nc <= svcntw()) {
    svzero_za();

    const int8_t* a_inc = (const int8_t*) lhs_packed;
    const int8_t* w0 = (const int8_t*) w;

#ifdef PREFETCH_W_MATRIX
    size_t prfm_base_0 = (size_t) (w0 - w_base);
    size_t prfm_end_0 = need_prfm ? (prfm_base_0 + w_block_size) : 0;
    size_t prfm_pos_0 = need_prfm
        ? ((prfm_base_0 + prfm_line_size - 1) & ~(size_t) (prfm_line_size - 1))
        : 0;
    size_t prfm_acc_0 = need_prfm ? (prfm_base_0 + upfront_prefetch_w_block) : 0;
    for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
      __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
    }
#endif

    const svint32_t vbias = svld1_s32(pg_32_all, (const int32_t*) w0); w0 += 4 * nr;
    const svfloat32_t vscale = svld1_f32(pg_32_all, (const float*) (w0 + k_block4_rounded * nr));

    const size_t k_full = kc / 16;
    uint32_t so = 0;
    #pragma clang loop unroll(disable)
    for (size_t k_iter = 0; k_iter < k_full; ++k_iter) {
      const int32_t* a_w = (const int32_t*) a_inc; a_inc += 16;

      svint8x2_t vb0_x2 = svld1_s8_x2(pcnt_8_all, w0); w0 += 4 * nr;

#ifdef PREFETCH_W_MATRIX
      if (prfm_acc_0 < prfm_end_0) {
        prfm_acc_0 += 4 * nr;
        if (prfm_acc_0 > prfm_end_0) {
          prfm_acc_0 = prfm_end_0;
        }
        for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
          __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
        }
      }
#endif

      svuint8_t vb0_raw0 = svreinterpret_u8(svget2(vb0_x2, 0));
      svuint8_t vb0_raw1 = svreinterpret_u8(svget2(vb0_x2, 1));

      svint8_t vb_0 = svluti2_lane_zt_s8(0, vb0_raw0, 0);
      svint8_t vb_1 = svluti2_lane_zt_s8(0, vb0_raw0, 2);
      svint8_t vb_2 = svluti2_lane_zt_s8(0, vb0_raw1, 0);
      svint8_t vb_3 = svluti2_lane_zt_s8(0, vb0_raw1, 2);

      svint8x2_t va_01 = svcreate2(svreinterpret_s8(svdup_n_s32(a_w[0])), svreinterpret_s8(svdup_n_s32(a_w[1])));
      svint8x2_t va_23 = svcreate2(svreinterpret_s8(svdup_n_s32(a_w[2])), svreinterpret_s8(svdup_n_s32(a_w[3])));

      svdot_za32_s8_vg1x2(so, svcreate2(vb_0, vb_1), va_01);
      svdot_za32_s8_vg1x2(so + 1, svcreate2(vb_2, vb_3), va_23);
      so ^= 2;
    }

    if (k_full * 16 < kc) {
      const svint32_t va = svreinterpret_s32(svld1rq_s8(svwhilelt_b8_u64(k_full * 16, kc), a_inc));

      svint8x2_t vb0_x2 = svld1_s8_x2(pcnt_8_all, w0); w0 += 4 * nr;

#ifdef PREFETCH_W_MATRIX
      if (prfm_acc_0 < prfm_end_0) {
        prfm_acc_0 += 4 * nr;
        if (prfm_acc_0 > prfm_end_0) {
          prfm_acc_0 = prfm_end_0;
        }
        for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
          __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
        }
      }
#endif

      svuint8_t vb0_raw0 = svreinterpret_u8(svget2(vb0_x2, 0));
      svuint8_t vb0_raw1 = svreinterpret_u8(svget2(vb0_x2, 1));

      svint8_t vb_0 = svluti2_lane_zt_s8(0, vb0_raw0, 0);
      svint8_t vb_1 = svluti2_lane_zt_s8(0, vb0_raw0, 2);
      svint8_t vb_2 = svluti2_lane_zt_s8(0, vb0_raw1, 0);
      svint8_t vb_3 = svluti2_lane_zt_s8(0, vb0_raw1, 2);

      svint8x2_t va_01 = svcreate2(svreinterpret_s8(svdup_lane_s32(va, 0)), svreinterpret_s8(svdup_lane_s32(va, 1)));
      svint8x2_t va_23 = svcreate2(svreinterpret_s8(svdup_lane_s32(va, 2)), svreinterpret_s8(svdup_lane_s32(va, 3)));

      svdot_za32_s8_vg1x2(so, svcreate2(vb_0, vb_1), va_01);
      svdot_za32_s8_vg1x2(so + 1, svcreate2(vb_2, vb_3), va_23);
    }

#ifdef PREFETCH_W_MATRIX
    for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
      __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
    }
#endif

    svint32x2_t vacc0_x2 = svread_za32_s32_vg1x2(0);
    svint32x2_t vacc1_x2 = svread_za32_s32_vg1x2(1);
    svint32x2_t vacc2_x2 = svread_za32_s32_vg1x2(2);
    svint32x2_t vacc3_x2 = svread_za32_s32_vg1x2(3);
    svint32_t vacc = svadd_s32_x(pg_32_all,
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, svget2_s32(vacc0_x2, 0), svget2_s32(vacc0_x2, 1)),
                    svadd_s32_x(pg_32_all, svget2_s32(vacc1_x2, 0), svget2_s32(vacc1_x2, 1))),
        svadd_s32_x(pg_32_all, svadd_s32_x(pg_32_all, svget2_s32(vacc2_x2, 0), svget2_s32(vacc2_x2, 1)),
                    svadd_s32_x(pg_32_all, svget2_s32(vacc3_x2, 0), svget2_s32(vacc3_x2, 1))));
    vacc = svadd_s32_x(pg_32_all, vacc, vbias);

    svfloat32_t vscaled = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc), vscale);
    svint32_t vquant = svadd_s32_x(pg_32_all, svcvt_s32_f32_x(pg_32_all, svrintn_f32_x(pg_32_all, vscaled)), voutput_zp_vec);

    svint8_t vout_8 = svqcvt_s8_s32_x4(svcreate4(vquant, vquant, vquant, vquant));
    vout_8 = svmax_s8_x(pg_8_all, vout_8, voutput_min_vec);
    vout_8 = svmin_s8_x(pg_8_all, vout_8, voutput_max_vec);

    svst1_s8(svwhilelt_b8_u64(0, nc), c_inc, vout_8);
    return;
  }

  do {
    svzero_za();

    const int8_t* a_inc = (const int8_t*) lhs_packed;
    const int8_t* w0 = (const int8_t*) w;
    svfloat32x2_t vscale_0;
    svfloat32x2_t vscale_1;

    if (nc > nr) {
      const int8_t* w1 = w0 + w_block_size;

#ifdef PREFETCH_W_MATRIX
      size_t prfm_base_0 = (size_t) (w0 - w_base);
      size_t prfm_end_0 = need_prfm ? (prfm_base_0 + w_block_size) : 0;
      size_t prfm_pos_0 = need_prfm
          ? ((prfm_base_0 + prfm_line_size - 1) & ~(size_t) (prfm_line_size - 1))
          : 0;
      size_t prfm_acc_0 = need_prfm ? (prfm_base_0 + upfront_prefetch_w_block) : 0;
      for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
      size_t prfm_base_1 = (size_t) (w1 - w_base);
      size_t prfm_end_1 = need_prfm ? (prfm_base_1 + w_block_size) : 0;
      size_t prfm_pos_1 = need_prfm
          ? ((prfm_base_1 + prfm_line_size - 1) & ~(size_t) (prfm_line_size - 1))
          : 0;
      size_t prfm_acc_1 = need_prfm ? (prfm_base_1 + upfront_prefetch_w_block) : 0;
      for (; prfm_pos_1 < prfm_acc_1; prfm_pos_1 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
      }
#endif

      svint32x2_t vbias_0 = svld1_s32_x2(pcnt_32_all, (const int32_t*) w0); w0 += 4 * nr;
      svint32x2_t vbias_1 = svld1_s32_x2(pcnt_32_all, (const int32_t*) w1); w1 += 4 * nr;
      svadd_za32_s32_vg1x2(0, vbias_0);
      svadd_za32_s32_vg1x2(2, vbias_1);

      #pragma clang loop unroll(disable)
      for (size_t k_iter = 0; k_iter < k_iters; ++k_iter) {
        svint8_t va = svld1rq_s8(svwhilelt_b8_u64(k_iter * 16, kc), a_inc); a_inc += 16;

        svint8x2_t vb0_x2 = svld1_s8_x2(pcnt_8_all, w0); w0 += 4 * nr;
        svint8x2_t vb1_x2 = svld1_s8_x2(pcnt_8_all, w1); w1 += 4 * nr;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * nr;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
        if (prfm_acc_1 < prfm_end_1) {
          prfm_acc_1 += 4 * nr;
          if (prfm_acc_1 > prfm_end_1) {
            prfm_acc_1 = prfm_end_1;
          }
          for (; prfm_pos_1 < prfm_acc_1; prfm_pos_1 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
          }
        }
#endif

        svuint8_t vb0_raw0 = svreinterpret_u8(svget2(vb0_x2, 0));
        svuint8_t vb0_raw1 = svreinterpret_u8(svget2(vb0_x2, 1));
        svuint8_t vb1_raw0 = svreinterpret_u8(svget2(vb1_x2, 0));
        svuint8_t vb1_raw1 = svreinterpret_u8(svget2(vb1_x2, 1));

        svint8x2_t vb0_0 = svluti2_lane_zt_s8_x2(0, vb0_raw0, 0);
        svint8x2_t vb1_0 = svluti2_lane_zt_s8_x2(0, vb1_raw0, 0);
        svint8x2_t vb0_1 = svluti2_lane_zt_s8_x2(0, vb0_raw0, 1);
        svint8x2_t vb1_1 = svluti2_lane_zt_s8_x2(0, vb1_raw0, 1);
        svint8x2_t vb0_2 = svluti2_lane_zt_s8_x2(0, vb0_raw1, 0);
        svint8x2_t vb1_2 = svluti2_lane_zt_s8_x2(0, vb1_raw1, 0);
        svint8x2_t vb0_3 = svluti2_lane_zt_s8_x2(0, vb0_raw1, 1);
        svint8x2_t vb1_3 = svluti2_lane_zt_s8_x2(0, vb1_raw1, 1);

        svdot_lane_za32_s8_vg1x2(0, vb0_0, va, 0);
        svdot_lane_za32_s8_vg1x2(2, vb1_0, va, 0);
        svdot_lane_za32_s8_vg1x2(o1, vb0_1, va, 1);
        svdot_lane_za32_s8_vg1x2(o3, vb1_1, va, 1);
        svdot_lane_za32_s8_vg1x2(0, vb0_2, va, 2);
        svdot_lane_za32_s8_vg1x2(2, vb1_2, va, 2);
        svdot_lane_za32_s8_vg1x2(o1, vb0_3, va, 3);
        svdot_lane_za32_s8_vg1x2(o3, vb1_3, va, 3);
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
      for (; prfm_pos_1 < prfm_end_1; prfm_pos_1 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
      }
#endif
      vscale_0 = svld1_f32_x2(pcnt_32_all, (const float*) w0);
      vscale_1 = svld1_f32_x2(pcnt_32_all, (const float*) w1);
      w = w1 + 4 * nr;
    } else {
#ifdef PREFETCH_W_MATRIX
      size_t prfm_base_0 = (size_t) (w0 - w_base);
      size_t prfm_end_0 = need_prfm ? (prfm_base_0 + w_block_size) : 0;
      size_t prfm_pos_0 = need_prfm
          ? ((prfm_base_0 + prfm_line_size - 1) & ~(size_t) (prfm_line_size - 1))
          : 0;
      size_t prfm_acc_0 = need_prfm ? (prfm_base_0 + upfront_prefetch_w_block) : 0;
      for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif

      svint32x2_t vbias_0 = svld1_s32_x2(pcnt_32_all, (const int32_t*) w0); w0 += 4 * nr;
      svadd_za32_s32_vg1x2(0, vbias_0);

      #pragma clang loop unroll(disable)
      for (size_t k_iter = 0; k_iter < k_iters; ++k_iter) {
        svint8_t va = svld1rq_s8(svwhilelt_b8_u64(k_iter * 16, kc), a_inc); a_inc += 16;

        svint8x2_t vb0_x2 = svld1_s8_x2(pcnt_8_all, w0); w0 += 4 * nr;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * nr;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
#endif

        svuint8_t vb0_raw0 = svreinterpret_u8(svget2(vb0_x2, 0));
        svuint8_t vb0_raw1 = svreinterpret_u8(svget2(vb0_x2, 1));

        svint8x2_t vb0_0 = svluti2_lane_zt_s8_x2(0, vb0_raw0, 0);
        svint8x2_t vb0_1 = svluti2_lane_zt_s8_x2(0, vb0_raw0, 1);
        svint8x2_t vb0_2 = svluti2_lane_zt_s8_x2(0, vb0_raw1, 0);
        svint8x2_t vb0_3 = svluti2_lane_zt_s8_x2(0, vb0_raw1, 1);

        svdot_lane_za32_s8_vg1x2(0, vb0_0, va, 0);
        svdot_lane_za32_s8_vg1x2(o1, vb0_1, va, 1);
        svdot_lane_za32_s8_vg1x2(0, vb0_2, va, 2);
        svdot_lane_za32_s8_vg1x2(o1, vb0_3, va, 3);
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif
      vscale_0 = svld1_f32_x2(pcnt_32_all, (const float*) w0);
      vscale_1 = vscale_0;
      w = w0 + 4 * nr;
    }

    svint32x2_t vacc0_x2 = svread_za32_s32_vg1x2(0);
    svint32x2_t vacc2_x2 = svread_za32_s32_vg1x2(2);
    svint32x4_t vacc_x4 = svcreate4_s32(svget2_s32(vacc0_x2, 0), svget2_s32(vacc0_x2, 1), svget2_s32(vacc2_x2, 0), svget2_s32(vacc2_x2, 1));
    if (o1 != 0) {
      svint32x2_t vacc1_x2 = svread_za32_s32_vg1x2(1);
      svint32x2_t vacc3_x2 = svread_za32_s32_vg1x2(3);
      vacc_x4 = svcreate4_s32(
          svadd_s32_x(pg_32_all, svget2_s32(vacc0_x2, 0), svget2_s32(vacc1_x2, 0)),
          svadd_s32_x(pg_32_all, svget2_s32(vacc0_x2, 1), svget2_s32(vacc1_x2, 1)),
          svadd_s32_x(pg_32_all, svget2_s32(vacc2_x2, 0), svget2_s32(vacc3_x2, 0)),
          svadd_s32_x(pg_32_all, svget2_s32(vacc2_x2, 1), svget2_s32(vacc3_x2, 1)));
    }

    svfloat32x4_t vout_x4 = svcvt_f32_s32_x4(vacc_x4);

    svfloat32x4_t scaled_x4 = svcreate4_f32(
        svmul_f32_x(pg_32_all, svget4_f32(vout_x4, 0), svget2_f32(vscale_0, 0)),
        svmul_f32_x(pg_32_all, svget4_f32(vout_x4, 1), svget2_f32(vscale_0, 1)),
        svmul_f32_x(pg_32_all, svget4_f32(vout_x4, 2), svget2_f32(vscale_1, 0)),
        svmul_f32_x(pg_32_all, svget4_f32(vout_x4, 3), svget2_f32(vscale_1, 1)));

    scaled_x4 = svrintn_f32_x4(scaled_x4);

    svint32x4_t vquant_x4 = svcvt_s32_f32_x4(scaled_x4);
    vquant_x4 = svadd_single_s32_x4(vquant_x4, voutput_zp_vec);

    svint8_t vout_8 = svqcvt_s8_s32_x4(vquant_x4);
    vout_8 = svmax_s8_x(pg_8_all, vout_8, voutput_min_vec);
    vout_8 = svmin_s8_x(pg_8_all, vout_8, voutput_max_vec);

    svst1_s8(svwhilelt_b8_u64(0, nc), c_inc, vout_8);

    if (nc > 2 * nr) {
      c_inc += 2 * nr;
      nc -= 2 * nr;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
