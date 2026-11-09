// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pf32_qc8w_gemm_minmax_ukernel_1x32__sme2_get_mr() {
  return 1;
}

size_t xnn_pf32_qc8w_gemm_minmax_ukernel_1x32__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
static void pf32_qc8w_gemm_1x32_sme2_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    float min,
    float max)
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
  (void) mr;
  (void) cm_stride;
  assert(mr == 1);
  assert(nc != 0);
  assert(kc != 0);

  size_t vl = svcntw();

  svbool_t pg_32_all = svptrue_b32();
  svcount_t pcnt_32_all = svptrue_c32();

  svfloat32_t vmin = svdup_n_f32(min);
  svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  float* c0 = c;

  const size_t k_unroll4 = kc / 4;
  const size_t k_remainder = kc % 4;
  const size_t w_block_size = 8 * vl + kc * 2 * vl + 8 * vl;

#ifdef PREFETCH_W_MATRIX
  const int8_t* w_base = (const int8_t*) w;
  const size_t nc_rup_nr = ((nc + 2 * vl - 1) / (2 * vl)) * (2 * vl);
  const size_t total_w_size = (nc_rup_nr / (2 * vl)) * w_block_size;
  const size_t prfm_dist_size = 64 * 8 * vl;
  const size_t upfront_prefetch_w_block = (prfm_dist_size < w_block_size) ? prfm_dist_size : w_block_size;
  const bool need_prfm = (total_w_size > prfm_min_w_size);
#endif

  do {
    svzero_za();

    const float* a0 = a_base;
    const int8_t* w0 = (const int8_t*) w;
    svfloat32x2_t vscale_0;
    svfloat32x2_t vscale_1;

    if (nc <= vl) {
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

      const svfloat32_t vbias = svld1_f32(pg_32_all, (const float*) w0); w0 += 2 * vl * sizeof(float);
      svwrite_hor_za32_f32_m(0, 0, pg_32_all, vbias);

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        const svfloat32_t va = svld1rq_f32(pg_32_all, a0); a0 += 4;

        const svfloat32x4_t vb = svcvt_f32_s32_x4(svcreate4_s32(
            svld1sb_vnum_s32(pg_32_all, w0, 0), svld1sb_vnum_s32(pg_32_all, w0, 2),
            svld1sb_vnum_s32(pg_32_all, w0, 4), svld1sb_vnum_s32(pg_32_all, w0, 6)));
        w0 += 8 * vl;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 8 * vl;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
#endif

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va, svget4_f32(vb, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va, svget4_f32(vb, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va, svget4_f32(vb, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va, svget4_f32(vb, 3));
      }

      if (k_remainder != 0) {
        const svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va, svcvt_f32_s32_x(pg_32_all, svld1sb_s32(pg_32_all, w0)));
        w0 += 2 * vl;
        if (k_remainder > 1) {
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va, svcvt_f32_s32_x(pg_32_all, svld1sb_s32(pg_32_all, w0)));
          w0 += 2 * vl;
        }
        if (k_remainder > 2) {
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va, svcvt_f32_s32_x(pg_32_all, svld1sb_s32(pg_32_all, w0)));
          w0 += 2 * vl;
        }
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif

      const svfloat32_t vscale = svld1_f32(pg_32_all, (const float*) w0);
      const svfloat32_t vsum = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all,
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 0),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 1)),
          svadd_f32_x(pg_32_all,
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 2),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 3)));
      svfloat32x4_t vout_x4 = svcreate4_f32(svmul_f32_x(pg_32_all, vsum, vscale), vmin, vmin, vmin);
      vout_x4 = svclamp_single_f32_x4(vout_x4, vmin, vmax);
      svst1_f32(svwhilelt_b32_u64(0, nc), c0, svget4_f32(vout_x4, 0));
      break;
    }

    if (nc > 2 * vl) {
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

      svfloat32x2_t vbias_0 = svld1_f32_x2(pcnt_32_all, (const float*) w0); w0 += 2 * vl * sizeof(float);
      svfloat32x2_t vbias_1 = svld1_f32_x2(pcnt_32_all, (const float*) w1); w1 += 2 * vl * sizeof(float);
      svadd_za32_f32_vg1x2(0, vbias_0);
      svadd_za32_f32_vg1x2(4, vbias_1);

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        svfloat32_t va = svld1rq_f32(pg_32_all, a0); a0 += 4;

        svint32x2_t vw0_0 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                      svld1sb_vnum_s32(pg_32_all, w0, 1));
        svint32x2_t vw1_0 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w1, 0),
                                      svld1sb_vnum_s32(pg_32_all, w1, 1));
        svint32x2_t vw0_1 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 2),
                                      svld1sb_vnum_s32(pg_32_all, w0, 3));
        svint32x2_t vw1_1 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w1, 2),
                                      svld1sb_vnum_s32(pg_32_all, w1, 3));
        svint32x2_t vw0_2 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 4),
                                      svld1sb_vnum_s32(pg_32_all, w0, 5));
        svint32x2_t vw1_2 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w1, 4),
                                      svld1sb_vnum_s32(pg_32_all, w1, 5));
        svint32x2_t vw0_3 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 6),
                                      svld1sb_vnum_s32(pg_32_all, w0, 7));
        svint32x2_t vw1_3 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w1, 6),
                                      svld1sb_vnum_s32(pg_32_all, w1, 7));
        w0 += 8 * vl;
        w1 += 8 * vl;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 8 * vl;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
        if (prfm_acc_1 < prfm_end_1) {
          prfm_acc_1 += 8 * vl;
          if (prfm_acc_1 > prfm_end_1) {
            prfm_acc_1 = prfm_end_1;
          }
          for (; prfm_pos_1 < prfm_acc_1; prfm_pos_1 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
          }
        }
#endif

        svfloat32x2_t vb0_0 = svcvt_f32_s32_x2(vw0_0);
        svfloat32x2_t vb1_0 = svcvt_f32_s32_x2(vw1_0);
        svfloat32x2_t vb0_1 = svcvt_f32_s32_x2(vw0_1);
        svfloat32x2_t vb1_1 = svcvt_f32_s32_x2(vw1_1);

        svfloat32x2_t vb0_2 = svcvt_f32_s32_x2(vw0_2);
        svfloat32x2_t vb1_2 = svcvt_f32_s32_x2(vw1_2);

        svfloat32x2_t vb0_3 = svcvt_f32_s32_x2(vw0_3);
        svfloat32x2_t vb1_3 = svcvt_f32_s32_x2(vw1_3);

        svmla_lane_za32_f32_vg1x2(0, vb0_0, va, 0);
        svmla_lane_za32_f32_vg1x2(1, vb0_1, va, 1);
        svmla_lane_za32_f32_vg1x2(2, vb0_2, va, 2);
        svmla_lane_za32_f32_vg1x2(3, vb0_3, va, 3);
        svmla_lane_za32_f32_vg1x2(4, vb1_0, va, 0);
        svmla_lane_za32_f32_vg1x2(5, vb1_1, va, 1);
        svmla_lane_za32_f32_vg1x2(6, vb1_2, va, 2);
        svmla_lane_za32_f32_vg1x2(7, vb1_3, va, 3);
      }

      if (k_remainder != 0) {
        svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);

        svfloat32x2_t vb0_0 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                                         svld1sb_vnum_s32(pg_32_all, w0, 1)));
        svfloat32x2_t vb1_0 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w1, 0),
                                                         svld1sb_vnum_s32(pg_32_all, w1, 1)));
        w0 += 2 * vl;
        w1 += 2 * vl;
        svmla_lane_za32_f32_vg1x2(0, vb0_0, va, 0);
        svmla_lane_za32_f32_vg1x2(4, vb1_0, va, 0);

        if (k_remainder > 1) {
          svfloat32x2_t vb0_1 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                                           svld1sb_vnum_s32(pg_32_all, w0, 1)));
          svfloat32x2_t vb1_1 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w1, 0),
                                                           svld1sb_vnum_s32(pg_32_all, w1, 1)));
          w0 += 2 * vl;
          w1 += 2 * vl;
          svmla_lane_za32_f32_vg1x2(1, vb0_1, va, 1);
          svmla_lane_za32_f32_vg1x2(5, vb1_1, va, 1);
        }
        if (k_remainder > 2) {
          svfloat32x2_t vb0_2 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                                           svld1sb_vnum_s32(pg_32_all, w0, 1)));
          svfloat32x2_t vb1_2 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w1, 0),
                                                           svld1sb_vnum_s32(pg_32_all, w1, 1)));
          w0 += 2 * vl;
          w1 += 2 * vl;
          svmla_lane_za32_f32_vg1x2(2, vb0_2, va, 2);
          svmla_lane_za32_f32_vg1x2(6, vb1_2, va, 2);
        }
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
      w = w1 + 2 * vl * sizeof(float);
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

      svfloat32x2_t vbias_0 = svld1_f32_x2(pcnt_32_all, (const float*) w0); w0 += 2 * vl * sizeof(float);
      svadd_za32_f32_vg1x2(0, vbias_0);

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        svfloat32_t va = svld1rq_f32(pg_32_all, a0); a0 += 4;

        svint32x2_t vw0_0 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                      svld1sb_vnum_s32(pg_32_all, w0, 1));
        svint32x2_t vw0_1 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 2),
                                      svld1sb_vnum_s32(pg_32_all, w0, 3));
        svint32x2_t vw0_2 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 4),
                                      svld1sb_vnum_s32(pg_32_all, w0, 5));
        svint32x2_t vw0_3 = svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 6),
                                      svld1sb_vnum_s32(pg_32_all, w0, 7));
        w0 += 8 * vl;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 8 * vl;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
#endif

        svfloat32x2_t vb0_0 = svcvt_f32_s32_x2(vw0_0);
        svfloat32x2_t vb0_1 = svcvt_f32_s32_x2(vw0_1);

        svmla_lane_za32_f32_vg1x2(0, vb0_0, va, 0);

        svfloat32x2_t vb0_2 = svcvt_f32_s32_x2(vw0_2);

        svmla_lane_za32_f32_vg1x2(1, vb0_1, va, 1);

        svfloat32x2_t vb0_3 = svcvt_f32_s32_x2(vw0_3);

        svmla_lane_za32_f32_vg1x2(2, vb0_2, va, 2);
        svmla_lane_za32_f32_vg1x2(3, vb0_3, va, 3);
      }

      if (k_remainder != 0) {
        svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);

        svfloat32x2_t vb0_0 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                                         svld1sb_vnum_s32(pg_32_all, w0, 1)));
        w0 += 2 * vl;
        svmla_lane_za32_f32_vg1x2(0, vb0_0, va, 0);

        if (k_remainder > 1) {
          svfloat32x2_t vb0_1 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                                           svld1sb_vnum_s32(pg_32_all, w0, 1)));
          w0 += 2 * vl;
          svmla_lane_za32_f32_vg1x2(1, vb0_1, va, 1);
        }
        if (k_remainder > 2) {
          svfloat32x2_t vb0_2 = svcvt_f32_s32_x2(svcreate2_s32(svld1sb_vnum_s32(pg_32_all, w0, 0),
                                                           svld1sb_vnum_s32(pg_32_all, w0, 1)));
          w0 += 2 * vl;
          svmla_lane_za32_f32_vg1x2(2, vb0_2, va, 2);
        }
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif

      vscale_0 = svld1_f32_x2(pcnt_32_all, (const float*) w0);
      vscale_1 = vscale_0;
      w = w0 + 2 * vl * sizeof(float);
    }

    svfloat32x2_t vacc0_x2 = svread_za32_f32_vg1x2(0);
    svfloat32x2_t vacc1_x2 = svread_za32_f32_vg1x2(1);
    svfloat32x2_t vacc2_x2 = svread_za32_f32_vg1x2(2);
    svfloat32x2_t vacc3_x2 = svread_za32_f32_vg1x2(3);

    svfloat32x4_t vout_x4;
    if (nc > 2 * vl) {
      svfloat32x2_t vacc4_x2 = svread_za32_f32_vg1x2(4);
      svfloat32x2_t vacc5_x2 = svread_za32_f32_vg1x2(5);
      svfloat32x2_t vacc6_x2 = svread_za32_f32_vg1x2(6);
      svfloat32x2_t vacc7_x2 = svread_za32_f32_vg1x2(7);
      svfloat32_t vsum_0 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc0_x2, 0), svget2_f32(vacc1_x2, 0)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc2_x2, 0), svget2_f32(vacc3_x2, 0)));
      svfloat32_t vsum_1 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc0_x2, 1), svget2_f32(vacc1_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc2_x2, 1), svget2_f32(vacc3_x2, 1)));
      svfloat32_t vsum_2 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc4_x2, 0), svget2_f32(vacc5_x2, 0)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc6_x2, 0), svget2_f32(vacc7_x2, 0)));
      svfloat32_t vsum_3 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc4_x2, 1), svget2_f32(vacc5_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc6_x2, 1), svget2_f32(vacc7_x2, 1)));
      vout_x4 = svcreate4_f32(
          svmul_f32_x(pg_32_all, vsum_0, svget2_f32(vscale_0, 0)),
          svmul_f32_x(pg_32_all, vsum_1, svget2_f32(vscale_0, 1)),
          svmul_f32_x(pg_32_all, vsum_2, svget2_f32(vscale_1, 0)),
          svmul_f32_x(pg_32_all, vsum_3, svget2_f32(vscale_1, 1)));
    } else {
      svfloat32_t vsum_0 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc0_x2, 0), svget2_f32(vacc1_x2, 0)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc2_x2, 0), svget2_f32(vacc3_x2, 0)));
      svfloat32_t vsum_1 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc0_x2, 1), svget2_f32(vacc1_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc2_x2, 1), svget2_f32(vacc3_x2, 1)));
      vout_x4 = svcreate4_f32(
          svmul_f32_x(pg_32_all, vsum_0, svget2_f32(vscale_0, 0)),
          svmul_f32_x(pg_32_all, vsum_1, svget2_f32(vscale_0, 1)),
          vmin,
          vmin);
    }

    vout_x4 = svclamp_single_f32_x4(vout_x4, vmin, vmax);

    svst1_f32_x4(svwhilelt_c32_u64(0, nc, 4), c0, vout_x4);

    if (nc > 4 * vl) {
      c0 += 4 * vl;
      nc -= 4 * vl;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_pf32_qc8w_gemm_minmax_ukernel_1x32__sme2(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
    if XNN_UNLIKELY(mr == 0 || nc == 0) {
      return;
    }
    assert(mr == 1);
    assert(cn_stride == sizeof(float));
    (void) cn_stride;
    pf32_qc8w_gemm_1x32_sme2_impl(
      mr, nc, kc / sizeof(float), a, w, c, cm_stride,
      params->scalar.min, params->scalar.max);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
