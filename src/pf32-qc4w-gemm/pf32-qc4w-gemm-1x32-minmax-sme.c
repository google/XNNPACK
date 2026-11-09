// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_1x32__sme_get_mr() {
  return 1;
}

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_1x32__sme_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
static void pf32_qc4w_gemm_1x32_sme_lut_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    float min,
    float max,
    int32_t kernel_zero_point)
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
  const svfloat32_t vlut = svcvt_f32_s32_x(pg_32_all, svsub_n_s32_x(pg_32_all, svindex_s32(0, 1), kernel_zero_point));
  const svuint32_t vrow = svindex_u32(0, 1);
  const svbool_t pg_row_0 = svcmpeq_n_u32(pg_32_all, vrow, 0);
  const svbool_t pg_row_1 = svcmpeq_n_u32(pg_32_all, vrow, 1);
  const svbool_t pg_row_2 = svcmpeq_n_u32(pg_32_all, vrow, 2);
  const svbool_t pg_row_3 = svcmpeq_n_u32(pg_32_all, vrow, 3);
  const svfloat32_t f32_one = svdup_n_f32(1.0f);

  svfloat32_t vmin = svdup_n_f32(min);
  svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  float* c0 = c;

  const size_t k_unroll4 = kc / 4;
  const size_t k_remainder = kc % 4;
  const size_t k_remainder_bytes = ((k_remainder + 1) / 2) * 2 * vl;
  const size_t w_block_size = 8 * vl + ((kc + 1) / 2) * 2 * vl + 8 * vl;

#ifdef PREFETCH_W_MATRIX
  const uint8_t* w_base = (const uint8_t*) w;
  const size_t nc_rup_nr = ((nc + 2 * vl - 1) / (2 * vl)) * (2 * vl);
  const size_t total_w_size = (nc_rup_nr / (2 * vl)) * w_block_size;
  const size_t prfm_dist_size = 64 * 8 * vl;
  const size_t upfront_prefetch_w_block = (prfm_dist_size < w_block_size) ? prfm_dist_size : w_block_size;
  const bool need_prfm = (total_w_size > prfm_min_w_size);
#endif

  do {
    svzero_za();

    const float* a0 = a_base;
    const uint8_t* w0 = (const uint8_t*) w;
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

        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw23 = svld1ub_vnum_u32(pg_32_all, w0, 2);
        w0 += 4 * vl;
        const svuint32_t vi01_lo = svand_n_u32_x(pg_32_all, vw01, 0xF);
        const svuint32_t vi01_hi = svlsr_n_u32_x(pg_32_all, vw01, 4);
        const svuint32_t vi23_lo = svand_n_u32_x(pg_32_all, vw23, 0xF);
        const svuint32_t vi23_hi = svlsr_n_u32_x(pg_32_all, vw23, 4);
        const svfloat32x4_t vb = svcreate4_f32(
            svtbl_f32(vlut, vi01_lo),
            svtbl_f32(vlut, vi01_hi),
            svtbl_f32(vlut, vi23_lo),
            svtbl_f32(vlut, vi23_hi));

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * vl;
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

        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vi01_lo = svand_n_u32_x(pg_32_all, vw01, 0xF);
        const svuint32_t vi01_hi = svlsr_n_u32_x(pg_32_all, vw01, 4);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va, svtbl_f32(vlut, vi01_lo));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va, svtbl_f32(vlut, vi01_hi));
        if (k_remainder > 2) {
          const svuint32_t vw23 = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vi23_lo = svand_n_u32_x(pg_32_all, vw23, 0xF);
          const svuint32_t vi23_hi = svlsr_n_u32_x(pg_32_all, vw23, 4);
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va, svtbl_f32(vlut, vi23_lo));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va, svtbl_f32(vlut, vi23_hi));
        }
        w0 += k_remainder_bytes;
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
      const svfloat32_t vout =
          svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, vsum, vscale), vmin), vmax);
      svst1_f32(svwhilelt_b32_u64(0, nc), c0, vout);
      break;
    }

    if (nc > 2 * vl) {
      const uint8_t* w1 = w0 + w_block_size;

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

      svfloat32x2_t vbias_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      w0 += 2 * vl * sizeof(float);
      svfloat32x2_t vbias_1 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w1, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w1, 1));
      w1 += 2 * vl * sizeof(float);
      svmopa_za32_f32_m(0, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 0));
      svmopa_za32_f32_m(1, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 1));
      svmopa_za32_f32_m(2, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_1, 0));
      svmopa_za32_f32_m(3, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_1, 1));

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        svfloat32_t va = svld1rq_f32(pg_32_all, a0); a0 += 4;

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vw01_2 = svld1ub_vnum_u32(pg_32_all, w1, 0);
        const svuint32_t vw01_3 = svld1ub_vnum_u32(pg_32_all, w1, 1);
        const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
        const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
        const svuint32_t vw23_2 = svld1ub_vnum_u32(pg_32_all, w1, 2);
        const svuint32_t vw23_3 = svld1ub_vnum_u32(pg_32_all, w1, 3);
        w0 += 4 * vl;
        w1 += 4 * vl;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * vl;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
        if (prfm_acc_1 < prfm_end_1) {
          prfm_acc_1 += 4 * vl;
          if (prfm_acc_1 > prfm_end_1) {
            prfm_acc_1 = prfm_end_1;
          }
          for (; prfm_pos_1 < prfm_acc_1; prfm_pos_1 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
          }
        }
#endif

        const svuint32_t vi01_0_lo = svand_n_u32_x(pg_32_all, vw01_0, 0xF);
        const svuint32_t vi01_0_hi = svlsr_n_u32_x(pg_32_all, vw01_0, 4);
        const svuint32_t vi01_1_lo = svand_n_u32_x(pg_32_all, vw01_1, 0xF);
        const svuint32_t vi01_1_hi = svlsr_n_u32_x(pg_32_all, vw01_1, 4);
        const svuint32_t vi01_2_lo = svand_n_u32_x(pg_32_all, vw01_2, 0xF);
        const svuint32_t vi01_2_hi = svlsr_n_u32_x(pg_32_all, vw01_2, 4);
        const svuint32_t vi01_3_lo = svand_n_u32_x(pg_32_all, vw01_3, 0xF);
        const svuint32_t vi01_3_hi = svlsr_n_u32_x(pg_32_all, vw01_3, 4);
        const svuint32_t vi23_0_lo = svand_n_u32_x(pg_32_all, vw23_0, 0xF);
        const svuint32_t vi23_0_hi = svlsr_n_u32_x(pg_32_all, vw23_0, 4);
        const svuint32_t vi23_1_lo = svand_n_u32_x(pg_32_all, vw23_1, 0xF);
        const svuint32_t vi23_1_hi = svlsr_n_u32_x(pg_32_all, vw23_1, 4);
        const svuint32_t vi23_2_lo = svand_n_u32_x(pg_32_all, vw23_2, 0xF);
        const svuint32_t vi23_2_hi = svlsr_n_u32_x(pg_32_all, vw23_2, 4);
        const svuint32_t vi23_3_lo = svand_n_u32_x(pg_32_all, vw23_3, 0xF);
        const svuint32_t vi23_3_hi = svlsr_n_u32_x(pg_32_all, vw23_3, 4);

        svfloat32x2_t vb0_0 = svcreate2_f32(svtbl_f32(vlut, vi01_0_lo), svtbl_f32(vlut, vi01_1_lo));
        svfloat32x2_t vb1_0 = svcreate2_f32(svtbl_f32(vlut, vi01_2_lo), svtbl_f32(vlut, vi01_3_lo));
        svfloat32x2_t vb0_1 = svcreate2_f32(svtbl_f32(vlut, vi01_0_hi), svtbl_f32(vlut, vi01_1_hi));
        svfloat32x2_t vb1_1 = svcreate2_f32(svtbl_f32(vlut, vi01_2_hi), svtbl_f32(vlut, vi01_3_hi));
        svfloat32x2_t vb0_2 = svcreate2_f32(svtbl_f32(vlut, vi23_0_lo), svtbl_f32(vlut, vi23_1_lo));
        svfloat32x2_t vb1_2 = svcreate2_f32(svtbl_f32(vlut, vi23_2_lo), svtbl_f32(vlut, vi23_3_lo));
        svfloat32x2_t vb0_3 = svcreate2_f32(svtbl_f32(vlut, vi23_0_hi), svtbl_f32(vlut, vi23_1_hi));
        svfloat32x2_t vb1_3 = svcreate2_f32(svtbl_f32(vlut, vi23_2_hi), svtbl_f32(vlut, vi23_3_hi));

        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 0));
        svmopa_za32_f32_m(3, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 1));
        svmopa_za32_f32_m(0, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(1, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 1));
        svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
        svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
        svmopa_za32_f32_m(2, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 0));
        svmopa_za32_f32_m(3, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 1));
        svmopa_za32_f32_m(0, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
        svmopa_za32_f32_m(1, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
        svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 0));
        svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 1));
      }

      if (k_remainder != 0) {
        svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vw01_2 = svld1ub_vnum_u32(pg_32_all, w1, 0);
        const svuint32_t vw01_3 = svld1ub_vnum_u32(pg_32_all, w1, 1);
        const svuint32_t vi01_0_lo = svand_n_u32_x(pg_32_all, vw01_0, 0xF);
        const svuint32_t vi01_0_hi = svlsr_n_u32_x(pg_32_all, vw01_0, 4);
        const svuint32_t vi01_1_lo = svand_n_u32_x(pg_32_all, vw01_1, 0xF);
        const svuint32_t vi01_1_hi = svlsr_n_u32_x(pg_32_all, vw01_1, 4);
        const svuint32_t vi01_2_lo = svand_n_u32_x(pg_32_all, vw01_2, 0xF);
        const svuint32_t vi01_2_hi = svlsr_n_u32_x(pg_32_all, vw01_2, 4);
        const svuint32_t vi01_3_lo = svand_n_u32_x(pg_32_all, vw01_3, 0xF);
        const svuint32_t vi01_3_hi = svlsr_n_u32_x(pg_32_all, vw01_3, 4);
        svfloat32x2_t vb0_0 = svcreate2_f32(svtbl_f32(vlut, vi01_0_lo), svtbl_f32(vlut, vi01_1_lo));
        svfloat32x2_t vb1_0 = svcreate2_f32(svtbl_f32(vlut, vi01_2_lo), svtbl_f32(vlut, vi01_3_lo));
        svfloat32x2_t vb0_1 = svcreate2_f32(svtbl_f32(vlut, vi01_0_hi), svtbl_f32(vlut, vi01_1_hi));
        svfloat32x2_t vb1_1 = svcreate2_f32(svtbl_f32(vlut, vi01_2_hi), svtbl_f32(vlut, vi01_3_hi));
        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 0));
        svmopa_za32_f32_m(3, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 1));
        svmopa_za32_f32_m(0, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(1, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 1));
        if (k_remainder > 2) {
          const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
          const svuint32_t vw23_2 = svld1ub_vnum_u32(pg_32_all, w1, 2);
          const svuint32_t vw23_3 = svld1ub_vnum_u32(pg_32_all, w1, 3);
          const svuint32_t vi23_0_lo = svand_n_u32_x(pg_32_all, vw23_0, 0xF);
          const svuint32_t vi23_0_hi = svlsr_n_u32_x(pg_32_all, vw23_0, 4);
          const svuint32_t vi23_1_lo = svand_n_u32_x(pg_32_all, vw23_1, 0xF);
          const svuint32_t vi23_1_hi = svlsr_n_u32_x(pg_32_all, vw23_1, 4);
          const svuint32_t vi23_2_lo = svand_n_u32_x(pg_32_all, vw23_2, 0xF);
          const svuint32_t vi23_2_hi = svlsr_n_u32_x(pg_32_all, vw23_2, 4);
          const svuint32_t vi23_3_lo = svand_n_u32_x(pg_32_all, vw23_3, 0xF);
          const svuint32_t vi23_3_hi = svlsr_n_u32_x(pg_32_all, vw23_3, 4);
          svfloat32x2_t vb0_2 = svcreate2_f32(svtbl_f32(vlut, vi23_0_lo), svtbl_f32(vlut, vi23_1_lo));
          svfloat32x2_t vb1_2 = svcreate2_f32(svtbl_f32(vlut, vi23_2_lo), svtbl_f32(vlut, vi23_3_lo));
          svfloat32x2_t vb0_3 = svcreate2_f32(svtbl_f32(vlut, vi23_0_hi), svtbl_f32(vlut, vi23_1_hi));
          svfloat32x2_t vb1_3 = svcreate2_f32(svtbl_f32(vlut, vi23_2_hi), svtbl_f32(vlut, vi23_3_hi));
          svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
          svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
          svmopa_za32_f32_m(2, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 0));
          svmopa_za32_f32_m(3, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 1));
          svmopa_za32_f32_m(0, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
          svmopa_za32_f32_m(1, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
          svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 0));
          svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 1));
        }
        w0 += k_remainder_bytes;
        w1 += k_remainder_bytes;
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
      for (; prfm_pos_1 < prfm_end_1; prfm_pos_1 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
      }
#endif

      vscale_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      vscale_1 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w1, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w1, 1));
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

      svfloat32x2_t vbias_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      w0 += 2 * vl * sizeof(float);
      svmopa_za32_f32_m(0, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 0));
      svmopa_za32_f32_m(1, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 1));

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        svfloat32_t va = svld1rq_f32(pg_32_all, a0); a0 += 4;

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
        const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
        w0 += 4 * vl;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * vl;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
#endif

        const svuint32_t vi01_0_lo = svand_n_u32_x(pg_32_all, vw01_0, 0xF);
        const svuint32_t vi01_0_hi = svlsr_n_u32_x(pg_32_all, vw01_0, 4);
        const svuint32_t vi01_1_lo = svand_n_u32_x(pg_32_all, vw01_1, 0xF);
        const svuint32_t vi01_1_hi = svlsr_n_u32_x(pg_32_all, vw01_1, 4);
        const svuint32_t vi23_0_lo = svand_n_u32_x(pg_32_all, vw23_0, 0xF);
        const svuint32_t vi23_0_hi = svlsr_n_u32_x(pg_32_all, vw23_0, 4);
        const svuint32_t vi23_1_lo = svand_n_u32_x(pg_32_all, vw23_1, 0xF);
        const svuint32_t vi23_1_hi = svlsr_n_u32_x(pg_32_all, vw23_1, 4);

        svfloat32x2_t vb0_0 = svcreate2_f32(svtbl_f32(vlut, vi01_0_lo), svtbl_f32(vlut, vi01_1_lo));
        svfloat32x2_t vb0_1 = svcreate2_f32(svtbl_f32(vlut, vi01_0_hi), svtbl_f32(vlut, vi01_1_hi));
        svfloat32x2_t vb0_2 = svcreate2_f32(svtbl_f32(vlut, vi23_0_lo), svtbl_f32(vlut, vi23_1_lo));
        svfloat32x2_t vb0_3 = svcreate2_f32(svtbl_f32(vlut, vi23_0_hi), svtbl_f32(vlut, vi23_1_hi));

        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
        svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
        svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
        svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
      }

      if (k_remainder != 0) {
        svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vi01_0_lo = svand_n_u32_x(pg_32_all, vw01_0, 0xF);
        const svuint32_t vi01_0_hi = svlsr_n_u32_x(pg_32_all, vw01_0, 4);
        const svuint32_t vi01_1_lo = svand_n_u32_x(pg_32_all, vw01_1, 0xF);
        const svuint32_t vi01_1_hi = svlsr_n_u32_x(pg_32_all, vw01_1, 4);
        svfloat32x2_t vb0_0 = svcreate2_f32(svtbl_f32(vlut, vi01_0_lo), svtbl_f32(vlut, vi01_1_lo));
        svfloat32x2_t vb0_1 = svcreate2_f32(svtbl_f32(vlut, vi01_0_hi), svtbl_f32(vlut, vi01_1_hi));
        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        if (k_remainder > 2) {
          const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
          const svuint32_t vi23_0_lo = svand_n_u32_x(pg_32_all, vw23_0, 0xF);
          const svuint32_t vi23_0_hi = svlsr_n_u32_x(pg_32_all, vw23_0, 4);
          const svuint32_t vi23_1_lo = svand_n_u32_x(pg_32_all, vw23_1, 0xF);
          const svuint32_t vi23_1_hi = svlsr_n_u32_x(pg_32_all, vw23_1, 4);
          svfloat32x2_t vb0_2 = svcreate2_f32(svtbl_f32(vlut, vi23_0_lo), svtbl_f32(vlut, vi23_1_lo));
          svfloat32x2_t vb0_3 = svcreate2_f32(svtbl_f32(vlut, vi23_0_hi), svtbl_f32(vlut, vi23_1_hi));
          svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
          svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
          svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
          svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
        }
        w0 += k_remainder_bytes;
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif

      vscale_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      vscale_1 = vscale_0;
      w = w0 + 2 * vl * sizeof(float);
    }

    svfloat32x4_t vacc0_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 3));
    svfloat32x4_t vacc1_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 3));
    svfloat32x4_t vacc2_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 3));
    svfloat32x4_t vacc3_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 3));

    svfloat32x4_t vout_x4;
    if (nc > 2 * vl) {
      vout_x4 = svcreate4_f32(
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 0), svget4_f32(vacc0_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 2), svget4_f32(vacc0_x4, 3))), svget2_f32(vscale_0, 0)),
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 0), svget4_f32(vacc1_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 2), svget4_f32(vacc1_x4, 3))), svget2_f32(vscale_0, 1)),
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc2_x4, 0), svget4_f32(vacc2_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc2_x4, 2), svget4_f32(vacc2_x4, 3))), svget2_f32(vscale_1, 0)),
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc3_x4, 0), svget4_f32(vacc3_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc3_x4, 2), svget4_f32(vacc3_x4, 3))), svget2_f32(vscale_1, 1)));
    } else {
      svfloat32_t vsum_0 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 0), svget4_f32(vacc2_x4, 1)),
          svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 2), svget4_f32(vacc2_x4, 3)));
      svfloat32_t vsum_1 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 0), svget4_f32(vacc3_x4, 1)),
          svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 2), svget4_f32(vacc3_x4, 3)));
      vout_x4 = svcreate4_f32(
          svmul_f32_x(pg_32_all, vsum_0, svget2_f32(vscale_0, 0)),
          svmul_f32_x(pg_32_all, vsum_1, svget2_f32(vscale_0, 1)),
          vmin,
          vmin);
    }

    vout_x4 = svcreate4_f32(
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 0), vmin), vmax),
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 1), vmin), vmax),
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 2), vmin), vmax),
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 3), vmin), vmax));

    svst1_f32(svwhilelt_b32_u64(0, nc), c0, svget4_f32(vout_x4, 0));
    svst1_f32(svwhilelt_b32_u64(vl, nc), c0 + vl, svget4_f32(vout_x4, 1));
    svst1_f32(svwhilelt_b32_u64(2 * vl, nc), c0 + 2 * vl, svget4_f32(vout_x4, 2));
    svst1_f32(svwhilelt_b32_u64(3 * vl, nc), c0 + 3 * vl, svget4_f32(vout_x4, 3));

    if (nc > 4 * vl) {
      c0 += 4 * vl;
      nc -= 4 * vl;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}

__arm_locally_streaming __arm_new("za")
static void pf32_qc4w_gemm_1x32_sme_fmagic_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    float min,
    float max,
    int32_t kernel_zero_point)
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
  const svbool_t pg_16_all = svptrue_b16();
  const svuint32_t vmagic = svdup_n_u32(UINT32_C(0x64006400));
  const svuint32_t vmask = svdup_n_u32(UINT32_C(0x000F000F));
  const uint16_t magic_bias_bits = (uint16_t) (UINT16_C(0x6400) + kernel_zero_point);
  const svfloat16_t vmagic_bias = svreinterpret_f16_u16(svdup_n_u16(magic_bias_bits));
  const svuint32_t vrow = svindex_u32(0, 1);
  const svbool_t pg_row_0 = svcmpeq_n_u32(pg_32_all, vrow, 0);
  const svbool_t pg_row_1 = svcmpeq_n_u32(pg_32_all, vrow, 1);
  const svbool_t pg_row_2 = svcmpeq_n_u32(pg_32_all, vrow, 2);
  const svbool_t pg_row_3 = svcmpeq_n_u32(pg_32_all, vrow, 3);
  const svfloat32_t f32_one = svdup_n_f32(1.0f);

  svfloat32_t vmin = svdup_n_f32(min);
  svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  float* c0 = c;

  const size_t k_unroll4 = kc / 4;
  const size_t k_remainder = kc % 4;
  const size_t k_remainder_bytes = ((k_remainder + 1) / 2) * 2 * vl;
  const size_t w_block_size = 8 * vl + ((kc + 1) / 2) * 2 * vl + 8 * vl;

#ifdef PREFETCH_W_MATRIX
  const uint8_t* w_base = (const uint8_t*) w;
  const size_t nc_rup_nr = ((nc + 2 * vl - 1) / (2 * vl)) * (2 * vl);
  const size_t total_w_size = (nc_rup_nr / (2 * vl)) * w_block_size;
  const size_t prfm_dist_size = 64 * 8 * vl;
  const size_t upfront_prefetch_w_block = (prfm_dist_size < w_block_size) ? prfm_dist_size : w_block_size;
  const bool need_prfm = (total_w_size > prfm_min_w_size);
#endif

  do {
    svzero_za();

    const float* a0 = a_base;
    const uint8_t* w0 = (const uint8_t*) w;
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

        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw23 = svld1ub_vnum_u32(pg_32_all, w0, 2);
        w0 += 4 * vl;
        const svuint32_t vt01 = svbsl_u32(svsli_n_u32(vw01, vw01, 12), vmagic, vmask);
        const svfloat16_t vh01 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01), vmagic_bias);
        const svuint32_t vt23 = svbsl_u32(svsli_n_u32(vw23, vw23, 12), vmagic, vmask);
        const svfloat16_t vh23 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23), vmagic_bias);
        const svfloat32x4_t vb = svcreate4_f32(
            svcvt_f32_f16_x(pg_32_all, vh01),
            svcvtlt_f32_f16_x(pg_32_all, vh01),
            svcvt_f32_f16_x(pg_32_all, vh23),
            svcvtlt_f32_f16_x(pg_32_all, vh23));

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * vl;
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

        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vt01 = svbsl_u32(svsli_n_u32(vw01, vw01, 12), vmagic, vmask);
        const svfloat16_t vh01 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01), vmagic_bias);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va, svcvt_f32_f16_x(pg_32_all, vh01));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va, svcvtlt_f32_f16_x(pg_32_all, vh01));
        if (k_remainder > 2) {
          const svuint32_t vw23 = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vt23 = svbsl_u32(svsli_n_u32(vw23, vw23, 12), vmagic, vmask);
          const svfloat16_t vh23 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23), vmagic_bias);
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va, svcvt_f32_f16_x(pg_32_all, vh23));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va, svcvtlt_f32_f16_x(pg_32_all, vh23));
        }
        w0 += k_remainder_bytes;
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
      const svfloat32_t vout =
          svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, vsum, vscale), vmin), vmax);
      svst1_f32(svwhilelt_b32_u64(0, nc), c0, vout);
      break;
    }

    if (nc > 2 * vl) {
      const uint8_t* w1 = w0 + w_block_size;

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

      svfloat32x2_t vbias_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      w0 += 2 * vl * sizeof(float);
      svfloat32x2_t vbias_1 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w1, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w1, 1));
      w1 += 2 * vl * sizeof(float);
      svmopa_za32_f32_m(0, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 0));
      svmopa_za32_f32_m(1, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 1));
      svmopa_za32_f32_m(2, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_1, 0));
      svmopa_za32_f32_m(3, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_1, 1));

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        svfloat32_t va = svld1rq_f32(pg_32_all, a0); a0 += 4;

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vw01_2 = svld1ub_vnum_u32(pg_32_all, w1, 0);
        const svuint32_t vw01_3 = svld1ub_vnum_u32(pg_32_all, w1, 1);
        const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
        const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
        const svuint32_t vw23_2 = svld1ub_vnum_u32(pg_32_all, w1, 2);
        const svuint32_t vw23_3 = svld1ub_vnum_u32(pg_32_all, w1, 3);
        w0 += 4 * vl;
        w1 += 4 * vl;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * vl;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
        if (prfm_acc_1 < prfm_end_1) {
          prfm_acc_1 += 4 * vl;
          if (prfm_acc_1 > prfm_end_1) {
            prfm_acc_1 = prfm_end_1;
          }
          for (; prfm_pos_1 < prfm_acc_1; prfm_pos_1 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
          }
        }
#endif

        const svuint32_t vt01_0 = svbsl_u32(svsli_n_u32(vw01_0, vw01_0, 12), vmagic, vmask);
        const svfloat16_t vh01_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_0), vmagic_bias);
        const svuint32_t vt01_1 = svbsl_u32(svsli_n_u32(vw01_1, vw01_1, 12), vmagic, vmask);
        const svfloat16_t vh01_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_1), vmagic_bias);
        const svuint32_t vt01_2 = svbsl_u32(svsli_n_u32(vw01_2, vw01_2, 12), vmagic, vmask);
        const svfloat16_t vh01_2 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_2), vmagic_bias);
        const svuint32_t vt01_3 = svbsl_u32(svsli_n_u32(vw01_3, vw01_3, 12), vmagic, vmask);
        const svfloat16_t vh01_3 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_3), vmagic_bias);
        const svuint32_t vt23_0 = svbsl_u32(svsli_n_u32(vw23_0, vw23_0, 12), vmagic, vmask);
        const svfloat16_t vh23_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_0), vmagic_bias);
        const svuint32_t vt23_1 = svbsl_u32(svsli_n_u32(vw23_1, vw23_1, 12), vmagic, vmask);
        const svfloat16_t vh23_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_1), vmagic_bias);
        const svuint32_t vt23_2 = svbsl_u32(svsli_n_u32(vw23_2, vw23_2, 12), vmagic, vmask);
        const svfloat16_t vh23_2 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_2), vmagic_bias);
        const svuint32_t vt23_3 = svbsl_u32(svsli_n_u32(vw23_3, vw23_3, 12), vmagic, vmask);
        const svfloat16_t vh23_3 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_3), vmagic_bias);

        svfloat32x2_t vb0_0 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh01_0), svcvt_f32_f16_x(pg_32_all, vh01_1));
        svfloat32x2_t vb1_0 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh01_2), svcvt_f32_f16_x(pg_32_all, vh01_3));
        svfloat32x2_t vb0_1 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh01_0), svcvtlt_f32_f16_x(pg_32_all, vh01_1));
        svfloat32x2_t vb1_1 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh01_2), svcvtlt_f32_f16_x(pg_32_all, vh01_3));
        svfloat32x2_t vb0_2 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh23_0), svcvt_f32_f16_x(pg_32_all, vh23_1));
        svfloat32x2_t vb1_2 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh23_2), svcvt_f32_f16_x(pg_32_all, vh23_3));
        svfloat32x2_t vb0_3 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh23_0), svcvtlt_f32_f16_x(pg_32_all, vh23_1));
        svfloat32x2_t vb1_3 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh23_2), svcvtlt_f32_f16_x(pg_32_all, vh23_3));

        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 0));
        svmopa_za32_f32_m(3, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 1));
        svmopa_za32_f32_m(0, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(1, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 1));
        svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
        svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
        svmopa_za32_f32_m(2, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 0));
        svmopa_za32_f32_m(3, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 1));
        svmopa_za32_f32_m(0, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
        svmopa_za32_f32_m(1, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
        svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 0));
        svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 1));
      }

      if (k_remainder != 0) {
        svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vw01_2 = svld1ub_vnum_u32(pg_32_all, w1, 0);
        const svuint32_t vw01_3 = svld1ub_vnum_u32(pg_32_all, w1, 1);
        const svuint32_t vt01_0 = svbsl_u32(svsli_n_u32(vw01_0, vw01_0, 12), vmagic, vmask);
        const svfloat16_t vh01_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_0), vmagic_bias);
        const svuint32_t vt01_1 = svbsl_u32(svsli_n_u32(vw01_1, vw01_1, 12), vmagic, vmask);
        const svfloat16_t vh01_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_1), vmagic_bias);
        const svuint32_t vt01_2 = svbsl_u32(svsli_n_u32(vw01_2, vw01_2, 12), vmagic, vmask);
        const svfloat16_t vh01_2 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_2), vmagic_bias);
        const svuint32_t vt01_3 = svbsl_u32(svsli_n_u32(vw01_3, vw01_3, 12), vmagic, vmask);
        const svfloat16_t vh01_3 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_3), vmagic_bias);
        svfloat32x2_t vb0_0 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh01_0), svcvt_f32_f16_x(pg_32_all, vh01_1));
        svfloat32x2_t vb1_0 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh01_2), svcvt_f32_f16_x(pg_32_all, vh01_3));
        svfloat32x2_t vb0_1 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh01_0), svcvtlt_f32_f16_x(pg_32_all, vh01_1));
        svfloat32x2_t vb1_1 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh01_2), svcvtlt_f32_f16_x(pg_32_all, vh01_3));
        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 0));
        svmopa_za32_f32_m(3, pg_row_0, pg_32_all, va, svget2_f32(vb1_0, 1));
        svmopa_za32_f32_m(0, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(1, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb1_1, 1));
        if (k_remainder > 2) {
          const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
          const svuint32_t vw23_2 = svld1ub_vnum_u32(pg_32_all, w1, 2);
          const svuint32_t vw23_3 = svld1ub_vnum_u32(pg_32_all, w1, 3);
          const svuint32_t vt23_0 = svbsl_u32(svsli_n_u32(vw23_0, vw23_0, 12), vmagic, vmask);
          const svfloat16_t vh23_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_0), vmagic_bias);
          const svuint32_t vt23_1 = svbsl_u32(svsli_n_u32(vw23_1, vw23_1, 12), vmagic, vmask);
          const svfloat16_t vh23_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_1), vmagic_bias);
          const svuint32_t vt23_2 = svbsl_u32(svsli_n_u32(vw23_2, vw23_2, 12), vmagic, vmask);
          const svfloat16_t vh23_2 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_2), vmagic_bias);
          const svuint32_t vt23_3 = svbsl_u32(svsli_n_u32(vw23_3, vw23_3, 12), vmagic, vmask);
          const svfloat16_t vh23_3 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_3), vmagic_bias);
          svfloat32x2_t vb0_2 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh23_0), svcvt_f32_f16_x(pg_32_all, vh23_1));
          svfloat32x2_t vb1_2 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh23_2), svcvt_f32_f16_x(pg_32_all, vh23_3));
          svfloat32x2_t vb0_3 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh23_0), svcvtlt_f32_f16_x(pg_32_all, vh23_1));
          svfloat32x2_t vb1_3 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh23_2), svcvtlt_f32_f16_x(pg_32_all, vh23_3));
          svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
          svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
          svmopa_za32_f32_m(2, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 0));
          svmopa_za32_f32_m(3, pg_row_2, pg_32_all, va, svget2_f32(vb1_2, 1));
          svmopa_za32_f32_m(0, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
          svmopa_za32_f32_m(1, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
          svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 0));
          svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb1_3, 1));
        }
        w0 += k_remainder_bytes;
        w1 += k_remainder_bytes;
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
      for (; prfm_pos_1 < prfm_end_1; prfm_pos_1 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
      }
#endif

      vscale_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      vscale_1 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w1, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w1, 1));
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

      svfloat32x2_t vbias_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      w0 += 2 * vl * sizeof(float);
      svmopa_za32_f32_m(0, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 0));
      svmopa_za32_f32_m(1, pg_row_0, pg_32_all, f32_one, svget2_f32(vbias_0, 1));

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        svfloat32_t va = svld1rq_f32(pg_32_all, a0); a0 += 4;

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
        const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
        w0 += 4 * vl;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 4 * vl;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
#endif

        const svuint32_t vt01_0 = svbsl_u32(svsli_n_u32(vw01_0, vw01_0, 12), vmagic, vmask);
        const svfloat16_t vh01_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_0), vmagic_bias);
        const svuint32_t vt01_1 = svbsl_u32(svsli_n_u32(vw01_1, vw01_1, 12), vmagic, vmask);
        const svfloat16_t vh01_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_1), vmagic_bias);
        const svuint32_t vt23_0 = svbsl_u32(svsli_n_u32(vw23_0, vw23_0, 12), vmagic, vmask);
        const svfloat16_t vh23_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_0), vmagic_bias);
        const svuint32_t vt23_1 = svbsl_u32(svsli_n_u32(vw23_1, vw23_1, 12), vmagic, vmask);
        const svfloat16_t vh23_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_1), vmagic_bias);

        svfloat32x2_t vb0_0 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh01_0), svcvt_f32_f16_x(pg_32_all, vh01_1));
        svfloat32x2_t vb0_1 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh01_0), svcvtlt_f32_f16_x(pg_32_all, vh01_1));
        svfloat32x2_t vb0_2 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh23_0), svcvt_f32_f16_x(pg_32_all, vh23_1));
        svfloat32x2_t vb0_3 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh23_0), svcvtlt_f32_f16_x(pg_32_all, vh23_1));

        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
        svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
        svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
        svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
      }

      if (k_remainder != 0) {
        svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, w0, 1);
        const svuint32_t vt01_0 = svbsl_u32(svsli_n_u32(vw01_0, vw01_0, 12), vmagic, vmask);
        const svfloat16_t vh01_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_0), vmagic_bias);
        const svuint32_t vt01_1 = svbsl_u32(svsli_n_u32(vw01_1, vw01_1, 12), vmagic, vmask);
        const svfloat16_t vh01_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_1), vmagic_bias);
        svfloat32x2_t vb0_0 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh01_0), svcvt_f32_f16_x(pg_32_all, vh01_1));
        svfloat32x2_t vb0_1 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh01_0), svcvtlt_f32_f16_x(pg_32_all, vh01_1));
        svmopa_za32_f32_m(0, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 0));
        svmopa_za32_f32_m(1, pg_row_0, pg_32_all, va, svget2_f32(vb0_0, 1));
        svmopa_za32_f32_m(2, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 0));
        svmopa_za32_f32_m(3, pg_row_1, pg_32_all, va, svget2_f32(vb0_1, 1));
        if (k_remainder > 2) {
          const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, w0, 3);
          const svuint32_t vt23_0 = svbsl_u32(svsli_n_u32(vw23_0, vw23_0, 12), vmagic, vmask);
          const svfloat16_t vh23_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_0), vmagic_bias);
          const svuint32_t vt23_1 = svbsl_u32(svsli_n_u32(vw23_1, vw23_1, 12), vmagic, vmask);
          const svfloat16_t vh23_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_1), vmagic_bias);
          svfloat32x2_t vb0_2 = svcreate2_f32(svcvt_f32_f16_x(pg_32_all, vh23_0), svcvt_f32_f16_x(pg_32_all, vh23_1));
          svfloat32x2_t vb0_3 = svcreate2_f32(svcvtlt_f32_f16_x(pg_32_all, vh23_0), svcvtlt_f32_f16_x(pg_32_all, vh23_1));
          svmopa_za32_f32_m(0, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 0));
          svmopa_za32_f32_m(1, pg_row_2, pg_32_all, va, svget2_f32(vb0_2, 1));
          svmopa_za32_f32_m(2, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 0));
          svmopa_za32_f32_m(3, pg_row_3, pg_32_all, va, svget2_f32(vb0_3, 1));
        }
        w0 += k_remainder_bytes;
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif

      vscale_0 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w0, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w0, 1));
      vscale_1 = vscale_0;
      w = w0 + 2 * vl * sizeof(float);
    }

    svfloat32x4_t vacc0_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 3));
    svfloat32x4_t vacc1_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 3));
    svfloat32x4_t vacc2_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 3));
    svfloat32x4_t vacc3_x4 = svcreate4_f32(
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 0),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 1),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 2),
        svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 3));

    svfloat32x4_t vout_x4;
    if (nc > 2 * vl) {
      vout_x4 = svcreate4_f32(
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 0), svget4_f32(vacc0_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 2), svget4_f32(vacc0_x4, 3))), svget2_f32(vscale_0, 0)),
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 0), svget4_f32(vacc1_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 2), svget4_f32(vacc1_x4, 3))), svget2_f32(vscale_0, 1)),
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc2_x4, 0), svget4_f32(vacc2_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc2_x4, 2), svget4_f32(vacc2_x4, 3))), svget2_f32(vscale_1, 0)),
          svmul_f32_x(pg_32_all, svadd_f32_x(pg_32_all,
              svadd_f32_x(pg_32_all, svget4_f32(vacc3_x4, 0), svget4_f32(vacc3_x4, 1)),
              svadd_f32_x(pg_32_all, svget4_f32(vacc3_x4, 2), svget4_f32(vacc3_x4, 3))), svget2_f32(vscale_1, 1)));
    } else {
      svfloat32_t vsum_0 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 0), svget4_f32(vacc2_x4, 1)),
          svadd_f32_x(pg_32_all, svget4_f32(vacc0_x4, 2), svget4_f32(vacc2_x4, 3)));
      svfloat32_t vsum_1 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 0), svget4_f32(vacc3_x4, 1)),
          svadd_f32_x(pg_32_all, svget4_f32(vacc1_x4, 2), svget4_f32(vacc3_x4, 3)));
      vout_x4 = svcreate4_f32(
          svmul_f32_x(pg_32_all, vsum_0, svget2_f32(vscale_0, 0)),
          svmul_f32_x(pg_32_all, vsum_1, svget2_f32(vscale_0, 1)),
          vmin,
          vmin);
    }

    vout_x4 = svcreate4_f32(
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 0), vmin), vmax),
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 1), vmin), vmax),
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 2), vmin), vmax),
        svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout_x4, 3), vmin), vmax));

    svst1_f32(svwhilelt_b32_u64(0, nc), c0, svget4_f32(vout_x4, 0));
    svst1_f32(svwhilelt_b32_u64(vl, nc), c0 + vl, svget4_f32(vout_x4, 1));
    svst1_f32(svwhilelt_b32_u64(2 * vl, nc), c0 + 2 * vl, svget4_f32(vout_x4, 2));
    svst1_f32(svwhilelt_b32_u64(3 * vl, nc), c0 + 3 * vl, svget4_f32(vout_x4, 3));

    if (nc > 4 * vl) {
      c0 += 4 * vl;
      nc -= 4 * vl;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_pf32_qc4w_gemm_minmax_ukernel_1x32__sme(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params)
{
    if XNN_UNLIKELY(mr == 0 || nc == 0) {
      return;
    }
    assert(mr == 1);
    assert(cn_stride == sizeof(float));
    (void) cn_stride;
    // With SVL >= 512 one Z register holds the whole 16-entry dequantization table (svtbl); smaller SVLs use
    // the fp16 magic-number unpack.
    if (svcntsw() >= 16) {
      pf32_qc4w_gemm_1x32_sme_lut_impl(
        mr, nc, kc / sizeof(float), a, w, c, cm_stride,
        params->scalar.min, params->scalar.max, params->scalar.kernel_zero_point);
    } else {
      pf32_qc4w_gemm_1x32_sme_fmagic_impl(
        mr, nc, kc / sizeof(float), a, w, c, cm_stride,
        params->scalar.min, params->scalar.max, params->scalar.kernel_zero_point);
    }
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
