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

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_1x32__sme2_get_mr() {
  return 1;
}

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_1x32__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za", "zt0")
static void pf32_qc4w_gemm_1x32_sme2_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    float min,
    float max,
    const float* lut)
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

  svbool_t pg_8_all = svptrue_b8();
  svbool_t pg_32_all = svptrue_b32();
  svcount_t pcnt_32_all = svptrue_c32();

  svfloat32_t vmin = svdup_n_f32(min);
  svfloat32_t vmax = svdup_n_f32(max);
  const svfloat32_t vzero = svdup_n_f32(0.0f);

  const float* a_base = (const float*) a;
  float* c0 = c;

  svldr_zt(0, lut);

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
      svadd_za32_f32_vg1x2(0, svzip_f32_x2(svcreate2_f32(vbias, vzero)));

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        uint64_t va_01_bits;
        uint64_t va_23_bits;
        __builtin_memcpy(&va_01_bits, a0, sizeof(va_01_bits));
        __builtin_memcpy(&va_23_bits, a0 + 2, sizeof(va_23_bits));
        a0 += 4;
        const svfloat32_t va_01 = svreinterpret_f32_u64(svdup_n_u64(va_01_bits));
        const svfloat32_t va_23 = svreinterpret_f32_u64(svdup_n_u64(va_23_bits));
        const svuint8_t vw = svld1_u8(pg_8_all, w0); w0 += 4 * vl;

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

        const svfloat32x2_t vb_01 = svluti4_lane_zt_f32_x2(0, vw, 0);
        const svfloat32x2_t vb_23 = svluti4_lane_zt_f32_x2(0, vw, 2);
        svmla_single_za32_f32_vg1x2(0, vb_01, va_01);
        svmla_single_za32_f32_vg1x2(1, vb_23, va_23);
      }

      if (k_remainder != 0) {
        const svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);
        const svuint8_t vw = svld1_u8(svwhilelt_b8_u64(0, k_remainder_bytes), w0); w0 += k_remainder_bytes;
        const svfloat32x2_t vb_01 = svluti4_lane_zt_f32_x2(0, vw, 0);
        svmla_single_za32_f32_vg1x2(0, vb_01, svreinterpret_f32_u64(svdup_lane_u64(svreinterpret_u64_f32(va), 0)));
        if (k_remainder > 2) {
          const svfloat32x2_t vb_23 = svluti4_lane_zt_f32_x2(0, vw, 2);
          svmla_single_za32_f32_vg1x2(1, vb_23, svreinterpret_f32_u64(svdup_lane_u64(svreinterpret_u64_f32(va), 1)));
        }
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif

      const svfloat32_t vscale = svld1_f32(pg_32_all, (const float*) w0);
      const svfloat32x2_t vacc_0 = svuzp_f32_x2(svread_za32_f32_vg1x2(0));
      const svfloat32x2_t vacc_1 = svuzp_f32_x2(svread_za32_f32_vg1x2(1));
      const svfloat32_t vsum = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc_0, 0), svget2_f32(vacc_0, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc_1, 0), svget2_f32(vacc_1, 1)));
      svfloat32x4_t vout_x4 = svcreate4_f32(svmul_f32_x(pg_32_all, vsum, vscale), vmin, vmin, vmin);
      vout_x4 = svclamp_single_f32_x4(vout_x4, vmin, vmax);
      svst1_f32(svwhilelt_b32_u64(0, nc), c0, svget4_f32(vout_x4, 0));
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

      const svfloat32x2_t vbias_0 = svld1_f32_x2(pcnt_32_all, (const float*) w0); w0 += 2 * vl * sizeof(float);
      const svfloat32x2_t vbias_1 = svld1_f32_x2(pcnt_32_all, (const float*) w1); w1 += 2 * vl * sizeof(float);
      svadd_za32_f32_vg1x2(0, svzip_f32_x2(svcreate2_f32(svget2_f32(vbias_0, 0), vzero)));
      svadd_za32_f32_vg1x2(1, svzip_f32_x2(svcreate2_f32(svget2_f32(vbias_0, 1), vzero)));
      svadd_za32_f32_vg1x2(2, svzip_f32_x2(svcreate2_f32(svget2_f32(vbias_1, 0), vzero)));
      svadd_za32_f32_vg1x2(3, svzip_f32_x2(svcreate2_f32(svget2_f32(vbias_1, 1), vzero)));

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        uint64_t va_01_bits;
        uint64_t va_23_bits;
        __builtin_memcpy(&va_01_bits, a0, sizeof(va_01_bits));
        __builtin_memcpy(&va_23_bits, a0 + 2, sizeof(va_23_bits));
        a0 += 4;
        const svfloat32_t va_01 = svreinterpret_f32_u64(svdup_n_u64(va_01_bits));
        const svfloat32_t va_23 = svreinterpret_f32_u64(svdup_n_u64(va_23_bits));
        const svuint8_t vw0 = svld1_u8(pg_8_all, w0);
        const svuint8_t vw1 = svld1_u8(pg_8_all, w1);
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

        const svfloat32x4_t vb0_01 = svluti4_lane_zt_f32_x4(0, vw0, 0);
        const svfloat32x4_t vb1_01 = svluti4_lane_zt_f32_x4(0, vw1, 0);
        const svfloat32x4_t vb0_23 = svluti4_lane_zt_f32_x4(0, vw0, 1);
        const svfloat32x4_t vb1_23 = svluti4_lane_zt_f32_x4(0, vw1, 1);

        svmla_single_za32_f32_vg1x2(0, svcreate2_f32(svget4_f32(vb0_01, 0), svget4_f32(vb0_01, 1)), va_01);
        svmla_single_za32_f32_vg1x2(1, svcreate2_f32(svget4_f32(vb0_01, 2), svget4_f32(vb0_01, 3)), va_01);
        svmla_single_za32_f32_vg1x2(2, svcreate2_f32(svget4_f32(vb1_01, 0), svget4_f32(vb1_01, 1)), va_01);
        svmla_single_za32_f32_vg1x2(3, svcreate2_f32(svget4_f32(vb1_01, 2), svget4_f32(vb1_01, 3)), va_01);
        svmla_single_za32_f32_vg1x2(4, svcreate2_f32(svget4_f32(vb0_23, 0), svget4_f32(vb0_23, 1)), va_23);
        svmla_single_za32_f32_vg1x2(5, svcreate2_f32(svget4_f32(vb0_23, 2), svget4_f32(vb0_23, 3)), va_23);
        svmla_single_za32_f32_vg1x2(6, svcreate2_f32(svget4_f32(vb1_23, 0), svget4_f32(vb1_23, 1)), va_23);
        svmla_single_za32_f32_vg1x2(7, svcreate2_f32(svget4_f32(vb1_23, 2), svget4_f32(vb1_23, 3)), va_23);
      }

      if (k_remainder != 0) {
        const svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);
        const svbool_t pg_rem = svwhilelt_b8_u64(0, k_remainder_bytes);
        const svuint8_t vw0 = svld1_u8(pg_rem, w0);
        const svuint8_t vw1 = svld1_u8(pg_rem, w1);
        w0 += k_remainder_bytes;
        w1 += k_remainder_bytes;
        const svfloat32x4_t vb0_01 = svluti4_lane_zt_f32_x4(0, vw0, 0);
        const svfloat32x4_t vb1_01 = svluti4_lane_zt_f32_x4(0, vw1, 0);
        const svfloat32_t va_01 = svreinterpret_f32_u64(svdup_lane_u64(svreinterpret_u64_f32(va), 0));
        svmla_single_za32_f32_vg1x2(0, svcreate2_f32(svget4_f32(vb0_01, 0), svget4_f32(vb0_01, 1)), va_01);
        svmla_single_za32_f32_vg1x2(1, svcreate2_f32(svget4_f32(vb0_01, 2), svget4_f32(vb0_01, 3)), va_01);
        svmla_single_za32_f32_vg1x2(2, svcreate2_f32(svget4_f32(vb1_01, 0), svget4_f32(vb1_01, 1)), va_01);
        svmla_single_za32_f32_vg1x2(3, svcreate2_f32(svget4_f32(vb1_01, 2), svget4_f32(vb1_01, 3)), va_01);
        if (k_remainder > 2) {
          const svfloat32x4_t vb0_23 = svluti4_lane_zt_f32_x4(0, vw0, 1);
          const svfloat32x4_t vb1_23 = svluti4_lane_zt_f32_x4(0, vw1, 1);
          const svfloat32_t va_23 = svreinterpret_f32_u64(svdup_lane_u64(svreinterpret_u64_f32(va), 1));
          svmla_single_za32_f32_vg1x2(4, svcreate2_f32(svget4_f32(vb0_23, 0), svget4_f32(vb0_23, 1)), va_23);
          svmla_single_za32_f32_vg1x2(5, svcreate2_f32(svget4_f32(vb0_23, 2), svget4_f32(vb0_23, 3)), va_23);
          svmla_single_za32_f32_vg1x2(6, svcreate2_f32(svget4_f32(vb1_23, 0), svget4_f32(vb1_23, 1)), va_23);
          svmla_single_za32_f32_vg1x2(7, svcreate2_f32(svget4_f32(vb1_23, 2), svget4_f32(vb1_23, 3)), va_23);
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

      const svfloat32x2_t vbias_0 = svld1_f32_x2(pcnt_32_all, (const float*) w0); w0 += 2 * vl * sizeof(float);
      svadd_za32_f32_vg1x2(0, svzip_f32_x2(svcreate2_f32(svget2_f32(vbias_0, 0), vzero)));
      svadd_za32_f32_vg1x2(1, svzip_f32_x2(svcreate2_f32(svget2_f32(vbias_0, 1), vzero)));

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        uint64_t va_01_bits;
        uint64_t va_23_bits;
        __builtin_memcpy(&va_01_bits, a0, sizeof(va_01_bits));
        __builtin_memcpy(&va_23_bits, a0 + 2, sizeof(va_23_bits));
        a0 += 4;
        const svfloat32_t va_01 = svreinterpret_f32_u64(svdup_n_u64(va_01_bits));
        const svfloat32_t va_23 = svreinterpret_f32_u64(svdup_n_u64(va_23_bits));
        const svuint8_t vw0 = svld1_u8(pg_8_all, w0);
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

        const svfloat32x4_t vb0_01 = svluti4_lane_zt_f32_x4(0, vw0, 0);
        const svfloat32x4_t vb0_23 = svluti4_lane_zt_f32_x4(0, vw0, 1);

        svmla_single_za32_f32_vg1x2(0, svcreate2_f32(svget4_f32(vb0_01, 0), svget4_f32(vb0_01, 1)), va_01);
        svmla_single_za32_f32_vg1x2(1, svcreate2_f32(svget4_f32(vb0_01, 2), svget4_f32(vb0_01, 3)), va_01);
        svmla_single_za32_f32_vg1x2(2, svcreate2_f32(svget4_f32(vb0_23, 0), svget4_f32(vb0_23, 1)), va_23);
        svmla_single_za32_f32_vg1x2(3, svcreate2_f32(svget4_f32(vb0_23, 2), svget4_f32(vb0_23, 3)), va_23);
      }

      if (k_remainder != 0) {
        const svfloat32_t va = svld1rq_f32(svwhilelt_b32_u64(0, k_remainder), a0);
        const svuint8_t vw0 = svld1_u8(svwhilelt_b8_u64(0, k_remainder_bytes), w0);
        w0 += k_remainder_bytes;
        const svfloat32x4_t vb0_01 = svluti4_lane_zt_f32_x4(0, vw0, 0);
        const svfloat32_t va_01 = svreinterpret_f32_u64(svdup_lane_u64(svreinterpret_u64_f32(va), 0));
        svmla_single_za32_f32_vg1x2(0, svcreate2_f32(svget4_f32(vb0_01, 0), svget4_f32(vb0_01, 1)), va_01);
        svmla_single_za32_f32_vg1x2(1, svcreate2_f32(svget4_f32(vb0_01, 2), svget4_f32(vb0_01, 3)), va_01);
        if (k_remainder > 2) {
          const svfloat32x4_t vb0_23 = svluti4_lane_zt_f32_x4(0, vw0, 1);
          const svfloat32_t va_23 = svreinterpret_f32_u64(svdup_lane_u64(svreinterpret_u64_f32(va), 1));
          svmla_single_za32_f32_vg1x2(2, svcreate2_f32(svget4_f32(vb0_23, 0), svget4_f32(vb0_23, 1)), va_23);
          svmla_single_za32_f32_vg1x2(3, svcreate2_f32(svget4_f32(vb0_23, 2), svget4_f32(vb0_23, 3)), va_23);
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

    svfloat32x2_t vacc0_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(0));
    svfloat32x2_t vacc1_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(1));
    svfloat32x2_t vacc2_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(2));
    svfloat32x2_t vacc3_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(3));

    svfloat32x4_t vout_x4;
    if (nc > 2 * vl) {
      svfloat32x2_t vacc4_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(4));
      svfloat32x2_t vacc5_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(5));
      svfloat32x2_t vacc6_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(6));
      svfloat32x2_t vacc7_x2 = svuzp_f32_x2(svread_za32_f32_vg1x2(7));
      svfloat32_t vsum_0 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc0_x2, 0), svget2_f32(vacc0_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc4_x2, 0), svget2_f32(vacc4_x2, 1)));
      svfloat32_t vsum_1 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc1_x2, 0), svget2_f32(vacc1_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc5_x2, 0), svget2_f32(vacc5_x2, 1)));
      svfloat32_t vsum_2 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc2_x2, 0), svget2_f32(vacc2_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc6_x2, 0), svget2_f32(vacc6_x2, 1)));
      svfloat32_t vsum_3 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc3_x2, 0), svget2_f32(vacc3_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc7_x2, 0), svget2_f32(vacc7_x2, 1)));
      vout_x4 = svcreate4_f32(
          svmul_f32_x(pg_32_all, vsum_0, svget2_f32(vscale_0, 0)),
          svmul_f32_x(pg_32_all, vsum_1, svget2_f32(vscale_0, 1)),
          svmul_f32_x(pg_32_all, vsum_2, svget2_f32(vscale_1, 0)),
          svmul_f32_x(pg_32_all, vsum_3, svget2_f32(vscale_1, 1)));
    } else {
      svfloat32_t vsum_0 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc0_x2, 0), svget2_f32(vacc0_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc2_x2, 0), svget2_f32(vacc2_x2, 1)));
      svfloat32_t vsum_1 = svadd_f32_x(pg_32_all,
          svadd_f32_x(pg_32_all, svget2_f32(vacc1_x2, 0), svget2_f32(vacc1_x2, 1)),
          svadd_f32_x(pg_32_all, svget2_f32(vacc3_x2, 0), svget2_f32(vacc3_x2, 1)));
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

void xnn_pf32_qc4w_gemm_minmax_ukernel_1x32__sme2(
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
    XNN_ALIGN(64) float lut[16];
    for (int32_t i = 0; i < 16; i++) {
      lut[i] = (float) (i - params->scalar.kernel_zero_point);
    }
    pf32_qc4w_gemm_1x32_sme2_impl(
      mr, nc, kc / sizeof(float), a, w, c, cm_stride,
      params->scalar.min, params->scalar.max, lut);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
