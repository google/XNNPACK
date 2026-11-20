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

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__sme_get_mr(void) {
  return 1;
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__sme_get_nr(void) {
  return 2 * svcntsw();
}

#if defined(__APPLE__)
#define PREFETCH_W_MATRIX
#endif

__arm_locally_streaming __arm_new("za")
void xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__sme(
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
  const size_t prfm_line_size = 128;
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
#endif
  if XNN_UNLIKELY(mr == 0 || nc == 0) {
    return;
  }
  assert(mr == 1);
  assert(kc != 0);
  assert(dst_stride_col == sizeof(int8_t));
  (void) mr;
  (void) dst_stride_row;
  (void) dst_stride_col;

  const int32_t output_zero_point = minmax_params->fp32_scalar.output_zero_point;
  const int32_t output_min = minmax_params->fp32_scalar.output_min;
  const int32_t output_max = minmax_params->fp32_scalar.output_max;

  const svfloat32_t vmagic_bias = svdup_n_f32(12582912.0f);
  const svint32_t vmagic_min = svdup_n_s32(INT32_C(0x4B400000) + (output_min - output_zero_point));
  const svint32_t vmagic_max = svdup_n_s32(INT32_C(0x4B400000) + (output_max - output_zero_point));
  const svint16_t vzp_s16 = svdup_n_s16((int16_t) output_zero_point);

  const size_t vl = svcntw();
  const size_t nr = 2 * vl;
  const svbool_t pg_8_all = svptrue_b8();
  const svbool_t pg_16_all = svptrue_b16();
  const svbool_t pg_32_all = svptrue_b32();
  const svuint8_t vrow_idx = svlsr_n_u8_x(pg_8_all, svindex_u8(0, 1), 2);
  const svbool_t pg_row_0 = svcmpeq_n_u8(pg_8_all, vrow_idx, 0);
  const svbool_t pg_row_1 = svcmpeq_n_u8(pg_8_all, vrow_idx, 1);
  const svbool_t pg_row_2 = svcmpeq_n_u8(pg_8_all, vrow_idx, 2);
  const svbool_t pg_row_3 = svcmpeq_n_u8(pg_8_all, vrow_idx, 3);

  const size_t k_block4 = (kc + 3) / 4;
  const size_t k_unroll4 = kc / 16;
  const size_t k_remainder = k_block4 - k_unroll4 * 4;
  const size_t k_tail = kc - k_unroll4 * 16;

  const size_t w_block_size = 4 * nr + k_block4 * 4 * nr + 4 * nr;

  int8_t* c_inc = (int8_t*) dst;

#ifdef PREFETCH_W_MATRIX
  const int8_t* w_base = (const int8_t*) w;
  const size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  const size_t total_w_size = (nc_rup_nr / nr) * w_block_size;
  const size_t prfm_dist_size = 16 * 16 * nr;
  const size_t upfront_prefetch_w_block = (prfm_dist_size < w_block_size) ? prfm_dist_size : w_block_size;
  const bool need_prfm = total_w_size > prfm_min_w_size;
#endif

  do {
    svzero_za();

    const int8_t* a_inc = (const int8_t*) lhs_packed;
    const int8_t* w0 = (const int8_t*) w;
    svfloat32_t vscale_0;
    svfloat32_t vscale_1;
    svfloat32_t vscale_2;
    svfloat32_t vscale_3;
    svint32_t vacc_0;
    svint32_t vacc_1;
    svint32_t vacc_2;
    svint32_t vacc_3;

    if (nc <= vl) {
      svwrite_hor_za32_s32_m(0, 0, pg_32_all, svld1_s32(pg_32_all, (const int32_t*) (const void*) w0));
      w0 += 4 * nr;

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        const svint8_t va = svld1rq_s8(pg_8_all, a_inc); a_inc += 16;
        const svint8_t vb0 = svld1_vnum_s8(pg_8_all, w0, 0);
        const svint8_t vb1 = svld1_vnum_s8(pg_8_all, w0, 2);
        const svint8_t vb2 = svld1_vnum_s8(pg_8_all, w0, 4);
        const svint8_t vb3 = svld1_vnum_s8(pg_8_all, w0, 6);
        w0 += 16 * nr;
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va, vb0);
        svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va, vb1);
        svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va, vb2);
        svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va, vb3);
      }

      if (k_remainder != 0) {
        const svint8_t va = svld1rq_s8(svwhilelt_b8_u64(0, k_tail), a_inc);
        svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
        w0 += 4 * nr;
        if (k_remainder > 1) {
          svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          w0 += 4 * nr;
        }
        if (k_remainder > 2) {
          svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          w0 += 4 * nr;
        }
        if (k_remainder > 3) {
          svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          w0 += 4 * nr;
        }
      }

      const svfloat32_t vscale = svld1_f32(pg_32_all, (const float*) (const void*) w0);
      const svint32_t vacc = svadd_s32_x(pg_32_all,
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, 0),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, 1)),
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, 2),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, 3)));
      const svfloat32_t vf = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc), vscale);
      const svint32_t vc = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf, vmagic_bias)), vmagic_min, vmagic_max);
      const svint16_t vh = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc), vzp_s16);
      svst1b_s32(svwhilelt_b32_u64(0, nc), c_inc, svreinterpret_s32_s16(vh));
      return;
    }

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

      svwrite_hor_za32_s32_m(0, 0, pg_32_all, svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w0, 0));
      svwrite_hor_za32_s32_m(1, 0, pg_32_all, svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w0, 1));
      svwrite_hor_za32_s32_m(2, 0, pg_32_all, svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w1, 0));
      svwrite_hor_za32_s32_m(3, 0, pg_32_all, svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w1, 1));
      w0 += 4 * nr;
      w1 += 4 * nr;

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        const svint8_t va = svld1rq_s8(pg_8_all, a_inc); a_inc += 16;

        const svint8_t vb0_0_0 = svld1_vnum_s8(pg_8_all, w0, 0);
        const svint8_t vb0_0_1 = svld1_vnum_s8(pg_8_all, w0, 1);
        const svint8_t vb1_0_0 = svld1_vnum_s8(pg_8_all, w1, 0);
        const svint8_t vb1_0_1 = svld1_vnum_s8(pg_8_all, w1, 1);
        const svint8_t vb0_1_0 = svld1_vnum_s8(pg_8_all, w0, 2);
        const svint8_t vb0_1_1 = svld1_vnum_s8(pg_8_all, w0, 3);
        const svint8_t vb1_1_0 = svld1_vnum_s8(pg_8_all, w1, 2);
        const svint8_t vb1_1_1 = svld1_vnum_s8(pg_8_all, w1, 3);
        const svint8_t vb0_2_0 = svld1_vnum_s8(pg_8_all, w0, 4);
        const svint8_t vb0_2_1 = svld1_vnum_s8(pg_8_all, w0, 5);
        const svint8_t vb1_2_0 = svld1_vnum_s8(pg_8_all, w1, 4);
        const svint8_t vb1_2_1 = svld1_vnum_s8(pg_8_all, w1, 5);
        const svint8_t vb0_3_0 = svld1_vnum_s8(pg_8_all, w0, 6);
        const svint8_t vb0_3_1 = svld1_vnum_s8(pg_8_all, w0, 7);
        const svint8_t vb1_3_0 = svld1_vnum_s8(pg_8_all, w1, 6);
        const svint8_t vb1_3_1 = svld1_vnum_s8(pg_8_all, w1, 7);
        w0 += 16 * nr;
        w1 += 16 * nr;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 16 * nr;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
        if (prfm_acc_1 < prfm_end_1) {
          prfm_acc_1 += 16 * nr;
          if (prfm_acc_1 > prfm_end_1) {
            prfm_acc_1 = prfm_end_1;
          }
          for (; prfm_pos_1 < prfm_acc_1; prfm_pos_1 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_1, 0, 2);
          }
        }
#endif

        svmopa_za32_s8_m(0, pg_row_0, pg_8_all, va, vb0_0_0);
        svmopa_za32_s8_m(1, pg_row_0, pg_8_all, va, vb0_0_1);
        svmopa_za32_s8_m(2, pg_row_0, pg_8_all, va, vb1_0_0);
        svmopa_za32_s8_m(3, pg_row_0, pg_8_all, va, vb1_0_1);
        svmopa_za32_s8_m(0, pg_row_1, pg_8_all, va, vb0_1_0);
        svmopa_za32_s8_m(1, pg_row_1, pg_8_all, va, vb0_1_1);
        svmopa_za32_s8_m(2, pg_row_1, pg_8_all, va, vb1_1_0);
        svmopa_za32_s8_m(3, pg_row_1, pg_8_all, va, vb1_1_1);
        svmopa_za32_s8_m(0, pg_row_2, pg_8_all, va, vb0_2_0);
        svmopa_za32_s8_m(1, pg_row_2, pg_8_all, va, vb0_2_1);
        svmopa_za32_s8_m(2, pg_row_2, pg_8_all, va, vb1_2_0);
        svmopa_za32_s8_m(3, pg_row_2, pg_8_all, va, vb1_2_1);
        svmopa_za32_s8_m(0, pg_row_3, pg_8_all, va, vb0_3_0);
        svmopa_za32_s8_m(1, pg_row_3, pg_8_all, va, vb0_3_1);
        svmopa_za32_s8_m(2, pg_row_3, pg_8_all, va, vb1_3_0);
        svmopa_za32_s8_m(3, pg_row_3, pg_8_all, va, vb1_3_1);
      }

      if (k_remainder != 0) {
        const svint8_t va = svld1rq_s8(svwhilelt_b8_u64(0, k_tail), a_inc);

        svmopa_za32_s8_m(0, pg_row_0, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
        svmopa_za32_s8_m(1, pg_row_0, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
        svmopa_za32_s8_m(2, pg_row_0, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 0));
        svmopa_za32_s8_m(3, pg_row_0, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 1));
        w0 += 4 * nr;
        w1 += 4 * nr;
        if (k_remainder > 1) {
          svmopa_za32_s8_m(0, pg_row_1, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          svmopa_za32_s8_m(1, pg_row_1, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
          svmopa_za32_s8_m(2, pg_row_1, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 0));
          svmopa_za32_s8_m(3, pg_row_1, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 1));
          w0 += 4 * nr;
          w1 += 4 * nr;
        }
        if (k_remainder > 2) {
          svmopa_za32_s8_m(0, pg_row_2, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          svmopa_za32_s8_m(1, pg_row_2, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
          svmopa_za32_s8_m(2, pg_row_2, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 0));
          svmopa_za32_s8_m(3, pg_row_2, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 1));
          w0 += 4 * nr;
          w1 += 4 * nr;
        }
        if (k_remainder > 3) {
          svmopa_za32_s8_m(0, pg_row_3, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          svmopa_za32_s8_m(1, pg_row_3, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
          svmopa_za32_s8_m(2, pg_row_3, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 0));
          svmopa_za32_s8_m(3, pg_row_3, pg_8_all, va, svld1_vnum_s8(pg_8_all, w1, 1));
          w0 += 4 * nr;
          w1 += 4 * nr;
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
      vscale_0 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w0, 0);
      vscale_1 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w0, 1);
      vscale_2 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w1, 0);
      vscale_3 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w1, 1);
      w = w1 + 4 * nr;

      vacc_0 = svadd_s32_x(pg_32_all,
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, 0),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, 1)),
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, 2),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, 3)));
      vacc_1 = svadd_s32_x(pg_32_all,
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, 0),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, 1)),
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, 2),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, 3)));
      vacc_2 = svadd_s32_x(pg_32_all,
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, 0),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, 1)),
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, 2),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, 3)));
      vacc_3 = svadd_s32_x(pg_32_all,
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, 0),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, 1)),
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, 2),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, 3)));
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

      svwrite_hor_za32_s32_m(0, 0, pg_32_all, svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w0, 0));
      svwrite_hor_za32_s32_m(1, 0, pg_32_all, svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w0, 1));
      w0 += 4 * nr;

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
        const svint8_t va = svld1rq_s8(pg_8_all, a_inc); a_inc += 16;

        const svint8_t vb0_0_0 = svld1_vnum_s8(pg_8_all, w0, 0);
        const svint8_t vb0_0_1 = svld1_vnum_s8(pg_8_all, w0, 1);
        const svint8_t vb0_1_0 = svld1_vnum_s8(pg_8_all, w0, 2);
        const svint8_t vb0_1_1 = svld1_vnum_s8(pg_8_all, w0, 3);
        const svint8_t vb0_2_0 = svld1_vnum_s8(pg_8_all, w0, 4);
        const svint8_t vb0_2_1 = svld1_vnum_s8(pg_8_all, w0, 5);
        const svint8_t vb0_3_0 = svld1_vnum_s8(pg_8_all, w0, 6);
        const svint8_t vb0_3_1 = svld1_vnum_s8(pg_8_all, w0, 7);
        w0 += 16 * nr;

#ifdef PREFETCH_W_MATRIX
        if (prfm_acc_0 < prfm_end_0) {
          prfm_acc_0 += 16 * nr;
          if (prfm_acc_0 > prfm_end_0) {
            prfm_acc_0 = prfm_end_0;
          }
          for (; prfm_pos_0 < prfm_acc_0; prfm_pos_0 += prfm_line_size) {
            __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
          }
        }
#endif

        svmopa_za32_s8_m(0, pg_row_0, pg_8_all, va, vb0_0_0);
        svmopa_za32_s8_m(1, pg_row_0, pg_8_all, va, vb0_0_1);
        svmopa_za32_s8_m(2, pg_row_1, pg_8_all, va, vb0_1_0);
        svmopa_za32_s8_m(3, pg_row_1, pg_8_all, va, vb0_1_1);
        svmopa_za32_s8_m(0, pg_row_2, pg_8_all, va, vb0_2_0);
        svmopa_za32_s8_m(1, pg_row_2, pg_8_all, va, vb0_2_1);
        svmopa_za32_s8_m(2, pg_row_3, pg_8_all, va, vb0_3_0);
        svmopa_za32_s8_m(3, pg_row_3, pg_8_all, va, vb0_3_1);
      }

      if (k_remainder != 0) {
        const svint8_t va = svld1rq_s8(svwhilelt_b8_u64(0, k_tail), a_inc);

        svmopa_za32_s8_m(0, pg_row_0, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
        svmopa_za32_s8_m(1, pg_row_0, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
        w0 += 4 * nr;
        if (k_remainder > 1) {
          svmopa_za32_s8_m(2, pg_row_1, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          svmopa_za32_s8_m(3, pg_row_1, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
          w0 += 4 * nr;
        }
        if (k_remainder > 2) {
          svmopa_za32_s8_m(0, pg_row_2, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          svmopa_za32_s8_m(1, pg_row_2, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
          w0 += 4 * nr;
        }
        if (k_remainder > 3) {
          svmopa_za32_s8_m(2, pg_row_3, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 0));
          svmopa_za32_s8_m(3, pg_row_3, pg_8_all, va, svld1_vnum_s8(pg_8_all, w0, 1));
          w0 += 4 * nr;
        }
      }

#ifdef PREFETCH_W_MATRIX
      for (; prfm_pos_0 < prfm_end_0; prfm_pos_0 += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos_0, 0, 2);
      }
#endif
      vscale_0 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w0, 0);
      vscale_1 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w0, 1);
      vscale_2 = vscale_0;
      vscale_3 = vscale_1;
      w = w0 + 4 * nr;

      vacc_0 = svadd_s32_x(pg_32_all,
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, 0),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, 1)),
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, 2),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, 3)));
      vacc_1 = svadd_s32_x(pg_32_all,
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, 0),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, 1)),
          svadd_s32_x(pg_32_all,
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, 2),
              svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, 3)));
      vacc_2 = vacc_0;
      vacc_3 = vacc_1;
    }

    const svfloat32_t vf_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc_0), vscale_0);
    const svfloat32_t vf_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc_1), vscale_1);
    const svfloat32_t vf_2 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc_2), vscale_2);
    const svfloat32_t vf_3 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc_3), vscale_3);
    const svint32_t vc_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf_0, vmagic_bias)), vmagic_min, vmagic_max);
    const svint32_t vc_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf_1, vmagic_bias)), vmagic_min, vmagic_max);
    const svint32_t vc_2 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf_2, vmagic_bias)), vmagic_min, vmagic_max);
    const svint32_t vc_3 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf_3, vmagic_bias)), vmagic_min, vmagic_max);
    const svint16_t vh_01 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc_0), svreinterpret_s16_s32(vc_1)), vzp_s16);
    const svint16_t vh_23 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc_2), svreinterpret_s16_s32(vc_3)), vzp_s16);
    const svint8_t vout = svuzp1_s8(svreinterpret_s8_s16(vh_01), svreinterpret_s8_s16(vh_23));

    svst1_s8(svwhilelt_b8_u64(0, nc), c_inc, vout);

    if (nc > 2 * nr) {
      c_inc += 2 * nr;
      nc -= 2 * nr;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
