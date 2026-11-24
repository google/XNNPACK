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

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__sme_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__sme_get_nr(void) {
  return 2 * svcntsw();
}

#if defined(__APPLE__)
#define PREFETCH_W_MATRIX
#endif

__arm_locally_streaming __arm_new("za")
void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__sme(
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
  const size_t prfm_distance_bytes = (16 * 1024);
#endif
  if XNN_UNLIKELY(mr == 0 || nc == 0) {
    return;
  }
  const size_t vl = svcntw();
  const size_t nr = 2 * vl;
  assert(mr <= 2 * vl);
  assert(kc != 0);
  assert(dst_stride_col == sizeof(int8_t));
  (void) dst_stride_col;

  const int32_t output_zero_point = minmax_params->fp32_scalar.output_zero_point;
  const int32_t output_min = minmax_params->fp32_scalar.output_min;
  const int32_t output_max = minmax_params->fp32_scalar.output_max;

  const svfloat32_t vmagic_bias = svdup_n_f32(12582912.0f);
  const svint32_t vmagic_min = svdup_n_s32(INT32_C(0x4B400000) + (output_min - output_zero_point));
  const svint32_t vmagic_max = svdup_n_s32(INT32_C(0x4B400000) + (output_max - output_zero_point));
  const svint16_t vzp_s16 = svdup_n_s16((int16_t) output_zero_point);

  const svbool_t pg_8_all = svptrue_b8();
  const svbool_t pg_16_all = svptrue_b16();
  const svbool_t pg_32_all = svptrue_b32();

  const size_t k_block4 = (kc + 3) / 4;
  const size_t k_unroll4 = k_block4 / 4;
  const size_t k_remainder = k_block4 % 4;

  const size_t k_rounded_32 = ((kc + 31) / 32) * 32;
  const size_t k_block4_rounded = k_rounded_32 / 4;
  const size_t k_tail_slices = k_block4_rounded - k_block4;
  const size_t w_block_size = 4 * nr + k_block4_rounded * 2 * nr + 4 * nr;

  uint8_t* c_inc = (uint8_t*) dst;

#ifdef PREFETCH_W_MATRIX
  const size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  const size_t total_w_size = (nc_rup_nr / nr) * w_block_size;
  const int8_t* w_base = (const int8_t*) w;
  const bool need_prfm = total_w_size > prfm_min_w_size;
  const size_t prfm_end_w_size = need_prfm ? total_w_size : 0;
  size_t prfm_pos = 0;
  size_t prfm_acc_w_size = need_prfm ? ((prfm_distance_bytes < total_w_size) ? prfm_distance_bytes : total_w_size) : 0;
  for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
    __builtin_prefetch(w_base + prfm_pos, 0, 2);
  }
#endif

  if (nc <= vl) {
nc_le_vl_tail:
    svzero_za();

    const svint32_t vbias = svld1_s32(pg_32_all, (const int32_t*) w);
    w = (const int8_t*) w + 4 * nr;
#ifdef PREFETCH_W_MATRIX
    if (prfm_pos < prfm_end_w_size) {
      prfm_acc_w_size += 4 * nr;
      if (prfm_acc_w_size > prfm_end_w_size) {
        prfm_acc_w_size = prfm_end_w_size;
      }
      for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos, 0, 2);
      }
    }
#endif

    const int8_t* a_inc = (const int8_t*) lhs_packed;

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
      const svint8_t vw0 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 0);
      const svint8_t vw1 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 1);
      const svint8_t vw2 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 2);
      const svint8_t vw3 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 3);
      w = (const int8_t*) w + 8 * nr;
      const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
      const svint8_t va0_1 = svld1_vnum_s8(pg_8_all, a_inc, 1);
      const svint8_t va1_0 = svld1_vnum_s8(pg_8_all, a_inc, 2);
      const svint8_t va1_1 = svld1_vnum_s8(pg_8_all, a_inc, 3);
      const svint8_t va2_0 = svld1_vnum_s8(pg_8_all, a_inc, 4);
      const svint8_t va2_1 = svld1_vnum_s8(pg_8_all, a_inc, 5);
      const svint8_t va3_0 = svld1_vnum_s8(pg_8_all, a_inc, 6);
      const svint8_t va3_1 = svld1_vnum_s8(pg_8_all, a_inc, 7);
      a_inc += 32 * vl;

#ifdef PREFETCH_W_MATRIX
      if (prfm_pos < prfm_end_w_size) {
        prfm_acc_w_size += 8 * nr;
        if (prfm_acc_w_size > prfm_end_w_size) {
          prfm_acc_w_size = prfm_end_w_size;
        }
        for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
          __builtin_prefetch(w_base + prfm_pos, 0, 2);
        }
      }
#endif

      svint8_t vb0 = svzip1_s8(svlsl_n_s8_x(pg_8_all, vw0, 4), svand_n_s8_x(pg_8_all, vw0, (int8_t) 0xF0));
      svint8_t vb1 = svzip1_s8(svlsl_n_s8_x(pg_8_all, vw1, 4), svand_n_s8_x(pg_8_all, vw1, (int8_t) 0xF0));
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb0), "+w"(vb1));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, vb0);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0_1, vb0);

      svint8_t vb2 = svzip1_s8(svlsl_n_s8_x(pg_8_all, vw2, 4), svand_n_s8_x(pg_8_all, vw2, (int8_t) 0xF0));
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb2));
#endif
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1_0, vb1);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va1_1, vb1);

      svint8_t vb3 = svzip1_s8(svlsl_n_s8_x(pg_8_all, vw3, 4), svand_n_s8_x(pg_8_all, vw3, (int8_t) 0xF0));
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb3));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2_0, vb2);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va2_1, vb2);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3_0, vb3);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va3_1, vb3);
    }

    for (size_t k_rem = 0; k_rem < k_remainder; ++k_rem) {
      const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
      const svint8_t va0_1 = svld1_vnum_s8(pg_8_all, a_inc, 1);
      a_inc += 8 * vl;
      const svint8_t vw = svld1_s8(pg_8_all, (const int8_t*) w); w = (const int8_t*) w + 4 * vl;

      const svint8_t vb = svzip1_s8(svlsl_n_s8_x(pg_8_all, vw, 4), svand_n_s8_x(pg_8_all, vw, (int8_t) 0xF0));

      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, vb);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0_1, vb);
    }

    if (k_tail_slices > 0) {
      w = (const int8_t*) w + k_tail_slices * 4 * vl;
    }
#ifdef PREFETCH_W_MATRIX
    if (prfm_pos < prfm_end_w_size) {
      prfm_acc_w_size += (k_remainder + k_tail_slices) * 4 * vl;
      if (prfm_acc_w_size > prfm_end_w_size) {
        prfm_acc_w_size = prfm_end_w_size;
      }
      for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos, 0, 2);
      }
    }
#endif

    const svfloat32_t vscale = svld1_f32(pg_32_all, (const float*) w);
#ifdef PREFETCH_W_MATRIX
    if (prfm_pos < prfm_end_w_size) {
      prfm_acc_w_size += 4 * nr;
      if (prfm_acc_w_size > prfm_end_w_size) {
        prfm_acc_w_size = prfm_end_w_size;
      }
      for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos, 0, 2);
      }
    }
#endif

    const svbool_t p_col_b32 = svwhilelt_b32_u64(0, nc);
    const size_t m_upper = (mr < vl) ? mr : vl;
    size_t i = 0;
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (; i + 4 <= m_upper; i += 4) {
      const uint32_t row = (uint32_t) i;
      const svint32_t vza0 = svadd_s32_x(pg_32_all,
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 0),
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 0));
      const svint32_t vza1 = svadd_s32_x(pg_32_all,
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 1),
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 1));
      const svint32_t vza2 = svadd_s32_x(pg_32_all,
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 2),
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 2));
      const svint32_t vza3 = svadd_s32_x(pg_32_all,
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 3),
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 3));
      const svint32_t vacc0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza0, 4), vbias);
      const svint32_t vacc1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza1, 4), vbias);
      const svint32_t vacc2 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza2, 4), vbias);
      const svint32_t vacc3 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza3, 4), vbias);
      const svfloat32_t vf0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0), vscale);
      const svfloat32_t vf1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc1), vscale);
      const svfloat32_t vf2 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc2), vscale);
      const svfloat32_t vf3 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc3), vscale);
      const svint32_t vc0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc2 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf2, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc3 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf3, vmagic_bias)), vmagic_min, vmagic_max);
      const svint16_t vh0 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc0), vzp_s16);
      const svint16_t vh1 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc1), vzp_s16);
      const svint16_t vh2 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc2), vzp_s16);
      const svint16_t vh3 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc3), vzp_s16);
      int8_t* c_r = (int8_t*) (c_inc + i * dst_stride_row);
      svst1b_s32(p_col_b32, c_r, svreinterpret_s32_s16(vh0));
      svst1b_s32(p_col_b32, c_r + dst_stride_row, svreinterpret_s32_s16(vh1));
      svst1b_s32(p_col_b32, c_r + 2 * dst_stride_row, svreinterpret_s32_s16(vh2));
      svst1b_s32(p_col_b32, c_r + 3 * dst_stride_row, svreinterpret_s32_s16(vh3));
    }
    for (; i < m_upper; ++i) {
      const uint32_t row = (uint32_t) i;
      const svint32_t vza0 = svadd_s32_x(pg_32_all,
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row),
          svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row));
      const svint32_t vacc0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza0, 4), vbias);
      const svfloat32_t vf0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0), vscale);
      const svint32_t vc0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint16_t vh0 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc0), vzp_s16);
      svst1b_s32(p_col_b32, (int8_t*) (c_inc + i * dst_stride_row), svreinterpret_s32_s16(vh0));
    }
    if (mr > vl) {
      const size_t m_lower = mr - vl;
      size_t j = 0;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (; j + 4 <= m_lower; j += 4) {
        const uint32_t row = (uint32_t) j;
        const svint32_t vza0 = svadd_s32_x(pg_32_all,
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 0),
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 0));
        const svint32_t vza1 = svadd_s32_x(pg_32_all,
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 1),
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 1));
        const svint32_t vza2 = svadd_s32_x(pg_32_all,
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 2),
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 2));
        const svint32_t vza3 = svadd_s32_x(pg_32_all,
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 3),
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 3));
        const svint32_t vacc0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza0, 4), vbias);
        const svint32_t vacc1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza1, 4), vbias);
        const svint32_t vacc2 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza2, 4), vbias);
        const svint32_t vacc3 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza3, 4), vbias);
        const svfloat32_t vf0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0), vscale);
        const svfloat32_t vf1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc1), vscale);
        const svfloat32_t vf2 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc2), vscale);
        const svfloat32_t vf3 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc3), vscale);
        const svint32_t vc0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc2 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf2, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc3 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf3, vmagic_bias)), vmagic_min, vmagic_max);
        const svint16_t vh0 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc0), vzp_s16);
        const svint16_t vh1 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc1), vzp_s16);
        const svint16_t vh2 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc2), vzp_s16);
        const svint16_t vh3 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc3), vzp_s16);
        int8_t* c_r = (int8_t*) (c_inc + (j + vl) * dst_stride_row);
        svst1b_s32(p_col_b32, c_r, svreinterpret_s32_s16(vh0));
        svst1b_s32(p_col_b32, c_r + dst_stride_row, svreinterpret_s32_s16(vh1));
        svst1b_s32(p_col_b32, c_r + 2 * dst_stride_row, svreinterpret_s32_s16(vh2));
        svst1b_s32(p_col_b32, c_r + 3 * dst_stride_row, svreinterpret_s32_s16(vh3));
      }
      for (; j < m_lower; ++j) {
        const uint32_t row = (uint32_t) j;
        const svint32_t vza0 = svadd_s32_x(pg_32_all,
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row),
            svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row));
        const svint32_t vacc0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, vza0, 4), vbias);
        const svfloat32_t vf0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0), vscale);
        const svint32_t vc0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0, vmagic_bias)), vmagic_min, vmagic_max);
        const svint16_t vh0 = svadd_s16_x(pg_16_all, svreinterpret_s16_s32(vc0), vzp_s16);
        svst1b_s32(p_col_b32, (int8_t*) (c_inc + (j + vl) * dst_stride_row), svreinterpret_s32_s16(vh0));
      }
    }
    return;
  }

  while (mr <= vl && nc > nr) {
    svzero_za();

    const int8_t* w0 = (const int8_t*) w;
    const int8_t* w1 = w0 + w_block_size;

    const svint32_t vbias_0 = svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w0, 0);
    const svint32_t vbias_1 = svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w0, 1);
    const svint32_t vbias_2 = svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w1, 0);
    const svint32_t vbias_3 = svld1_vnum_s32(pg_32_all, (const int32_t*) (const void*) w1, 1);
    w0 += 4 * nr;
    w1 += 4 * nr;

    const int8_t* a_inc = (const int8_t*) lhs_packed;

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
      const svint8_t va0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
      const svint8_t va1 = svld1_vnum_s8(pg_8_all, a_inc, 2);
      const svint8_t va2 = svld1_vnum_s8(pg_8_all, a_inc, 4);
      const svint8_t va3 = svld1_vnum_s8(pg_8_all, a_inc, 6);
      a_inc += 32 * vl;
      const svint8_t vw0_0 = svld1_vnum_s8(pg_8_all, w0, 0);
      const svint8_t vw1_0 = svld1_vnum_s8(pg_8_all, w1, 0);
      const svint8_t vw0_1 = svld1_vnum_s8(pg_8_all, w0, 1);
      const svint8_t vw1_1 = svld1_vnum_s8(pg_8_all, w1, 1);
      const svint8_t vw0_2 = svld1_vnum_s8(pg_8_all, w0, 2);
      const svint8_t vw1_2 = svld1_vnum_s8(pg_8_all, w1, 2);
      const svint8_t vw0_3 = svld1_vnum_s8(pg_8_all, w0, 3);
      const svint8_t vw1_3 = svld1_vnum_s8(pg_8_all, w1, 3);
      w0 += 8 * nr;
      w1 += 8 * nr;

      const svint8_t vt0_0_lo = svlsl_n_s8_x(pg_8_all, vw0_0, 4);
      const svint8_t vt0_0_hi = svand_n_s8_x(pg_8_all, vw0_0, (int8_t) 0xF0);
      const svint8_t vt1_0_lo = svlsl_n_s8_x(pg_8_all, vw1_0, 4);
      const svint8_t vt1_0_hi = svand_n_s8_x(pg_8_all, vw1_0, (int8_t) 0xF0);
      svint8_t vb0_0 = svzip1_s8(vt0_0_lo, vt0_0_hi);
      svint8_t vb0_1 = svzip2_s8(vt0_0_lo, vt0_0_hi);
      svint8_t vb1_0 = svzip1_s8(vt1_0_lo, vt1_0_hi);
      svint8_t vb1_1 = svzip2_s8(vt1_0_lo, vt1_0_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb0_0), "+w"(vb0_1), "+w"(vb1_0), "+w"(vb1_1));
#endif

      const svint8_t vt0_1_lo = svlsl_n_s8_x(pg_8_all, vw0_1, 4);
      const svint8_t vt0_1_hi = svand_n_s8_x(pg_8_all, vw0_1, (int8_t) 0xF0);
      const svint8_t vt1_1_lo = svlsl_n_s8_x(pg_8_all, vw1_1, 4);
      const svint8_t vt1_1_hi = svand_n_s8_x(pg_8_all, vw1_1, (int8_t) 0xF0);
      svint8_t vb0_2 = svzip1_s8(vt0_1_lo, vt0_1_hi);
      svint8_t vb0_3 = svzip2_s8(vt0_1_lo, vt0_1_hi);
      svint8_t vb1_2 = svzip1_s8(vt1_1_lo, vt1_1_hi);
      svint8_t vb1_3 = svzip2_s8(vt1_1_lo, vt1_1_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb0_2), "+w"(vb0_3), "+w"(vb1_2), "+w"(vb1_3));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0, vb0_0);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0, vb0_1);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0, vb1_0);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va0, vb1_1);

      const svint8_t vt0_2_lo = svlsl_n_s8_x(pg_8_all, vw0_2, 4);
      const svint8_t vt0_2_hi = svand_n_s8_x(pg_8_all, vw0_2, (int8_t) 0xF0);
      const svint8_t vt1_2_lo = svlsl_n_s8_x(pg_8_all, vw1_2, 4);
      const svint8_t vt1_2_hi = svand_n_s8_x(pg_8_all, vw1_2, (int8_t) 0xF0);
      svint8_t vb0_4 = svzip1_s8(vt0_2_lo, vt0_2_hi);
      svint8_t vb0_5 = svzip2_s8(vt0_2_lo, vt0_2_hi);
      svint8_t vb1_4 = svzip1_s8(vt1_2_lo, vt1_2_hi);
      svint8_t vb1_5 = svzip2_s8(vt1_2_lo, vt1_2_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb0_4), "+w"(vb0_5), "+w"(vb1_4), "+w"(vb1_5));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va1, vb0_2);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1, vb0_3);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va1, vb1_2);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va1, vb1_3);

      const svint8_t vt0_3_lo = svlsl_n_s8_x(pg_8_all, vw0_3, 4);
      const svint8_t vt0_3_hi = svand_n_s8_x(pg_8_all, vw0_3, (int8_t) 0xF0);
      const svint8_t vt1_3_lo = svlsl_n_s8_x(pg_8_all, vw1_3, 4);
      const svint8_t vt1_3_hi = svand_n_s8_x(pg_8_all, vw1_3, (int8_t) 0xF0);
      svint8_t vb0_6 = svzip1_s8(vt0_3_lo, vt0_3_hi);
      svint8_t vb0_7 = svzip2_s8(vt0_3_lo, vt0_3_hi);
      svint8_t vb1_6 = svzip1_s8(vt1_3_lo, vt1_3_hi);
      svint8_t vb1_7 = svzip2_s8(vt1_3_lo, vt1_3_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb0_6), "+w"(vb0_7), "+w"(vb1_6), "+w"(vb1_7));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2, vb0_4);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va2, vb0_5);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va2, vb1_4);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va2, vb1_5);
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va3, vb0_6);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3, vb0_7);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va3, vb1_6);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va3, vb1_7);
    }

    for (size_t k_rem = 0; k_rem < k_remainder; ++k_rem) {
      const svint8_t va0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
      a_inc += 8 * vl;
      const svint8_t vw0 = svld1_s8(pg_8_all, w0); w0 += 4 * vl;
      const svint8_t vw1 = svld1_s8(pg_8_all, w1); w1 += 4 * vl;
      const svint8_t vt0_lo = svlsl_n_s8_x(pg_8_all, vw0, 4);
      const svint8_t vt0_hi = svand_n_s8_x(pg_8_all, vw0, (int8_t) 0xF0);
      const svint8_t vt1_lo = svlsl_n_s8_x(pg_8_all, vw1, 4);
      const svint8_t vt1_hi = svand_n_s8_x(pg_8_all, vw1, (int8_t) 0xF0);
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0, svzip1_s8(vt0_lo, vt0_hi));
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0, svzip2_s8(vt0_lo, vt0_hi));
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0, svzip1_s8(vt1_lo, vt1_hi));
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va0, svzip2_s8(vt1_lo, vt1_hi));
    }

    if (k_tail_slices > 0) {
      w0 += k_tail_slices * 4 * vl;
      w1 += k_tail_slices * 4 * vl;
    }

    const svfloat32_t vscale_0 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w0, 0);
    const svfloat32_t vscale_1 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w0, 1);
    const svfloat32_t vscale_2 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w1, 0);
    const svfloat32_t vscale_3 = svld1_vnum_f32(pg_32_all, (const float*) (const void*) w1, 1);
    w = w1 + 4 * nr;

#ifdef PREFETCH_W_MATRIX
    if (prfm_pos < prfm_end_w_size) {
      prfm_acc_w_size += 2 * w_block_size;
      if (prfm_acc_w_size > prfm_end_w_size) {
        prfm_acc_w_size = prfm_end_w_size;
      }
      for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos, 0, 2);
      }
    }
#endif

    const svbool_t p_col = svwhilelt_b8_u64(0, nc);
    size_t i = 0;
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (; i + 2 <= mr; i += 2) {
      const uint32_t row = (uint32_t) i;
      const svint32_t vacc0_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 0), 4), vbias_0);
      const svint32_t vacc0_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 0), 4), vbias_1);
      const svint32_t vacc0_2 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 0), 4), vbias_2);
      const svint32_t vacc0_3 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 0), 4), vbias_3);
      const svint32_t vacc1_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 1), 4), vbias_0);
      const svint32_t vacc1_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 1), 4), vbias_1);
      const svint32_t vacc1_2 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 1), 4), vbias_2);
      const svint32_t vacc1_3 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 1), 4), vbias_3);
      const svfloat32_t vf0_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_0), vscale_0);
      const svfloat32_t vf0_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_1), vscale_1);
      const svfloat32_t vf0_2 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_2), vscale_2);
      const svfloat32_t vf0_3 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_3), vscale_3);
      const svfloat32_t vf1_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc1_0), vscale_0);
      const svfloat32_t vf1_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc1_1), vscale_1);
      const svfloat32_t vf1_2 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc1_2), vscale_2);
      const svfloat32_t vf1_3 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc1_3), vscale_3);
      const svint32_t vc0_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_2 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_2, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_3 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_3, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc1_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc1_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc1_2 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_2, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc1_3 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_3, vmagic_bias)), vmagic_min, vmagic_max);
      const svint16_t vh0_01 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_0), svreinterpret_s16_s32(vc0_1)), vzp_s16);
      const svint16_t vh0_23 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_2), svreinterpret_s16_s32(vc0_3)), vzp_s16);
      const svint16_t vh1_01 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc1_0), svreinterpret_s16_s32(vc1_1)), vzp_s16);
      const svint16_t vh1_23 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc1_2), svreinterpret_s16_s32(vc1_3)), vzp_s16);
      int8_t* c_r = (int8_t*) (c_inc + i * dst_stride_row);
      svst1_s8(p_col, c_r, svuzp1_s8(svreinterpret_s8_s16(vh0_01), svreinterpret_s8_s16(vh0_23)));
      svst1_s8(p_col, c_r + dst_stride_row, svuzp1_s8(svreinterpret_s8_s16(vh1_01), svreinterpret_s8_s16(vh1_23)));
    }
    if (i < mr) {
      const uint32_t row = (uint32_t) i;
      const svint32_t vacc0_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row), 4), vbias_0);
      const svint32_t vacc0_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row), 4), vbias_1);
      const svint32_t vacc0_2 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row), 4), vbias_2);
      const svint32_t vacc0_3 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row), 4), vbias_3);
      const svfloat32_t vf0_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_0), vscale_0);
      const svfloat32_t vf0_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_1), vscale_1);
      const svfloat32_t vf0_2 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_2), vscale_2);
      const svfloat32_t vf0_3 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vacc0_3), vscale_3);
      const svint32_t vc0_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_2 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_2, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_3 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_3, vmagic_bias)), vmagic_min, vmagic_max);
      const svint16_t vh0_01 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_0), svreinterpret_s16_s32(vc0_1)), vzp_s16);
      const svint16_t vh0_23 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_2), svreinterpret_s16_s32(vc0_3)), vzp_s16);
      svst1_s8(p_col, (int8_t*) (c_inc + i * dst_stride_row), svuzp1_s8(svreinterpret_s8_s16(vh0_01), svreinterpret_s8_s16(vh0_23)));
    }

    if (nc > 2 * nr) {
      c_inc += 2 * nr;
      nc -= 2 * nr;
      if (nc <= vl) {
        goto nc_le_vl_tail;
      }
    } else {
      return;
    }
  }

  do {
    svzero_za();

    const svint32_t vbias_0 = svld1_s32(pg_32_all, (const int32_t*) w);
    const svint32_t vbias_1 = svld1_s32(pg_32_all, (const int32_t*) w + vl);
    w = (const int8_t*) w + 4 * nr;
#ifdef PREFETCH_W_MATRIX
    if (prfm_pos < prfm_end_w_size) {
      prfm_acc_w_size += 4 * nr;
      if (prfm_acc_w_size > prfm_end_w_size) {
        prfm_acc_w_size = prfm_end_w_size;
      }
      for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos, 0, 2);
      }
    }
#endif

    const int8_t* a_inc = (const int8_t*) lhs_packed;

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
      const svint8_t vw0 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 0);
      const svint8_t vw1 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 1);
      const svint8_t vw2 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 2);
      const svint8_t vw3 = svld1_vnum_s8(pg_8_all, (const int8_t*) w, 3);
      w = (const int8_t*) w + 8 * nr;
      const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
      const svint8_t va0_1 = svld1_vnum_s8(pg_8_all, a_inc, 1);
      const svint8_t va1_0 = svld1_vnum_s8(pg_8_all, a_inc, 2);
      const svint8_t va1_1 = svld1_vnum_s8(pg_8_all, a_inc, 3);
      const svint8_t va2_0 = svld1_vnum_s8(pg_8_all, a_inc, 4);
      const svint8_t va2_1 = svld1_vnum_s8(pg_8_all, a_inc, 5);
      const svint8_t va3_0 = svld1_vnum_s8(pg_8_all, a_inc, 6);
      const svint8_t va3_1 = svld1_vnum_s8(pg_8_all, a_inc, 7);
      a_inc += 32 * vl;

#ifdef PREFETCH_W_MATRIX
      if (prfm_pos < prfm_end_w_size) {
        prfm_acc_w_size += 8 * nr;
        if (prfm_acc_w_size > prfm_end_w_size) {
          prfm_acc_w_size = prfm_end_w_size;
        }
        for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
          __builtin_prefetch(w_base + prfm_pos, 0, 2);
        }
      }
#endif

      const svint8_t vt0_lo = svlsl_n_s8_x(pg_8_all, vw0, 4);
      const svint8_t vt0_hi = svand_n_s8_x(pg_8_all, vw0, (int8_t) 0xF0);
      svint8_t vb0_0 = svzip1_s8(vt0_lo, vt0_hi);
      svint8_t vb0_1 = svzip2_s8(vt0_lo, vt0_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb0_0), "+w"(vb0_1));
#endif

      const svint8_t vt1_lo = svlsl_n_s8_x(pg_8_all, vw1, 4);
      const svint8_t vt1_hi = svand_n_s8_x(pg_8_all, vw1, (int8_t) 0xF0);
      svint8_t vb1_0 = svzip1_s8(vt1_lo, vt1_hi);
      svint8_t vb1_1 = svzip2_s8(vt1_lo, vt1_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb1_0), "+w"(vb1_1));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, vb0_0);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0_0, vb0_1);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0_1, vb0_0);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va0_1, vb0_1);

      const svint8_t vt2_lo = svlsl_n_s8_x(pg_8_all, vw2, 4);
      const svint8_t vt2_hi = svand_n_s8_x(pg_8_all, vw2, (int8_t) 0xF0);
      svint8_t vb2_0 = svzip1_s8(vt2_lo, vt2_hi);
      svint8_t vb2_1 = svzip2_s8(vt2_lo, vt2_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb2_0), "+w"(vb2_1));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va1_0, vb1_0);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va1_0, vb1_1);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va1_1, vb1_0);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va1_1, vb1_1);

      const svint8_t vt3_lo = svlsl_n_s8_x(pg_8_all, vw3, 4);
      const svint8_t vt3_hi = svand_n_s8_x(pg_8_all, vw3, (int8_t) 0xF0);
      svint8_t vb3_0 = svzip1_s8(vt3_lo, vt3_hi);
      svint8_t vb3_1 = svzip2_s8(vt3_lo, vt3_hi);
#if defined(__GNUC__) || defined(__clang__)
      __asm__("" : "+w"(vb3_0), "+w"(vb3_1));
#endif
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va2_0, vb2_0);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va2_0, vb2_1);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va2_1, vb2_0);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va2_1, vb2_1);

      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va3_0, vb3_0);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va3_0, vb3_1);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va3_1, vb3_0);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va3_1, vb3_1);
    }

    for (size_t k_rem = 0; k_rem < k_remainder; ++k_rem) {
      const svint8_t va0_0 = svld1_vnum_s8(pg_8_all, a_inc, 0);
      const svint8_t va0_1 = svld1_vnum_s8(pg_8_all, a_inc, 1);
      a_inc += 8 * vl;
      const svint8_t vw = svld1_s8(pg_8_all, (const int8_t*) w); w = (const int8_t*) w + 4 * vl;

      const svint8_t vt_lo = svlsl_n_s8_x(pg_8_all, vw, 4);
      const svint8_t vt_hi = svand_n_s8_x(pg_8_all, vw, (int8_t) 0xF0);
      const svint8_t vb0_0 = svzip1_s8(vt_lo, vt_hi);
      const svint8_t vb0_1 = svzip2_s8(vt_lo, vt_hi);

      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, va0_0, vb0_0);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, va0_0, vb0_1);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, va0_1, vb0_0);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, va0_1, vb0_1);
    }

    if (k_tail_slices > 0) {
      w = (const int8_t*) w + k_tail_slices * 4 * vl;
    }
#ifdef PREFETCH_W_MATRIX
    if (prfm_pos < prfm_end_w_size) {
      prfm_acc_w_size += (k_remainder + k_tail_slices) * 4 * vl;
      if (prfm_acc_w_size > prfm_end_w_size) {
        prfm_acc_w_size = prfm_end_w_size;
      }
      for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos, 0, 2);
      }
    }
#endif

    const svfloat32_t vscale_0 = svld1_f32(pg_32_all, (const float*) w);
    const svfloat32_t vscale_1 = svld1_f32(pg_32_all, (const float*) w + vl);
    w = (const int8_t*) w + 4 * nr;

#ifdef PREFETCH_W_MATRIX
    if (prfm_pos < prfm_end_w_size) {
      prfm_acc_w_size += 4 * nr;
      if (prfm_acc_w_size > prfm_end_w_size) {
        prfm_acc_w_size = prfm_end_w_size;
      }
      for (; prfm_pos < prfm_acc_w_size; prfm_pos += prfm_line_size) {
        __builtin_prefetch(w_base + prfm_pos, 0, 2);
      }
    }
#endif

    const size_t nc_cur = (nc < nr) ? nc : nr;
    const svbool_t p_row = svwhilelt_b16_u64(0, nc_cur);

    const size_t m_upper = (mr < vl) ? mr : vl;
    size_t i = 0;
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (; i + 4 <= m_upper; i += 4) {
      const uint32_t row = (uint32_t) i;
      const svint32_t vi0_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 0), 4), vbias_0);
      const svint32_t vi0_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 0), 4), vbias_1);
      const svint32_t vi1_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 1), 4), vbias_0);
      const svint32_t vi1_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 1), 4), vbias_1);
      const svint32_t vi2_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 2), 4), vbias_0);
      const svint32_t vi2_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 2), 4), vbias_1);
      const svint32_t vi3_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row + 3), 4), vbias_0);
      const svint32_t vi3_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row + 3), 4), vbias_1);
      const svfloat32_t vf0_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_0), vscale_0);
      const svfloat32_t vf0_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_1), vscale_1);
      const svfloat32_t vf1_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi1_0), vscale_0);
      const svfloat32_t vf1_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi1_1), vscale_1);
      const svfloat32_t vf2_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi2_0), vscale_0);
      const svfloat32_t vf2_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi2_1), vscale_1);
      const svfloat32_t vf3_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi3_0), vscale_0);
      const svfloat32_t vf3_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi3_1), vscale_1);
      const svint32_t vc0_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc1_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc1_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc2_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf2_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc2_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf2_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc3_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf3_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc3_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf3_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint16_t vout0 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_0), svreinterpret_s16_s32(vc0_1)), vzp_s16);
      const svint16_t vout1 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc1_0), svreinterpret_s16_s32(vc1_1)), vzp_s16);
      const svint16_t vout2 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc2_0), svreinterpret_s16_s32(vc2_1)), vzp_s16);
      const svint16_t vout3 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc3_0), svreinterpret_s16_s32(vc3_1)), vzp_s16);
      int8_t* c_r = (int8_t*) (c_inc + i * dst_stride_row);
      svst1b_s16(p_row, c_r, vout0);
      svst1b_s16(p_row, c_r + dst_stride_row, vout1);
      svst1b_s16(p_row, c_r + 2 * dst_stride_row, vout2);
      svst1b_s16(p_row, c_r + 3 * dst_stride_row, vout3);
    }
    for (; i < m_upper; ++i) {
      const uint32_t row = (uint32_t) i;
      const svint32_t vi0_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 0, row), 4), vbias_0);
      const svint32_t vi0_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 1, row), 4), vbias_1);
      const svfloat32_t vf0_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_0), vscale_0);
      const svfloat32_t vf0_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_1), vscale_1);
      const svint32_t vc0_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_0, vmagic_bias)), vmagic_min, vmagic_max);
      const svint32_t vc0_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_1, vmagic_bias)), vmagic_min, vmagic_max);
      const svint16_t vout0 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_0), svreinterpret_s16_s32(vc0_1)), vzp_s16);
      svst1b_s16(p_row, (int8_t*) (c_inc + i * dst_stride_row), vout0);
    }

    if (mr > vl) {
      const size_t m_lower = mr - vl;
      size_t j = 0;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (; j + 4 <= m_lower; j += 4) {
        const uint32_t row = (uint32_t) j;
        const svint32_t vi0_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 0), 4), vbias_0);
        const svint32_t vi0_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 0), 4), vbias_1);
        const svint32_t vi1_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 1), 4), vbias_0);
        const svint32_t vi1_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 1), 4), vbias_1);
        const svint32_t vi2_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 2), 4), vbias_0);
        const svint32_t vi2_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 2), 4), vbias_1);
        const svint32_t vi3_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row + 3), 4), vbias_0);
        const svint32_t vi3_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row + 3), 4), vbias_1);
        const svfloat32_t vf0_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_0), vscale_0);
        const svfloat32_t vf0_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_1), vscale_1);
        const svfloat32_t vf1_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi1_0), vscale_0);
        const svfloat32_t vf1_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi1_1), vscale_1);
        const svfloat32_t vf2_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi2_0), vscale_0);
        const svfloat32_t vf2_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi2_1), vscale_1);
        const svfloat32_t vf3_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi3_0), vscale_0);
        const svfloat32_t vf3_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi3_1), vscale_1);
        const svint32_t vc0_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_0, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc0_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_1, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc1_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_0, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc1_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf1_1, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc2_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf2_0, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc2_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf2_1, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc3_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf3_0, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc3_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf3_1, vmagic_bias)), vmagic_min, vmagic_max);
        const svint16_t vout0 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_0), svreinterpret_s16_s32(vc0_1)), vzp_s16);
        const svint16_t vout1 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc1_0), svreinterpret_s16_s32(vc1_1)), vzp_s16);
        const svint16_t vout2 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc2_0), svreinterpret_s16_s32(vc2_1)), vzp_s16);
        const svint16_t vout3 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc3_0), svreinterpret_s16_s32(vc3_1)), vzp_s16);
        int8_t* c_r = (int8_t*) (c_inc + (j + vl) * dst_stride_row);
        svst1b_s16(p_row, c_r, vout0);
        svst1b_s16(p_row, c_r + dst_stride_row, vout1);
        svst1b_s16(p_row, c_r + 2 * dst_stride_row, vout2);
        svst1b_s16(p_row, c_r + 3 * dst_stride_row, vout3);
      }
      for (; j < m_lower; ++j) {
        const uint32_t row = (uint32_t) j;
        const svint32_t vi0_0 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 2, row), 4), vbias_0);
        const svint32_t vi0_1 = svadd_s32_x(pg_32_all, svasr_n_s32_x(pg_32_all, svread_hor_za32_s32_m(svundef_s32(), pg_32_all, 3, row), 4), vbias_1);
        const svfloat32_t vf0_0 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_0), vscale_0);
        const svfloat32_t vf0_1 = svmul_f32_x(pg_32_all, svcvt_f32_s32_x(pg_32_all, vi0_1), vscale_1);
        const svint32_t vc0_0 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_0, vmagic_bias)), vmagic_min, vmagic_max);
        const svint32_t vc0_1 = svclamp_s32(svreinterpret_s32_f32(svadd_f32_x(pg_32_all, vf0_1, vmagic_bias)), vmagic_min, vmagic_max);
        const svint16_t vout0 = svadd_s16_x(pg_16_all, svuzp1_s16(svreinterpret_s16_s32(vc0_0), svreinterpret_s16_s32(vc0_1)), vzp_s16);
        svst1b_s16(p_row, (int8_t*) (c_inc + (j + vl) * dst_stride_row), vout0);
      }
    }

    if (nc > nr) {
      c_inc += nr;
      nc -= nr;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
