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

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__sme2_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__sme2_get_nr(void) {
  return 2 * svcntsw();
}

#if defined(__APPLE__)
#define PREFETCH_W_MATRIX
#endif

__arm_locally_streaming __arm_new("za", "zt0")
void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__sme2(
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

  static const int8_t lut[64] = {
      0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0,
      4,  0, 0, 0, 5,  0, 0, 0, 6,  0, 0,  0, 7,  0, 0,  0,
      -8, 0, 0, 0, -7, 0, 0, 0, -6, 0, 0,  0, -5, 0, 0,  0,
      -4, 0, 0, 0, -3, 0, 0, 0, -2, 0, 0,  0, -1, 0, 0,  0};
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

  const size_t k_block4 = (kc + 3) / 4;
  const size_t k_unroll4 = k_block4 / 4;
  const size_t k_remainder = k_block4 % 4;

  const size_t k_rounded_32 = ((kc + 31) / 32) * 32;
  const size_t k_block4_rounded = k_rounded_32 / 4;
  const size_t k_tail_slices = k_block4_rounded - k_block4;

  uint8_t* c_inc = (uint8_t*) dst;

#ifdef PREFETCH_W_MATRIX
  const size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  const size_t w_block_size = 4 * nr + k_block4_rounded * 2 * nr + 4 * nr;
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

    svaddha_za32_s32_m(0, pg_32_all, pg_32_all, vbias);
    svaddha_za32_s32_m(2, pg_32_all, pg_32_all, vbias);

    const int8_t* a_inc = (const int8_t*) lhs_packed;

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
      svint8x4_t vb_x4 = svld1_s8_x4(pcnt_8_all, w); w = (const int8_t*) w + 8 * nr;
      svint8x4_t va_x4_0 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += 16 * vl;
      svint8x4_t va_x4_1 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += 16 * vl;

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

      svint8_t vb0 = svluti4_lane_zt_s8(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 0)), 0);
      svint8_t vb1 = svluti4_lane_zt_s8(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 1)), 0);
      svint8_t vb2 = svluti4_lane_zt_s8(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 2)), 0);
      svint8_t vb3 = svluti4_lane_zt_s8(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 3)), 0);

      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, svget4_s8(va_x4_0, 0), vb0);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, svget4_s8(va_x4_0, 1), vb0);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, svget4_s8(va_x4_0, 2), vb1);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, svget4_s8(va_x4_0, 3), vb1);
      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, svget4_s8(va_x4_1, 0), vb2);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, svget4_s8(va_x4_1, 1), vb2);
      svmopa_za32_s8_m(1, pg_8_all, pg_8_all, svget4_s8(va_x4_1, 2), vb3);
      svmopa_za32_s8_m(3, pg_8_all, pg_8_all, svget4_s8(va_x4_1, 3), vb3);
    }

    for (size_t k_rem = 0; k_rem < k_remainder; ++k_rem) {
      svint8x2_t va_x2 = svld1_s8_x2(pcnt_8_all, a_inc); a_inc += 8 * vl;
      svint8_t vb_nibbles = svld1_s8(pg_8_all, (const int8_t*) w); w = (const int8_t*) w + 4 * vl;

      svint8_t vb0 = svluti4_lane_zt_s8(0, svreinterpret_u8_s8(vb_nibbles), 0);

      svmopa_za32_s8_m(0, pg_8_all, pg_8_all, svget2_s8(va_x2, 0), vb0);
      svmopa_za32_s8_m(2, pg_8_all, pg_8_all, svget2_s8(va_x2, 1), vb0);
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

    const svbool_t p_row = svwhilelt_b16_u64(0, nc);
    const bool two_halves = mr > vl;
    const size_t m_upper = (mr < vl) ? mr : vl;
#if defined(__APPLE__)
    const svfloat32_t vzero = svdup_n_f32(0.0f);
    const svfloat32x2_t vzero_x2 = svcreate2_f32(vzero, vzero);
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t s = 0; s < vl / 2; s++) {
      const uint32_t slice = (uint32_t) (4 * s);
      const svint32x2_t vt0 = svread_za32_s32_vg1x2(slice + 0);
      const svint32x2_t vt1 = svread_za32_s32_vg1x2(slice + 1);
      const svfloat32x2_t vf0 = svcvt_f32_s32_x2(svcreate2_s32(
          svadd_s32_x(pg_32_all, svget2_s32(vt0, 0), svget2_s32(vt1, 0)),
          svadd_s32_x(pg_32_all, svget2_s32(vt0, 1), svget2_s32(vt1, 1))));
      svwrite_za32_f32_vg1x2(slice + 0, vzero_x2);
      svmla_single_za32_f32_vg1x2(slice + 0, vf0, vscale);
      if (two_halves) {
        const svint32x2_t vt2 = svread_za32_s32_vg1x2(slice + 2);
        const svint32x2_t vt3 = svread_za32_s32_vg1x2(slice + 3);
        const svfloat32x2_t vf2 = svcvt_f32_s32_x2(svcreate2_s32(
            svadd_s32_x(pg_32_all, svget2_s32(vt2, 0), svget2_s32(vt3, 0)),
            svadd_s32_x(pg_32_all, svget2_s32(vt2, 1), svget2_s32(vt3, 1))));
        svwrite_za32_f32_vg1x2(slice + 2, vzero_x2);
        svmla_single_za32_f32_vg1x2(slice + 2, vf2, vscale);
      }
    }

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t i = 0; i < m_upper; i += 4) {
      const svint32x4_t vacc = svcvt_s32_f32_x4(svrintn_f32_x4(svread_hor_za32_f32_vg4(0, (uint32_t) i)));
      const svint32_t vq0 = svqadd_s32(svget4_s32(vacc, 0), voutput_zp_vec);
      const svint32_t vq1 = svqadd_s32(svget4_s32(vacc, 1), voutput_zp_vec);
      const svint32_t vq2 = svqadd_s32(svget4_s32(vacc, 2), voutput_zp_vec);
      const svint32_t vq3 = svqadd_s32(svget4_s32(vacc, 3), voutput_zp_vec);
      svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(vq0, vq0, vq1, vq1));
      svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(vq2, vq2, vq3, vq3));
      vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
      vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
      const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
      const uintptr_t c_r = (uintptr_t) (c_inc + i * dst_stride_row);
      const size_t rows = m_upper - i;
      svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
      if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
      if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
      if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
    }
    if (two_halves) {
      const size_t m_lower = mr - vl;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t i = 0; i < m_lower; i += 4) {
        const svint32x4_t vacc = svcvt_s32_f32_x4(svrintn_f32_x4(svread_hor_za32_f32_vg4(2, (uint32_t) i)));
        const svint32_t vq0 = svqadd_s32(svget4_s32(vacc, 0), voutput_zp_vec);
        const svint32_t vq1 = svqadd_s32(svget4_s32(vacc, 1), voutput_zp_vec);
        const svint32_t vq2 = svqadd_s32(svget4_s32(vacc, 2), voutput_zp_vec);
        const svint32_t vq3 = svqadd_s32(svget4_s32(vacc, 3), voutput_zp_vec);
        svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(vq0, vq0, vq1, vq1));
        svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(vq2, vq2, vq3, vq3));
        vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
        vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
        const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
        const uintptr_t c_r = (uintptr_t) (c_inc + (i + vl) * dst_stride_row);
        const size_t rows = m_lower - i;
        svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
        if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
        if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
        if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
      }
    }
#else
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t i = 0; i < m_upper; i += 4) {
      const svint32x4_t vi0 = svread_hor_za32_s32_vg4(0, (uint32_t) i);
      const svint32x4_t vi1 = svread_hor_za32_s32_vg4(1, (uint32_t) i);
      const svfloat32x4_t vf = svcvt_f32_s32_x4(svcreate4_s32(
          svadd_s32_x(pg_32_all, svget4_s32(vi0, 0), svget4_s32(vi1, 0)),
          svadd_s32_x(pg_32_all, svget4_s32(vi0, 1), svget4_s32(vi1, 1)),
          svadd_s32_x(pg_32_all, svget4_s32(vi0, 2), svget4_s32(vi1, 2)),
          svadd_s32_x(pg_32_all, svget4_s32(vi0, 3), svget4_s32(vi1, 3))));
      const svfloat32x4_t vscaled = svcreate4_f32(
          svmul_f32_x(pg_32_all, svget4_f32(vf, 0), vscale),
          svmul_f32_x(pg_32_all, svget4_f32(vf, 1), vscale),
          svmul_f32_x(pg_32_all, svget4_f32(vf, 2), vscale),
          svmul_f32_x(pg_32_all, svget4_f32(vf, 3), vscale));
      const svint32x4_t vacc = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled));
      const svint32_t vq0 = svqadd_s32(svget4_s32(vacc, 0), voutput_zp_vec);
      const svint32_t vq1 = svqadd_s32(svget4_s32(vacc, 1), voutput_zp_vec);
      const svint32_t vq2 = svqadd_s32(svget4_s32(vacc, 2), voutput_zp_vec);
      const svint32_t vq3 = svqadd_s32(svget4_s32(vacc, 3), voutput_zp_vec);
      svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(vq0, vq0, vq1, vq1));
      svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(vq2, vq2, vq3, vq3));
      vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
      vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
      const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
      const uintptr_t c_r = (uintptr_t) (c_inc + i * dst_stride_row);
      const size_t rows = m_upper - i;
      svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
      if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
      if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
      if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
    }
    if (two_halves) {
      const size_t m_lower = mr - vl;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t i = 0; i < m_lower; i += 4) {
        const svint32x4_t vi2 = svread_hor_za32_s32_vg4(2, (uint32_t) i);
        const svint32x4_t vi3 = svread_hor_za32_s32_vg4(3, (uint32_t) i);
        const svfloat32x4_t vf = svcvt_f32_s32_x4(svcreate4_s32(
            svadd_s32_x(pg_32_all, svget4_s32(vi2, 0), svget4_s32(vi3, 0)),
            svadd_s32_x(pg_32_all, svget4_s32(vi2, 1), svget4_s32(vi3, 1)),
            svadd_s32_x(pg_32_all, svget4_s32(vi2, 2), svget4_s32(vi3, 2)),
            svadd_s32_x(pg_32_all, svget4_s32(vi2, 3), svget4_s32(vi3, 3))));
        const svfloat32x4_t vscaled = svcreate4_f32(
            svmul_f32_x(pg_32_all, svget4_f32(vf, 0), vscale),
            svmul_f32_x(pg_32_all, svget4_f32(vf, 1), vscale),
            svmul_f32_x(pg_32_all, svget4_f32(vf, 2), vscale),
            svmul_f32_x(pg_32_all, svget4_f32(vf, 3), vscale));
        const svint32x4_t vacc = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled));
        const svint32_t vq0 = svqadd_s32(svget4_s32(vacc, 0), voutput_zp_vec);
        const svint32_t vq1 = svqadd_s32(svget4_s32(vacc, 1), voutput_zp_vec);
        const svint32_t vq2 = svqadd_s32(svget4_s32(vacc, 2), voutput_zp_vec);
        const svint32_t vq3 = svqadd_s32(svget4_s32(vacc, 3), voutput_zp_vec);
        svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(vq0, vq0, vq1, vq1));
        svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(vq2, vq2, vq3, vq3));
        vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
        vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
        const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
        const uintptr_t c_r = (uintptr_t) (c_inc + (i + vl) * dst_stride_row);
        const size_t rows = m_lower - i;
        svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
        if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
        if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
        if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
      }
    }
#endif
    return;
  }

  do {
    svzero_za();

    svint32_t vbias_0 = svld1_s32(pg_32_all, (const int32_t*) w);
    svint32_t vbias_1 = svld1_s32(pg_32_all, (const int32_t*) w + vl);
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

    svaddha_za32_s32_m(0, pg_32_all, pg_32_all, vbias_0);
    svaddha_za32_s32_m(1, pg_32_all, pg_32_all, vbias_1);
    svaddha_za32_s32_m(2, pg_32_all, pg_32_all, vbias_0);
    svaddha_za32_s32_m(3, pg_32_all, pg_32_all, vbias_1);

    const int8_t* a_inc = (const int8_t*) lhs_packed;

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t k_iter = 0; k_iter < k_unroll4; ++k_iter) {
      svint8x4_t vb_x4 = svld1_s8_x4(pcnt_8_all, w); w = (const int8_t*) w + 8 * nr;
      svint8x4_t va_x4_0 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += 16 * vl;
      svint8x4_t va_x4_1 = svld1_s8_x4(pcnt_8_all, a_inc); a_inc += 16 * vl;

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

      svint8x2_t luti_0 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 0)), 0);
      svint8x2_t luti_1 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 1)), 0);
      svint8x2_t luti_2 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 2)), 0);
      svint8x2_t luti_3 = svluti4_lane_zt_s8_x2(0, svreinterpret_u8_s8(svget4_s8(vb_x4, 3)), 0);

      svint8_t vb0_0 = svget2_s8(luti_0, 0);
      svint8_t vb0_1 = svget2_s8(luti_0, 1);
      svint8_t vb1_0 = svget2_s8(luti_1, 0);
      svint8_t vb1_1 = svget2_s8(luti_1, 1);

      svint8_t vb2_0 = svget2_s8(luti_2, 0);
      svint8_t vb2_1 = svget2_s8(luti_2, 1);
      svint8_t vb3_0 = svget2_s8(luti_3, 0);
      svint8_t vb3_1 = svget2_s8(luti_3, 1);

      svint8_t va0_0 = svget4_s8(va_x4_0, 0);
      svint8_t va0_1 = svget4_s8(va_x4_0, 1);
      svint8_t va1_0 = svget4_s8(va_x4_0, 2);
      svint8_t va1_1 = svget4_s8(va_x4_0, 3);

      svint8_t va2_0 = svget4_s8(va_x4_1, 0);
      svint8_t va2_1 = svget4_s8(va_x4_1, 1);
      svint8_t va3_0 = svget4_s8(va_x4_1, 2);
      svint8_t va3_1 = svget4_s8(va_x4_1, 3);

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

    for (size_t k_rem = 0; k_rem < k_remainder; ++k_rem) {
      svint8x2_t va_x2 = svld1_s8_x2(pcnt_8_all, a_inc); a_inc += 8 * vl;
      svint8_t vb_nibbles = svld1_s8(pg_8_all, (const int8_t*) w); w = (const int8_t*) w + 4 * vl;

      svint8x2_t luti_rem = svluti4_lane_zt_s8_x2(0, svreinterpret_u8_s8(vb_nibbles), 0);
      svint8_t vb0_0 = svget2_s8(luti_rem, 0);
      svint8_t vb0_1 = svget2_s8(luti_rem, 1);

      svint8_t va0_0 = svget2_s8(va_x2, 0);
      svint8_t va0_1 = svget2_s8(va_x2, 1);

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

    svfloat32_t vscale_0 = svld1_f32(pg_32_all, (const float*) w);
    svfloat32_t vscale_1 = svld1_f32(pg_32_all, (const float*) w + vl);
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

#if defined(__APPLE__)
    const size_t num_tiles = (mr > vl) ? 4 : 2;
    const size_t num_pairs = vl / 2;
    const svfloat32_t vzero = svdup_n_f32(0.0f);
    const svfloat32x2_t vzero_x2 = svcreate2_f32(vzero, vzero);
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t s = 0; s < num_pairs; s++) {
      const uint32_t slice = (uint32_t) (4 * s);
      if (num_tiles == 4) {
        const svfloat32x2_t vf0 = svcvt_f32_s32_x2(svread_za32_s32_vg1x2(slice + 0));
        const svfloat32x2_t vf1 = svcvt_f32_s32_x2(svread_za32_s32_vg1x2(slice + 1));
        const svfloat32x2_t vf2 = svcvt_f32_s32_x2(svread_za32_s32_vg1x2(slice + 2));
        const svfloat32x2_t vf3 = svcvt_f32_s32_x2(svread_za32_s32_vg1x2(slice + 3));
        svwrite_za32_f32_vg1x2(slice + 0, vzero_x2);
        svwrite_za32_f32_vg1x2(slice + 1, vzero_x2);
        svwrite_za32_f32_vg1x2(slice + 2, vzero_x2);
        svwrite_za32_f32_vg1x2(slice + 3, vzero_x2);
        svmla_single_za32_f32_vg1x2(slice + 0, vf0, vscale_0);
        svmla_single_za32_f32_vg1x2(slice + 1, vf1, vscale_1);
        svmla_single_za32_f32_vg1x2(slice + 2, vf2, vscale_0);
        svmla_single_za32_f32_vg1x2(slice + 3, vf3, vscale_1);
      } else {
        const svfloat32x2_t vf0 = svcvt_f32_s32_x2(svread_za32_s32_vg1x2(slice + 0));
        const svfloat32x2_t vf1 = svcvt_f32_s32_x2(svread_za32_s32_vg1x2(slice + 1));
        svwrite_za32_f32_vg1x2(slice + 0, vzero_x2);
        svwrite_za32_f32_vg1x2(slice + 1, vzero_x2);
        svmla_single_za32_f32_vg1x2(slice + 0, vf0, vscale_0);
        svmla_single_za32_f32_vg1x2(slice + 1, vf1, vscale_1);
      }
    }

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t i = 0; i < m_upper; i += 4) {
      const svint32x4_t vacc_lo = svcvt_s32_f32_x4(svrintn_f32_x4(svread_hor_za32_f32_vg4(0, (uint32_t) i)));
      const svint32x4_t vacc_hi = svcvt_s32_f32_x4(svrintn_f32_x4(svread_hor_za32_f32_vg4(1, (uint32_t) i)));
      svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(
          svqadd_s32(svget4_s32(vacc_lo, 0), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 0), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_lo, 1), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 1), voutput_zp_vec)));
      svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(
          svqadd_s32(svget4_s32(vacc_lo, 2), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 2), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_lo, 3), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 3), voutput_zp_vec)));
      vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
      vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
      const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
      const uintptr_t c_r = (uintptr_t) (c_inc + i * dst_stride_row);
      const size_t rows = m_upper - i;
      svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
      if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
      if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
      if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
    }

    if (mr > vl) {
      const size_t m_lower = mr - vl;

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t i = 0; i < m_lower; i += 4) {
        const svint32x4_t vacc_lo = svcvt_s32_f32_x4(svrintn_f32_x4(svread_hor_za32_f32_vg4(2, (uint32_t) i)));
        const svint32x4_t vacc_hi = svcvt_s32_f32_x4(svrintn_f32_x4(svread_hor_za32_f32_vg4(3, (uint32_t) i)));
        svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(
            svqadd_s32(svget4_s32(vacc_lo, 0), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 0), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_lo, 1), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 1), voutput_zp_vec)));
        svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(
            svqadd_s32(svget4_s32(vacc_lo, 2), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 2), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_lo, 3), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 3), voutput_zp_vec)));
        vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
        vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
        const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
        const uintptr_t c_r = (uintptr_t) (c_inc + (i + vl) * dst_stride_row);
        const size_t rows = m_lower - i;
        svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
        if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
        if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
        if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
      }
    }
#else
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t i = 0; i < m_upper; i += 4) {
      const svfloat32x4_t vf_lo = svcvt_f32_s32_x4(svread_hor_za32_s32_vg4(0, (uint32_t) i));
      const svfloat32x4_t vf_hi = svcvt_f32_s32_x4(svread_hor_za32_s32_vg4(1, (uint32_t) i));
      const svfloat32x4_t vscaled_lo = svcreate4_f32(
          svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 0), vscale_0),
          svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 1), vscale_0),
          svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 2), vscale_0),
          svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 3), vscale_0));
      const svfloat32x4_t vscaled_hi = svcreate4_f32(
          svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 0), vscale_1),
          svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 1), vscale_1),
          svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 2), vscale_1),
          svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 3), vscale_1));
      const svint32x4_t vacc_lo = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled_lo));
      const svint32x4_t vacc_hi = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled_hi));
      svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(
          svqadd_s32(svget4_s32(vacc_lo, 0), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 0), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_lo, 1), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 1), voutput_zp_vec)));
      svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(
          svqadd_s32(svget4_s32(vacc_lo, 2), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 2), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_lo, 3), voutput_zp_vec),
          svqadd_s32(svget4_s32(vacc_hi, 3), voutput_zp_vec)));
      vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
      vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
      const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
      const uintptr_t c_r = (uintptr_t) (c_inc + i * dst_stride_row);
      const size_t rows = m_upper - i;
      svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
      if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
      if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
      if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
    }

    if (mr > vl) {
      const size_t m_lower = mr - vl;

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t i = 0; i < m_lower; i += 4) {
        const svfloat32x4_t vf_lo = svcvt_f32_s32_x4(svread_hor_za32_s32_vg4(2, (uint32_t) i));
        const svfloat32x4_t vf_hi = svcvt_f32_s32_x4(svread_hor_za32_s32_vg4(3, (uint32_t) i));
        const svfloat32x4_t vscaled_lo = svcreate4_f32(
            svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 0), vscale_0),
            svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 1), vscale_0),
            svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 2), vscale_0),
            svmul_f32_x(pg_32_all, svget4_f32(vf_lo, 3), vscale_0));
        const svfloat32x4_t vscaled_hi = svcreate4_f32(
            svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 0), vscale_1),
            svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 1), vscale_1),
            svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 2), vscale_1),
            svmul_f32_x(pg_32_all, svget4_f32(vf_hi, 3), vscale_1));
        const svint32x4_t vacc_lo = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled_lo));
        const svint32x4_t vacc_hi = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled_hi));
        svint8_t vout_01 = svqcvt_s8_s32_x4(svcreate4_s32(
            svqadd_s32(svget4_s32(vacc_lo, 0), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 0), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_lo, 1), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 1), voutput_zp_vec)));
        svint8_t vout_23 = svqcvt_s8_s32_x4(svcreate4_s32(
            svqadd_s32(svget4_s32(vacc_lo, 2), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 2), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_lo, 3), voutput_zp_vec),
            svqadd_s32(svget4_s32(vacc_hi, 3), voutput_zp_vec)));
        vout_01 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_01, voutput_min_vec), voutput_max_vec);
        vout_23 = svmin_s8_x(pg_8_all, svmax_s8_x(pg_8_all, vout_23, voutput_min_vec), voutput_max_vec);
        const svint8x2_t vout = svcreate2_s8(vout_01, vout_23);
        const uintptr_t c_r = (uintptr_t) (c_inc + (i + vl) * dst_stride_row);
        const size_t rows = m_lower - i;
        svst1b_s16(p_row, (int8_t*) c_r, svunpklo_s16(svget2_s8(vout, 0)));
        if (rows > 1) svst1b_s16(p_row, (int8_t*) (c_r + dst_stride_row), svunpkhi_s16(svget2_s8(vout, 0)));
        if (rows > 2) svst1b_s16(p_row, (int8_t*) (c_r + 2 * dst_stride_row), svunpklo_s16(svget2_s8(vout, 1)));
        if (rows > 3) svst1b_s16(p_row, (int8_t*) (c_r + 3 * dst_stride_row), svunpkhi_s16(svget2_s8(vout, 1)));
      }
    }
#endif

    if (nc > nr) {
      c_inc += nr;
      nc -= nr;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
