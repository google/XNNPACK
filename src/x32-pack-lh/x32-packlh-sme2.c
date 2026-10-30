// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

size_t xnn_x32_pack_lh_size__sme2(size_t m, size_t k, size_t mr_packed,
                                  size_t kr, size_t sr) {
  assert(kr == 1);
  assert(sr == 1);
  assert(mr_packed == 1 || mr_packed == 2 * svcntsw());
  (void) kr;
  (void) sr;
  return round_up(m, mr_packed) * k * sizeof(float);
}

size_t xnn_x32_pack_lh_offset__sme2(size_t m, size_t k, size_t mr_packed,
                                    size_t kr, size_t sr) {
  assert(kr == 1);
  assert(sr == 1);
  assert(mr_packed == 1 || mr_packed == 2 * svcntsw());
  (void) mr_packed;
  (void) kr;
  (void) sr;
  return m * k * sizeof(float);
}

__arm_locally_streaming __arm_new("za")
static void xnn_x32_packlh_sme_m32__sme2(
    size_t m, size_t k, const float* lhs, size_t lhs_row_stride, float* lhs_packed)
{
    const size_t vl = svcntw();
    const size_t mr2 = 2 * vl;
#ifdef PREFETCH_A_MATRIX
    bool need_prfm = (k >= 4 * vl);
#endif

    size_t m_roundup_mr = ((m + mr2 - 1) / mr2) * mr2;

    for (size_t mr_blk_idx = 0; mr_blk_idx < m_roundup_mr; mr_blk_idx = mr_blk_idx + mr2) {

      size_t current_m = (mr_blk_idx == (m_roundup_mr - mr2)) ? (m - mr_blk_idx) : mr2;

      const float* lhs_blk = lhs + mr_blk_idx * lhs_row_stride;
      float* lhs_packed_0 = lhs_packed + mr_blk_idx * k;

#ifdef PREFETCH_A_MATRIX
      if (need_prfm) {
        for (size_t r = 0; r < current_m; r++) {
          const float* row_ptr = lhs_blk + (r * lhs_row_stride);
          __builtin_prefetch(row_ptr + 0, 0, 2);
          __builtin_prefetch(row_ptr + vl, 0, 2);
          __builtin_prefetch(row_ptr + 2 * vl, 0, 2);
          __builtin_prefetch(row_ptr + 3 * vl, 0, 2);
        }
      }
#endif

      svbool_t pg_32_all = svptrue_b32();

      //===== x32 to hor ZA -> ZA ver out

      ptrdiff_t k_left = (ptrdiff_t) k;

      size_t k_idx = 0;
      if (XNN_LIKELY(current_m == mr2)) {

        for (; k_idx + 4 * vl <= k; k_idx += 4 * vl) {
          k_left -= 4 * vl;
          float* out_block = lhs_packed_0;
          lhs_packed_0 += mr2 * 4 * vl;

          const float* lhs_offset = lhs_blk + k_idx;

#ifdef PREFETCH_A_MATRIX
          if (need_prfm) {
            for (size_t r = 0; r < mr2; r++) {
              const float* r_ptr = lhs_offset + (r * lhs_row_stride);
              if (k_idx + 4 * vl < k) __builtin_prefetch(r_ptr + 4 * vl, 0, 2);
              if (k_idx + 5 * vl < k) __builtin_prefetch(r_ptr + 5 * vl, 0, 2);
              if (k_idx + 6 * vl < k) __builtin_prefetch(r_ptr + 6 * vl, 0, 2);
              if (k_idx + 7 * vl < k) __builtin_prefetch(r_ptr + 7 * vl, 0, 2);
            }
          }
#endif

          for (size_t p = 0; p < 2; p++) {
            const float* lhs_offset_p = lhs_offset + p * vl * lhs_row_stride;
            for (size_t row_idx = 0; row_idx < vl; row_idx += 4) {
              const float* src0 = lhs_offset_p + (row_idx * lhs_row_stride);
              const float* src1 = lhs_offset_p + ((row_idx + 1) * lhs_row_stride);
              const float* src2 = lhs_offset_p + ((row_idx + 2) * lhs_row_stride);
              const float* src3 = lhs_offset_p + ((row_idx + 3) * lhs_row_stride);

              svfloat32x4_t r0_x4 = svld1_f32_x4(svptrue_c32(), src0);
              svfloat32x4_t r1_x4 = svld1_f32_x4(svptrue_c32(), src1);
              svfloat32x4_t r2_x4 = svld1_f32_x4(svptrue_c32(), src2);
              svfloat32x4_t r3_x4 = svld1_f32_x4(svptrue_c32(), src3);

              svwrite_hor_za32_f32_vg4(0, row_idx, svcreate4(svget4(r0_x4, 0), svget4(r1_x4, 0), svget4(r2_x4, 0), svget4(r3_x4, 0)));
              svwrite_hor_za32_f32_vg4(1, row_idx, svcreate4(svget4(r0_x4, 1), svget4(r1_x4, 1), svget4(r2_x4, 1), svget4(r3_x4, 1)));
              svwrite_hor_za32_f32_vg4(2, row_idx, svcreate4(svget4(r0_x4, 2), svget4(r1_x4, 2), svget4(r2_x4, 2), svget4(r3_x4, 2)));
              svwrite_hor_za32_f32_vg4(3, row_idx, svcreate4(svget4(r0_x4, 3), svget4(r1_x4, 3), svget4(r2_x4, 3), svget4(r3_x4, 3)));
            }

            for (size_t c = 0; c < vl; c += 1) {
              svst1_ver_za32(0, c, pg_32_all, out_block + c * mr2 + p * vl);
              svst1_ver_za32(1, c, pg_32_all, out_block + (vl + c) * mr2 + p * vl);
              svst1_ver_za32(2, c, pg_32_all, out_block + (2 * vl + c) * mr2 + p * vl);
              svst1_ver_za32(3, c, pg_32_all, out_block + (3 * vl + c) * mr2 + p * vl);
            }
          }
        }

      }

      if (current_m < mr2) {
        svzero_za();
      }
      size_t row_end = (current_m + 1) & ~(size_t) 1;

      for (; k_idx < k; k_idx += mr2) {
          size_t len_2x = (k_left >= (ptrdiff_t) mr2) ? mr2 : (k_left > 0 ? (size_t) k_left : 0);
          k_left = k_left - mr2;

          const float* src_ptr_0;
          const float* src_ptr_1;
          const float* lhs_offset = lhs_blk + k_idx;

          for (size_t row_idx = 0; row_idx < row_end; row_idx = row_idx + 2) {
            size_t m_valid_0 = 1;
            size_t m_valid_1 = 1;

            if (current_m >= (row_idx+1)) {
              src_ptr_0 = lhs_offset + (lhs_row_stride * row_idx);
            } else {
              src_ptr_0 = lhs_offset;
              m_valid_0 = 0;
            }

            if (current_m >= (row_idx+2)) {
              src_ptr_1 = lhs_offset + (lhs_row_stride * (row_idx+1));
            } else {
              src_ptr_1 = lhs_offset;
              m_valid_1 = 0;
            }

            svfloat32_t s0_l, s0_h, s1_l, s1_h;
            if (XNN_LIKELY(len_2x == mr2 && m_valid_0 && m_valid_1)) {
              svfloat32x2_t s0_x2 = svld1_f32_x2(svptrue_c32(), src_ptr_0);
              svfloat32x2_t s1_x2 = svld1_f32_x2(svptrue_c32(), src_ptr_1);
              s0_l = svget2(s0_x2, 0);
              s0_h = svget2(s0_x2, 1);
              s1_l = svget2(s1_x2, 0);
              s1_h = svget2(s1_x2, 1);
            } else {
              svbool_t pg_0_low  = svwhilelt_b32_u64(0, m_valid_0 ? len_2x : 0);
              svbool_t pg_0_high = svwhilelt_b32_u64(vl, m_valid_0 ? len_2x : 0);
              svbool_t pg_1_low  = svwhilelt_b32_u64(0, m_valid_1 ? len_2x : 0);
              svbool_t pg_1_high = svwhilelt_b32_u64(vl, m_valid_1 ? len_2x : 0);

              s0_l = svld1_f32(pg_0_low,  src_ptr_0 + 0);
              s0_h = svld1_f32(pg_0_high, src_ptr_0 + vl);
              s1_l = svld1_f32(pg_1_low,  src_ptr_1 + 0);
              s1_h = svld1_f32(pg_1_high, src_ptr_1 + vl);
            }

#ifdef PREFETCH_A_MATRIX
            if (need_prfm) {
              if (k_idx + mr2 < k) {
                if (m_valid_0) __builtin_prefetch(src_ptr_0 + mr2, 0, 2);
                if (m_valid_1) __builtin_prefetch(src_ptr_1 + mr2, 0, 2);
              }
              if (k_idx + 3 * vl < k) {
                if (m_valid_0) __builtin_prefetch(src_ptr_0 + 3 * vl, 0, 2);
                if (m_valid_1) __builtin_prefetch(src_ptr_1 + 3 * vl, 0, 2);
              }
            }
#endif

            if (row_idx < vl) {
              svwrite_hor_za32_f32_vg2(0, row_idx,      svcreate2(s0_l, s1_l));
              svwrite_hor_za32_f32_vg2(1, row_idx,      svcreate2(s0_h, s1_h));
            } else {
              svwrite_hor_za32_f32_vg2(2, row_idx - vl, svcreate2(s0_l, s1_l));
              svwrite_hor_za32_f32_vg2(3, row_idx - vl, svcreate2(s0_h, s1_h));
            }
          }

          size_t k_out = k - k_idx;
          size_t len0 = (k_out < vl) ? k_out : vl;
          size_t len1 = (k_out > vl) ? ((k_out - vl < vl) ? k_out - vl : vl) : 0;

          for (size_t k_idx_out = 0; k_idx_out < len0; ++k_idx_out) {
            svst1_ver_za32(0, k_idx_out, pg_32_all, lhs_packed_0); lhs_packed_0 += vl;
            svst1_ver_za32(2, k_idx_out, pg_32_all, lhs_packed_0); lhs_packed_0 += vl;
          }
          for (size_t k_idx_out = 0; k_idx_out < len1; ++k_idx_out) {
            svst1_ver_za32(1, k_idx_out, pg_32_all, lhs_packed_0); lhs_packed_0 += vl;
            svst1_ver_za32(3, k_idx_out, pg_32_all, lhs_packed_0); lhs_packed_0 += vl;
          }
        }
    }
}

void xnn_x32_pack_lh_ukernel__sme2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed)
{
  assert(kr == 1);
  assert(sr == 1);
  assert(mr == 1 || mr == 2 * svcntsw());
  assert(m_idx_start == 0);
  assert(lhs_stride % sizeof(float) == 0);
  assert(lhs_stride >= k * sizeof(float));
  (void) kr;
  (void) sr;
  (void) m_idx_start;
  if (mr == 1) {
    float* out = (float*) lhs_packed;
    for (size_t row = 0; row < m; row++) {
      memcpy(out + row * k, (const float*) ((uintptr_t) lhs + row * lhs_stride), k * sizeof(float));
    }
  } else {
    xnn_x32_packlh_sme_m32__sme2(m, k, lhs, lhs_stride / sizeof(float), lhs_packed);
  }
}

#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
