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
#include <string.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

size_t xnn_x32_pack_lh_size__sme(size_t m, size_t k, size_t mr_packed,
                                  size_t kr, size_t sr) {
  assert(kr == 1);
  assert(sr == 1);
  assert(mr_packed == 1 || mr_packed == 2 * svcntsw());
  (void) kr;
  (void) sr;
  return round_up(m, mr_packed) * k * sizeof(float);
}

size_t xnn_x32_pack_lh_offset__sme(size_t m, size_t k, size_t mr_packed,
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
static void xnn_x32_packlh_sme_m32__sme(
    size_t m, size_t k, const float* lhs, size_t lhs_row_stride, float* lhs_packed)
{
    const size_t vl = svcntw();
    const size_t mr2 = 2 * vl;
#ifdef PREFETCH_A_MATRIX
#if defined(__APPLE__)
    const size_t prfm_line_size = 128;
    const size_t prfm_min_lhs_size = (1024 * 1024);
#else
    const size_t prfm_line_size = 64;
    const size_t prfm_min_lhs_size = (128 * 1024);
#endif
    const size_t prfm_line = prfm_line_size / sizeof(float);
    const size_t prfm_dist = 4 * vl;
    const bool need_prfm = (m * k * sizeof(float) >= prfm_min_lhs_size);
#endif

    size_t m_roundup_mr = ((m + mr2 - 1) / mr2) * mr2;

    for (size_t mr_blk_idx = 0; mr_blk_idx < m_roundup_mr; mr_blk_idx = mr_blk_idx + mr2) {

      size_t current_m = (mr_blk_idx == (m_roundup_mr - mr2)) ? (m - mr_blk_idx) : mr2;

      const float* lhs_blk = lhs + mr_blk_idx * lhs_row_stride;
      float* lhs_packed_0 = lhs_packed + mr_blk_idx * k;

#ifdef PREFETCH_A_MATRIX
      size_t pf_col = 0;
      if (need_prfm) {
        const size_t pf_end = (k < prfm_dist) ? k : prfm_dist;
        for (; pf_col < pf_end; pf_col += prfm_line) {
          for (size_t r = 0; r < current_m; r++) {
            __builtin_prefetch(lhs_blk + r * lhs_row_stride + pf_col, 0, 2);
          }
        }
      }
#endif

      svbool_t pg_32_all = svptrue_b32();


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
            const size_t pf_end = (k_idx + 4 * vl + prfm_dist < k) ? (k_idx + 4 * vl + prfm_dist) : k;
            for (; pf_col < pf_end; pf_col += prfm_line) {
              for (size_t r = 0; r < mr2; r++) {
                __builtin_prefetch(lhs_blk + r * lhs_row_stride + pf_col, 0, 2);
              }
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

              svfloat32x4_t r0_x4 = svcreate4_f32(
                  svld1_vnum_f32(pg_32_all, src0, 0),
                  svld1_vnum_f32(pg_32_all, src0, 1),
                  svld1_vnum_f32(pg_32_all, src0, 2),
                  svld1_vnum_f32(pg_32_all, src0, 3));
              svfloat32x4_t r1_x4 = svcreate4_f32(
                  svld1_vnum_f32(pg_32_all, src1, 0),
                  svld1_vnum_f32(pg_32_all, src1, 1),
                  svld1_vnum_f32(pg_32_all, src1, 2),
                  svld1_vnum_f32(pg_32_all, src1, 3));
              svfloat32x4_t r2_x4 = svcreate4_f32(
                  svld1_vnum_f32(pg_32_all, src2, 0),
                  svld1_vnum_f32(pg_32_all, src2, 1),
                  svld1_vnum_f32(pg_32_all, src2, 2),
                  svld1_vnum_f32(pg_32_all, src2, 3));
              svfloat32x4_t r3_x4 = svcreate4_f32(
                  svld1_vnum_f32(pg_32_all, src3, 0),
                  svld1_vnum_f32(pg_32_all, src3, 1),
                  svld1_vnum_f32(pg_32_all, src3, 2),
                  svld1_vnum_f32(pg_32_all, src3, 3));

              svwrite_hor_za32_f32_m(0, row_idx, pg_32_all, svget4_f32(r0_x4, 0));
              svwrite_hor_za32_f32_m(0, row_idx + 1, pg_32_all, svget4_f32(r1_x4, 0));
              svwrite_hor_za32_f32_m(0, row_idx + 2, pg_32_all, svget4_f32(r2_x4, 0));
              svwrite_hor_za32_f32_m(0, row_idx + 3, pg_32_all, svget4_f32(r3_x4, 0));
              svwrite_hor_za32_f32_m(1, row_idx, pg_32_all, svget4_f32(r0_x4, 1));
              svwrite_hor_za32_f32_m(1, row_idx + 1, pg_32_all, svget4_f32(r1_x4, 1));
              svwrite_hor_za32_f32_m(1, row_idx + 2, pg_32_all, svget4_f32(r2_x4, 1));
              svwrite_hor_za32_f32_m(1, row_idx + 3, pg_32_all, svget4_f32(r3_x4, 1));
              svwrite_hor_za32_f32_m(2, row_idx, pg_32_all, svget4_f32(r0_x4, 2));
              svwrite_hor_za32_f32_m(2, row_idx + 1, pg_32_all, svget4_f32(r1_x4, 2));
              svwrite_hor_za32_f32_m(2, row_idx + 2, pg_32_all, svget4_f32(r2_x4, 2));
              svwrite_hor_za32_f32_m(2, row_idx + 3, pg_32_all, svget4_f32(r3_x4, 2));
              svwrite_hor_za32_f32_m(3, row_idx, pg_32_all, svget4_f32(r0_x4, 3));
              svwrite_hor_za32_f32_m(3, row_idx + 1, pg_32_all, svget4_f32(r1_x4, 3));
              svwrite_hor_za32_f32_m(3, row_idx + 2, pg_32_all, svget4_f32(r2_x4, 3));
              svwrite_hor_za32_f32_m(3, row_idx + 3, pg_32_all, svget4_f32(r3_x4, 3));
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

#ifdef PREFETCH_A_MATRIX
          if (need_prfm) {
            const size_t pf_end = (k_idx + mr2 + prfm_dist < k) ? (k_idx + mr2 + prfm_dist) : k;
            for (; pf_col < pf_end; pf_col += prfm_line) {
              for (size_t r = 0; r < current_m; r++) {
                __builtin_prefetch(lhs_blk + r * lhs_row_stride + pf_col, 0, 2);
              }
            }
          }
#endif

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
              svfloat32x2_t s0_x2 = svcreate2_f32(
                  svld1_vnum_f32(pg_32_all, src_ptr_0, 0),
                  svld1_vnum_f32(pg_32_all, src_ptr_0, 1));
              svfloat32x2_t s1_x2 = svcreate2_f32(
                  svld1_vnum_f32(pg_32_all, src_ptr_1, 0),
                  svld1_vnum_f32(pg_32_all, src_ptr_1, 1));
              s0_l = svget2_f32(s0_x2, 0);
              s0_h = svget2_f32(s0_x2, 1);
              s1_l = svget2_f32(s1_x2, 0);
              s1_h = svget2_f32(s1_x2, 1);
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

            if (row_idx < vl) {
              svwrite_hor_za32_f32_m(0, row_idx, pg_32_all, s0_l);
              svwrite_hor_za32_f32_m(0, row_idx + 1, pg_32_all, s1_l);
              svwrite_hor_za32_f32_m(1, row_idx, pg_32_all, s0_h);
              svwrite_hor_za32_f32_m(1, row_idx + 1, pg_32_all, s1_h);
            } else {
              svwrite_hor_za32_f32_m(2, row_idx - vl, pg_32_all, s0_l);
              svwrite_hor_za32_f32_m(2, row_idx - vl + 1, pg_32_all, s1_l);
              svwrite_hor_za32_f32_m(3, row_idx - vl, pg_32_all, s0_h);
              svwrite_hor_za32_f32_m(3, row_idx - vl + 1, pg_32_all, s1_h);
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

void xnn_x32_pack_lh_ukernel__sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed)
{
  if XNN_UNLIKELY(m == 0 || k == 0) {
    return;
  }
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
    xnn_x32_packlh_sme_m32__sme(m, k, lhs, lhs_stride / sizeof(float), lhs_packed);
  }
}

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
