// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

size_t xnn_x16_pack_lh_size__sme2(size_t m, size_t k, size_t mr_packed,
                                  size_t kr, size_t sr) {
  assert(sr == 1);
  assert(mr_packed == 1 || (mr_packed == 2 * svcntsw() && kr == 2));
  (void) sr;
  if (mr_packed == 1) {
    return m * k * sizeof(xnn_float16);
  }
  return round_up(m, mr_packed) * round_up(k, kr) * sizeof(xnn_float16);
}

size_t xnn_x16_pack_lh_offset__sme2(size_t m, size_t k, size_t mr_packed,
                                    size_t kr, size_t sr) {
  assert(sr == 1);
  assert(mr_packed == 1 || (mr_packed == 2 * svcntsw() && kr == 2));
  (void) sr;
  if (mr_packed == 1) {
    return m * k * sizeof(xnn_float16);
  }
  return m * round_up(k, kr) * sizeof(xnn_float16);
}

__arm_locally_streaming __arm_new("za")
static void xnn_x16_packlh_mr2_kr2__sme2(
    size_t m, size_t k,
    const uint16_t* XNN_RESTRICT lhs, size_t ld,
    uint16_t* XNN_RESTRICT lhs_packed)
{
  const size_t vl = svcntw();
  const size_t mr2 = 2 * vl;
  const size_t k_rounded_2 = ((k + 1) / 2) * 2;
  const size_t m_rounded = ((m + mr2 - 1) / mr2) * mr2;
  const svbool_t pg_16_all = svptrue_b16();
  const svbool_t pg_32_all = svptrue_b32();
  const svcount_t pcnt_16_all = svptrue_c16();
  const svuint32_t vzero = svdup_u32(0);
  const svuint16_t vzero_u16 = svdup_u16(0);

  for (size_t mr_blk_idx = 0; mr_blk_idx < m_rounded; mr_blk_idx += mr2) {
    const size_t valid_rows = (mr_blk_idx + mr2 <= m) ? mr2 : (m - mr_blk_idx);
    const uint16_t* lhs_blk = lhs + mr_blk_idx * ld;
    uint16_t* lhs_packed_inc = lhs_packed + mr_blk_idx * k_rounded_2;
    size_t k_idx = 0;

    if (valid_rows <= vl) {
      if (valid_rows < vl) {
        svzero_za();
      }
      for (; k_idx + 4 * vl <= k; k_idx += 4 * vl) {
        const uint16_t* src = lhs_blk + k_idx;
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows; row_idx += 2) {
          const uint16_t* src1 = src + ld;
          svuint16_t r0_lo = svld1_u16(pg_16_all, src);
          svuint16_t r1_lo = svld1_u16(pg_16_all, src1);
          svuint16_t r0_hi = svld1_u16(pg_16_all, src + 2 * vl);
          svuint16_t r1_hi = svld1_u16(pg_16_all, src1 + 2 * vl);
          src += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0_lo, r1_lo)));
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r0_hi, r1_hi)));
        }
        if (row_idx < valid_rows) {
          svuint16_t r0_lo = svld1_u16(pg_16_all, src);
          svuint16_t r0_hi = svld1_u16(pg_16_all, src + 2 * vl);
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0_lo, vzero_u16)));
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r0_hi, vzero_u16)));
        }

        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), vzero, svget4(v0, 1), vzero);
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), vzero, svget4(v0, 3), vzero);
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(1, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), vzero, svget4(v0, 1), vzero);
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), vzero, svget4(v0, 3), vzero);
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
      }

      for (; k_idx + 2 * vl <= k; k_idx += 2 * vl) {
        const uint16_t* src = lhs_blk + k_idx;
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows; row_idx += 2) {
          svuint16_t r0 = svld1_u16(pg_16_all, src);
          svuint16_t r1 = svld1_u16(pg_16_all, src + ld);
          src += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, r1)));
        }
        if (row_idx < valid_rows) {
          svuint16_t r0 = svld1_u16(pg_16_all, src);
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, vzero_u16)));
        }

        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), vzero, svget4(v0, 1), vzero);
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), vzero, svget4(v0, 3), vzero);
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
      }

      if (k_idx < k_rounded_2) {
        const size_t k_rem = (k > k_idx) ? (k - k_idx) : 0;
        const svbool_t pg = svwhilelt_b16_u64(0, k_rem);
        const size_t n_cols = (k_rounded_2 - k_idx) / 2;

        const uint16_t* src = lhs_blk + k_idx;
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows; row_idx += 2) {
          svuint16_t r0 = svld1_u16(pg, src);
          svuint16_t r1 = svld1_u16(pg, src + ld);
          src += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, r1)));
        }
        if (row_idx < valid_rows) {
          svuint16_t r0 = svld1_u16(pg, src);
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, vzero_u16)));
        }

        size_t col = 0;
        for (; col + 4 <= n_cols; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), vzero, svget4(v0, 1), vzero);
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), vzero, svget4(v0, 3), vzero);
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
        for (; col + 2 <= n_cols; col += 2) {
          svuint32x2_t v0 = svread_ver_za32_u32_vg2(0, col);
          svuint32x4_t comb0 = svcreate4(svget2(v0, 0), vzero, svget2(v0, 1), vzero);
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
        }
        if (col < n_cols) {
          svuint32_t v00 = svread_ver_za32_u32_m(svundef_u32(), pg_32_all, 0, col);
          svst1_u16_x2(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(svcreate2(v00, vzero)));
        }
      }
    } else if (XNN_LIKELY(valid_rows == mr2)) {
      for (; k_idx + 8 * vl <= k; k_idx += 8 * vl) {
        uint16_t* out_block = lhs_packed_inc;
        lhs_packed_inc += mr2 * 8 * vl;
        for (size_t p = 0; p < 2; p++) {
          const uint16_t* rp = lhs_blk + p * vl * ld + k_idx;
          for (size_t r = 0; r < vl; r++) {
            svld1_hor_za32(0, r, pg_32_all, rp);
            svld1_hor_za32(1, r, pg_32_all, rp + 2 * vl);
            svld1_hor_za32(2, r, pg_32_all, rp + 4 * vl);
            svld1_hor_za32(3, r, pg_32_all, rp + 6 * vl);
            rp += ld;
          }
          uint16_t* out_p = out_block + p * mr2;
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(0, c, pg_32_all, out_p + c * 2 * mr2);
          }
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(1, c, pg_32_all, out_p + (vl + c) * 2 * mr2);
          }
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(2, c, pg_32_all, out_p + (2 * vl + c) * 2 * mr2);
          }
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(3, c, pg_32_all, out_p + (3 * vl + c) * 2 * mr2);
          }
        }
      }

      for (; k_idx + 4 * vl <= k; k_idx += 4 * vl) {
        const uint16_t* src0 = lhs_blk + k_idx;
        const uint16_t* src1 = src0 + vl * ld;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint16_t* src0_1 = src0 + ld;
          svuint16_t r0_lo = svld1_u16(pg_16_all, src0);
          svuint16_t r1_lo = svld1_u16(pg_16_all, src0_1);
          svuint16_t r0_hi = svld1_u16(pg_16_all, src0 + 2 * vl);
          svuint16_t r1_hi = svld1_u16(pg_16_all, src0_1 + 2 * vl);
          src0 += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0_lo, r1_lo)));
          svwrite_hor_za32_u32_vg2(2, row_idx, svreinterpret_u32(svcreate2(r0_hi, r1_hi)));
        }
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint16_t* src1_1 = src1 + ld;
          svuint16_t r2_lo = svld1_u16(pg_16_all, src1);
          svuint16_t r3_lo = svld1_u16(pg_16_all, src1_1);
          svuint16_t r2_hi = svld1_u16(pg_16_all, src1 + 2 * vl);
          svuint16_t r3_hi = svld1_u16(pg_16_all, src1_1 + 2 * vl);
          src1 += 2 * ld;
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r2_lo, r3_lo)));
          svwrite_hor_za32_u32_vg2(3, row_idx, svreinterpret_u32(svcreate2(r2_hi, r3_hi)));
        }

        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t v1 = svread_ver_za32_u32_vg4(1, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), svget4(v1, 0), svget4(v0, 1), svget4(v1, 1));
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), svget4(v1, 2), svget4(v0, 3), svget4(v1, 3));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(2, col);
          svuint32x4_t v1 = svread_ver_za32_u32_vg4(3, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), svget4(v1, 0), svget4(v0, 1), svget4(v1, 1));
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), svget4(v1, 2), svget4(v0, 3), svget4(v1, 3));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
      }

      for (; k_idx + 2 * vl <= k; k_idx += 2 * vl) {
        const uint16_t* src0 = lhs_blk + k_idx;
        const uint16_t* src1 = src0 + vl * ld;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          svuint16_t r0 = svld1_u16(pg_16_all, src0);
          svuint16_t r1 = svld1_u16(pg_16_all, src0 + ld);
          src0 += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, r1)));
        }
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          svuint16_t r2 = svld1_u16(pg_16_all, src1);
          svuint16_t r3 = svld1_u16(pg_16_all, src1 + ld);
          src1 += 2 * ld;
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r2, r3)));
        }

        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t v1 = svread_ver_za32_u32_vg4(1, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), svget4(v1, 0), svget4(v0, 1), svget4(v1, 1));
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), svget4(v1, 2), svget4(v0, 3), svget4(v1, 3));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
      }

      if (k_idx < k_rounded_2) {
        const size_t k_rem = (k > k_idx) ? (k - k_idx) : 0;
        const svbool_t pg = svwhilelt_b16_u64(0, k_rem);
        const size_t n_cols = (k_rounded_2 - k_idx) / 2;

        const uint16_t* src0 = lhs_blk + k_idx;
        const uint16_t* src1 = src0 + vl * ld;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          svuint16_t r0 = svld1_u16(pg, src0);
          svuint16_t r1 = svld1_u16(pg, src0 + ld);
          src0 += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, r1)));

          svuint16_t r2 = svld1_u16(pg, src1);
          svuint16_t r3 = svld1_u16(pg, src1 + ld);
          src1 += 2 * ld;
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r2, r3)));
        }

        size_t col = 0;
        for (; col + 4 <= n_cols; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t v1 = svread_ver_za32_u32_vg4(1, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), svget4(v1, 0), svget4(v0, 1), svget4(v1, 1));
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), svget4(v1, 2), svget4(v0, 3), svget4(v1, 3));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
        for (; col + 2 <= n_cols; col += 2) {
          svuint32x2_t v0 = svread_ver_za32_u32_vg2(0, col);
          svuint32x2_t v1 = svread_ver_za32_u32_vg2(1, col);
          svuint32x4_t comb0 = svcreate4(svget2(v0, 0), svget2(v1, 0), svget2(v0, 1), svget2(v1, 1));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
        }
        if (col < n_cols) {
          svuint32_t v00 = svread_ver_za32_u32_m(svundef_u32(), pg_32_all, 0, col);
          svuint32_t v10 = svread_ver_za32_u32_m(svundef_u32(), pg_32_all, 1, col);
          svst1_u16_x2(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(svcreate2(v00, v10)));
        }
      }
    } else {
      svzero_za();
      const size_t valid_rows_1 = valid_rows - vl;
      for (; k_idx + 4 * vl <= k; k_idx += 4 * vl) {
        const uint16_t* src0 = lhs_blk + k_idx;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint16_t* src0_1 = src0 + ld;
          svuint16_t r0_lo = svld1_u16(pg_16_all, src0);
          svuint16_t r1_lo = svld1_u16(pg_16_all, src0_1);
          svuint16_t r0_hi = svld1_u16(pg_16_all, src0 + 2 * vl);
          svuint16_t r1_hi = svld1_u16(pg_16_all, src0_1 + 2 * vl);
          src0 += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0_lo, r1_lo)));
          svwrite_hor_za32_u32_vg2(2, row_idx, svreinterpret_u32(svcreate2(r0_hi, r1_hi)));
        }
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows_1; row_idx += 2) {
          const uint16_t* src0_1 = src0 + ld;
          svuint16_t r0_lo = svld1_u16(pg_16_all, src0);
          svuint16_t r1_lo = svld1_u16(pg_16_all, src0_1);
          svuint16_t r0_hi = svld1_u16(pg_16_all, src0 + 2 * vl);
          svuint16_t r1_hi = svld1_u16(pg_16_all, src0_1 + 2 * vl);
          src0 += 2 * ld;
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r0_lo, r1_lo)));
          svwrite_hor_za32_u32_vg2(3, row_idx, svreinterpret_u32(svcreate2(r0_hi, r1_hi)));
        }
        if (row_idx < valid_rows_1) {
          svuint16_t r0_lo = svld1_u16(pg_16_all, src0);
          svuint16_t r0_hi = svld1_u16(pg_16_all, src0 + 2 * vl);
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r0_lo, vzero_u16)));
          svwrite_hor_za32_u32_vg2(3, row_idx, svreinterpret_u32(svcreate2(r0_hi, vzero_u16)));
        }

        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t v1 = svread_ver_za32_u32_vg4(1, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), svget4(v1, 0), svget4(v0, 1), svget4(v1, 1));
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), svget4(v1, 2), svget4(v0, 3), svget4(v1, 3));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
        for (size_t col = 0; col < vl; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(2, col);
          svuint32x4_t v1 = svread_ver_za32_u32_vg4(3, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), svget4(v1, 0), svget4(v0, 1), svget4(v1, 1));
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), svget4(v1, 2), svget4(v0, 3), svget4(v1, 3));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
      }

      for (; k_idx < k_rounded_2; k_idx += 2 * vl) {
        const size_t k_rem = (k > k_idx) ? (k - k_idx) : 0;
        const svbool_t pg = (k_rem >= 2 * vl) ? pg_16_all : svwhilelt_b16_u64(0, k_rem);
        const size_t n_cols = (k_rounded_2 - k_idx < 2 * vl) ? ((k_rounded_2 - k_idx) / 2) : vl;

        const uint16_t* src0 = lhs_blk + k_idx;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          svuint16_t r0 = svld1_u16(pg, src0);
          svuint16_t r1 = svld1_u16(pg, src0 + ld);
          src0 += 2 * ld;
          svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, r1)));
        }
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows_1; row_idx += 2) {
          svuint16_t r0 = svld1_u16(pg, src0);
          svuint16_t r1 = svld1_u16(pg, src0 + ld);
          src0 += 2 * ld;
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r0, r1)));
        }
        if (row_idx < valid_rows_1) {
          svuint16_t r0 = svld1_u16(pg, src0);
          svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r0, vzero_u16)));
        }

        size_t col = 0;
        for (; col + 4 <= n_cols; col += 4) {
          svuint32x4_t v0 = svread_ver_za32_u32_vg4(0, col);
          svuint32x4_t v1 = svread_ver_za32_u32_vg4(1, col);
          svuint32x4_t comb0 = svcreate4(svget4(v0, 0), svget4(v1, 0), svget4(v0, 1), svget4(v1, 1));
          svuint32x4_t comb1 = svcreate4(svget4(v0, 2), svget4(v1, 2), svget4(v0, 3), svget4(v1, 3));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb1));
          lhs_packed_inc += 8 * vl;
        }
        for (; col + 2 <= n_cols; col += 2) {
          svuint32x2_t v0 = svread_ver_za32_u32_vg2(0, col);
          svuint32x2_t v1 = svread_ver_za32_u32_vg2(1, col);
          svuint32x4_t comb0 = svcreate4(svget2(v0, 0), svget2(v1, 0), svget2(v0, 1), svget2(v1, 1));
          svst1_u16_x4(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(comb0));
          lhs_packed_inc += 8 * vl;
        }
        if (col < n_cols) {
          svuint32_t v00 = svread_ver_za32_u32_m(svundef_u32(), pg_32_all, 0, col);
          svuint32_t v10 = svread_ver_za32_u32_m(svundef_u32(), pg_32_all, 1, col);
          svst1_u16_x2(pcnt_16_all, lhs_packed_inc, svreinterpret_u16(svcreate2(v00, v10)));
          lhs_packed_inc += 4 * vl;
        }
      }
    }
  }
}

void xnn_x16_pack_lh_ukernel__sme2(
    size_t m, size_t k_f16, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const xnn_float16* XNN_RESTRICT lhs_fp16, size_t lhs_stride,
    void* XNN_RESTRICT lhs_packed_fp16)
{
  const size_t k = k_f16;
  const uint16_t* lhs = (const uint16_t*) lhs_fp16;
  uint16_t* lhs_packed = (uint16_t*) lhs_packed_fp16;
  assert(sr == 1);
  assert(mr == 1 || (mr == 2 * svcntsw() && kr == 2));
  assert(m_idx_start == 0);
  assert(lhs_stride % sizeof(uint16_t) == 0);
  assert(lhs_stride >= k * sizeof(uint16_t));
  (void) kr;
  (void) sr;
  (void) m_idx_start;

  if (mr == 1) {
    for (size_t row = 0; row < m; row++) {
      memcpy(lhs_packed + row * k,
             (const uint16_t*) ((uintptr_t) lhs + row * lhs_stride),
             k * sizeof(uint16_t));
    }
  } else {
    xnn_x16_packlh_mr2_kr2__sme2(
        m, k, lhs, lhs_stride / sizeof(uint16_t), lhs_packed);
  }
}

#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
