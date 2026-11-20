// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

size_t xnn_x8_pack_lh_size__sme(size_t m, size_t k, size_t mr_packed,
                                size_t kr, size_t sr) {
  assert(kr == 4);
  assert(sr == 1);
  assert(mr_packed == 1 || mr_packed == 2 * svcntsw());
  (void) kr;
  (void) sr;
  if (mr_packed == 1) {
    return m * k * sizeof(int8_t);
  }
  return round_up(m, mr_packed) * round_up_po2(k, 4) * sizeof(int8_t);
}

size_t xnn_x8_pack_lh_offset__sme(size_t m, size_t k, size_t mr_packed,
                                  size_t kr, size_t sr) {
  assert(kr == 4);
  assert(sr == 1);
  assert(mr_packed == 1 || mr_packed == 2 * svcntsw());
  (void) kr;
  (void) sr;
  if (mr_packed == 1) {
    return m * k * sizeof(int8_t);
  }
  return m * round_up_po2(k, 4) * sizeof(int8_t);
}

__arm_locally_streaming __arm_new("za")
static void x8_packlh_m_large_sme_impl(
    size_t m, size_t k,
    const int8_t* XNN_RESTRICT lhs, size_t lhs_stride,
    int8_t* XNN_RESTRICT lhs_packed)
{
  const size_t vl = svcntw();
  const size_t mr2 = 2 * vl;
  const size_t k_rounded_4 = ((k + 3) / 4) * 4;
  const size_t m_rounded_32 = ((m + mr2 - 1) / mr2) * mr2;
  const svbool_t pg_8_all = svptrue_b8();
  const svbool_t pg_32_all = svptrue_b32();
  const svint8_t vzero_s8 = svdup_n_s8(0);

  for (size_t mr_blk_idx = 0; mr_blk_idx < m_rounded_32; mr_blk_idx += mr2) {
    const size_t valid_rows = (mr_blk_idx + mr2 <= m) ? mr2 : (m - mr_blk_idx);
    const int8_t* lhs_blk = lhs + mr_blk_idx * lhs_stride;
    int8_t* lhs_packed_inc = lhs_packed + mr_blk_idx * k_rounded_4;
    size_t k_idx = 0;

    if (valid_rows <= vl) {
      if (valid_rows < vl) {
        svzero_za();
      }
      for (; k_idx + 8 * vl <= k; k_idx += 8 * vl) {
        const int8_t* src = lhs_blk + k_idx;
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          const int8_t* src1 = src + lhs_stride;
          svint8_t r0_lo = svld1_s8(pg_8_all, src);
          svint8_t r1_lo = svld1_s8(pg_8_all, src1);
          svint8_t r0_hi = svld1_s8(pg_8_all, src + 4 * vl);
          svint8_t r1_hi = svld1_s8(pg_8_all, src1 + 4 * vl);
          src += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0_lo));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1_lo));
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r0_hi));
          svwrite_hor_za32_u32_m(1, row + 1, pg_32_all, svreinterpret_u32_s8(r1_hi));
        }
        if (row_idx < valid_rows) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0_lo = svld1_s8(pg_8_all, src);
          svint8_t r0_hi = svld1_s8(pg_8_all, src + 4 * vl);
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0_lo));
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r0_hi));
        }

        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_s8(pg_8_all, lhs_packed_inc + 4 * vl, vzero_s8);
          lhs_packed_inc += 8 * vl;
        }
        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(1, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_s8(pg_8_all, lhs_packed_inc + 4 * vl, vzero_s8);
          lhs_packed_inc += 8 * vl;
        }
      }

      for (; k_idx + 4 * vl <= k; k_idx += 4 * vl) {
        const int8_t* src = lhs_blk + k_idx;
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0 = svld1_s8(pg_8_all, src);
          svint8_t r1 = svld1_s8(pg_8_all, src + lhs_stride);
          src += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1));
        }
        if (row_idx < valid_rows) {
          svint8_t r0 = svld1_s8(pg_8_all, src);
          svwrite_hor_za32_u32_m(0, (uint32_t) row_idx, pg_32_all, svreinterpret_u32_s8(r0));
        }

        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_s8(pg_8_all, lhs_packed_inc + 4 * vl, vzero_s8);
          lhs_packed_inc += 8 * vl;
        }
      }

      if (k_idx < k_rounded_4) {
        const size_t k_rem = (k > k_idx) ? (k - k_idx) : 0;
        const svbool_t pg = svwhilelt_b8_u64(0, k_rem);
        const size_t n_cols = (k_rounded_4 - k_idx) / 4;

        const int8_t* src = lhs_blk + k_idx;
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0 = svld1_s8(pg, src);
          svint8_t r1 = svld1_s8(pg, src + lhs_stride);
          src += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1));
        }
        if (row_idx < valid_rows) {
          svint8_t r0 = svld1_s8(pg, src);
          svwrite_hor_za32_u32_m(0, (uint32_t) row_idx, pg_32_all, svreinterpret_u32_s8(r0));
        }

        for (size_t col = 0; col < n_cols; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_s8(pg_8_all, lhs_packed_inc + 4 * vl, vzero_s8);
          lhs_packed_inc += 8 * vl;
        }
      }
    } else if (XNN_LIKELY(valid_rows == mr2)) {
      for (; k_idx + 16 * vl <= k; k_idx += 16 * vl) {
        int8_t* out_block = lhs_packed_inc;
        lhs_packed_inc += mr2 * 16 * vl;
        for (size_t p = 0; p < 2; p++) {
          const int8_t* rp = lhs_blk + p * vl * lhs_stride + k_idx;
          for (size_t r = 0; r < vl; r++) {
            const uint32_t row = (uint32_t) r;
            svld1_hor_za32(0, row, pg_32_all, rp);
            svld1_hor_za32(1, row, pg_32_all, rp + 4 * vl);
            svld1_hor_za32(2, row, pg_32_all, rp + 8 * vl);
            svld1_hor_za32(3, row, pg_32_all, rp + 12 * vl);
            rp += lhs_stride;
          }
          int8_t* out_p = out_block + p * 4 * vl;
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(0, (uint32_t) c, pg_32_all, out_p + c * 4 * mr2);
          }
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(1, (uint32_t) c, pg_32_all, out_p + (vl + c) * 4 * mr2);
          }
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(2, (uint32_t) c, pg_32_all, out_p + (2 * vl + c) * 4 * mr2);
          }
          for (size_t c = 0; c < vl; c++) {
            svst1_ver_za32(3, (uint32_t) c, pg_32_all, out_p + (3 * vl + c) * 4 * mr2);
          }
        }
      }

      for (; k_idx + 8 * vl <= k; k_idx += 8 * vl) {
        const int8_t* src0 = lhs_blk + k_idx;
        const int8_t* src1 = src0 + vl * lhs_stride;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          const int8_t* src0_1 = src0 + lhs_stride;
          svint8_t r0_lo = svld1_s8(pg_8_all, src0);
          svint8_t r1_lo = svld1_s8(pg_8_all, src0_1);
          svint8_t r0_hi = svld1_s8(pg_8_all, src0 + 4 * vl);
          svint8_t r1_hi = svld1_s8(pg_8_all, src0_1 + 4 * vl);
          src0 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0_lo));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1_lo));
          svwrite_hor_za32_u32_m(2, row, pg_32_all, svreinterpret_u32_s8(r0_hi));
          svwrite_hor_za32_u32_m(2, row + 1, pg_32_all, svreinterpret_u32_s8(r1_hi));
        }
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          const int8_t* src1_1 = src1 + lhs_stride;
          svint8_t r2_lo = svld1_s8(pg_8_all, src1);
          svint8_t r3_lo = svld1_s8(pg_8_all, src1_1);
          svint8_t r2_hi = svld1_s8(pg_8_all, src1 + 4 * vl);
          svint8_t r3_hi = svld1_s8(pg_8_all, src1_1 + 4 * vl);
          src1 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r2_lo));
          svwrite_hor_za32_u32_m(1, row + 1, pg_32_all, svreinterpret_u32_s8(r3_lo));
          svwrite_hor_za32_u32_m(3, row, pg_32_all, svreinterpret_u32_s8(r2_hi));
          svwrite_hor_za32_u32_m(3, row + 1, pg_32_all, svreinterpret_u32_s8(r3_hi));
        }

        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_ver_za32(1, (uint32_t) col, pg_32_all, lhs_packed_inc + 4 * vl);
          lhs_packed_inc += 8 * vl;
        }
        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(2, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_ver_za32(3, (uint32_t) col, pg_32_all, lhs_packed_inc + 4 * vl);
          lhs_packed_inc += 8 * vl;
        }
      }

      for (; k_idx + 4 * vl <= k; k_idx += 4 * vl) {
        const int8_t* src0 = lhs_blk + k_idx;
        const int8_t* src1 = src0 + vl * lhs_stride;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0 = svld1_s8(pg_8_all, src0);
          svint8_t r1 = svld1_s8(pg_8_all, src0 + lhs_stride);
          src0 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1));
        }
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r2 = svld1_s8(pg_8_all, src1);
          svint8_t r3 = svld1_s8(pg_8_all, src1 + lhs_stride);
          src1 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r2));
          svwrite_hor_za32_u32_m(1, row + 1, pg_32_all, svreinterpret_u32_s8(r3));
        }

        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_ver_za32(1, (uint32_t) col, pg_32_all, lhs_packed_inc + 4 * vl);
          lhs_packed_inc += 8 * vl;
        }
      }

      if (k_idx < k_rounded_4) {
        const size_t k_rem = (k > k_idx) ? (k - k_idx) : 0;
        const svbool_t pg = svwhilelt_b8_u64(0, k_rem);
        const size_t n_cols = (k_rounded_4 - k_idx) / 4;

        const int8_t* src0 = lhs_blk + k_idx;
        const int8_t* src1 = src0 + vl * lhs_stride;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0 = svld1_s8(pg, src0);
          svint8_t r1 = svld1_s8(pg, src0 + lhs_stride);
          src0 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1));

          svint8_t r2 = svld1_s8(pg, src1);
          svint8_t r3 = svld1_s8(pg, src1 + lhs_stride);
          src1 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r2));
          svwrite_hor_za32_u32_m(1, row + 1, pg_32_all, svreinterpret_u32_s8(r3));
        }

        for (size_t col = 0; col < n_cols; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_ver_za32(1, (uint32_t) col, pg_32_all, lhs_packed_inc + 4 * vl);
          lhs_packed_inc += 8 * vl;
        }
      }
    } else {
      svzero_za();
      const size_t valid_rows_1 = valid_rows - vl;
      for (; k_idx + 8 * vl <= k; k_idx += 8 * vl) {
        const int8_t* src0 = lhs_blk + k_idx;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          const int8_t* src0_1 = src0 + lhs_stride;
          svint8_t r0_lo = svld1_s8(pg_8_all, src0);
          svint8_t r1_lo = svld1_s8(pg_8_all, src0_1);
          svint8_t r0_hi = svld1_s8(pg_8_all, src0 + 4 * vl);
          svint8_t r1_hi = svld1_s8(pg_8_all, src0_1 + 4 * vl);
          src0 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0_lo));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1_lo));
          svwrite_hor_za32_u32_m(2, row, pg_32_all, svreinterpret_u32_s8(r0_hi));
          svwrite_hor_za32_u32_m(2, row + 1, pg_32_all, svreinterpret_u32_s8(r1_hi));
        }
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows_1; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          const int8_t* src0_1 = src0 + lhs_stride;
          svint8_t r0_lo = svld1_s8(pg_8_all, src0);
          svint8_t r1_lo = svld1_s8(pg_8_all, src0_1);
          svint8_t r0_hi = svld1_s8(pg_8_all, src0 + 4 * vl);
          svint8_t r1_hi = svld1_s8(pg_8_all, src0_1 + 4 * vl);
          src0 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r0_lo));
          svwrite_hor_za32_u32_m(1, row + 1, pg_32_all, svreinterpret_u32_s8(r1_lo));
          svwrite_hor_za32_u32_m(3, row, pg_32_all, svreinterpret_u32_s8(r0_hi));
          svwrite_hor_za32_u32_m(3, row + 1, pg_32_all, svreinterpret_u32_s8(r1_hi));
        }
        if (row_idx < valid_rows_1) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0_lo = svld1_s8(pg_8_all, src0);
          svint8_t r0_hi = svld1_s8(pg_8_all, src0 + 4 * vl);
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r0_lo));
          svwrite_hor_za32_u32_m(3, row, pg_32_all, svreinterpret_u32_s8(r0_hi));
        }

        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_ver_za32(1, (uint32_t) col, pg_32_all, lhs_packed_inc + 4 * vl);
          lhs_packed_inc += 8 * vl;
        }
        for (size_t col = 0; col < vl; col++) {
          svst1_ver_za32(2, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_ver_za32(3, (uint32_t) col, pg_32_all, lhs_packed_inc + 4 * vl);
          lhs_packed_inc += 8 * vl;
        }
      }

      for (; k_idx < k_rounded_4; k_idx += 4 * vl) {
        const size_t k_rem = (k > k_idx) ? (k - k_idx) : 0;
        const svbool_t pg = (k_rem >= 4 * vl) ? pg_8_all : svwhilelt_b8_u64(0, k_rem);
        const size_t n_cols = (k_rounded_4 - k_idx < 4 * vl) ? ((k_rounded_4 - k_idx) / 4) : vl;

        const int8_t* src0 = lhs_blk + k_idx;
        for (size_t row_idx = 0; row_idx < vl; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0 = svld1_s8(pg, src0);
          svint8_t r1 = svld1_s8(pg, src0 + lhs_stride);
          src0 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(0, row, pg_32_all, svreinterpret_u32_s8(r0));
          svwrite_hor_za32_u32_m(0, row + 1, pg_32_all, svreinterpret_u32_s8(r1));
        }
        size_t row_idx = 0;
        for (; row_idx + 2 <= valid_rows_1; row_idx += 2) {
          const uint32_t row = (uint32_t) row_idx;
          svint8_t r0 = svld1_s8(pg, src0);
          svint8_t r1 = svld1_s8(pg, src0 + lhs_stride);
          src0 += 2 * lhs_stride;
          svwrite_hor_za32_u32_m(1, row, pg_32_all, svreinterpret_u32_s8(r0));
          svwrite_hor_za32_u32_m(1, row + 1, pg_32_all, svreinterpret_u32_s8(r1));
        }
        if (row_idx < valid_rows_1) {
          svint8_t r0 = svld1_s8(pg, src0);
          svwrite_hor_za32_u32_m(1, (uint32_t) row_idx, pg_32_all, svreinterpret_u32_s8(r0));
        }

        for (size_t col = 0; col < n_cols; col++) {
          svst1_ver_za32(0, (uint32_t) col, pg_32_all, lhs_packed_inc);
          svst1_ver_za32(1, (uint32_t) col, pg_32_all, lhs_packed_inc + 4 * vl);
          lhs_packed_inc += 8 * vl;
        }
      }
    }
  }
}

void xnn_x8_pack_lh_ukernel__sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const int8_t* XNN_RESTRICT lhs, size_t lhs_stride,
    void* XNN_RESTRICT lhs_packed)
{
  assert(kr == 4);
  assert(sr == 1);
  assert(mr == 1 || mr == 2 * svcntsw());
  assert(m_idx_start == 0);
  (void) kr;
  (void) sr;
  (void) m_idx_start;

  if XNN_UNLIKELY(m == 0 || k == 0) {
    return;
  }
  assert(lhs_stride >= k);

  if (mr == 1) {
    int8_t* out = (int8_t*) lhs_packed;
    for (size_t row = 0; row < m; row++) {
      memcpy(out + row * k, lhs + row * lhs_stride, k * sizeof(int8_t));
    }
  } else {
    x8_packlh_m_large_sme_impl(m, k, lhs, lhs_stride, (int8_t*) lhs_packed);
  }
}

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
