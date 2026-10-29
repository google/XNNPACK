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
  (void)sr;
  if (mr_packed == 1) {
    return m * k * sizeof(xnn_float16);
  }
  return round_up(m, mr_packed) * round_up(k, kr) * sizeof(xnn_float16);
}

size_t xnn_x16_pack_lh_offset__sme2(size_t m, size_t k, size_t mr_packed,
                                    size_t kr, size_t sr) {
  assert(sr == 1);
  assert(mr_packed == 1 || (mr_packed == 2 * svcntsw() && kr == 2));
  (void)sr;
  if (mr_packed == 1) {
    return m * k * sizeof(xnn_float16);
  }
  return m * round_up(k, kr) * sizeof(xnn_float16);
}

__arm_locally_streaming __arm_new("za") static void xnn_x16_packlh_sme_2vl(
    size_t m, size_t k, const uint8_t* lhs_ptr, size_t lhs_stride,
    uint16_t* dst_base) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t k_padded = (k + 1) & ~(size_t)1;
  const svbool_t pg32 = svptrue_b32();

  for (size_t i_m = 0; i_m < m; i_m += m_step) {
    const size_t valid_rows = m - i_m < m_step ? m - i_m : m_step;
    uint32_t* dst_blk = (uint32_t*)(dst_base + i_m * k_padded);
    size_t k_idx = 0;

    if (XNN_LIKELY(valid_rows == m_step)) {
      for (; k_idx + 8 * vl <= k; k_idx += 8 * vl) {
        const size_t kp = k_idx / 2;
        for (size_t p = 0; p < 2; ++p) {
          const uint8_t* rp =
              lhs_ptr + (i_m + p * vl) * lhs_stride + k_idx * sizeof(uint16_t);
          for (size_t r = 0; r < vl; ++r) {
            const uint32_t* row = (const uint32_t*)rp;
            svld1_hor_za32(0, r, pg32, row + 0 * vl);
            svld1_hor_za32(1, r, pg32, row + 1 * vl);
            svld1_hor_za32(2, r, pg32, row + 2 * vl);
            svld1_hor_za32(3, r, pg32, row + 3 * vl);
            rp += lhs_stride;
          }
          uint32_t* out_p = dst_blk + kp * m_step + p * vl;
          for (size_t c = 0; c < vl; ++c) {
            svst1_ver_za32(0, c, pg32, out_p + (0 * vl + c) * m_step);
            svst1_ver_za32(1, c, pg32, out_p + (1 * vl + c) * m_step);
            svst1_ver_za32(2, c, pg32, out_p + (2 * vl + c) * m_step);
            svst1_ver_za32(3, c, pg32, out_p + (3 * vl + c) * m_step);
          }
        }
      }
    } else {
      svzero_za();
    }

    const size_t rows0 = valid_rows < vl ? valid_rows : vl;
    const size_t rows1 = valid_rows > vl ? valid_rows - vl : 0;
    for (; k_idx < k_padded; k_idx += 4 * vl) {
      const size_t k_rem =
          k > k_idx ? (k - k_idx < 4 * vl ? k - k_idx : 4 * vl) : 0;
      const size_t kp_rem =
          (k_padded - k_idx < 4 * vl ? k_padded - k_idx : 4 * vl) / 2;
      const svbool_t pg16_lo = svwhilelt_b16_u64(0, k_rem);
      const svbool_t pg16_hi = svwhilelt_b16_u64(2 * vl, k_rem);
      const uint8_t* rp0 =
          lhs_ptr + i_m * lhs_stride + k_idx * sizeof(uint16_t);
      for (size_t r = 0; r < rows0; ++r) {
        const uint16_t* row = (const uint16_t*)rp0;
        svwrite_hor_za32_u32_m(0, r, pg32,
                               svreinterpret_u32_u16(svld1_u16(pg16_lo, row)));
        svwrite_hor_za32_u32_m(
            1, r, pg32,
            svreinterpret_u32_u16(svld1_u16(pg16_hi, row + 2 * vl)));
        rp0 += lhs_stride;
      }
      const uint8_t* rp1 =
          lhs_ptr + (i_m + vl) * lhs_stride + k_idx * sizeof(uint16_t);
      for (size_t r = 0; r < rows1; ++r) {
        const uint16_t* row = (const uint16_t*)rp1;
        svwrite_hor_za32_u32_m(2, r, pg32,
                               svreinterpret_u32_u16(svld1_u16(pg16_lo, row)));
        svwrite_hor_za32_u32_m(
            3, r, pg32,
            svreinterpret_u32_u16(svld1_u16(pg16_hi, row + 2 * vl)));
        rp1 += lhs_stride;
      }
      const size_t len0 = kp_rem < vl ? kp_rem : vl;
      const size_t len1 = kp_rem > vl ? kp_rem - vl : 0;
      uint32_t* out_ptr = dst_blk + (k_idx / 2) * m_step;
      for (size_t c = 0; c < len0; ++c) {
        svst1_ver_za32(0, c, pg32, out_ptr);
        svst1_ver_za32(2, c, pg32, out_ptr + vl);
        out_ptr += m_step;
      }
      for (size_t c = 0; c < len1; ++c) {
        svst1_ver_za32(1, c, pg32, out_ptr);
        svst1_ver_za32(3, c, pg32, out_ptr + vl);
        out_ptr += m_step;
      }
    }
  }
}

void xnn_x16_pack_lh_ukernel__sme2(size_t m, size_t k_f16, size_t mr, size_t kr,
                                   size_t sr, size_t m_idx_start,
                                   const xnn_float16* XNN_RESTRICT lhs_fp16,
                                   size_t lhs_stride,
                                   void* XNN_RESTRICT lhs_packed_fp16) {
  const size_t k = k_f16;
  const uint16_t* lhs = (const uint16_t*)lhs_fp16;
  uint16_t* lhs_packed = (uint16_t*)lhs_packed_fp16;
  assert(sr == 1);
  assert(mr == 1 || (mr == 2 * svcntsw() && kr == 2));
  assert(m_idx_start == 0);
  assert(lhs_stride % sizeof(uint16_t) == 0);
  assert(lhs_stride >= k * sizeof(uint16_t));
  (void)kr;
  (void)sr;
  (void)m_idx_start;

  if (mr == 1) {
    for (size_t row = 0; row < m; ++row) {
      memcpy(lhs_packed + row * k,
             (const uint16_t*)((uintptr_t)lhs + row * lhs_stride),
             k * sizeof(uint16_t));
    }
  } else {
    xnn_x16_packlh_sme_2vl(m, k, (const uint8_t*)lhs, lhs_stride, lhs_packed);
  }
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
