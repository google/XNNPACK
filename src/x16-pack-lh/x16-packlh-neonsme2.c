// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.h"
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_x16_packlh_sme_2vl(size_t m, size_t k, const uint8_t* lhs_ptr,
                       size_t lhs_stride, uint16_t* dst_base) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t k_padded = (k + 1) & ~(size_t)1;
  const svbool_t pg16 = svptrue_b16();
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
            const uint16_t* row = (const uint16_t*)rp;
            svld1_hor_za16(0, 2 * r + 0, pg16, row + 0 * 2 * vl);
            svld1_hor_za16(1, 2 * r + 0, pg16, row + 1 * 2 * vl);
            svld1_hor_za16(0, 2 * r + 1, pg16, row + 2 * 2 * vl);
            svld1_hor_za16(1, 2 * r + 1, pg16, row + 3 * 2 * vl);
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
        svld1_hor_za16(0, 2 * r + 0, pg16_lo, row);
        svld1_hor_za16(1, 2 * r + 0, pg16_hi, row + 2 * vl);
        rp0 += lhs_stride;
      }
      const uint8_t* rp1 =
          lhs_ptr + (i_m + vl) * lhs_stride + k_idx * sizeof(uint16_t);
      for (size_t r = 0; r < rows1; ++r) {
        const uint16_t* row = (const uint16_t*)rp1;
        svld1_hor_za16(0, 2 * r + 1, pg16_lo, row);
        svld1_hor_za16(1, 2 * r + 1, pg16_hi, row + 2 * vl);
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
#endif  // XNN_ENABLE_KLEIDIAI

// This function just wraps KleidiAI's `kai_run_lhs_pack_x16p2vlx2_x16_sme`, but
// with a name that is recognized by our tooling.
void xnn_x16_pack_lh_ukernel__neonsme2(size_t m, size_t k, size_t mr_packed,
                                       size_t kr, size_t sr, size_t m_idx_start,
                                       const xnn_float16* XNN_RESTRICT lhs,
                                       size_t lhs_stride,
                                       void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    memcpy(lhs_packed, lhs, sizeof(xnn_float16) * k);
  } else {
    kai_run_lhs_pack_x16p2vlx2_x16_sme(m, k, mr_packed, kr, sr, m_idx_start,
                                       lhs, lhs_stride, lhs_packed);
  }
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  assert(m_idx_start == 0);
  assert(kr == 2);
  assert(sr == 1);
  (void)m_idx_start;
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    uint16_t* out = (uint16_t*)lhs_packed;
    for (size_t row = 0; row < m; ++row) {
      memcpy(out + row * k, (const uint8_t*)lhs + row * lhs_stride,
             k * sizeof(xnn_float16));
    }
    return;
  }
  assert(mr_packed == 2 * svcntsw());
  (void)mr_packed;
  xnn_x16_packlh_sme_2vl(m, k, (const uint8_t*)lhs, lhs_stride,
                         (uint16_t*)lhs_packed);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x16_pack_lh_size__neonsme2(size_t m, size_t k, size_t mr_packed,
                                      size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  } else {
    return kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme(m, k, mr_packed,
                                                              kr, sr);
  }
#elif XNN_ENABLE_ARM_SME_ACLE
  (void)sr;
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  }
  return round_up(m, mr_packed) * round_up_po2(k, kr) * sizeof(xnn_float16);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x16_pack_lh_offset__neonsme2(size_t m, size_t k, size_t mr_packed,
                                        size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  } else {
    return kai_get_lhs_packed_offset_lhs_pack_x16p2vlx2_x16_sme(m, k, mr_packed,
                                                                kr, sr);
  }
#elif XNN_ENABLE_ARM_SME_ACLE
  (void)sr;
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  }
  return m * round_up_po2(k, kr) * sizeof(xnn_float16);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
