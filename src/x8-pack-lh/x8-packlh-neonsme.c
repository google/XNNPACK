// Copyright 2025 Google LLC
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
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_x8_packlh_sme_2vl(size_t m, size_t k, const int8_t* lhs_ptr,
                      size_t lhs_stride, int8_t* dst_base) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t k_padded = (k + 3) & ~(size_t)3;
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();

  for (size_t i_m = 0; i_m < m; i_m += m_step) {
    const size_t valid_rows = m - i_m < m_step ? m - i_m : m_step;
    uint32_t* dst_blk = (uint32_t*)(dst_base + i_m * k_padded);
    size_t k_idx = 0;

    if (XNN_LIKELY(valid_rows == m_step)) {
      for (; k_idx + 16 * vl <= k; k_idx += 16 * vl) {
        const size_t kp = k_idx / 4;
        for (size_t p = 0; p < 2; ++p) {
          const int8_t* rp = lhs_ptr + (i_m + p * vl) * lhs_stride + k_idx;
          for (size_t r = 0; r < vl; ++r) {
            svld1_hor_za8(0, 4 * r + 0, pg8, rp + 0 * 4 * vl);
            svld1_hor_za8(0, 4 * r + 1, pg8, rp + 1 * 4 * vl);
            svld1_hor_za8(0, 4 * r + 2, pg8, rp + 2 * 4 * vl);
            svld1_hor_za8(0, 4 * r + 3, pg8, rp + 3 * 4 * vl);
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
    for (; k_idx < k_padded; k_idx += 8 * vl) {
      const size_t k_rem =
          k > k_idx ? (k - k_idx < 8 * vl ? k - k_idx : 8 * vl) : 0;
      const size_t kp_rem =
          (k_padded - k_idx < 8 * vl ? k_padded - k_idx : 8 * vl) / 4;
      const svbool_t pg8_lo = svwhilelt_b8_u64(0, k_rem);
      const svbool_t pg8_hi = svwhilelt_b8_u64(4 * vl, k_rem);
      const int8_t* rp0 = lhs_ptr + i_m * lhs_stride + k_idx;
      for (size_t r = 0; r < rows0; ++r) {
        svld1_hor_za8(0, 4 * r + 0, pg8_lo, rp0);
        svld1_hor_za8(0, 4 * r + 1, pg8_hi, rp0 + 4 * vl);
        rp0 += lhs_stride;
      }
      const int8_t* rp1 = lhs_ptr + (i_m + vl) * lhs_stride + k_idx;
      for (size_t r = 0; r < rows1; ++r) {
        svld1_hor_za8(0, 4 * r + 2, pg8_lo, rp1);
        svld1_hor_za8(0, 4 * r + 3, pg8_hi, rp1 + 4 * vl);
        rp1 += lhs_stride;
      }
      const size_t len0 = kp_rem < vl ? kp_rem : vl;
      const size_t len1 = kp_rem > vl ? kp_rem - vl : 0;
      uint32_t* out_ptr = dst_blk + (k_idx / 4) * m_step;
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

// This function just wraps KleidiAI's `kai_run_lhs_pack_x8p2vlx4_x8_sme`, but
// with a name that is recognized by our tooling.
void xnn_x8_pack_lh_ukernel__neonsme(size_t m, size_t k, size_t mr_packed,
                                     size_t kr, size_t sr, size_t m_idx_start,
                                     const int8_t* XNN_RESTRICT lhs,
                                     size_t lhs_stride,
                                     void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    for (size_t i = 0; i < m; ++i) {
      memcpy((int8_t*)lhs_packed + i * k, (const int8_t*)lhs + i * lhs_stride,
             sizeof(int8_t) * k);
    }
  } else {
    kai_run_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed, kr, sr, m_idx_start, lhs,
                                     lhs_stride, lhs_packed);
  }
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  assert(m_idx_start == 0);
  assert(kr == 4);
  assert(sr == 1);
  (void)m_idx_start;
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    for (size_t i = 0; i < m; ++i) {
      memcpy((int8_t*)lhs_packed + i * k, lhs + i * lhs_stride,
             sizeof(int8_t) * k);
    }
    return;
  }
  assert(mr_packed == 2 * svcntsw());
  (void)mr_packed;
  xnn_x8_packlh_sme_2vl(m, k, lhs, lhs_stride, (int8_t*)lhs_packed);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_size__neonsme(size_t m, size_t k, size_t mr_packed,
                                    size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    return kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed, kr,
                                                            sr);
  }
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  (void)sr;
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  }
  return round_up(m, mr_packed) * round_up_po2(k, kr) * sizeof(int8_t);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_offset__neonsme(size_t m, size_t k, size_t mr_packed,
                                      size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    return kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed,
                                                              kr, sr);
  }
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  (void)sr;
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  }
  return m * round_up_po2(k, kr) * sizeof(int8_t);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
