// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_x8_packlh_igemm_sme_2vl(size_t m, size_t kc, size_t ks,
                            const void** XNN_RESTRICT a, size_t a_offset,
                            const void* zero, int8_t* dst_base) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t k_padded = (kc + 3) & ~(size_t)3;
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();

  for (size_t i_m = 0; i_m < m; i_m += m_step) {
    const size_t valid_rows = m - i_m < m_step ? m - i_m : m_step;
    const size_t rows0 = valid_rows < vl ? valid_rows : vl;
    const size_t rows1 = valid_rows > vl ? valid_rows - vl : 0;
    if (XNN_UNLIKELY(valid_rows < m_step)) {
      svzero_za();
    }

    for (size_t i_ks = 0; i_ks < ks; ++i_ks) {
      // Resolve indirection inline on GP registers to avoid SSVE-to-GP stalls.
      const void** a_ks = a + i_m * ks + i_ks * m_step;
      uint32_t* dst_blk = (uint32_t*)dst_base;
      dst_base += m_step * k_padded;

      size_t k_idx = 0;
      if (XNN_LIKELY(valid_rows == m_step)) {
        for (; k_idx + 16 * vl <= kc; k_idx += 16 * vl) {
          const size_t kp = k_idx / 4;
          for (size_t p = 0; p < 2; ++p) {
            for (size_t r = 0; r < vl; ++r) {
              const void* ptr = a_ks[p * vl + r];
              const int8_t* row =
                  (const int8_t*)((ptr != zero) ? ((uintptr_t)ptr + a_offset)
                                                : (uintptr_t)ptr) +
                  k_idx;
              svld1_hor_za8(0, 4 * r + 0, pg8, row + 0 * 4 * vl);
              svld1_hor_za8(0, 4 * r + 1, pg8, row + 1 * 4 * vl);
              svld1_hor_za8(0, 4 * r + 2, pg8, row + 2 * 4 * vl);
              svld1_hor_za8(0, 4 * r + 3, pg8, row + 3 * 4 * vl);
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
      }

      for (; k_idx < k_padded; k_idx += 8 * vl) {
        const size_t k_rem =
            kc > k_idx ? (kc - k_idx < 8 * vl ? kc - k_idx : 8 * vl) : 0;
        const size_t kp_rem =
            (k_padded - k_idx < 8 * vl ? k_padded - k_idx : 8 * vl) / 4;
        const svbool_t pg8_lo = svwhilelt_b8_u64(0, k_rem);
        const svbool_t pg8_hi = svwhilelt_b8_u64(4 * vl, k_rem);
        for (size_t r = 0; r < rows0; ++r) {
          const void* ptr = a_ks[r];
          const int8_t* row =
              (const int8_t*)((ptr != zero) ? ((uintptr_t)ptr + a_offset)
                                            : (uintptr_t)ptr) +
              k_idx;
          svld1_hor_za8(0, 4 * r + 0, pg8_lo, row);
          svld1_hor_za8(0, 4 * r + 1, pg8_hi, row + 4 * vl);
        }
        for (size_t r = 0; r < rows1; ++r) {
          const void* ptr = a_ks[vl + r];
          const int8_t* row =
              (const int8_t*)((ptr != zero) ? ((uintptr_t)ptr + a_offset)
                                            : (uintptr_t)ptr) +
              k_idx;
          svld1_hor_za8(0, 4 * r + 2, pg8_lo, row);
          svld1_hor_za8(0, 4 * r + 3, pg8_hi, row + 4 * vl);
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
}
#endif  // XNN_ENABLE_KLEIDIAI

// This function just wraps KleidiAI's `kai_run_lhs_pack_x8p2vlx4_x8_sme`, but
// with a name that is recognized by our tooling.
void xnn_x8_pack_lh_ukernel__igemm_neonsme(size_t m, size_t kc, size_t ks,
                                           size_t mr_packed, size_t kr,
                                           size_t sr, const void** restrict a,
                                           size_t a_offset, const void* zero,
                                           void* lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m, ks, kc, a, a_offset, zero,
                                            lhs_packed);
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  assert(kr == 4);
  assert(mr_packed == 0 || mr_packed == 2 * svcntsw());
  (void)mr_packed;
  (void)kr;
  (void)sr;
  xnn_x8_packlh_igemm_sme_2vl(m, kc, ks, a, a_offset, zero,
                              (int8_t*)lhs_packed);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_size__igemm_neonsme(size_t m, size_t kc, size_t ks,
                                          size_t mr_packed, size_t kr,
                                          size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  return kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m, ks, kc);
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  assert(kr == 4);
  (void)sr;
  const size_t m_step = mr_packed ? mr_packed : (2 * svcntsb() / 4);
  return round_up(m, m_step) * ks * round_up_po2(kc, kr) * sizeof(int8_t);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_offset__igemm_neonsme(size_t m, size_t kc, size_t ks,
                                            size_t mr_packed, size_t kr,
                                            size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  return kai_get_lhs_packed_offset_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m, ks, kc);
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  assert(kr == 4);
  (void)mr_packed;
  (void)sr;
  return m * ks * round_up_po2(kc, kr) * sizeof(int8_t);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
