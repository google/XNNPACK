// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#elif XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#endif  // XNN_ENABLE_KLEIDIAI

// This function just wraps KleidiAI's `kai_run_lhs_pack_x8p2vlx4_x8_sme`, but
// with a name that is recognized by our tooling.
void xnn_x8_pack_lh_ukernel__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                            size_t mr_packed, size_t kr,
                                            size_t sr, const void** restrict a,
                                            size_t a_offset, const void* zero,
                                            void* lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m, ks, kc, a, a_offset, zero,
                                            lhs_packed);
#elif XNN_ENABLE_ARM_SME_ACLE
  assert(kr == 4);
  (void)sr;
  const size_t m_step = mr_packed ? mr_packed : (2 * svcntsb() / 4);
  const size_t k_padded = round_up_po2(kc, kr);
  int8_t* dst_base = (int8_t*)lhs_packed;
  const uint32x4_t vzero = vdupq_n_u32(0);
  for (size_t i_m = 0; i_m < m; i_m += m_step) {
    const size_t height = math_min_s32((int32_t)(m - i_m), (int32_t)m_step);
    for (size_t i_ks = 0; i_ks < ks; ++i_ks) {
      int8_t* dst = dst_base;
      uint32_t* dst_u32 = (uint32_t*)dst;
      dst_base += m_step * k_padded;
      size_t y = 0;
      for (; y + 4 <= height; y += 4) {
        const void* p0 = a[i_m * ks + i_ks * m_step + y + 0];
        const void* p1 = a[i_m * ks + i_ks * m_step + y + 1];
        const void* p2 = a[i_m * ks + i_ks * m_step + y + 2];
        const void* p3 = a[i_m * ks + i_ks * m_step + y + 3];
        const int8_t* r0 =
            (const int8_t*)((p0 != zero) ? ((uintptr_t)p0 + a_offset)
                                         : (uintptr_t)p0);
        const int8_t* r1 =
            (const int8_t*)((p1 != zero) ? ((uintptr_t)p1 + a_offset)
                                         : (uintptr_t)p1);
        const int8_t* r2 =
            (const int8_t*)((p2 != zero) ? ((uintptr_t)p2 + a_offset)
                                         : (uintptr_t)p2);
        const int8_t* r3 =
            (const int8_t*)((p3 != zero) ? ((uintptr_t)p3 + a_offset)
                                         : (uintptr_t)p3);
        size_t x = 0;
        for (; x + 16 <= kc; x += 16) {
          const uint32x4_t v0 = vreinterpretq_u32_s8(vld1q_s8(r0 + x));
          const uint32x4_t v1 = vreinterpretq_u32_s8(vld1q_s8(r1 + x));
          const uint32x4_t v2 = vreinterpretq_u32_s8(vld1q_s8(r2 + x));
          const uint32x4_t v3 = vreinterpretq_u32_s8(vld1q_s8(r3 + x));
          const uint32x4_t t0 = vtrn1q_u32(v0, v1);
          const uint32x4_t t1 = vtrn2q_u32(v0, v1);
          const uint32x4_t t2 = vtrn1q_u32(v2, v3);
          const uint32x4_t t3 = vtrn2q_u32(v2, v3);
          const uint32x4_t c0 = vreinterpretq_u32_u64(
              vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
          const uint32x4_t c1 = vreinterpretq_u32_u64(
              vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
          const uint32x4_t c2 = vreinterpretq_u32_u64(
              vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
          const uint32x4_t c3 = vreinterpretq_u32_u64(
              vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
          uint32_t* d = dst_u32 + (x / 4) * m_step + y;
          vst1q_u32(d + 0 * m_step, c0);
          vst1q_u32(d + 1 * m_step, c1);
          vst1q_u32(d + 2 * m_step, c2);
          vst1q_u32(d + 3 * m_step, c3);
        }
        for (; x < k_padded; x += kr) {
          int8_t* d = dst + x * m_step + y * 4;
          d[0] = (x + 0 < kc) ? r0[x + 0] : 0;
          d[1] = (x + 1 < kc) ? r0[x + 1] : 0;
          d[2] = (x + 2 < kc) ? r0[x + 2] : 0;
          d[3] = (x + 3 < kc) ? r0[x + 3] : 0;
          d[4] = (x + 0 < kc) ? r1[x + 0] : 0;
          d[5] = (x + 1 < kc) ? r1[x + 1] : 0;
          d[6] = (x + 2 < kc) ? r1[x + 2] : 0;
          d[7] = (x + 3 < kc) ? r1[x + 3] : 0;
          d[8] = (x + 0 < kc) ? r2[x + 0] : 0;
          d[9] = (x + 1 < kc) ? r2[x + 1] : 0;
          d[10] = (x + 2 < kc) ? r2[x + 2] : 0;
          d[11] = (x + 3 < kc) ? r2[x + 3] : 0;
          d[12] = (x + 0 < kc) ? r3[x + 0] : 0;
          d[13] = (x + 1 < kc) ? r3[x + 1] : 0;
          d[14] = (x + 2 < kc) ? r3[x + 2] : 0;
          d[15] = (x + 3 < kc) ? r3[x + 3] : 0;
        }
      }
      for (; y < height; ++y) {
        const void* ptr = a[i_m * ks + i_ks * m_step + y];
        const int8_t* row =
            (const int8_t*)((ptr != zero) ? ((uintptr_t)ptr + a_offset)
                                          : (uintptr_t)ptr);
        for (size_t x = 0; x < k_padded; x += kr) {
          int8_t* d = dst + x * m_step + y * 4;
          d[0] = (x + 0 < kc) ? row[x + 0] : 0;
          d[1] = (x + 1 < kc) ? row[x + 1] : 0;
          d[2] = (x + 2 < kc) ? row[x + 2] : 0;
          d[3] = (x + 3 < kc) ? row[x + 3] : 0;
        }
      }
      for (; y + 4 <= m_step; y += 4) {
        for (size_t x = 0; x < k_padded / 4; ++x) {
          vst1q_u32(dst_u32 + x * m_step + y, vzero);
        }
      }
      for (; y < m_step; ++y) {
        for (size_t x = 0; x < k_padded / 4; ++x) {
          dst_u32[x * m_step + y] = 0;
        }
      }
    }
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_size__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                           size_t mr_packed, size_t kr,
                                           size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  return kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m, ks, kc);
#elif XNN_ENABLE_ARM_SME_ACLE
  assert(kr == 4);
  (void)sr;
  const size_t m_step = mr_packed ? mr_packed : (2 * svcntsb() / 4);
  return round_up(m, m_step) * ks * round_up_po2(kc, kr) * sizeof(int8_t);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_offset__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                             size_t mr_packed, size_t kr,
                                             size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  return kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m, ks, kc);
#elif XNN_ENABLE_ARM_SME_ACLE
  assert(kr == 4);
  (void)sr;
  const size_t m_step = mr_packed ? mr_packed : (2 * svcntsb() / 4);
  return round_up(m, m_step) * ks * round_up_po2(kc, kr) * sizeof(int8_t);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
