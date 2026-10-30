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
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.h"
#elif XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#endif  // XNN_ENABLE_KLEIDIAI

// This function just wraps KleidiAI's
// `kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme`, but with a name that is
// recognized by our tooling.
void xnn_x32_pack_lh_ukernel__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                             size_t mr_packed, size_t kr,
                                             size_t sr, const void** restrict a,
                                             size_t a_offset, const void* zero,
                                             void* lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 1);
  kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme(m, ks, kc, a, a_offset, zero,
                                              lhs_packed);
#elif XNN_ENABLE_ARM_SME_ACLE
  assert(kr == 1);
  (void)sr;
  const size_t m_step = mr_packed ? mr_packed : 2 * svcntsw();
  float* dst_base = (float*)lhs_packed;
  const uint32x4_t vzero = vdupq_n_u32(0);
  for (size_t i_m = 0; i_m < m; i_m += m_step) {
    const size_t height = math_min_s32((int32_t)(m - i_m), (int32_t)m_step);
    for (size_t i_ks = 0; i_ks < ks; ++i_ks) {
      float* dst = dst_base;
      uint32_t* dst_u32 = (uint32_t*)dst;
      dst_base += m_step * kc;
      size_t y = 0;
      for (; y + 4 <= height; y += 4) {
        const void* p0 = a[i_m * ks + i_ks * m_step + y + 0];
        const void* p1 = a[i_m * ks + i_ks * m_step + y + 1];
        const void* p2 = a[i_m * ks + i_ks * m_step + y + 2];
        const void* p3 = a[i_m * ks + i_ks * m_step + y + 3];
        const uint32_t* r0 =
            (const uint32_t*)((p0 != zero) ? ((uintptr_t)p0 + a_offset)
                                           : (uintptr_t)p0);
        const uint32_t* r1 =
            (const uint32_t*)((p1 != zero) ? ((uintptr_t)p1 + a_offset)
                                           : (uintptr_t)p1);
        const uint32_t* r2 =
            (const uint32_t*)((p2 != zero) ? ((uintptr_t)p2 + a_offset)
                                           : (uintptr_t)p2);
        const uint32_t* r3 =
            (const uint32_t*)((p3 != zero) ? ((uintptr_t)p3 + a_offset)
                                           : (uintptr_t)p3);
        size_t x = 0;
        for (; x + 4 <= kc; x += 4) {
          const uint32x4_t v0 = vld1q_u32(r0 + x);
          const uint32x4_t v1 = vld1q_u32(r1 + x);
          const uint32x4_t v2 = vld1q_u32(r2 + x);
          const uint32x4_t v3 = vld1q_u32(r3 + x);
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
          uint32_t* d = dst_u32 + x * m_step + y;
          vst1q_u32(d + 0 * m_step, c0);
          vst1q_u32(d + 1 * m_step, c1);
          vst1q_u32(d + 2 * m_step, c2);
          vst1q_u32(d + 3 * m_step, c3);
        }
        for (; x < kc; ++x) {
          uint32_t* d = dst_u32 + x * m_step + y;
          d[0] = r0[x];
          d[1] = r1[x];
          d[2] = r2[x];
          d[3] = r3[x];
        }
      }
      for (; y < height; ++y) {
        const void* ptr = a[i_m * ks + i_ks * m_step + y];
        const float* row =
            (const float*)((ptr != zero) ? ((uintptr_t)ptr + a_offset)
                                         : (uintptr_t)ptr);
        for (size_t x = 0; x < kc; ++x) {
          dst[x * m_step + y] = row[x];
        }
      }
      for (; y + 4 <= m_step; y += 4) {
        for (size_t x = 0; x < kc; ++x) {
          vst1q_u32(dst_u32 + x * m_step + y, vzero);
        }
      }
      for (; y < m_step; ++y) {
        for (size_t x = 0; x < kc; ++x) {
          dst[x * m_step + y] = 0.0f;
        }
      }
    }
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x32_pack_lh_size__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                            size_t mr_packed, size_t kr,
                                            size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 1);
  return kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme(m, ks, kc);
#elif XNN_ENABLE_ARM_SME_ACLE
  assert(kr == 1);
  (void)sr;
  const size_t m_step = mr_packed ? mr_packed : 2 * svcntsw();
  return round_up(m, m_step) * ks * kc * sizeof(float);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x32_pack_lh_offset__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                              size_t mr_packed, size_t kr,
                                              size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 1);
  return kai_get_lhs_packed_offset_lhs_imatmul_pack_x32p2vlx1_x32p_sme(m, ks,
                                                                       kc);
#elif XNN_ENABLE_ARM_SME_ACLE
  assert(kr == 1);
  (void)mr_packed;
  (void)sr;
  return m * ks * kc * sizeof(float);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
