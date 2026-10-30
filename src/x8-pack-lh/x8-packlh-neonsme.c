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
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
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
#elif XNN_ENABLE_ARM_SME_ACLE
  assert(m_idx_start == 0);
  assert(kr == 4);
  assert(sr == 1);
  (void)m_idx_start;
  (void)sr;
  if (mr_packed == 1) {
    for (size_t i = 0; i < m; ++i) {
      memcpy((int8_t*)lhs_packed + i * k, (const int8_t*)lhs + i * lhs_stride,
             sizeof(int8_t) * k);
    }
    return;
  }
  const size_t k_padded = round_up_po2(k, kr);
  const uint8_t* lhs_ptr = (const uint8_t*)lhs;
  int8_t* dst_base = (int8_t*)lhs_packed;
  const uint32x4_t vzero = vdupq_n_u32(0);
  for (size_t i_m = 0; i_m < m; i_m += mr_packed) {
    const size_t height = math_min_s32((int32_t)(m - i_m), (int32_t)mr_packed);
    int8_t* dst = dst_base;
    uint32_t* dst_u32 = (uint32_t*)dst;
    dst_base += mr_packed * k_padded;
    size_t y = 0;
    for (; y + 4 <= height; y += 4) {
      const int8_t* r0 = (const int8_t*)(lhs_ptr + (i_m + y + 0) * lhs_stride);
      const int8_t* r1 = (const int8_t*)(lhs_ptr + (i_m + y + 1) * lhs_stride);
      const int8_t* r2 = (const int8_t*)(lhs_ptr + (i_m + y + 2) * lhs_stride);
      const int8_t* r3 = (const int8_t*)(lhs_ptr + (i_m + y + 3) * lhs_stride);
      size_t x = 0;
      for (; x + 16 <= k; x += 16) {
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
        uint32_t* d = dst_u32 + (x / 4) * mr_packed + y;
        vst1q_u32(d + 0 * mr_packed, c0);
        vst1q_u32(d + 1 * mr_packed, c1);
        vst1q_u32(d + 2 * mr_packed, c2);
        vst1q_u32(d + 3 * mr_packed, c3);
      }
      for (; x < k_padded; x += kr) {
        int8_t* d = dst + x * mr_packed + y * 4;
        d[0] = (x + 0 < k) ? r0[x + 0] : 0;
        d[1] = (x + 1 < k) ? r0[x + 1] : 0;
        d[2] = (x + 2 < k) ? r0[x + 2] : 0;
        d[3] = (x + 3 < k) ? r0[x + 3] : 0;
        d[4] = (x + 0 < k) ? r1[x + 0] : 0;
        d[5] = (x + 1 < k) ? r1[x + 1] : 0;
        d[6] = (x + 2 < k) ? r1[x + 2] : 0;
        d[7] = (x + 3 < k) ? r1[x + 3] : 0;
        d[8] = (x + 0 < k) ? r2[x + 0] : 0;
        d[9] = (x + 1 < k) ? r2[x + 1] : 0;
        d[10] = (x + 2 < k) ? r2[x + 2] : 0;
        d[11] = (x + 3 < k) ? r2[x + 3] : 0;
        d[12] = (x + 0 < k) ? r3[x + 0] : 0;
        d[13] = (x + 1 < k) ? r3[x + 1] : 0;
        d[14] = (x + 2 < k) ? r3[x + 2] : 0;
        d[15] = (x + 3 < k) ? r3[x + 3] : 0;
      }
    }
    for (; y < height; ++y) {
      const int8_t* row = (const int8_t*)(lhs_ptr + (i_m + y) * lhs_stride);
      for (size_t x = 0; x < k_padded; x += kr) {
        int8_t* d = dst + x * mr_packed + y * 4;
        d[0] = (x + 0 < k) ? row[x + 0] : 0;
        d[1] = (x + 1 < k) ? row[x + 1] : 0;
        d[2] = (x + 2 < k) ? row[x + 2] : 0;
        d[3] = (x + 3 < k) ? row[x + 3] : 0;
      }
    }
    for (; y + 4 <= mr_packed; y += 4) {
      for (size_t x = 0; x < k_padded / 4; ++x) {
        vst1q_u32(dst_u32 + x * mr_packed + y, vzero);
      }
    }
    for (; y < mr_packed; ++y) {
      for (size_t x = 0; x < k_padded / 4; ++x) {
        dst_u32[x * mr_packed + y] = 0;
      }
    }
  }
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
#elif XNN_ENABLE_ARM_SME_ACLE
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
#elif XNN_ENABLE_ARM_SME_ACLE
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
