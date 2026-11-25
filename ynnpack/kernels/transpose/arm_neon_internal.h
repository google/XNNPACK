// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_ARM_NEON_INTERNAL_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_ARM_NEON_INTERNAL_H_

#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/arm_vec128.h"
#include "ynnpack/kernels/transpose/generic.h"

#ifdef YNN_ARCH_ARM64

namespace ynn {

namespace internal {

static YNN_ALWAYS_INLINE void prefetch_read_arm(const void* addr) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(addr, 0, 3);
#elif defined(_MSC_VER) && \
    (defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC))
  __prefetch(const_cast<void*>(addr));
#else
  (void)addr;
#endif
}

static YNN_ALWAYS_INLINE uint32x4_t load_u32x4_neon(const void* ptr) {
  return vreinterpretq_u32_u8(vld1q_u8(static_cast<const uint8_t*>(ptr)));
}

static YNN_ALWAYS_INLINE void store_u32x4_neon(void* ptr, uint32x4_t v) {
  vst1q_u8(static_cast<uint8_t*>(ptr), vreinterpretq_u8_u32(v));
}

static YNN_ALWAYS_INLINE void transpose_4x4_u32_neon(uint32x4_t& r0,
                                                     uint32x4_t& r1,
                                                     uint32x4_t& r2,
                                                     uint32x4_t& r3) {
  uint32x4_t t0 = vtrn1q_u32(r0, r1);
  uint32x4_t t1 = vtrn2q_u32(r0, r1);
  uint32x4_t t2 = vtrn1q_u32(r2, r3);
  uint32x4_t t3 = vtrn2q_u32(r2, r3);
  r0 = vreinterpretq_u32_u64(
      vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
  r1 = vreinterpretq_u32_u64(
      vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
  r2 = vreinterpretq_u32_u64(
      vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
  r3 = vreinterpretq_u32_u64(
      vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
}

// Transposes 16 input rows x 4 input cols -> 4 output rows x 16 output cols
// (64 contiguous bytes per output row) directly in registers.
static YNN_ALWAYS_INLINE void transpose_4x16_u32_neon(size_t stride_a,
                                                      const void* a,
                                                      size_t stride_x,
                                                      void* x) {
  uint32x4_t v[4][4];
  const void* a_r = a;
  YNN_UNROLL
  for (size_t b = 0; b < 4; ++b) {
    YNN_UNROLL
    for (size_t r = 0; r < 4; ++r) {
      v[b][r] = load_u32x4_neon(a_r);
      a_r = offset_bytes(a_r, stride_a);
    }
    transpose_4x4_u32_neon(v[b][0], v[b][1], v[b][2], v[b][3]);
  }

  void* x_r = x;
  YNN_UNROLL
  for (size_t r = 0; r < 4; ++r) {
    YNN_UNROLL
    for (size_t b = 0; b < 4; ++b) {
      store_u32x4_neon(offset_bytes(x_r, b * 16), v[b][r]);
    }
    x_r = offset_bytes(x_r, stride_x);
  }
}

// Blocked 4x16 main loop for full-bounds (m_4 x n_16) 32-bit transposes.
// Writes 64 B full cache-line bursts per output row while keeping the active
// input/output working set within L1/L2 cache and L2 TLB.
static YNN_ALWAYS_INLINE void transpose_x32_arm64_blocked(
    size_t m_4, size_t n_16, size_t stride_a, const void* a, size_t stride_x,
    void* x) {
  constexpr size_t kMBlock = 512;
  for (size_t i_blk = 0; i_blk < m_4; i_blk += kMBlock) {
    const size_t i_end = std::min(i_blk + kMBlock, m_4);
    for (size_t j = 0; j < n_16; j += 16) {
      const void* a_ij = offset_bytes(a, j * stride_a + i_blk * 4);
      void* x_ij = offset_bytes(x, i_blk * stride_x + j * 4);
      size_t i = i_blk;
      for (; i + 16 <= i_end; i += 16) {
        if (i + 64 < i_end) {
          const void* pf = offset_bytes(a_ij, 256);
          for (size_t r = 0; r < 16; ++r) {
            prefetch_read_arm(pf);
            pf = offset_bytes(pf, stride_a);
          }
        }
        YNN_UNROLL
        for (size_t u = 0; u < 4; ++u) {
          transpose_4x16_u32_neon(stride_a, a_ij, stride_x, x_ij);
          a_ij = offset_bytes(a_ij, 16);
          x_ij = offset_bytes(x_ij, 4 * stride_x);
        }
      }
      for (; i < i_end; i += 4) {
        transpose_4x16_u32_neon(stride_a, a_ij, stride_x, x_ij);
        a_ij = offset_bytes(a_ij, 16);
        x_ij = offset_bytes(x_ij, 4 * stride_x);
      }
    }
  }
}

// Internal linkage ensures NEON and SVE TUs each instantiate against their own
// simd::load/store overloads without ODR conflicts.
static void transpose_x32_arm64_strips(size_t m, size_t n, size_t n_bytes_a,
                                       size_t stride_a, const void* a,
                                       size_t stride_x, void* x) {
  for (size_t j = 0; j < n; j += 32) {
    const size_t n_j = std::min<size_t>(32, n - j);
    transpose<std::array<simd::u8x16, 4>>(m, n_j, n_bytes_a, stride_a,
                                          offset_bytes(a, j * stride_a),
                                          stride_x, offset_bytes(x, j * 4),
                                          std::integral_constant<size_t, 32>{});
  }
}

static YNN_ALWAYS_INLINE void transpose_x32_arm64(size_t m, size_t n,
                                                  size_t n_bytes_a,
                                                  size_t stride_a,
                                                  const void* a,
                                                  size_t stride_x, void* x) {
  const size_t m_full = std::min(m, n_bytes_a / 4);
  if (m_full >= 16 && n >= 16) {
    const size_t m_4 = align_down<size_t>(m_full, 4);
    const size_t n_16 = align_down<size_t>(n, 16);
    transpose_x32_arm64_blocked(m_4, n_16, stride_a, a, stride_x, x);
    if (n > n_16) {
      transpose<std::array<simd::u8x16, 4>>(
          m_4, n - n_16, m_4 * 4, stride_a, offset_bytes(a, n_16 * stride_a),
          stride_x, offset_bytes(x, n_16 * 4),
          std::integral_constant<size_t, 32>{});
    }
    if (m > m_4) {
      transpose_x32_arm64_strips(m - m_4, n, sub_sat(n_bytes_a, m_4 * 4),
                                 stride_a, offset_bytes(a, m_4 * 4), stride_x,
                                 offset_bytes(x, m_4 * stride_x));
    }
    return;
  }
  transpose_x32_arm64_strips(m, n, n_bytes_a, stride_a, a, stride_x, x);
}

}  // namespace internal

}  // namespace ynn

#endif  // YNN_ARCH_ARM64

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_ARM_NEON_INTERNAL_H_
