// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_ARM64_SME_INTERNAL_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_ARM64_SME_INTERNAL_H_

#include <arm_sme.h>

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"

#ifndef YNN_DISABLE_SME

namespace ynn {

namespace internal {

#define YNN_SME_TRANSPOSE_HELPER __arm_streaming __arm_inout("za")

static YNN_ALWAYS_INLINE const void* sme_offset_bytes(
    const void* ptr, size_t offset) YNN_SME_TRANSPOSE_HELPER {
  return static_cast<const uint8_t*>(ptr) + offset;
}

static YNN_ALWAYS_INLINE void* sme_offset_bytes(
    void* ptr, size_t offset) YNN_SME_TRANSPOSE_HELPER {
  return static_cast<uint8_t*>(ptr) + offset;
}

static YNN_ALWAYS_INLINE void sme_prefetch_read(const void* addr)
    YNN_SME_TRANSPOSE_HELPER {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(addr, 0, 3);
#else
  (void)addr;
#endif
}

// General streaming-mode tail handler for arbitrary m, n, and byte-granularity
// n_bytes_a using ZA tile 0 in chunks of up to svl x svl.
static YNN_ALWAYS_INLINE void transpose_x32_sme_strip(
    size_t m, size_t n, size_t n_bytes_a, size_t stride_a, const void* a,
    size_t stride_x, void* x) YNN_SME_TRANSPOSE_HELPER {
  const size_t svl = svcntsw();
  const size_t svl_bytes = svl * 4;
  const svbool_t p32 = svptrue_b32();

  for (size_t i = 0; i < m; i += svl) {
    const size_t m_i = (m - i) < svl ? (m - i) : svl;
    const size_t i_bytes = i * 4;
    const size_t rem_bytes = n_bytes_a > i_bytes ? (n_bytes_a - i_bytes) : 0;
    const size_t block_bytes = rem_bytes < svl_bytes ? rem_bytes : svl_bytes;
    const size_t m_bytes = m_i * 4;
    const size_t valid_bytes = block_bytes < m_bytes ? block_bytes : m_bytes;

    void* x_i = sme_offset_bytes(x, i * stride_x);

    if (valid_bytes == 0) {
      // Zero-fill with SVE byte stores to avoid calling libc memset in
      // streaming mode and to stay safe for unaligned output pointers.
      const svuint8_t zero = svdup_n_u8(0);
      const size_t row_bytes = n * 4;
      for (size_t c = 0; c < m_i; ++c) {
        void* x_c = sme_offset_bytes(x_i, c * stride_x);
        for (size_t b = 0; b < row_bytes; b += svl_bytes) {
          const svbool_t pg = svwhilelt_b8_u64(b, row_bytes);
          svst1_u8(pg, static_cast<uint8_t*>(sme_offset_bytes(x_c, b)), zero);
        }
      }
      continue;
    }

    const svbool_t pg_b = svwhilelt_b8_u64(0, valid_bytes);
    const void* a_i = sme_offset_bytes(a, i_bytes);

    for (size_t j = 0; j < n; j += svl) {
      const size_t n_j = (n - j) < svl ? (n - j) : svl;
      const svbool_t pg_out = n_j == svl ? p32 : svwhilelt_b32_u64(0, n_j);
      const void* a_r = sme_offset_bytes(a_i, j * stride_a);

      for (size_t r = 0; r < n_j; ++r) {
        svuint8_t row_b = svld1_u8(pg_b, static_cast<const uint8_t*>(a_r));
        svwrite_hor_za32_u32_m(0, r, p32, svreinterpret_u32_u8(row_b));
        a_r = sme_offset_bytes(a_r, stride_a);
      }

      void* x_c = sme_offset_bytes(x_i, j * 4);
      for (size_t c = 0; c < m_i; ++c) {
        svst1_ver_za32(0, c, pg_out, x_c);
        x_c = sme_offset_bytes(x_c, stride_x);
      }
    }
  }
}

template <typename Loader>
__arm_new("za") __arm_locally_streaming void transpose_x32_sme_generic(
    size_t m, size_t n, size_t n_bytes_a, size_t stride_a, const void* a,
    size_t stride_x, void* x) {
  const size_t svl = svcntsw();
  const size_t svl2 = svl * 2;
  const size_t svl4 = svl * 4;
  const size_t svl_bytes = svl * 4;
  const size_t svl2_bytes = svl2 * 4;
  const svbool_t p32 = svptrue_b32();

  const size_t m_full = m < (n_bytes_a / 4) ? m : (n_bytes_a / 4);
  const size_t n_2svl = n & ~(svl2 - 1);

  // Main 2D blocked loop: 2*svl x 2*svl tiles across all 4 ZA32 tiles
  // (32x32 = 128 B contiguous reads and writes when SVL = 64 B).
  if (n_2svl > 0) {
    const size_t m_2svl = m_full & ~(svl2 - 1);
    if (m_2svl > 0) {
      const size_t m_blk = svl2 < 512 ? (512 & ~(svl2 - 1)) : svl2;
      for (size_t i_blk = 0; i_blk < m_2svl; i_blk += m_blk) {
        const size_t i_end =
            (i_blk + m_blk) < m_2svl ? (i_blk + m_blk) : m_2svl;
        for (size_t j = 0; j < n_2svl; j += svl2) {
          const void* a_ij = sme_offset_bytes(a, j * stride_a + i_blk * 4);
          void* x_ij = sme_offset_bytes(x, i_blk * stride_x + j * 4);
          for (size_t i = i_blk; i < i_end; i += svl2) {
            const bool pf = (i + svl2) < i_end;
            const void* a_r = a_ij;
            for (size_t r = 0; r < svl; ++r) {
              if (pf) {
                sme_prefetch_read(sme_offset_bytes(a_r, svl2_bytes));
              }
              Loader::template load_2svl<0, 1>(r, a_r, svl_bytes);
              a_r = sme_offset_bytes(a_r, stride_a);
            }
            for (size_t r = 0; r < svl; ++r) {
              if (pf) {
                sme_prefetch_read(sme_offset_bytes(a_r, svl2_bytes));
              }
              Loader::template load_2svl<2, 3>(r, a_r, svl_bytes);
              a_r = sme_offset_bytes(a_r, stride_a);
            }

            void* x_r = x_ij;
            for (size_t c = 0; c < svl; ++c) {
              svst1_ver_za32(0, c, p32, x_r);
              svst1_ver_za32(2, c, p32, sme_offset_bytes(x_r, svl_bytes));
              x_r = sme_offset_bytes(x_r, stride_x);
            }
            for (size_t c = 0; c < svl; ++c) {
              svst1_ver_za32(1, c, p32, x_r);
              svst1_ver_za32(3, c, p32, sme_offset_bytes(x_r, svl_bytes));
              x_r = sme_offset_bytes(x_r, stride_x);
            }

            a_ij = sme_offset_bytes(a_ij, svl2_bytes);
            x_ij = sme_offset_bytes(x_ij, svl2 * stride_x);
          }
        }
      }
    }
    if (m > m_2svl) {
      const size_t n_bytes_rem =
          n_bytes_a > (m_2svl * 4) ? (n_bytes_a - m_2svl * 4) : 0;
      transpose_x32_sme_strip(m - m_2svl, n_2svl, n_bytes_rem, stride_a,
                              sme_offset_bytes(a, m_2svl * 4), stride_x,
                              sme_offset_bytes(x, m_2svl * stride_x));
    }
  }

  // Handle a single svl-wide column strip using 4*svl x svl tiles across all 4
  // ZA32 tiles (fast path for n == svl, e.g., SME dot LHS packing).
  size_t n_done = n_2svl;
  if (n - n_done >= svl) {
    const void* a_j = sme_offset_bytes(a, n_done * stride_a);
    void* x_j = sme_offset_bytes(x, n_done * 4);
    const size_t m_4svl = m_full & ~(svl4 - 1);
    for (size_t i = 0; i < m_4svl; i += svl4) {
      const bool pf = (i + svl4) < m_4svl;
      const void* a_r = sme_offset_bytes(a_j, i * 4);
      for (size_t r = 0; r < svl; ++r) {
        if (pf) {
          sme_prefetch_read(sme_offset_bytes(a_r, svl4 * 4));
        }
        Loader::load_4svl(r, a_r, svl_bytes);
        a_r = sme_offset_bytes(a_r, stride_a);
      }

      void* x_r = sme_offset_bytes(x_j, i * stride_x);
      for (size_t c = 0; c < svl; ++c) {
        svst1_ver_za32(0, c, p32, x_r);
        x_r = sme_offset_bytes(x_r, stride_x);
      }
      for (size_t c = 0; c < svl; ++c) {
        svst1_ver_za32(1, c, p32, x_r);
        x_r = sme_offset_bytes(x_r, stride_x);
      }
      for (size_t c = 0; c < svl; ++c) {
        svst1_ver_za32(2, c, p32, x_r);
        x_r = sme_offset_bytes(x_r, stride_x);
      }
      for (size_t c = 0; c < svl; ++c) {
        svst1_ver_za32(3, c, p32, x_r);
        x_r = sme_offset_bytes(x_r, stride_x);
      }
    }
    if (m > m_4svl) {
      const size_t n_bytes_rem =
          n_bytes_a > (m_4svl * 4) ? (n_bytes_a - m_4svl * 4) : 0;
      transpose_x32_sme_strip(m - m_4svl, svl, n_bytes_rem, stride_a,
                              sme_offset_bytes(a_j, m_4svl * 4), stride_x,
                              sme_offset_bytes(x_j, m_4svl * stride_x));
    }
    n_done += svl;
  }

  // Final partial column strip (1 <= n - n_done < svl).
  if (n > n_done) {
    transpose_x32_sme_strip(m, n - n_done, n_bytes_a, stride_a,
                            sme_offset_bytes(a, n_done * stride_a), stride_x,
                            sme_offset_bytes(x, n_done * 4));
  }
}

}  // namespace internal

}  // namespace ynn

#endif  // YNN_DISABLE_SME

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_ARM64_SME_INTERNAL_H_
