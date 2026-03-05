// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DOT_X86_AMX_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_X86_AMX_H_

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"

#if YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
// Workaround for GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=122446
#define YNN_TILE_DP_IMPL(name, dst, src1, src2)                             \
  __asm__ volatile(                                                         \
      "{t" #name "\t%%tmm%c[_src2], %%tmm%c[_src1], %%tmm%c[_dst]|t" #name    \
      "\t%%tmm%c[_dst], %%tmm%c[_src1], %%tmm%c[_src1]}" ::[_dst] "i"(dst), \
      [_src1] "i"(src1), [_src2] "i"(src2))

#define YNN_TILE_DPBF16PS(a, b, c) YNN_TILE_DP_IMPL(dpbf16ps, a, b, c)
#define YNN_TILE_DPFP16PS(a, b, c) YNN_TILE_DP_IMPL(dpfp16ps, a, b, c)
#define YNN_TILE_DPBSSD(a, b, c) YNN_TILE_DP_IMPL(dpbssd, a, b, c)
#define YNN_TILE_DPBUSD(a, b, c) YNN_TILE_DP_IMPL(dpbusd, a, b, c)
#else
#define YNN_TILE_DPBF16PS(a, b, c) _tile_dpbf16ps(a, b, c)
#define YNN_TILE_DPFP16PS(a, b, c) _tile_dpfp16ps(a, b, c)
#define YNN_TILE_DPBSSD(a, b, c) _tile_dpbssd(a, b, c)
#define YNN_TILE_DPBUSD(a, b, c) _tile_dpbusd(a, b, c)
#endif

namespace ynn {

namespace internal {

constexpr size_t tile_row_bytes = 64;

struct tile_config {
  std::uint8_t palette_id;
  std::uint8_t start_row;
  std::uint8_t reserved_0[14];
  std::uint16_t colsb[8];
  std::uint16_t reserved_1[8];
  std::uint8_t rows[8];
  std::uint8_t reserved_2[8];
};

static_assert(sizeof(tile_config) == 64, "");
static_assert(offsetof(tile_config, colsb) == 16, "");
static_assert(offsetof(tile_config, rows) == 48, "");

template <typename TA, typename TB, typename TC>
static void load_tile_config_1x4(size_t m, size_t n, size_t ktail) {
  alignas(64) tile_config config = {0};
  config.palette_id = 1;

  // c tiles
  config.rows[0] = m;
  config.rows[1] = m;
  config.rows[2] = m;
  config.rows[3] = m;
  config.colsb[0] = n * sizeof(TC);
  config.colsb[1] = n * sizeof(TC);
  config.colsb[2] = n * sizeof(TC);
  config.colsb[3] = n * sizeof(TC);

  // a, a tail
  config.rows[4] = m;
  config.colsb[4] = 64;
  config.rows[6] = m;
  config.colsb[6] = ktail * sizeof(TA);

  // b, b tail
  config.rows[5] = 16;
  config.colsb[5] = n * sizeof(TC);
  config.rows[7] = ktail * sizeof(TB) / 4;
  config.colsb[7] = n * sizeof(TC);

  _tile_loadconfig(&config);
}

}  // namespace internal

template <typename TAB, typename TC, template <int, int, int> class TileOp>
YNN_ALWAYS_INLINE static void x86_amx_dot_1x4(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  // AMX is structured as 16x16x4 byte tiles. Each row is 64 bytes. This will
  // represent 64 / sizeof(T) elements.
  constexpr size_t k_block = internal::tile_row_bytes / sizeof(TAB);

  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  assert(M <= 16);

  constexpr size_t align_k = 4 / sizeof(TAB);
  assert(K1 % align_k == 0);

  const size_t B_stride_k1_block = B_stride_k1 * k_block;

  // We load this many rows of B at a time.
  B_stride_k1 *= align_k;
  assert(B_stride_k1 % internal::tile_row_bytes == 0 || K1 == 1);

  // The size of the remainder in the K loop.
  const size_t k_tail = (K1 & (k_block - 1)) ? (K1 & (k_block - 1)) : k_block;

  // Prepare the config for the main loop (4 tiles).
  internal::load_tile_config_1x4<TAB, TAB, TC>(M, 16, k_tail);
  while (N >= 64) {
    if (C_in) {
      _tile_loadd(0, offset_bytes(C_in, 0 * internal::tile_row_bytes),
                  C_in_stride_m);
      _tile_loadd(1, offset_bytes(C_in, 1 * internal::tile_row_bytes),
                  C_in_stride_m);
      _tile_loadd(2, offset_bytes(C_in, 2 * internal::tile_row_bytes),
                  C_in_stride_m);
      _tile_loadd(3, offset_bytes(C_in, 3 * internal::tile_row_bytes),
                  C_in_stride_m);
    } else {
      _tile_zero(0);
      _tile_zero(1);
      _tile_zero(2);
      _tile_zero(3);
    }
    const void* B_k3 = B;
    const void* A_k3 = A;
    size_t k3 = K3;
    do {
      const void* B_k2 = B_k3;
      const void* A_k2 = A_k3;
      size_t k2 = K2;
      do {
        const void* B_k1 = B_k2;
        const void* A_k1 = A_k2;
        std::ptrdiff_t k1 = K1;
        while (k1 >= k_block) {
          _tile_loadd(4, A_k1, A_stride_m);

          _tile_loadd(5, offset_bytes(B_k1, (0 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<0, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (1 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<1, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (2 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<2, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (3 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<3, 4, 5>()();

          k1 -= k_block;
          B_k1 = offset_bytes(B_k1, B_stride_k1_block);
          A_k1 = offset_bytes(A_k1, internal::tile_row_bytes);
        }
        if (k1 > 0) {
          _tile_loadd(6, A_k1, A_stride_m);
          _tile_loadd(7, offset_bytes(B_k1, (0 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<0, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (1 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<1, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (2 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<2, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (3 * internal::tile_row_bytes)),
                      B_stride_k1);
          TileOp<3, 6, 7>()();
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);
    _tile_stored(0, offset_bytes(C_out, 0 * internal::tile_row_bytes),
                 C_out_stride_m);
    _tile_stored(1, offset_bytes(C_out, 1 * internal::tile_row_bytes),
                 C_out_stride_m);
    _tile_stored(2, offset_bytes(C_out, 2 * internal::tile_row_bytes),
                 C_out_stride_m);
    _tile_stored(3, offset_bytes(C_out, 3 * internal::tile_row_bytes),
                 C_out_stride_m);
#if YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
    // msan doesn't support amx, avoid false positives.
    for (size_t i = 0; i < M; ++i) {
      __msan_unpoison(offset_bytes(C_out, i * C_out_stride_m),
                      4 * internal::kAmxTileRowBytes);
    }
#endif
    C_in = C_in ? offset_bytes(C_in, 4 * internal::tile_row_bytes) : nullptr;
    C_out = offset_bytes(C_out, 4 * internal::tile_row_bytes);
    B = offset_bytes(B, 4 * internal::tile_row_bytes);
    N -= 64;
  }
  while (N > 0) {
    // We might need to handle a less-than-tile here.
    internal::load_tile_config_1x4<TAB, TAB, TC>(M, std::min<size_t>(N, 16),
                                                 k_tail);
    if (C_in) {
      _tile_loadd(0, C_in, C_in_stride_m);
    } else {
      _tile_zero(0);
    }
    const void* B_k3 = B;
    const void* A_k3 = A;
    size_t k3 = K3;
    do {
      const void* B_k2 = B_k3;
      const void* A_k2 = A_k3;
      size_t k2 = K2;
      do {
        const void* B_k1 = B_k2;
        const void* A_k1 = A_k2;
        std::ptrdiff_t k1 = K1;
        while (k1 >= k_block) {
          _tile_loadd(4, A_k1, A_stride_m);
          _tile_loadd(5, B_k1, B_stride_k1);
          TileOp<0, 4, 5>()();

          k1 -= k_block;
          B_k1 = offset_bytes(B_k1, B_stride_k1_block);
          A_k1 = offset_bytes(A_k1, internal::tile_row_bytes);
        }
        if (k1 > 0) {
          _tile_loadd(6, A_k1, A_stride_m);
          _tile_loadd(7, B_k1, B_stride_k1);
          TileOp<0, 6, 7>()();
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);
    _tile_stored(0, C_out, C_out_stride_m);
    #if YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
    // msan doesn't support amx, avoid false positives.
    for (size_t i = 0; i < M; ++i) {
      __msan_unpoison(offset_bytes(C_out, i * C_out_stride_m), N * sizeof(TC));
    }
    #endif
    C_in = C_in ? offset_bytes(C_in, internal::tile_row_bytes) : nullptr;
    C_out = offset_bytes(C_out, internal::tile_row_bytes);
    B = offset_bytes(B, internal::tile_row_bytes);
    N = sub_sat(N, 16);
  }
  _tile_release();
}

namespace internal {

// Loads the tile configuration for a 2x2 scenario with the given dimensions.
//
// For a 2x2 output tile, tile ids are as follows:
//   - 0, 1, 2, 3: C tiles.
//   - 4, 6: A tiles.
//   - 5, 7: B tiles.
//
// Parameters:
//   m: number of rows in A and C.
//   n: number of columns within a tile of B and C.
//   k_len: length of the K dimension (must be <= 16).
//   align_k: alignment of the K dimension in the B matrix.
template <typename TAB, typename TC>
static void load_config_2x2(size_t m, size_t n, size_t k_len, size_t align_k) {
  const size_t m0 = std::min<size_t>(m, 16);
  const size_t m1 = sub_sat(m, 16);

  const size_t n0 = std::min<size_t>(n, 16);
  const size_t n1 = sub_sat(n, 16);

  assert(m0 > 0);
  assert(n > 0);
  assert(k_len > 0);
  assert(m0 <= 16);
  assert(m1 <= 16);
  assert(n0 <= 16);
  assert(n1 <= 16);

  alignas(64) tile_config config = {0};
  config.palette_id = 1;

  // The configuration here must exactly match the tile IDs used in the kernel.
  //
  // If we configure all tiles to be used and then in the kernel, zero out the
  // tiles we don't need, AMX will treat this as invalid and crash.
  //
  // If we configure a tile to be unused here and then in the kernel, use the
  // tile, AMX will also crash.

  // C tiles.
  config.rows[0] = m0;
  config.colsb[0] = n0 * sizeof(TC);
  if (n1 > 0) {
    config.rows[1] = m0;
    config.colsb[1] = n1 * sizeof(TC);
  }

  if (m1 > 0) {
    config.rows[2] = m1;
    config.colsb[2] = n0 * sizeof(TC);
    if (n1 > 0) {
      config.rows[3] = m1;
      config.colsb[3] = n1 * sizeof(TC);
    }
  }

  // A tiles.
  config.rows[4] = m0;
  config.colsb[4] = k_len * sizeof(TAB);
  if (m1 > 0) {
    config.rows[6] = m1;
    config.colsb[6] = k_len * sizeof(TAB);
  }

  // B tiles.
  config.rows[5] = k_len / align_k;
  config.colsb[5] = n0 * sizeof(TC);
  if (n1 > 0) {
    config.rows[7] = k_len / align_k;
    config.colsb[7] = n1 * sizeof(TC);
  }

  _tile_loadconfig(&config);
}

// Calculates the dot product of A and B over the K dimension.
//
// If HasM1 && HasN1, loops over the K dimension of a 2x2 output tile.
// If HasM1 && !HasN1, loops over the K dimension of a 2x1 output tile.
// If !HasM1 && HasN1, loops over the K dimension of a 1x2 output tile.
// If !HasM1 && !HasN1, loops over the K dimension of a 1x1 output tile.
//
// Tiles 0, 1, 2, and 3 are the accumulator tiles.
template <template <int, int, int> class TileOp, bool HasM1, bool HasN1>
YNN_ALWAYS_INLINE static void k_loops_impl(
    size_t K3, size_t K2, size_t k1_iters, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, size_t A_stride_k1, const void* A,
    size_t B_stride_k3, size_t B_stride_k2, size_t B_stride_k1,
    size_t B_stride_k1_block, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  // Initialize accumulator tiles.
  if (C_in) {
    _tile_loadd(0, offset_bytes(C_in, 0), C_in_stride_m);
    if constexpr (HasN1) {
      _tile_loadd(1, offset_bytes(C_in, tile_row_bytes), C_in_stride_m);
    }
    if constexpr (HasM1) {
      C_in = offset_bytes(C_in, 16 * C_in_stride_m);
      _tile_loadd(2, C_in, C_in_stride_m);
      if constexpr (HasN1) {
        _tile_loadd(3, offset_bytes(C_in, tile_row_bytes), C_in_stride_m);
      }
    }
  } else {
    _tile_zero(0);
    if constexpr (HasN1) _tile_zero(1);
    if constexpr (HasM1) _tile_zero(2);
    if constexpr (HasM1 && HasN1) _tile_zero(3);
  }

  // Loop over the K dimension.
  const void* B_k3 = B;
  const void* A_k3 = A;
  size_t k3 = K3;
  do {
    const void* B_k2 = B_k3;
    const void* A_k2 = A_k3;
    size_t k2 = K2;
    do {
      const void* B_k1 = B_k2;
      const void* A_k1 = A_k2;
      size_t k = k1_iters;
      while (k--) {
        if constexpr (HasM1 && HasN1) {  // 2x2 case.
          _tile_loadd(4, A_k1, A_stride_m);
          _tile_loadd(5, offset_bytes(B_k1, 0), B_stride_k1);
          _tile_loadd(6, offset_bytes(A_k1, 16 * A_stride_m), A_stride_m);
          _tile_loadd(7, offset_bytes(B_k1, tile_row_bytes), B_stride_k1);
          TileOp<0, 4, 5>()();
          TileOp<2, 6, 5>()();
          TileOp<1, 4, 7>()();
          TileOp<3, 6, 7>()();
        } else if constexpr (!HasM1 && HasN1) {  // 1x2 case.
          _tile_loadd(4, A_k1, A_stride_m);
          _tile_loadd(5, offset_bytes(B_k1, 0), B_stride_k1);
          _tile_loadd(7, offset_bytes(B_k1, tile_row_bytes), B_stride_k1);
          TileOp<0, 4, 5>()();
          TileOp<1, 4, 7>()();
        } else if constexpr (HasM1 && !HasN1) {  // 2x1 case.
          _tile_loadd(4, A_k1, A_stride_m);
          _tile_loadd(5, B_k1, B_stride_k1);
          _tile_loadd(6, offset_bytes(A_k1, 16 * A_stride_m), A_stride_m);
          TileOp<0, 4, 5>()();
          TileOp<2, 6, 5>()();
        } else {  // 1x1 case.
          _tile_loadd(4, A_k1, A_stride_m);
          _tile_loadd(5, B_k1, B_stride_k1);
          TileOp<0, 4, 5>()();
        }
        B_k1 = offset_bytes(B_k1, B_stride_k1_block);
        A_k1 = offset_bytes(A_k1, A_stride_k1);
      }
      k2 -= 1;
      B_k2 = offset_bytes(B_k2, B_stride_k2);
      A_k2 = offset_bytes(A_k2, A_stride_k2);
    } while (k2 > 0);
    k3 -= 1;
    B_k3 = offset_bytes(B_k3, B_stride_k3);
    A_k3 = offset_bytes(A_k3, A_stride_k3);
  } while (k3 > 0);

  // Store the accumulator tiles.
  _tile_stored(0, offset_bytes(C_out, 0), C_out_stride_m);
  if constexpr (HasN1) {
    _tile_stored(1, offset_bytes(C_out, tile_row_bytes), C_out_stride_m);
  }
  if constexpr (HasM1) {
    C_out = offset_bytes(C_out, 16 * C_out_stride_m);
    _tile_stored(2, C_out, C_out_stride_m);
    if constexpr (HasN1) {
      _tile_stored(3, offset_bytes(C_out, tile_row_bytes), C_out_stride_m);
    }
  }
}

template <typename TAB, typename TC, template <int, int, int> class TileOp,
          bool HasM1>
YNN_ALWAYS_INLINE static void n_loops_impl(
    size_t M, size_t n_loops, size_t n_tail, size_t K3, size_t K2,
    size_t k_iters, size_t k_len, size_t align_k, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, size_t A_stride_k1, const void* A,
    size_t B_stride_k3, size_t B_stride_k2, size_t B_stride_k1,
    size_t B_stride_k1_block, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  load_config_2x2<TAB, TC>(M, 32, k_len, align_k);

  for (size_t i = 0; i < n_loops; ++i) {
    k_loops_impl<TileOp, HasM1, /*HasN1=*/true>(
        K3, K2, k_iters, A_stride_m, A_stride_k3, A_stride_k2, A_stride_k1, A,
        B_stride_k3, B_stride_k2, B_stride_k1, B_stride_k1_block, B,
        C_in_stride_m, C_in, C_out_stride_m, C_out);

    if (C_in) {
      C_in = offset_bytes(C_in, 2 * tile_row_bytes);
    }
    C_out = offset_bytes(C_out, 2 * tile_row_bytes);
    B = offset_bytes(B, 2 * tile_row_bytes);
  }

  if (n_tail > 0) {
    load_config_2x2<TAB, TC>(M, n_tail, k_len, align_k);
    if (n_tail > 16) {
      k_loops_impl<TileOp, HasM1, /*HasN1=*/true>(
          K3, K2, k_iters, A_stride_m, A_stride_k3, A_stride_k2, A_stride_k1, A,
          B_stride_k3, B_stride_k2, B_stride_k1, B_stride_k1_block, B,
          C_in_stride_m, C_in, C_out_stride_m, C_out);
    } else {
      k_loops_impl<TileOp, HasM1, /*HasN1=*/false>(
          K3, K2, k_iters, A_stride_m, A_stride_k3, A_stride_k2, A_stride_k1, A,
          B_stride_k3, B_stride_k2, B_stride_k1, B_stride_k1_block, B,
          C_in_stride_m, C_in, C_out_stride_m, C_out);
    }
  }
}

// Computes the dot product of two matrices A and B with the given dimensions
// M, N, K3, K2 and K1.
// If HasM1 is true, M is assumed to be > 16 and the function will compute
// the second row of a 2x2 output tile.
template <typename TAB, typename TC, template <int, int, int> class TileOp,
          bool HasM1>
YNN_ALWAYS_INLINE static void x86_amx_dot_2x2_impl(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  constexpr size_t k_block = tile_row_bytes / sizeof(TAB);
  constexpr size_t align_k = 4 / sizeof(TAB);

  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  assert(M <= 32);

  const size_t B_stride_k1_block = B_stride_k1 * k_block;

  // We load this many rows of B at a time.
  B_stride_k1 *= align_k;
  assert(K1 % align_k == 0);

  const size_t k_tail = K1 & (k_block - 1);
  const size_t k1_iters = K1 / k_block;
  const size_t n_loops = N / 32;
  const size_t n_tail = N % 32;

  // 1. Handle M == 32 (or M == 16 if !HasM1), N <= 32, K == 32.
  if (k1_iters > 0) {
    n_loops_impl<TAB, TC, TileOp, HasM1>(
        M, n_loops, n_tail, K3, K2, k1_iters, k_block, align_k, A_stride_m,
        A_stride_k3, A_stride_k2, tile_row_bytes, A, B_stride_k3, B_stride_k2,
        B_stride_k1, B_stride_k1_block, B, C_in_stride_m, C_in, C_out_stride_m,
        C_out);

    // For the tail, we want to read C_out instead of C_in.
    C_in = C_out;
    C_in_stride_m = C_out_stride_m;

    // And we only want to handle the tail.
    A = offset_bytes(A, k1_iters * tile_row_bytes);
    B = offset_bytes(B, k1_iters * B_stride_k1_block);
  }

  // 2. Handle M == 32 (or M == 16 if !HasM1), N <= 32, K < 32.
  if (k_tail > 0) {
    n_loops_impl<TAB, TC, TileOp, HasM1>(
        M, n_loops, n_tail, K3, K2, /*k_iters=*/1, k_tail, align_k, A_stride_m,
        A_stride_k3, A_stride_k2, /*A_stride_k1=*/0, A, B_stride_k3,
        B_stride_k2, B_stride_k1, /*B_stride_k1_block=*/0, B, C_in_stride_m,
        C_in, C_out_stride_m, C_out);
  }

  _tile_release();
}
}  // namespace internal

template <typename TAB, typename TC, template <int, int, int> class TileOp>
YNN_ALWAYS_INLINE static void x86_amx_dot_2x2(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  if (M > 16) {
    internal::x86_amx_dot_2x2_impl<TAB, TC, TileOp, /*HasM1=*/true>(
        M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
        B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m,
        C_out);
  } else {
    internal::x86_amx_dot_2x2_impl<TAB, TC, TileOp, /*HasM1=*/false>(
        M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
        B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m,
        C_out);
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_X86_AMX_H_
