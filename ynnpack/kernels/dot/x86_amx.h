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

template <typename TA, typename TB, typename TC>
static void load_tile_config(size_t m, size_t n, size_t ktail) {
  struct tile_config {
    std::uint8_t palette_id;
    std::uint8_t start_row;
    std::uint8_t reserved_0[14];
    std::uint16_t colsb[8];
    std::uint16_t reserved_1[8];
    std::uint8_t rows[8];
    std::uint8_t reserved_2[8];
  };

  YNN_ALIGN(64) tile_config config = {0};
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

template <typename TAB, typename TC, template <int, int, int> class TileOp>
YNN_ALWAYS_INLINE static void x86_amx_dot(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  // AMX is structured as 16x16x4 byte tiles. Each row is 64 bytes. This will
  // represent 64 / sizeof(T) elements.
  constexpr size_t row_bytes = 64;
  constexpr size_t k_block = row_bytes / sizeof(TAB);

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
  assert(B_stride_k1 % row_bytes == 0 || K1 == 1);

  // The size of the remainder in the K loop.
  const size_t k_tail = (K1 & (k_block - 1)) ? (K1 & (k_block - 1)) : k_block;

  // Prepare the config for the main loop (4 tiles).
  load_tile_config<TAB, TAB, TC>(M, 16, k_tail);
  while (N >= 64) {
    if (C_in) {
      _tile_loadd(0, offset_bytes(C_in, 0 * row_bytes), C_in_stride_m);
      _tile_loadd(1, offset_bytes(C_in, 1 * row_bytes), C_in_stride_m);
      _tile_loadd(2, offset_bytes(C_in, 2 * row_bytes), C_in_stride_m);
      _tile_loadd(3, offset_bytes(C_in, 3 * row_bytes), C_in_stride_m);
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

          _tile_loadd(5, offset_bytes(B_k1, (0 * row_bytes)), B_stride_k1);
          TileOp<0, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (1 * row_bytes)), B_stride_k1);
          TileOp<1, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (2 * row_bytes)), B_stride_k1);
          TileOp<2, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (3 * row_bytes)), B_stride_k1);
          TileOp<3, 4, 5>()();

          k1 -= k_block;
          B_k1 = offset_bytes(B_k1, B_stride_k1_block);
          A_k1 = offset_bytes(A_k1, row_bytes);
        }
        if (k1 > 0) {
          _tile_loadd(6, A_k1, A_stride_m);

          _tile_loadd(7, offset_bytes(B_k1, (0 * row_bytes)), B_stride_k1);
          TileOp<0, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (1 * row_bytes)), B_stride_k1);
          TileOp<1, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (2 * row_bytes)), B_stride_k1);
          TileOp<2, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (3 * row_bytes)), B_stride_k1);
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
    _tile_stored(0, offset_bytes(C_out, 0 * row_bytes), C_out_stride_m);
    _tile_stored(1, offset_bytes(C_out, 1 * row_bytes), C_out_stride_m);
    _tile_stored(2, offset_bytes(C_out, 2 * row_bytes), C_out_stride_m);
    _tile_stored(3, offset_bytes(C_out, 3 * row_bytes), C_out_stride_m);
    #if YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
    // msan doesn't support amx, avoid false positives.
    for (size_t i = 0; i < M; ++i) {
      __msan_unpoison(offset_bytes(C_out, i * C_out_stride_m), 4 * row_bytes);
    }
    #endif
    C_in = C_in ? offset_bytes(C_in, 4 * row_bytes) : nullptr;
    C_out = offset_bytes(C_out, 4 * row_bytes);
    B = offset_bytes(B, 4 * row_bytes);
    N -= 64;
  }
  while (N > 0) {
    // We might need to handle a less-than-tile here.
    load_tile_config<TAB, TAB, TC>(M, std::min<size_t>(N, 16), k_tail);
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
          A_k1 = offset_bytes(A_k1, row_bytes);
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
    C_in = C_in ? offset_bytes(C_in, row_bytes) : nullptr;
    C_out = offset_bytes(C_out, row_bytes);
    B = offset_bytes(B, row_bytes);
    N = sub_sat(N, 16);
  }
  _tile_release();
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_X86_AMX_H_
