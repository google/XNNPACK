// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/x86_vec512.h"
#include "ynnpack/kernels/transpose/generic.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

using simd::u8x16;
using simd::u8x32;
using simd::u8x64;

void transpose_x4_avx512(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x16, 32>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                   std::integral_constant<size_t, 4>{});
}
void transpose_x8_avx512(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x16, 16>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                   std::integral_constant<size_t, 8>{});
}
void transpose_x16_avx512(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                          const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x32, 16>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                   std::integral_constant<size_t, 16>{});
}
void transpose_x32_avx512(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                          const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x64, 16>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                   std::integral_constant<size_t, 32>{});
}
void transpose_x64_avx512(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                          const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x64, 8>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 64>{});
}
void transpose_x128_avx512(size_t m, size_t n, size_t n_bytes_a,
                           size_t stride_a, const void* a, size_t stride_x,
                           void* x) {
  transpose<std::array<u8x64, 4>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 128>{});
}
void transpose_x256_avx512(size_t m, size_t n, size_t n_bytes_a,
                           size_t stride_a, const void* a, size_t stride_x,
                           void* x) {
  transpose<std::array<u8x64, 2>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 256>{});
}

void interleave2_x2_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x64, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 2>{});
}

void interleave2_x4_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x64, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 4>{});
}

void interleave2_x8_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x64, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 8>{});
}

void interleave2_x16_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                            const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x64, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 16>{});
}

void interleave2_x32_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                            const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x64, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 32>{});
}

void interleave4_x2_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x64, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 2>{});
}

void interleave4_x4_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x64, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 4>{});
}

void interleave4_x8_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x64, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 8>{});
}

void interleave4_x16_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                            const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x64, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 16>{});
}

void interleave4_x32_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                            const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x64, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 32>{});
}

void interleave8_x2_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 8);
  interleave<std::array<u8x64, 8>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 2>{});
}

void interleave8_x4_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                           const void* a, void* x) {
  assert(factor == 8);
  interleave<std::array<u8x64, 8>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 4>{});
}

void interleave16_x2_avx512(size_t factor, size_t m, size_t n, size_t stride_a,
                            const void* a, void* x) {
  assert(factor == 16);
  interleave<std::array<u8x64, 16>>(m, n, stride_a, a, x,
                                    std::integral_constant<size_t, 2>{});
}

// Fused VNNI packing for 16-bit elements, 2 rows at a time.
//
// Produces, for each output row r and column j:
//   out32[r][j] = a[2r][j] | (a[2r + 1][j] << 16)
//
// Unlike the `interleave` template, this never crosses 128-bit lanes: the
// widening load places element j of the source into 32-bit lane j directly, so
// the `unpacklo/hi_epi16` + `vpermt2q` fixup sequence is not needed. That is
// only possible because the load granularity is 256 bits rather than 512, which
// is why this cannot be expressed as a drop-in replacement for the
// `interleave(integral_constant<16>, ...)` primitive.
void interleave2_x16_block_avx512(size_t factor, size_t m, size_t n,
                                  size_t tile_n, size_t stride_a, const void* a,
                                  size_t stride_x, void* x) {
  assert(factor == 2);
  (void)factor;
  assert(n <= tile_n);

  // Columns we can handle with unmasked 16-wide loads and stores.
  const size_t n_fast = std::min(tile_n, n) & ~size_t{15};

  for (size_t i = 0; i < m; i += 2) {
    const uint16_t* a0 =
        reinterpret_cast<const uint16_t*>(offset_bytes(a, i * stride_a));
    const bool has_a1 = (i + 1) < m;
    const uint16_t* a1 = has_a1 ? reinterpret_cast<const uint16_t*>(
                                      offset_bytes(a, (i + 1) * stride_a))
                                : a0;
    uint8_t* out =
        reinterpret_cast<uint8_t*>(offset_bytes(x, (i / 2) * stride_x));

    size_t j = 0;
    for (; j < n_fast; j += 16) {
      const __m512i r0 = _mm512_cvtepu16_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0 + j)));
      const __m512i r1 =
          has_a1 ? _mm512_slli_epi32(
                       _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                           reinterpret_cast<const __m256i*>(a1 + j))),
                       16)
                 : _mm512_setzero_si512();
      _mm512_storeu_si512(reinterpret_cast<void*>(out + j * 4),
                          _mm512_or_si512(r0, r1));
    }
    // Remaining columns: partially available input and/or column padding.
    for (; j < tile_n; j += 16) {
      const size_t cols = std::min<size_t>(16, tile_n - j);
      const size_t avail = j < n ? std::min<size_t>(cols, n - j) : 0;
      const __mmask16 load_mask =
          static_cast<__mmask16>((uint32_t{1} << avail) - 1);
      const __mmask16 store_mask =
          static_cast<__mmask16>((uint32_t{1} << cols) - 1);
      const __m512i r0 =
          _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(load_mask, a0 + j));
      const __m512i r1 =
          has_a1 ? _mm512_slli_epi32(
                       _mm512_cvtepu16_epi32(
                           _mm256_maskz_loadu_epi16(load_mask, a1 + j)),
                       16)
                 : _mm512_setzero_si512();
      _mm512_mask_storeu_epi32(reinterpret_cast<void*>(out + j * 4), store_mask,
                               _mm512_or_si512(r0, r1));
    }
  }
}

}  // namespace ynn
