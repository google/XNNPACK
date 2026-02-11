// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_X86_SSE2_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_X86_SSE2_H_

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"

namespace ynn {

using row = __m128i;

static std::tuple<row, row> interleave(std::integral_constant<size_t, 64>,
                                       row x0, row x1) {
  return {_mm_unpacklo_epi64(x0, x1), _mm_unpackhi_epi64(x0, x1)};
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 32>,
                                       row x0, row x1) {
  return {_mm_unpacklo_epi32(x0, x1), _mm_unpackhi_epi32(x0, x1)};
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 16>,
                                       row x0, row x1) {
  return {_mm_unpacklo_epi16(x0, x1), _mm_unpackhi_epi16(x0, x1)};
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 8>,
                                       row x0, row x1) {
  return {_mm_unpacklo_epi8(x0, x1), _mm_unpackhi_epi8(x0, x1)};
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 4>,
                                       row x0, row x1) {
  __m128i even0 = _mm_and_si128(x0, _mm_set1_epi8(0x0f));
  __m128i even1 = _mm_and_si128(x1, _mm_set1_epi8(0x0f));
  __m128i odd0 = _mm_and_si128(x0, _mm_set1_epi8(0xf0));
  __m128i odd1 = _mm_and_si128(x1, _mm_set1_epi8(0xf0));
  return interleave(std::integral_constant<size_t, 8>{},
                    _mm_or_si128(_mm_slli_epi16(even1, 4), even0),
                    _mm_or_si128(odd1, _mm_srli_epi16(odd0, 4)));
}

template <size_t M>
static std::array<row, M> load(
    std::array<row, M>, const void* a, size_t stride, size_t m,
    std::integral_constant<size_t, 16> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  std::array<row, M> x;
  x[0] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
  for (size_t i = 1; i < M; ++i) {
    x[i] = i < m ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                       offset_bytes(a, i * stride)))
                 : _mm_setzero_si128();
  }
  return x;
}

template <size_t M>
static void store(std::array<row, M> x, void* a, size_t stride, size_t m,
                  std::integral_constant<size_t, 16> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(a), x[0]);
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(offset_bytes(a, i * stride)),
                       x[i]);
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_X86_SSE2_H_
