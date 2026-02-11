// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_X86_AVX2_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_X86_AVX2_H_

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"

namespace ynn {

using row = __m256i;

static std::tuple<row, row> interleave(std::integral_constant<size_t, 128>,
                                       row x0, row x1) {
  return {_mm256_permute2x128_si256(x0, x1, 32),
          _mm256_permute2x128_si256(x0, x1, 49)};
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 64>,
                                       row x0, row x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    _mm256_unpacklo_epi64(x0, x1),
                    _mm256_unpackhi_epi64(x0, x1));
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 32>,
                                       row x0, row x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    _mm256_unpacklo_epi32(x0, x1),
                    _mm256_unpackhi_epi32(x0, x1));
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 16>,
                                       row x0, row x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    _mm256_unpacklo_epi16(x0, x1),
                    _mm256_unpackhi_epi16(x0, x1));
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 8>,
                                       row x0, row x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    _mm256_unpacklo_epi8(x0, x1), _mm256_unpackhi_epi8(x0, x1));
}
static std::tuple<row, row> interleave(std::integral_constant<size_t, 4>,
                                       row x0, row x1) {
  __m256i even0 = _mm256_and_si256(x0, _mm256_set1_epi8(0x0f));
  __m256i even1 = _mm256_and_si256(x1, _mm256_set1_epi8(0x0f));
  __m256i odd0 = _mm256_and_si256(x0, _mm256_set1_epi8(0xf0));
  __m256i odd1 = _mm256_and_si256(x1, _mm256_set1_epi8(0xf0));
  return interleave(std::integral_constant<size_t, 8>{},
                    _mm256_or_si256(_mm256_slli_epi16(even1, 4), even0),
                    _mm256_or_si256(odd1, _mm256_srli_epi16(odd0, 4)));
}

template <size_t M>
static std::array<row, M> load(
    std::array<row, M>, const void* a, size_t stride, size_t m,
    std::integral_constant<size_t, 32> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  std::array<row, M> x;
  x[0] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
  for (size_t i = 1; i < M; ++i) {
    x[i] = i < m ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                       offset_bytes(a, i * stride)))
                 : _mm256_setzero_si256();
  }
  return x;
}

template <size_t M>
static void store(std::array<row, M> x, void* a, size_t stride, size_t m,
                  std::integral_constant<size_t, 32> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(a), x[0]);
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(offset_bytes(a, i * stride)), x[i]);
    }
  }
}

template <typename Tile>
static Tile load(Tile, const void* a, size_t stride, size_t m, size_t n_bytes) {
  Tile result;
  memset(&result, 0, sizeof(Tile));
  for (size_t i = 0; i < m; ++i) {
    memcpy(&result[i], offset_bytes(a, i * stride), n_bytes);
  }
  return result;
}

template <typename Tile>
static void store(Tile tile, void* x, size_t stride, size_t m, size_t n_bytes) {
  for (size_t i = 0; i < m; ++i) {
    memcpy(offset_bytes(x, i * stride), &tile[i], n_bytes);
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_X86_AVX2_H_
