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
#include <type_traits>

#include "ynnpack/base/arithmetic.h"

namespace ynn {

static std::array<__m256i, 2> interleave(std::integral_constant<size_t, 128>,
                                         std::array<__m256i, 2> x) {
  return {{_mm256_permute2x128_si256(x[0], x[1], 32),
           _mm256_permute2x128_si256(x[0], x[1], 49)}};
}
static std::array<__m256i, 2> interleave(std::integral_constant<size_t, 64>,
                                         std::array<__m256i, 2> x) {
  return interleave(
      std::integral_constant<size_t, 128>{},
      {_mm256_unpacklo_epi64(x[0], x[1]), _mm256_unpackhi_epi64(x[0], x[1])});
}
static std::array<__m256i, 2> interleave(std::integral_constant<size_t, 32>,
                                         std::array<__m256i, 2> x) {
  return interleave(
      std::integral_constant<size_t, 128>{},
      {_mm256_unpacklo_epi32(x[0], x[1]), _mm256_unpackhi_epi32(x[0], x[1])});
}
static std::array<__m256i, 2> interleave(std::integral_constant<size_t, 16>,
                                         std::array<__m256i, 2> x) {
  return interleave(
      std::integral_constant<size_t, 128>{},
      {_mm256_unpacklo_epi16(x[0], x[1]), _mm256_unpackhi_epi16(x[0], x[1])});
}
static std::array<__m256i, 2> interleave(std::integral_constant<size_t, 8>,
                                         std::array<__m256i, 2> x) {
  return interleave(
      std::integral_constant<size_t, 128>{},
      {_mm256_unpacklo_epi8(x[0], x[1]), _mm256_unpackhi_epi8(x[0], x[1])});
}
static std::array<__m256i, 2> interleave(std::integral_constant<size_t, 4>,
                                         std::array<__m256i, 2> x) {
  __m256i even0 = _mm256_and_si256(x[0], _mm256_set1_epi8(0x0f));
  __m256i even1 = _mm256_and_si256(x[1], _mm256_set1_epi8(0x0f));
  __m256i odd0 = _mm256_and_si256(x[0], _mm256_set1_epi8(0xf0));
  __m256i odd1 = _mm256_and_si256(x[1], _mm256_set1_epi8(0xf0));
  return interleave(std::integral_constant<size_t, 8>{},
                    {_mm256_or_si256(_mm256_slli_epi16(even1, 4), even0),
                     _mm256_or_si256(odd1, _mm256_srli_epi16(odd0, 4))});
}

template <size_t M>
static std::array<__m256i, M> load(
    std::array<__m256i, M>, const void* a, size_t stride, size_t m,
    std::integral_constant<size_t, 32> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  std::array<__m256i, M> x;
  x[0] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
  for (size_t i = 1; i < M; ++i) {
    x[i] = i < m ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                       offset_bytes(a, i * stride)))
                 : _mm256_setzero_si256();
  }
  return x;
}

template <size_t M>
static void store(std::array<__m256i, M> x, void* a, size_t stride, size_t m,
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

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_X86_AVX2_H_
