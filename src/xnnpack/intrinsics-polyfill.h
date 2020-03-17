// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


#ifdef __AVX512F__
#include <immintrin.h>

// GCC pre-7, Clang pre-8, Apple Clang pre-11, and ICC pre-18
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 7)) || \
    (defined(__clang__) && !defined(__apple_build_version__) && (__clang_major__ < 8)) || \
    (defined(__clang__) && defined(__apple_build_version__) && (__apple_build_version__ < 11000000)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1800))

static inline __mmask16 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_cvtu32_mask16(unsigned int mask) {
  return (__mmask16) mask;
}

#endif  // GCC pre-7, Clang pre-8, Apple Clang pre-10, and ICC pre-18

// GCC pre-7, Clang pre-4, and ICC pre-18
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 7)) || \
    (defined(__clang__) && (__clang_major__ < 4)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1800))

static inline float __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm512_reduce_add_ps(__m512 v) {
#if __AVX512DQ__
  const __m256 sum2 = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
#else
  const __m256 sum2 = _mm256_add_ps(_mm512_castps512_ps256(v), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)));
#endif
  const __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum2), _mm256_extractf128_ps(sum2, 1));
  const __m128 sum8 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  const __m128 sum16 = _mm_add_ss(sum8, _mm_movehdup_ps(sum8));
  return _mm_cvtss_f32(sum16);
}

static inline float __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm512_reduce_max_ps(__m512 v) {
#if __AVX512DQ__
  const __m256 sum2 = _mm256_max_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
#else
  const __m256 sum2 = _mm256_max_ps(_mm512_castps512_ps256(v), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)));
#endif
  const __m128 sum4 = _mm_max_ps(_mm256_castps256_ps128(sum2), _mm256_extractf128_ps(sum2, 1));
  const __m128 sum8 = _mm_max_ps(sum4, _mm_movehl_ps(sum4, sum4));
  const __m128 sum16 = _mm_max_ss(sum8, _mm_movehdup_ps(sum8));
  return _mm_cvtss_f32(sum16);
}

#endif  // GCC pre-7, Clang pre-4, and ICC pre-18

#endif  // __AVX512F__
