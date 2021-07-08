// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/common.h>


#if defined(__SSE2__)
#include <emmintrin.h>

// GCC pre-11, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, and ICC pre-16
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && __GNUC__ < 11) || \
    (defined(__clang__) && !defined(__apple_build_version__) && (__clang_major__ < 8)) || \
    (defined(__clang__) && defined(__ANDROID__) && (__clang_major__ == 8) && (__clang_minor__ == 0) && (__clang_patchlevel__ < 7)) || \
    (defined(__clang__) && defined(__apple_build_version__) && (__apple_build_version__ < 11000000)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1600))

static XNN_INTRINSIC
__m128i _mm_loadu_si32(const void* address) {
  return _mm_cvtsi32_si128(*((const int*) address));
}

static XNN_INTRINSIC
void _mm_storeu_si32(const void* address, __m128i v) {
  *((int*) address) = _mm_cvtsi128_si32(v);
}
#endif  // GCC pre-11, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, and ICC pre-16
#endif  // SSE2

#ifdef __AVX512F__
#include <immintrin.h>

// GCC pre-7, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, and ICC pre-18
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 7)) || \
    (defined(__clang__) && !defined(__apple_build_version__) && (__clang_major__ < 8)) || \
    (defined(__clang__) && defined(__ANDROID__) && (__clang_major__ == 8) && (__clang_minor__ == 0) && (__clang_patchlevel__ < 7)) || \
    (defined(__clang__) && defined(__apple_build_version__) && (__apple_build_version__ < 11000000)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1800)) || \
    (defined(_MSC_VER) && !defined(__clang__) && !defined(__GNUC__))

static XNN_INTRINSIC
__mmask16 _cvtu32_mask16(unsigned int mask) {
  return (__mmask16) mask;
}

static XNN_INTRINSIC
__mmask64 _cvtu64_mask64(unsigned long long mask) {
  return (__mmask64) mask;
}

static XNN_INTRINSIC
__mmask64 _kshiftli_mask64(__mmask64 a, unsigned int count) {
  return (__mmask64) ((unsigned long long) a << count);
}

static XNN_INTRINSIC
__mmask64 _kshiftri_mask64(__mmask64 a, unsigned int count) {
  return (__mmask64) ((unsigned long long) a >> count);
}

#endif  // GCC pre-7, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, and ICC pre-18

// GCC pre-7, Clang pre-4, and ICC pre-18
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 7)) || \
    (defined(__clang__) && (__clang_major__ < 4)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1800))

static XNN_INTRINSIC
float _mm512_reduce_add_ps(__m512 v) {
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

static XNN_INTRINSIC
float _mm512_reduce_max_ps(__m512 v) {
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

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 9)
static XNN_INTRINSIC
__m512i _mm512_set_epi8(
  char e63, char e62, char e61, char e60,
  char e59, char e58, char e57, char e56,
  char e55, char e54, char e53, char e52,
  char e51, char e50, char e49, char e48,
  char e47, char e46, char e45, char e44,
  char e43, char e42, char e41, char e40,
  char e39, char e38, char e37, char e36,
  char e35, char e34, char e33, char e32,
  char e31, char e30, char e29, char e28,
  char e27, char e26, char e25, char e24,
  char e23, char e22, char e21, char e20,
  char e19, char e18, char e17, char e16,
  char e15, char e14, char e13, char e12,
  char e11, char e10, char e09, char e08,
  char e07, char e06, char e05, char e04,
  char e03, char e02, char e01, char e00)
{
  return (__m512i) (__v64qi) {
    e00, e01, e02, e03, e04, e05, e06, e07,
    e08, e09, e10, e11, e12, e13, e14, e15,
    e16, e17, e18, e19, e20, e21, e22, e23,
    e24, e25, e26, e27, e28, e29, e30, e31,
    e32, e33, e34, e35, e36, e37, e38, e39,
    e40, e41, e42, e43, e44, e45, e46, e47,
    e48, e49, e50, e51, e52, e53, e54, e55,
    e56, e57, e58, e59, e60, e61, e62, e63
  };
}
#endif  // GCC pre-9

// AArch32 GCC, see
// - https://gcc.gnu.org/bugzilla/show_bug.cgi?id=71233
// - https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95399
#if XNN_ARCH_ARM && defined(__GNUC__) && !defined(__clang__) && defined(__ARM_NEON__)
#include <arm_neon.h>

static XNN_INTRINSIC
int32x4_t vcvtnq_s32_f32(float32x4_t v) {
  return vcvtq_s32_f32(vrndnq_f32(v));
}
#endif  // AArch32 GCC

#endif  // __AVX512F__
