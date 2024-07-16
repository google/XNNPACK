// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_H_

#include "xnnpack/simd/f32-avx-base.h"
#include "xnnpack/common.h"

// Whether or not this architecture has native fused multiply-add support.
#define XNN_SIMD_HAS_NATIVE_FMA 0

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}

// This intrinsic is not particularly efficient. If you really need this
// intrinsic, consider using `avx2` instead.
static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  __m128i a_lo = _mm256_castsi256_si128(_mm256_castps_si256(a));
  __m128i a_hi = _mm256_extractf128_si256(_mm256_castps_si256(a), 1);
  __m128i res_lo = _mm_slli_epi32(a_lo, bits);
  __m128i res_hi = _mm_slli_epi32(a_hi, bits);
  return _mm256_castsi256_ps(
      _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1));
}

// This intrinsic is not particularly efficient. If you really need this
// intrinsic, consider using `avx2` instead.
static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  __m128i a_lo = _mm256_castsi256_si128(_mm256_castps_si256(a));
  __m128i a_hi = _mm256_extractf128_si256(_mm256_castps_si256(a), 1);
  __m128i res_lo = _mm_srli_epi32(a_lo, bits);
  __m128i res_hi = _mm_srli_epi32(a_hi, bits);
  return _mm256_castsi256_ps(
      _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1));
}

// This intrinsic is not particularly efficient. If you really need this
// intrinsic, consider using `avx2` instead.
static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  __m128i a_lo = _mm256_castsi256_si128(_mm256_castps_si256(a));
  __m128i a_hi = _mm256_extractf128_si256(_mm256_castps_si256(a), 1);
  __m128i res_lo = _mm_srai_epi32(a_lo, bits);
  __m128i res_hi = _mm_srai_epi32(a_hi, bits);
  return _mm256_castsi256_ps(
      _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_H_
