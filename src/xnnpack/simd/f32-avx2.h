// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX2_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX2_H_

#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f32-avx-base.h"  // IWYU pragma: export
#include "src/xnnpack/simd/f32-common.h"

// Whether or not this architecture has native fused multiply-add support.
#ifdef __FMA__
#define XNN_SIMD_HAS_NATIVE_FMA 1
#else
#define XNN_SIMD_HAS_NATIVE_FMA 0
#endif  // __FMA__

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
#ifdef __FMA__
  return _mm256_fmadd_ps(a, b, c);
#else
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif  // __FMA__
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
#ifdef __FMA__
  return _mm256_fnmadd_ps(a, b, c);
#else
  return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif  // __FMA__
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
#ifdef __FMA__
  return _mm256_fmsub_ps(a, b, c);
#else
  return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
#endif  // __FMA__
}

static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_castps_si256(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return _mm256_castsi256_ps(
      _mm256_cmpeq_epi32(_mm256_castps_si256(a), _mm256_castps_si256(b)));
}

// This is not faster than _mm256_rsqrt_ps, but it is numerically consistent
// across all CPUs.
static XNN_INLINE xnn_simd_f32_t
xnn_approx_reciprocal_sqrt_f32(xnn_simd_f32_t a) {
  const __m256i magic =
      _mm256_load_si256((const __m256i*)approx_reciprocal_sqrt_magic);
  return _mm256_castsi256_ps(
      _mm256_sub_epi32(magic, _mm256_srai_epi32(_mm256_castps_si256(a), 1)));
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX2_H_
