// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX2_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX2_H_

#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/simd/f32-avx-base.h"

// Whether or not this architecture has native fused multiply-add support.
#ifdef __FMA3__
#define XNN_SIMD_HAS_NATIVE_FMA 1
#else
#define XNN_SIMD_HAS_NATIVE_FMA 0
#endif  // __FMA3__

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
#ifdef __FMA3__
  return _mm256_fmadd_ps(a, b, c);
#else
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif  // __FMA3__
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
#ifdef __FMA3__
  return _mm256_fnmadd_ps(a, b, c);
#else
  return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif  // __FMA3__
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
#ifdef __FMA3__
  return _mm256_fmsub_ps(a, b, c);
#else
  return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
#endif  // __FMA3__
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

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX2_H_
