// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_FMA_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_FMA_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/f32-sse2-base.h"

// Whether or not this architecture has native fused multiply-add support.
// Here, we do not, but we are emulating it for numerical consistency with FMA
// targets.
#define XNN_SIMD_HAS_NATIVE_FMA 1

// TODO: Do we need to implement this, to preserve the double rounding behavior?
// https://guillaume.melquiond.fr/doc/08-tc.pdf
// https://drilian.com/posts/2024.12.31-emulating-the-fmadd-instruction-part-1-32-bit-floats/)
static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  __m128d a_lo = _mm_cvtps_pd(a);
  __m128d a_hi = _mm_cvtps_pd(_mm_movehl_ps(a, a));
  __m128d b_lo = _mm_cvtps_pd(b);
  __m128d b_hi = _mm_cvtps_pd(_mm_movehl_ps(b, b));
  __m128d c_lo = _mm_cvtps_pd(c);
  __m128d c_hi = _mm_cvtps_pd(_mm_movehl_ps(c, c));
  __m128d product_lo = _mm_mul_pd(a_lo, b_lo);
  __m128d product_hi = _mm_mul_pd(a_hi, b_hi);
  __m128d sum_lo = _mm_add_pd(product_lo, c_lo);
  __m128d sum_hi = _mm_add_pd(product_hi, c_hi);
  return _mm_movelh_ps(_mm_cvtpd_ps(sum_lo), _mm_cvtpd_ps(sum_hi));
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return xnn_fmadd_f32(xnn_neg_f32(a), b, c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return xnn_fmadd_f32(a, b, xnn_neg_f32(c));
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_FMA_H_
