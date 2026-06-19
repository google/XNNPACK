// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_FMA_H_
#define XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_FMA_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/f32-sse2-base.h"  // IWYU pragma: export

// Whether or not this architecture has native fused multiply-add support.
// Here, we do not, but we are emulating it for numerical consistency with FMA
// targets.
#define XNN_SIMD_HAS_NATIVE_FMA 1

// Implementation of https://guillaume.melquiond.fr/doc/08-tc.pdf (with help
// from
// https://drilian.com/posts/2024.12.31-emulating-the-fmadd-instruction-part-1-32-bit-floats/)
// to emulate exact fused multiply-add results for 32-bit floats.
static XNN_INLINE __m128d xnn_round_to_odd_f64(__m128d value, __m128d error) {
  // We want to add (if error is > 0) or subtract one (otherwise) from the
  // mantissa, if the mantissa of the value is even.
  const __m128d lowest_bit = _mm_castsi128_pd(_mm_set1_epi64x(0x1));
  const __m128d nonsign_mask =
      _mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffull));
  const __m128d inf = _mm_castsi128_pd(_mm_set1_epi64x(0x7ff0000000000000ull));

  // This value is a 1 if the mantissa of value is even, or 0 otherwise.
  const __m128d odd = _mm_andnot_pd(value, lowest_bit);

  // Get the sign of the error.
  const __m128i sign = _mm_castpd_si128(_mm_cmplt_pd(error, _mm_setzero_pd()));
  __m128i adjustment = _mm_xor_si128(sign, _mm_castpd_si128(odd));
  adjustment = _mm_sub_epi64(adjustment, sign);

  // Mask the adjustment if we have no error or the value is inf/nan.
  const __m128i has_error =
      _mm_castpd_si128(_mm_cmpneq_pd(error, _mm_setzero_pd()));
  const __m128i not_inf_or_nan =
      _mm_castpd_si128(_mm_cmplt_pd(_mm_and_pd(value, nonsign_mask), inf));
  adjustment = _mm_and_si128(adjustment, has_error);
  adjustment = _mm_and_si128(adjustment, not_inf_or_nan);
  return _mm_castsi128_pd(_mm_add_epi64(adjustment, _mm_castpd_si128(value)));
}

static XNN_INLINE __m128d xnn_add_with_error_f64(__m128d a, __m128d b,
                                                 __m128d* error) {
  __m128d sum = _mm_add_pd(a, b);
  __m128d residual = _mm_sub_pd(sum, a);
  __m128d err1 = _mm_sub_pd(b, residual);
  __m128d err2 = _mm_sub_pd(a, _mm_sub_pd(sum, residual));
  *error = _mm_add_pd(err1, err2);
  return sum;
}

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
  __m128d error_lo, error_hi;
  __m128d sum_lo = xnn_add_with_error_f64(product_lo, c_lo, &error_lo);
  __m128d sum_hi = xnn_add_with_error_f64(product_hi, c_hi, &error_hi);
  sum_lo = xnn_round_to_odd_f64(sum_lo, error_lo);
  sum_hi = xnn_round_to_odd_f64(sum_hi, error_hi);
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

#endif  // XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_FMA_H_
