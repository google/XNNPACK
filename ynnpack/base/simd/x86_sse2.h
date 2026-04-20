// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_H_

#include <cstdint>
#include <limits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_sse2_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_partial_load_store.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_saturate_cast.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x8 = vec<float, 8>;
using s32x8 = vec<int32_t, 8>;
using s16x16 = vec<int16_t, 16>;
using bf16x16 = vec<bfloat16, 16>;
using f16x16 = vec<half, 16>;
using s8x32 = vec<int8_t, 32>;
using u8x32 = vec<uint8_t, 32>;
using f64x4 = vec<double, 4>;

using s32x16 = vec<int32_t, 16>;

YNN_ALWAYS_INLINE f32x8 cast(bf16x8 b, float) {
  __m128i zero = _mm_setzero_si128();

  return {
      f32x4{_mm_castsi128_ps(_mm_unpacklo_epi16(zero, b.v))},
      f32x4{_mm_castsi128_ps(_mm_unpackhi_epi16(zero, b.v))},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  __m128i i8_lo = _mm_unpacklo_epi8(a.v, a.v);
  __m128i i8_hi = _mm_unpackhi_epi8(a.v, a.v);

  return {
      {s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_lo, i8_lo), 24)},
       s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_lo, i8_lo), 24)}},
      {s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_hi, i8_hi), 24)},
       s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_hi, i8_hi), 24)}},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  const __m128i zero = _mm_setzero_si128();
  __m128i i16_lo = _mm_unpacklo_epi8(a.v, zero);
  __m128i i16_hi = _mm_unpackhi_epi8(a.v, zero);

  return {
      {s32x4{_mm_unpacklo_epi16(i16_lo, zero)},
       s32x4{_mm_unpackhi_epi16(i16_lo, zero)}},
      {s32x4{_mm_unpacklo_epi16(i16_hi, zero)},
       s32x4{_mm_unpackhi_epi16(i16_hi, zero)}},
  };
}

YNN_ALWAYS_INLINE f64x4 cast(f32x4 x, double) {
  return {f64x2{_mm_cvtps_pd(x.v)},
          f64x2{_mm_cvtps_pd(_mm_movehl_ps(x.v, x.v))}};
}
YNN_ALWAYS_INLINE f32x4 cast(f64x4 x, float) {
  return f32x4{_mm_movelh_ps(_mm_cvtpd_ps(x[0].v), _mm_cvtpd_ps(x[1].v))};
}

YNN_ALWAYS_INLINE f32x4 floor_log2(f32x4 a) {
  __m128 sign_mask = _mm_set1_ps(-0.0f);
  __m128 is_zero = _mm_cmpeq_ps(a.v, _mm_setzero_ps());
  a.v = _mm_or_ps(_mm_and_ps(is_zero, sign_mask), a.v);

  __m128i sign_and_exp_mask = _mm_set1_epi32(0xFF800000);
  __m128i exp = _mm_and_si128(_mm_castps_si128(a.v), sign_and_exp_mask);

  __m128 infinity = _mm_set1_ps(std::numeric_limits<float>::infinity());
  __m128 is_inf = _mm_cmpeq_ps(a.v, infinity);

  exp = _mm_srai_epi32(exp, 8);

  __m128 bias_256 = _mm_set1_ps(256.0f);
  __m128 bias_383 = _mm_set1_ps(383.0f);
  __m128 res = _mm_sub_ps(_mm_or_ps(bias_256, _mm_castsi128_ps(exp)), bias_383);
  return f32x4{
      _mm_or_ps(_mm_andnot_ps(is_inf, res), _mm_and_ps(is_inf, infinity))};
}
YNN_ALWAYS_INLINE f64x2 floor_log2(f64x2 a) {
  double a0 = _mm_cvtsd_f64(a.v);
  double a1 = _mm_cvtsd_f64(_mm_shuffle_pd(a.v, a.v, _MM_SHUFFLE2(1, 1)));
  return f64x2(_mm_set_pd(ynn::floor_log2(a1), ynn::floor_log2(a0)));
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE_H_
