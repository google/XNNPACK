// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_avx_partial_load_store.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_partial_load_store.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;
using bf16x32 = vec<bfloat16, 32>;
using f16x32 = vec<half, 32>;
using s8x64 = vec<int8_t, 64>;
using u8x64 = vec<uint8_t, 64>;

YNN_ALWAYS_INLINE f64x4 cast(f32x4 a, double) {
  return f64x4{_mm256_cvtps_pd(a.v)};
}
YNN_ALWAYS_INLINE f32x4 cast(f64x4 a, float) {
  return f32x4{_mm256_cvtpd_ps(a.v)};
}

#ifdef __F16C__
YNN_ALWAYS_INLINE f32x8 cast(f16x8 a, float) {
  return f32x8{_mm256_cvtph_ps(a.v)};
}
YNN_ALWAYS_INLINE f16x8 cast(f32x8 a, half) {
  return f16x8{_mm256_cvtps_ph(a.v, _MM_FROUND_TO_NEAREST_INT)};
}
#endif  // __F16C__

YNN_ALWAYS_INLINE s32x8 select(s32x8 cond, s32x8 a, s32x8 b) {
  __m256 mc = _mm256_castsi256_ps(cond.v);
  __m256 ma = _mm256_castsi256_ps(a.v);
  __m256 mb = _mm256_castsi256_ps(b.v);
  return s32x8{_mm256_castps_si256(
      _mm256_or_ps(_mm256_and_ps(mc, ma), _mm256_andnot_ps(mc, mb)))};
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
  __m128d sign_mask = _mm_set1_pd(-0.0);
  __m128d is_zero = _mm_cmpeq_pd(a.v, _mm_setzero_pd());
  a.v = _mm_or_pd(_mm_and_pd(is_zero, sign_mask), a.v);

  __m128i sign_and_exp_mask = _mm_set1_epi64x(0xFFF0000000000000);
  __m128i exp = _mm_and_si128(_mm_castpd_si128(a.v), sign_and_exp_mask);

  __m128d infinity = _mm_set1_pd(std::numeric_limits<double>::infinity());
  __m128d is_inf = _mm_cmpeq_pd(a.v, infinity);

  exp = _mm_srai_epi32(exp, 11);

  __m128d bias_2048 = _mm_set1_pd(2048.0);
  __m128d bias_3071 = _mm_set1_pd(3071.0);
  __m128d res =
      _mm_sub_pd(_mm_or_pd(bias_2048, _mm_castsi128_pd(exp)), bias_3071);
  return f64x2{
      _mm_or_pd(_mm_andnot_pd(is_inf, res), _mm_and_pd(is_inf, infinity))};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
