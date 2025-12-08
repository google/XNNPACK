// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_ONLY_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_ONLY_H_

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_sse2.h"

namespace ynn {

namespace simd {

using f32x4x2 = multi_vec<f32x4, 2>;
YNN_ALWAYS_INLINE f32x4x2 convert(bf16x8 a, float) {
  f32x4x2 result;
  __m128i zero = _mm_setzero_si128();
  __m128i lo = _mm_unpacklo_epi16(a.v, zero);
  __m128i hi = _mm_unpackhi_epi16(a.v, zero);
  result.v[0].v = _mm_castsi128_ps(_mm_slli_epi32(lo, 16));
  result.v[1].v = _mm_castsi128_ps(_mm_slli_epi32(hi, 16));
  return result;
}

using s32x4x4 = multi_vec<s32x4, 4>;
YNN_ALWAYS_INLINE s32x4x4 convert(s8x16 a, int32_t) {
  s32x4x4 result;
  __m128i lo = _mm_unpacklo_epi8(a.v, a.v);
  __m128i hi = _mm_unpackhi_epi8(a.v, a.v);

  result.v[0].v = _mm_srai_epi32(_mm_unpacklo_epi16(lo, lo), 24);
  result.v[1].v = _mm_srai_epi32(_mm_unpackhi_epi16(lo, lo), 24);
  result.v[2].v = _mm_srai_epi32(_mm_unpacklo_epi16(hi, hi), 24);
  result.v[3].v = _mm_srai_epi32(_mm_unpackhi_epi16(hi, hi), 24);
  return result;
}
YNN_ALWAYS_INLINE s32x4x4 convert(u8x16 a, int32_t) {
  s32x4x4 result;
  const __m128i zero = _mm_setzero_si128();
  __m128i lo = _mm_unpacklo_epi8(a.v, zero);
  __m128i hi = _mm_unpackhi_epi8(a.v, zero);

  result.v[0].v = _mm_unpacklo_epi16(lo, zero);
  result.v[1].v = _mm_unpackhi_epi16(lo, zero);
  result.v[2].v = _mm_unpacklo_epi16(hi, zero);
  result.v[3].v = _mm_unpackhi_epi16(hi, zero);
  return result;
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_ONLY_H_
