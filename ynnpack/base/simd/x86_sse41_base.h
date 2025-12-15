// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_BASE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_BASE_H_

#include <immintrin.h>

#include <cassert>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/x86_sse2_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

YNN_ALWAYS_INLINE s32x4& operator*=(s32x4& a, s32x4 b) {
  a.v = _mm_mullo_epi32(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s32x4 operator*(s32x4 a, s32x4 b) { return a *= b; }

YNN_ALWAYS_INLINE s8x16 min(s8x16 a, s8x16 b) {
  return s8x16{_mm_min_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 max(s8x16 a, s8x16 b) {
  return s8x16{_mm_max_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 min(s32x4 a, s32x4 b) {
  return s32x4{_mm_min_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 max(s32x4 a, s32x4 b) {
  return s32x4{_mm_max_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE u8x16 abs(s8x16 a) { return u8x16{_mm_abs_epi8(a.v)}; }

YNN_ALWAYS_INLINE int8_t horizontal_max(s8x16 a) {
  const __m128i max8 = _mm_max_epi8(a.v, _mm_srli_si128(a.v, 8));
  const __m128i max4 = _mm_max_epi8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epi8(max4, _mm_srli_si128(max4, 2));
  const __m128i max1 = _mm_max_epi8(max2, _mm_srli_si128(max2, 1));
  return static_cast<int8_t>(_mm_cvtsi128_si32(max1));
}
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x16 a) {
  const __m128i min8 = _mm_min_epi8(a.v, _mm_srli_si128(a.v, 8));
  const __m128i min4 = _mm_min_epi8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epi8(min4, _mm_srli_si128(min4, 2));
  const __m128i min1 = _mm_min_epi8(min2, _mm_srli_si128(min2, 1));
  return static_cast<int8_t>(_mm_cvtsi128_si32(min1));
}
YNN_ALWAYS_INLINE int32_t horizontal_max(s32x4 a) {
  const __m128i max4 = _mm_max_epi32(a.v, _mm_srli_si128(a.v, 8));
  return _mm_cvtsi128_si32(_mm_max_epi32(max4, _mm_srli_si128(max4, 4)));
}
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x4 a) {
  const __m128i min4 = _mm_min_epi32(a.v, _mm_srli_si128(a.v, 8));
  return _mm_cvtsi128_si32(_mm_min_epi32(min4, _mm_srli_si128(min4, 4)));
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_BASE_H_
