// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_BASE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_BASE_H_

#include <immintrin.h>

#include <cassert>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

namespace internal {

YNN_ALWAYS_INLINE s32x8 unpacklo(s32x8 a, s32x8 b) {
  return s32x8{_mm256_unpacklo_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x8 unpackhi(s32x8 a, s32x8 b) {
  return s32x8{_mm256_unpackhi_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE s8x32 unpacklo(s8x32 a, s8x32 b) {
  return s8x32{_mm256_unpacklo_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 unpackhi(s8x32 a, s8x32 b) {
  return s8x32{_mm256_unpackhi_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE u8x32 unpacklo(u8x32 a, u8x32 b) {
  return u8x32{_mm256_unpacklo_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 unpackhi(u8x32 a, u8x32 b) {
  return u8x32{_mm256_unpackhi_epi8(a.v, b.v)};
}

}  // namespace internal

YNN_ALWAYS_INLINE s32x8& operator+=(s32x8& a, s32x8 b) {
  a.v = _mm256_add_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x32& operator+=(s8x32& a, s8x32 b) {
  a.v = _mm256_add_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x32& operator+=(u8x32& a, u8x32 b) {
  a.v = _mm256_add_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s32x8& operator-=(s32x8& a, s32x8 b) {
  a.v = _mm256_sub_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x32& operator-=(s8x32& a, s8x32 b) {
  a.v = _mm256_sub_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x32& operator-=(u8x32& a, u8x32 b) {
  a.v = _mm256_sub_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s32x8& operator*=(s32x8& a, s32x8 b) {
  a.v = _mm256_mullo_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x8 operator+(s32x8 a, s32x8 b) { return a += b; }
YNN_ALWAYS_INLINE s8x32 operator+(s8x32 a, s8x32 b) { return a += b; }
YNN_ALWAYS_INLINE u8x32 operator+(u8x32 a, u8x32 b) { return a += b; }

YNN_ALWAYS_INLINE s32x8 operator-(s32x8 a, s32x8 b) { return a -= b; }
YNN_ALWAYS_INLINE s8x32 operator-(s8x32 a, s8x32 b) { return a -= b; }
YNN_ALWAYS_INLINE u8x32 operator-(u8x32 a, u8x32 b) { return a -= b; }

YNN_ALWAYS_INLINE s32x8 operator*(s32x8 a, s32x8 b) { return a *= b; }

YNN_ALWAYS_INLINE s16x16 operator>>(s16x16 a, int b) {
  return s16x16{_mm256_srai_epi16(a.v, b)};
}

YNN_ALWAYS_INLINE s32x8 min(s32x8 a, s32x8 b) {
  return s32x8{_mm256_min_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x16 min(s16x16 a, s16x16 b) {
  return s16x16{_mm256_min_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 min(s8x32 a, s8x32 b) {
  return s8x32{_mm256_min_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 min(u8x32 a, u8x32 b) {
  return u8x32{_mm256_min_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x8 max(s32x8 a, s32x8 b) {
  return s32x8{_mm256_max_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x16 max(s16x16 a, s16x16 b) {
  return s16x16{_mm256_max_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 max(s8x32 a, s8x32 b) {
  return s8x32{_mm256_max_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 max(u8x32 a, u8x32 b) {
  return u8x32{_mm256_max_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE u8x32 abs(s8x32 a) { return u8x32{_mm256_abs_epi8(a.v)}; }

YNN_ALWAYS_INLINE f32x8 convert(bf16x8 a, float) {
  return f32x8{_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a.v),
      16))};
}

YNN_ALWAYS_INLINE int16_t horizontal_max(s16x16 a) {
  const __m128i max8_vals = _mm_max_epi16(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i max4_vals =
      _mm_max_epi16(max8_vals, _mm_srli_si128(max8_vals, 8));
  const __m128i max2_vals =
      _mm_max_epi16(max4_vals, _mm_srli_si128(max4_vals, 4));
  return static_cast<int16_t>(_mm_cvtsi128_si32(
      _mm_max_epi16(max2_vals, _mm_srli_si128(max2_vals, 2))));
}
YNN_ALWAYS_INLINE int16_t horizontal_min(s16x16 a) {
  const __m128i min8_vals = _mm_min_epi16(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i min4_vals =
      _mm_min_epi16(min8_vals, _mm_srli_si128(min8_vals, 8));
  const __m128i min2_vals =
      _mm_min_epi16(min4_vals, _mm_srli_si128(min4_vals, 4));
  return static_cast<int16_t>(_mm_cvtsi128_si32(
      _mm_min_epi16(min2_vals, _mm_srli_si128(min2_vals, 2))));
}

YNN_ALWAYS_INLINE int32_t horizontal_max(s32x8 a) {
  const __m128i max8_vals = _mm_max_epi32(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i max4_vals =
      _mm_max_epi32(max8_vals, _mm_srli_si128(max8_vals, 8));
  return _mm_cvtsi128_si32(
      _mm_max_epi32(max4_vals, _mm_srli_si128(max4_vals, 4)));
}
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x8 a) {
  const __m128i min8_vals = _mm_min_epi32(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i min4_vals =
      _mm_min_epi32(min8_vals, _mm_srli_si128(min8_vals, 8));
  return static_cast<int32_t>(_mm_cvtsi128_si32(
      _mm_min_epi32(min4_vals, _mm_srli_si128(min4_vals, 4))));
}

YNN_ALWAYS_INLINE int8_t horizontal_max(s8x32 a) {
  const __m128i max16 = _mm_max_epi8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i max8 = _mm_max_epi8(max16, _mm_srli_si128(max16, 8));
  const __m128i max4 = _mm_max_epi8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epi8(max4, _mm_srli_si128(max4, 2));
  return static_cast<int8_t>(
      _mm_cvtsi128_si32(_mm_max_epi8(max2, _mm_srli_si128(max2, 1))));
}
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x32 a) {
  const __m128i min16 = _mm_min_epi8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i min8 = _mm_min_epi8(min16, _mm_srli_si128(min16, 8));
  const __m128i min4 = _mm_min_epi8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epi8(min4, _mm_srli_si128(min4, 2));
  return static_cast<int8_t>(
      _mm_cvtsi128_si32(_mm_min_epi8(min2, _mm_srli_si128(min2, 1))));
}

YNN_ALWAYS_INLINE uint8_t horizontal_max(u8x32 a) {
  const __m128i max16 = _mm_max_epu8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i max8 = _mm_max_epu8(max16, _mm_srli_si128(max16, 8));
  const __m128i max4 = _mm_max_epu8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epu8(max4, _mm_srli_si128(max4, 2));
  return static_cast<uint8_t>(
      _mm_cvtsi128_si32(_mm_max_epu8(max2, _mm_srli_si128(max2, 1))));
}
YNN_ALWAYS_INLINE uint8_t horizontal_min(u8x32 a) {
  const __m128i min16 = _mm_min_epu8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i min8 = _mm_min_epu8(min16, _mm_srli_si128(min16, 8));
  const __m128i min4 = _mm_min_epu8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epu8(min4, _mm_srli_si128(min4, 2));
  return static_cast<uint8_t>(
      _mm_cvtsi128_si32(_mm_min_epu8(min2, _mm_srli_si128(min2, 1))));
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_BASE_H_
