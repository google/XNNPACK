// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_BASE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_BASE_H_

#include <immintrin.h>

#include <cassert>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

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

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_BASE_H_
