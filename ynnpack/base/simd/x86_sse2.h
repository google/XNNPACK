// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_H_

#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_sse2_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x8 = multi_vec<f32x4, 2>;
using s32x8 = multi_vec<s32x4, 2>;
using s16x16 = multi_vec<s16x8, 2>;
using bf16x16 = multi_vec<bf16x8, 2>;
using f16x16 = multi_vec<f16x8, 2>;
using s8x32 = multi_vec<s8x16, 2>;
using u8x32 = multi_vec<u8x16, 2>;

using s32x16 = multi_vec<s32x4, 4>;

YNN_ALWAYS_INLINE f32x8 convert(bf16x8 b, float) {
  __m128i zero = _mm_setzero_si128();

  return {
      f32x4{_mm_castsi128_ps(_mm_unpacklo_epi16(zero, b.v))},
      f32x4{_mm_castsi128_ps(_mm_unpackhi_epi16(zero, b.v))},
  };
}

YNN_ALWAYS_INLINE s32x16 convert(s8x16 a, int32_t) {
  __m128i i8_lo = _mm_unpacklo_epi8(a.v, a.v);
  __m128i i8_hi = _mm_unpackhi_epi8(a.v, a.v);

  return {
      s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_lo, i8_lo), 24)},
      s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_lo, i8_lo), 24)},
      s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_hi, i8_hi), 24)},
      s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_hi, i8_hi), 24)},
  };
}

YNN_ALWAYS_INLINE s32x16 convert(u8x16 a, int32_t) {
  const __m128i zero = _mm_setzero_si128();
  __m128i i16_lo = _mm_unpacklo_epi8(a.v, zero);
  __m128i i16_hi = _mm_unpackhi_epi8(a.v, zero);

  return {
      s32x4{_mm_unpacklo_epi16(i16_lo, zero)},
      s32x4{_mm_unpackhi_epi16(i16_lo, zero)},
      s32x4{_mm_unpacklo_epi16(i16_hi, zero)},
      s32x4{_mm_unpackhi_epi16(i16_hi, zero)},
  };
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE_H_
