// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_CAST_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_CAST_H_

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"

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
using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;

YNN_ALWAYS_INLINE s16x8 cast(s32x8 a, int16_t) {
  return s16x8{_mm_packs_epi32(lo(a).v, hi(a).v)};
}

YNN_ALWAYS_INLINE s8x16 cast(s16x16 a, int8_t) {
  return s8x16{_mm_packs_epi16(lo(a).v, hi(a).v)};
}

YNN_ALWAYS_INLINE u8x16 cast(s16x16 a, uint8_t) {
  return u8x16{_mm_packus_epi16(lo(a).v, hi(a).v)};
}

YNN_ALWAYS_INLINE s16x8 cast(f32x8 f, int16_t) {
  const s32x4 i0 = cast(lo(f), int32_t());
  const s32x4 i1 = cast(hi(f), int32_t());
  return cast(s32x8(i0, i1), int16_t());
}

YNN_ALWAYS_INLINE s8x16 cast(f32x16 f, int8_t) {
  const s16x8 i01 = cast(lo(f), int16_t());
  const s16x8 i23 = cast(hi(f), int16_t());
  return cast(s16x16(i01, i23), int8_t());
}

YNN_ALWAYS_INLINE u8x16 cast(f32x16 f, uint8_t) {
  const s32x4 i0 = cast(lo(lo(f)), int32_t());
  const s32x4 i1 = cast(hi(lo(f)), int32_t());
  const s32x4 i2 = cast(lo(hi(f)), int32_t());
  const s32x4 i3 = cast(hi(hi(f)), int32_t());
  const __m128i i01_16 = _mm_packs_epi32(i0.v, i1.v);
  const __m128i i23_16 = _mm_packs_epi32(i2.v, i3.v);
  return u8x16{_mm_packus_epi16(i01_16, i23_16)};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_CAST_H_
