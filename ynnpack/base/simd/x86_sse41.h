// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_H_

#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_sse2_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_partial_load_store.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_cast.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse41_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using s32x16 = vec<int32_t, 16>;

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  return {
      {s32x4{_mm_cvtepi8_epi32(a.v)},
       s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 4))}},
      {s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 8))},
       s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 12))}},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  return {
      {s32x4{_mm_cvtepu8_epi32(a.v)},
       s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 4))}},
      {s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 8))},
       s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 12))}},
  };
}

YNN_ALWAYS_INLINE f32x4 select(s32x4 cond, f32x4 a, f32x4 b) {
  return f32x4{_mm_blendv_ps(b.v, a.v, _mm_castsi128_ps(cond.v))};
}
YNN_ALWAYS_INLINE s32x4 select(s32x4 cond, s32x4 a, s32x4 b) {
  return s32x4{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u32x4 select(s32x4 cond, u32x4 a, u32x4 b) {
  return u32x4{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE s16x8 select(s16x8 cond, s16x8 a, s16x8 b) {
  return s16x8{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u16x8 select(s16x8 cond, u16x8 a, u16x8 b) {
  return u16x8{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE s8x16 select(s8x16 cond, s8x16 a, s8x16 b) {
  return s8x16{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u8x16 select(s8x16 cond, u8x16 a, u8x16 b) {
  return u8x16{_mm_blendv_epi8(b.v, a.v, cond.v)};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_H_
