// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_

#include <cstdint>
#include <limits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx2_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_avx_partial_load_store.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_partial_load_store.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;
using s32x32 = vec<int32_t, 32>;
using f32x32 = vec<float, 32>;
using s16x32 = vec<int16_t, 32>;

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  return {
      s32x8{_mm256_cvtepi8_epi32(a.v)},
      s32x8{_mm256_cvtepi8_epi32(_mm_srli_si128(a.v, 8))},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  return {
      s32x8{_mm256_cvtepu8_epi32(a.v)},
      s32x8{_mm256_cvtepu8_epi32(_mm_srli_si128(a.v, 8))},
  };
}

YNN_ALWAYS_INLINE f32x8 cast(s32x8 x, float) {
  return f32x8{_mm256_cvtepi32_ps(x.v)};
}

YNN_ALWAYS_INLINE s32x8 cast(f32x8 x, int32_t) {
  return s32x8{_mm256_cvttps_epi32(x.v)};
}

YNN_ALWAYS_INLINE bf16x16 cast(f32x16 a, bfloat16) {
  const __m256 rounding_multiplier =
      _mm256_set1_ps(bfloat16::rounding_multiplier);
  const __m256 b1 = _mm256_mul_ps(a.lo().v, rounding_multiplier);
  const __m256 b2 = _mm256_mul_ps(a.hi().v, rounding_multiplier);
  const __m256i c1 = _mm256_srli_epi32(_mm256_castps_si256(b1), 16);
  const __m256i c2 = _mm256_srli_epi32(_mm256_castps_si256(b2), 16);
  const __m256i d = _mm256_packus_epi32(c1, c2);
  return bf16x16{_mm256_permute4x64_epi64(d, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s16x16 saturate_cast(s32x16 a, int16_t) {
  const __m256i r = _mm256_packs_epi32(a.lo().v, a.hi().v);
  return s16x16{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s8x32 saturate_cast(s16x32 a, int8_t) {
  const __m256i r = _mm256_packs_epi16(a.lo().v, a.hi().v);
  return s8x32{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE u8x32 saturate_cast(s16x32 a, uint8_t) {
  const __m256i r = _mm256_packus_epi16(a.lo().v, a.hi().v);
  return u8x32{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s16x16 round_float_to_int(f32x16 f, int16_t) {
  const __m256 max_int16 = _mm256_set1_ps((float)((1 << 15) - 1));
  const __m256i i0 = _mm256_cvtps_epi32(_mm256_min_ps(f.lo().v, max_int16));
  const __m256i i1 = _mm256_cvtps_epi32(_mm256_min_ps(f.hi().v, max_int16));
  return saturate_cast(s32x16(s32x8(i0), s32x8(i1)), int16_t());
}

YNN_ALWAYS_INLINE s8x32 round_float_to_int(f32x32 f, int8_t) {
  const s16x16 i01 =
      round_float_to_int(f32x16(f.lo().lo(), f.lo().hi()), int16_t());
  const s16x16 i23 =
      round_float_to_int(f32x16(f.hi().lo(), f.hi().hi()), int16_t());
  return saturate_cast(s16x32(i01, i23), int8_t());
}

YNN_ALWAYS_INLINE u8x32 round_float_to_int(f32x32 f, uint8_t) {
  const __m256 max_uint16 = _mm256_set1_ps((float)((1 << 16) - 1));
  const __m256i i0 =
      _mm256_cvtps_epi32(_mm256_min_ps(f.lo().lo().v, max_uint16));
  const __m256i i1 =
      _mm256_cvtps_epi32(_mm256_min_ps(f.lo().hi().v, max_uint16));
  const __m256i i2 =
      _mm256_cvtps_epi32(_mm256_min_ps(f.hi().lo().v, max_uint16));
  const __m256i i3 =
      _mm256_cvtps_epi32(_mm256_min_ps(f.hi().hi().v, max_uint16));
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  const __m256i i23_16 = _mm256_packs_epi32(i2, i3);
  const __m256i r = _mm256_packus_epi16(i01_16, i23_16);
  return u8x32{_mm256_permutevar8x32_epi32(
      r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))};
}

YNN_ALWAYS_INLINE f32x8 floor_log2(f32x8 a) {
  __m256 sign_mask = _mm256_set1_ps(-0.0f);
  __m256 is_zero = _mm256_cmp_ps(a.v, _mm256_setzero_ps(), _CMP_EQ_OQ);
  a.v = _mm256_or_ps(_mm256_and_ps(is_zero, sign_mask), a.v);

  __m256i sign_and_exp_mask = _mm256_set1_epi32(0xFF800000);
  __m256i exp = _mm256_and_si256(_mm256_castps_si256(a.v), sign_and_exp_mask);

  __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256 is_inf = _mm256_cmp_ps(a.v, infinity, _CMP_EQ_OQ);

  exp = _mm256_srai_epi32(exp, 8);

  __m256 bias_256 = _mm256_set1_ps(256.0f);
  __m256 bias_383 = _mm256_set1_ps(383.0f);
  __m256 res =
      _mm256_sub_ps(_mm256_or_ps(bias_256, _mm256_castsi256_ps(exp)), bias_383);
  return f32x8{_mm256_blendv_ps(res, infinity, is_inf)};
}
YNN_ALWAYS_INLINE f64x2 floor_log2(f64x2 a) {
  double a0 = _mm_cvtsd_f64(a.v);
  double a1 = _mm_cvtsd_f64(_mm_shuffle_pd(a.v, a.v, _MM_SHUFFLE2(1, 1)));
  return f64x2(_mm_set_pd(ynn::floor_log2(a1), ynn::floor_log2(a0)));
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_
