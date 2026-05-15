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

YNN_ALWAYS_INLINE bf16x16 cast(f32x16 a, bfloat16) {
  __m256 nan_mask_lo = _mm256_cmp_ps(a.lo().v, a.lo().v, _CMP_UNORD_Q);
  __m256i u_lo = _mm256_castps_si256(a.lo().v);
  __m256i lsb_lo =
      _mm256_and_si256(_mm256_srli_epi32(u_lo, 16), _mm256_set1_epi32(1));
  __m256i bias_lo = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb_lo);
  __m256i res_lo = _mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(_mm256_add_epi32(u_lo, bias_lo)),
      _mm256_castsi256_ps(_mm256_or_si256(u_lo, _mm256_set1_epi32(0x00010000))),
      nan_mask_lo));
  __m256i c1 = _mm256_srli_epi32(res_lo, 16);

  __m256 nan_mask_hi = _mm256_cmp_ps(a.hi().v, a.hi().v, _CMP_UNORD_Q);
  __m256i u_hi = _mm256_castps_si256(a.hi().v);
  __m256i lsb_hi =
      _mm256_and_si256(_mm256_srli_epi32(u_hi, 16), _mm256_set1_epi32(1));
  __m256i bias_hi = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb_hi);
  __m256i res_hi = _mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(_mm256_add_epi32(u_hi, bias_hi)),
      _mm256_castsi256_ps(_mm256_or_si256(u_hi, _mm256_set1_epi32(0x00010000))),
      nan_mask_hi));
  __m256i c2 = _mm256_srli_epi32(res_hi, 16);

  const __m256i d = _mm256_packus_epi32(c1, c2);
  return bf16x16{_mm256_permute4x64_epi64(d, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s16x16 cast(s32x16 a, int16_t) {
  const __m256i r = _mm256_packs_epi32(a.lo().v, a.hi().v);
  return s16x16{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s8x32 cast(s16x32 a, int8_t) {
  const __m256i r = _mm256_packs_epi16(a.lo().v, a.hi().v);
  return s8x32{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE u8x32 cast(s16x32 a, uint8_t) {
  const __m256i r = _mm256_packus_epi16(a.lo().v, a.hi().v);
  return u8x32{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s32x8 cast(f32x8 f, int32_t) {
  const __m256 threshold = _mm256_set1_ps(2147483520.0f);
  const __m256 mask = _mm256_cmp_ps(f.v, threshold, _CMP_GT_OQ);
  const __m256 rounded =
      _mm256_round_ps(f.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m256i res = _mm256_cvttps_epi32(rounded);
  return s32x8{_mm256_blendv_epi8(res, _mm256_set1_epi32(0x7fffffff),
                                  _mm256_castps_si256(mask))};
}

YNN_ALWAYS_INLINE s16x16 cast(f32x16 f, int16_t) {
  const s32x8 i0 = cast(f.lo(), int32_t());
  const s32x8 i1 = cast(f.hi(), int32_t());
  return cast(s32x16(i0, i1), int16_t());
}

YNN_ALWAYS_INLINE s8x32 cast(f32x32 f, int8_t) {
  const s16x16 i01 = cast(f.lo(), int16_t());
  const s16x16 i23 = cast(f.hi(), int16_t());
  return cast(s16x32(i01, i23), int8_t());
}

YNN_ALWAYS_INLINE u8x32 cast(f32x32 f, uint8_t) {
  const s32x8 i0 = cast(f.lo().lo(), int32_t());
  const s32x8 i1 = cast(f.lo().hi(), int32_t());
  const s32x8 i2 = cast(f.hi().lo(), int32_t());
  const s32x8 i3 = cast(f.hi().hi(), int32_t());
  const __m256i i01_16 = _mm256_packs_epi32(i0.v, i1.v);
  const __m256i i23_16 = _mm256_packs_epi32(i2.v, i3.v);
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

YNN_ALWAYS_INLINE s8x32 cast(s2x32 from, int8_t) {
  // 1. Broadcast the 64-bit GPR directly across the entire 256-bit register.
  __m256i dup = _mm256_set1_epi64x(static_cast<long long>(from.v));

  // 2. Duplicate the bytes so each 2-bit value has its own 8-bit lane.
  const __m256i mask_dup =
      _mm256_set_epi8(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3,
                      3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  dup = _mm256_shuffle_epi8(dup, mask_dup);

  // 3. The cross-byte spill trick.
  __m256i shifted = _mm256_srli_epi32(dup, 4);
  __m256i blended = _mm256_blend_epi16(dup, shifted, 0xAA);
  __m256i masked = _mm256_and_si256(blended, _mm256_set1_epi32(0x0C030C03));

  // 4. Final sign-extension LUT
  const __m256i lut =
      _mm256_set_epi8(0, 0, 0, -1, 0, 0, 0, -2, 0, 0, 0, 1, -1, -2, 1, 0, 0, 0,
                      0, -1, 0, 0, 0, -2, 0, 0, 0, 1, -1, -2, 1, 0);

  return s8x32{_mm256_shuffle_epi8(lut, masked)};
}

YNN_ALWAYS_INLINE s8x32 cast(s4x32 from, int8_t) {
  // 1. Broadcast the 128-bit input to both lanes of the 256-bit register.
  __m256i dup = _mm256_broadcastsi128_si256(from.v);

  // 2. Duplicate each byte 2 times inside each 128-bit lane.
  const __m256i mask_dup =
      _mm256_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
                       9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
  dup = _mm256_shuffle_epi8(dup, mask_dup);

  // 3. Shift right and mask-blend bytes
  __m256i shifted = _mm256_srli_epi16(dup, 4);

  __m256i sel0 = _mm256_setr_epi8(
      0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
      0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0);
  __m256i sel1 = _mm256_setr_epi8(
      0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
      0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff);
  __m256i blended = _mm256_or_si256(_mm256_and_si256(dup, sel0),
                                    _mm256_and_si256(shifted, sel1));

  __m256i indices = _mm256_and_si256(blended, _mm256_set1_epi8(0x0f));
  const __m256i lut = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1));

  return s8x32{_mm256_shuffle_epi8(lut, indices)};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_
