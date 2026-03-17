// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_

#include <cstdint>

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

YNN_ALWAYS_INLINE s32x16 convert(s8x16 a, int32_t) {
  return {
      s32x8{_mm256_cvtepi8_epi32(a.v)},
      s32x8{_mm256_cvtepi8_epi32(_mm_srli_si128(a.v, 8))},
  };
}

YNN_ALWAYS_INLINE s32x16 convert(u8x16 a, int32_t) {
  return {
      s32x8{_mm256_cvtepu8_epi32(a.v)},
      s32x8{_mm256_cvtepu8_epi32(_mm_srli_si128(a.v, 8))},
  };
}

YNN_ALWAYS_INLINE f32x8 convert(s32x8 x, float) {
  return f32x8{_mm256_cvtepi32_ps(x.v)};
}

YNN_ALWAYS_INLINE s32x8 convert(f32x8 x, int32_t) {
  return s32x8{_mm256_cvttps_epi32(x.v)};
}

YNN_ALWAYS_INLINE bf16x16 convert(f32x16 a, bfloat16) {
  const __m256 rounding_multiplier = _mm256_set1_ps(1.0f + 0.5f / 128.0f);
  __m256 a1 = _mm256_mul_ps(a.lo().v, rounding_multiplier);
  __m256 a2 = _mm256_mul_ps(a.hi().v, rounding_multiplier);
  const __m256i ai = _mm256_castps_si256(a1);
  const __m256i bi = _mm256_castps_si256(a2);
  const __m256i as = _mm256_srli_epi32(ai, 16);
  const __m256i bs = _mm256_srli_epi32(bi, 16);
  const __m256i r = _mm256_packus_epi32(as, bs);
  return bf16x16{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_
