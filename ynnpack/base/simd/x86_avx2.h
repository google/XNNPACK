// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_

#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx2_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x16 = multi_vec<f32x8, 2>;
using s32x16 = multi_vec<s32x8, 2>;
using s32x32 = multi_vec<s32x8, 4>;

YNN_ALWAYS_INLINE f32x16 convert(bf16x16 a, float) {
  return {
      convert(extract<0>(a, bf16x8{}), float{}),
      convert(extract<1>(a, bf16x8{}), float{}),
  };
}

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

YNN_ALWAYS_INLINE s32x32 convert(s8x32 a, int32_t) {
  s32x16 lo = convert(extract<0>(a, s8x16{}), int32_t{});
  s32x16 hi = convert(extract<1>(a, s8x16{}), int32_t{});
  return {lo[0], lo[1], hi[0], hi[1]};
}
YNN_ALWAYS_INLINE s32x32 convert(u8x32 a, int32_t) {
  s32x16 lo = convert(extract<0>(a, u8x16{}), int32_t{});
  s32x16 hi = convert(extract<1>(a, u8x16{}), int32_t{});
  return {lo[0], lo[1], hi[0], hi[1]};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX2_H_
