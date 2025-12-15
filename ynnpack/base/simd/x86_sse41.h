// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_H_

#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_sse2_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse41_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using s32x16 = multi_vec<s32x4, 4>;

YNN_ALWAYS_INLINE s32x16 convert(s8x16 a, int32_t) {
  return {
      s32x4{_mm_cvtepi8_epi32(a.v)},
      s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 4))},
      s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 8))},
      s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 12))},
  };
}

YNN_ALWAYS_INLINE s32x16 convert(u8x16 a, int32_t) {
  return {
      s32x4{_mm_cvtepu8_epi32(a.v)},
      s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 4))},
      s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 8))},
      s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 12))},
  };
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE41_H_
