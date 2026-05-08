// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/simd/x86_sse2.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static s32x4 reduce_add(
    s32x4 a, s8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i b2x = _mm_maddubs_epi16(_mm_set1_epi8(1), b.v);
  return a += s32x4{_mm_madd_epi16(_mm_set1_epi16(1), b2x)};
}

static s32x4 reduce_add(
    s32x4 a, u8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i b2x = _mm_maddubs_epi16(b.v, _mm_set1_epi8(1));
  return a += s32x4{_mm_madd_epi16(_mm_set1_epi16(1), b2x)};
}

}  // namespace simd

using simd::s32x16;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

SUM_KERNEL(sum_int8_int32_ssse3, s32x4, int8_t, int32_t, 16);
SUM_KERNEL(sum_uint8_int32_ssse3, s32x4, uint8_t, int32_t, 16);

}  // namespace ynn
