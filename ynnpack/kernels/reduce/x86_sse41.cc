// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/simd/x86_vec128.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static s32x16 reduce_add(
    s32x16 a, u8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  s32x16 b_s32 = cast(b, int32_t{});
  s32x4 b0 = extract<0>(b_s32, s32x4::N);
  s32x4 b1 = extract<1>(b_s32, s32x4::N);
  s32x4 b2 = extract<2>(b_s32, s32x4::N);
  s32x4 b3 = extract<3>(b_s32, s32x4::N);

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  return a + concat(s32x4{_mm_madd_epi16(b0.v, b0.v)},
                    s32x4{_mm_madd_epi16(b1.v, b1.v)},
                    s32x4{_mm_madd_epi16(b2.v, b2.v)},
                    s32x4{_mm_madd_epi16(b3.v, b3.v)});

  return a;
}

static s32x16 reduce_add(s32x16 a, s8x16 b, square map_fn,
                         std::integral_constant<size_t, 1> horizontal_factor) {
  // We're squaring, we can take the absolute value and use the unsigned reduce.
  return reduce_add(a, abs(b), map_fn, horizontal_factor);
}

static s32x4 reduce_add(
    s32x4 a, s8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i lo = _mm_cvtepi8_epi16(b.v);
  __m128i hi = _mm_cvtepi8_epi16(_mm_srli_si128(b.v, 8));
  return a +=
         s32x4(_mm_hadd_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi)));
}

static s32x4 reduce_add(
    s32x4 a, u8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i lo = _mm_cvtepu8_epi16(b.v);
  __m128i hi = _mm_cvtepu8_epi16(_mm_srli_si128(b.v, 8));
  return a +=
         s32x4(_mm_hadd_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi)));
}

YNN_ALWAYS_INLINE s8x16 sign_complement(s8x16 x) {
  __m128i zero = _mm_setzero_si128();
  __m128i sign = _mm_cmpgt_epi8(zero, x.v);
  __m128i mask = _mm_set1_epi8(0x7F);
  __m128i abs_val = _mm_and_si128(x.v, mask);
  return s8x16(_mm_xor_si128(abs_val, sign));
}

}  // namespace simd

using simd::s32x16;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

using xf8x16 = sign_magnitude<s8x16>;

MIN_MAX_K1_KERNEL(min_max_k1_int8_sse41, s8x16, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(min_max_kn_int8_sse41, s8x16, s8x16, int8_t, 16);
MIN_MAX_K1_KERNEL(min_max_k1_xf8_sse41, xf8x16, xf8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(min_max_kn_xf8_sse41, xf8x16, xf8x16, int8_t, 16);

MIN_MAX_K1_KERNEL(min_k1_int8_sse41, s8x16, dummy_t, int8_t, 16);
MIN_MAX_KN_KERNEL(min_kn_int8_sse41, s8x16, dummy_t, int8_t, 16);
MIN_MAX_K1_KERNEL(min_k1_xf8_sse41, xf8x16, dummy_t, int8_t, 16);
MIN_MAX_KN_KERNEL(min_kn_xf8_sse41, xf8x16, dummy_t, int8_t, 16);

MIN_MAX_K1_KERNEL(max_k1_int8_sse41, dummy_t, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(max_kn_int8_sse41, dummy_t, s8x16, int8_t, 16);
MIN_MAX_K1_KERNEL(max_k1_xf8_sse41, dummy_t, xf8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(max_kn_xf8_sse41, dummy_t, xf8x16, int8_t, 16);

SUM_KN_KERNEL(sum_kn_int8_int32_sse41, int8_t, int32_t, 16, identity);
SUM_KN_KERNEL(sum_kn_uint8_int32_sse41, uint8_t, int32_t, 16, identity);

SUM_K1_KERNEL(sum_squared_k1_uint8_int32_sse41, uint8_t, int32_t, 4, 4, square);
SUM_KN_KERNEL(sum_squared_kn_uint8_int32_sse41, uint8_t, int32_t, 16, square);
SUM_K1_KERNEL(sum_squared_k1_int8_int32_sse41, int8_t, int32_t, 4, 4, square);
SUM_KN_KERNEL(sum_squared_kn_int8_int32_sse41, int8_t, int32_t, 16, square);

}  // namespace ynn
