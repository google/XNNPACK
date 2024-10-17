// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include <smmintrin.h>

#include "xnnpack/math.h"
#include "xnnpack/requantization.h"
#include "xnnpack/requantization-stubs.h"

static inline __m128i clamp16(__m128i result, __m128i zero_point, __m128i min_less_zero_point, __m128i max_less_zero_point) {
  result = _mm_max_epi16(result, min_less_zero_point);
  result = _mm_min_epi16(result, max_less_zero_point);
  return result + zero_point;
}

void xnn_qu8_requantize_rndnu16__sse41(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output)
{
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  struct F32 f32 = parse_f32(scale);
  int exp = f32.exp;
  int16_t m16 = f32.multiplier >> 9;

  __m128i m16_vec = _mm_set1_epi16(m16);
  __m128i zero_point16 = _mm_set1_epi16(zero_point);
  __m128i min_less_zero_point16 = _mm_set1_epi16(qmin) - zero_point16;
  __m128i max_less_zero_point16 = _mm_set1_epi16(qmax) - zero_point16;

  const int right_preshift = -exp - 1;
  for (; n != 0; n -= 16) {
      __m128i x = _mm_loadu_si128((const __m128i*)input);
      input += 4;

      __m128i y = _mm_loadu_si128((const __m128i*)input);
      input += 4;

      __m128i z = _mm_loadu_si128((const __m128i*)input);
      input += 4;

      __m128i w = _mm_loadu_si128((const __m128i*)input);
      input += 4;

      __m128i preshifted_x = _mm_srai_epi32(x, right_preshift);
      __m128i preshifted_y = _mm_srai_epi32(y, right_preshift);
      __m128i preshifted_z = _mm_srai_epi32(z, right_preshift);
      __m128i preshifted_w = _mm_srai_epi32(w, right_preshift);

      __m128i xy16 = _mm_packs_epi32(preshifted_x, preshifted_y);
      __m128i zw16 = _mm_packs_epi32(preshifted_z, preshifted_w);

      __m128i xy_upper_half16 = _mm_mulhrs_epi16(xy16, m16_vec);
      __m128i zw_upper_half16 = _mm_mulhrs_epi16(zw16, m16_vec);

      __m128i xy_res16 = clamp16(xy_upper_half16, zero_point16, min_less_zero_point16, max_less_zero_point16);
      __m128i zw_res16 = clamp16(zw_upper_half16, zero_point16, min_less_zero_point16, max_less_zero_point16);

      __m128i xyzw_output = _mm_packus_epi16(xy_res16, zw_res16);
      _mm_storeu_si128((__m128i*)output, xyzw_output);
      output += 16;
  }
}
