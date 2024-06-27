// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include <smmintrin.h>

#include "xnnpack/math.h"
#include "xnnpack/requantization-stubs.h"


void xnn_qs8_requantize_rndnu__sse41_sra(
    size_t n,
    const int32_t* input,
    float scale,
    int8_t zero_point,
    int8_t qmin,
    int8_t qmax,
    int8_t* output)
{
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = float_as_uint32(scale);
  const int32_t multiplier = ((int32_t) (scale_bits << 7) & INT32_C(0x3FFFFF80)) | INT32_C(0x40000000);
  const uint32_t shift = 127 + 30 - (scale_bits >> 23);
  assert(shift >= 31);
  assert(shift < 63);
  const int64_t rounding = INT64_C(1) << (shift - 1);

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);
  const __m128i vzero_point = _mm_set1_epi16((short) zero_point);
  const __m128i vqmin = _mm_set1_epi8((char) qmin);
  const __m128i vqmax = _mm_set1_epi8((char) qmax);
  const __m128i vshift = _mm_cvtsi32_si128((int) (shift - 31));
  const __m128i vrounding = _mm_set1_epi64x(rounding);
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*) input);
    const __m128i y = _mm_loadu_si128((const __m128i*) (input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*) (input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*) (input + 12));
    input += 16;

    const __m128i x_odd = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i y_odd = _mm_shuffle_epi32(y, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i z_odd = _mm_shuffle_epi32(z, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i w_odd = _mm_shuffle_epi32(w, _MM_SHUFFLE(3, 3, 1, 1));

    const __m128i x_product02 = _mm_mul_epi32(x, vmultiplier);
    const __m128i y_product02 = _mm_mul_epi32(y, vmultiplier);
    const __m128i z_product02 = _mm_mul_epi32(z, vmultiplier);
    const __m128i w_product02 = _mm_mul_epi32(w, vmultiplier);

    const __m128i x_product13 = _mm_mul_epi32(x_odd, vmultiplier);
    const __m128i y_product13 = _mm_mul_epi32(y_odd, vmultiplier);
    const __m128i z_product13 = _mm_mul_epi32(z_odd, vmultiplier);
    const __m128i w_product13 = _mm_mul_epi32(w_odd, vmultiplier);

    const __m128i x_prescaled02 = _mm_srli_epi64(_mm_add_epi64(x_product02, vrounding), 31);
    const __m128i x_prescaled13 = _mm_slli_epi64(_mm_add_epi64(x_product13, vrounding), 1);
    const __m128i y_prescaled02 = _mm_srli_epi64(_mm_add_epi64(y_product02, vrounding), 31);
    const __m128i y_prescaled13 = _mm_slli_epi64(_mm_add_epi64(y_product13, vrounding), 1);
    const __m128i z_prescaled02 = _mm_srli_epi64(_mm_add_epi64(z_product02, vrounding), 31);
    const __m128i z_prescaled13 = _mm_slli_epi64(_mm_add_epi64(z_product13, vrounding), 1);
    const __m128i w_prescaled02 = _mm_srli_epi64(_mm_add_epi64(w_product02, vrounding), 31);
    const __m128i w_prescaled13 = _mm_slli_epi64(_mm_add_epi64(w_product13, vrounding), 1);

    const __m128i x_prescaled = _mm_blend_epi16(x_prescaled02, x_prescaled13, 0xCC);
    const __m128i y_prescaled = _mm_blend_epi16(y_prescaled02, y_prescaled13, 0xCC);
    const __m128i z_prescaled = _mm_blend_epi16(z_prescaled02, z_prescaled13, 0xCC);
    const __m128i w_prescaled = _mm_blend_epi16(w_prescaled02, w_prescaled13, 0xCC);

    const __m128i x_scaled = _mm_sra_epi32(x_prescaled, vshift);
    const __m128i y_scaled = _mm_sra_epi32(y_prescaled, vshift);
    const __m128i z_scaled = _mm_sra_epi32(z_prescaled, vshift);
    const __m128i w_scaled = _mm_sra_epi32(w_prescaled, vshift);

    const __m128i xy_packed = _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    const __m128i zw_packed = _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    const __m128i xyzw_packed = _mm_packs_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped = _mm_max_epi8(_mm_min_epi8(xyzw_packed, vqmax), vqmin);

    // 4x PSHUFD
    // 8x PMULDQ
    // 8x PADDQ
    // 4x PSRLQ
    // 4x PSLLQ
    // 4x PBLENDW
    // 4x PSRAD
    // 2x PACKSSDW
    // 2x PADDSW
    // 1x PACKSSWB
    // 1x PMAXSB
    // 1x PMINSB
    // ---------------------
    // 43 instructions total

    _mm_storeu_si128((__m128i*) output, xyzw_clamped);
    output += 16;
  }
}
