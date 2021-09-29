// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <smmintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_f32_cvt__sse41_int16(
    size_t n,
    const void* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const __m128i vsign_mask = _mm_set1_epi16(0x8000);
  const __m128i vexp_offset = _mm_set1_epi16(0x7000);
  const __m128 vexp_scale = _mm_set1_ps(0x1.0p-112f);
  const __m128i vmagic_mask = _mm_set1_epi16(0x3F00);
  const __m128 vmagic_bias = _mm_set1_ps(0.5f);
  const __m128i vdenorm_cutoff = _mm_set1_epi16(0x0400);

  const uint16_t* i = (const uint16_t*) input;
  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128i vsign = _mm_and_si128(vh, vsign_mask);

    const __m128i vnonsign = _mm_xor_si128(vh, vsign);

    const __m128i vprenorm_lo = _mm_slli_epi16(vnonsign, 13);
    const __m128i vprenorm_hi = _mm_add_epi16(_mm_srli_epi16(vnonsign, 3), vexp_offset);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign, vmagic_mask)), vmagic_bias));

    const __m128i vmask = _mm_cmpgt_epi16(vnonsign, vdenorm_cutoff);
    const __m128i vf_lo = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_lo, vnorm_lo, _mm_cvtepi16_epi32(vmask)));
    const __m128i vf_hi = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_hi, vnorm_hi, _mm_unpackhi_epi16(vmask, vmask)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf_lo));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf_hi));
    output += 8;
  }
}
