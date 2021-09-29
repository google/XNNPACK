// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_f32_cvt__sse2_int32(
    size_t n,
    const void* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const __m128i vsign_mask = _mm_set1_epi32(0x80000000);
  const __m128i vexp_offset = _mm_set1_epi32(0x70000000);
  const __m128 vexp_scale = _mm_set1_ps(0x1.0p-112f);
  const __m128i vmagic_mask = _mm_set1_epi32(0x3F000000);
  const __m128 vmagic_bias = _mm_set1_ps(0.5f);
  const __m128i vdenorm_cutoff = _mm_set1_epi32(0x04000000);

  const uint16_t* i = (const uint16_t*) input;
  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128i vw_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), vh);
    const __m128i vw_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), vh);

    const __m128i vsign_lo = _mm_and_si128(vw_lo, vsign_mask);
    const __m128i vsign_hi = _mm_and_si128(vw_hi, vsign_mask);

    const __m128i vnonsign_lo = _mm_xor_si128(vw_lo, vsign_lo);
    const __m128i vnonsign_hi = _mm_xor_si128(vw_hi, vsign_hi);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign_lo, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign_hi, 3), vexp_offset)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign_lo, 16), vmagic_mask)), vmagic_bias));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign_hi, 16), vmagic_mask)), vmagic_bias));

    const __m128i vmask_lo = _mm_cmpgt_epi32(vnonsign_lo, vdenorm_cutoff);
    const __m128i vmask_hi = _mm_cmpgt_epi32(vnonsign_hi, vdenorm_cutoff);

    const __m128i vf_lo = _mm_or_si128(vsign_lo,
      _mm_or_si128(_mm_and_si128(vmask_lo, vnorm_lo), _mm_andnot_si128(vmask_lo, vdenorm_lo)));
    const __m128i vf_hi = _mm_or_si128(vsign_hi,
      _mm_or_si128(_mm_and_si128(vmask_hi, vnorm_hi), _mm_andnot_si128(vmask_hi, vdenorm_hi)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf_lo));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf_hi));
    output += 8;
  }
}
