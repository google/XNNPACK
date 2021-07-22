// Auto-generated file. Do not edit!
//   Template: src/qs8-vaddc/sse-mul32-ld32.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vadd.h>


void xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_x16(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4_mul32.a_multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse4_mul32.rounding);
  const __m128i vshift = _mm_loadu_si32(params->sse4_mul32.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4_mul32.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4_mul32.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4_mul32.output_max);

  __m128i vbias = _mm_cvtsi32_si128(params->sse4_mul32.b_multiplier[0] * (int32_t) *input_b);
  vbias = _mm_shuffle_epi32(vbias, _MM_SHUFFLE(0, 0, 0, 0));
  vbias = _mm_add_epi32(vbias, _mm_load_si128((const __m128i*) params->sse4_mul32.bias));
  for (; n >= 16 * sizeof(int8_t); n -= 16 * sizeof(int8_t)) {
    const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
    const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));
    const __m128i va89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 8));
    const __m128i vaCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 12));
    input_a += 16;
    input_b += 16;

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));
    __m128i vacc89AB = _mm_add_epi32(vbias, _mm_mullo_epi32(va89AB, va_multiplier));
    __m128i vaccCDEF = _mm_add_epi32(vbias, _mm_mullo_epi32(vaCDEF, va_multiplier));

    vacc0123 = _mm_sra_epi32(_mm_add_epi32(vacc0123, vrounding), vshift);
    vacc4567 = _mm_sra_epi32(_mm_add_epi32(vacc4567, vrounding), vshift);
    vacc89AB = _mm_sra_epi32(_mm_add_epi32(vacc89AB, vrounding), vshift);
    vaccCDEF = _mm_sra_epi32(_mm_add_epi32(vaccCDEF, vrounding), vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    const __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
      const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));
      input_a += 8;

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

      vacc0123 = _mm_sra_epi32(_mm_add_epi32(vacc0123, vrounding), vshift);
      vacc4567 = _mm_sra_epi32(_mm_add_epi32(vacc4567, vrounding), vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(n >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        n -= 8 * sizeof(int8_t);
      } else {
        if (n & (4 * sizeof(int8_t))) {
          *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (n & (2 * sizeof(int8_t))) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (n & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}
