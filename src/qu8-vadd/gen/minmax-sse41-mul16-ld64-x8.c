// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/sse-mul16-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/vadd.h>


void xnn_qu8_vadd_minmax_ukernel__sse41_mul16_ld64_x8(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse2.bias);
  const __m128i va_multiplier_lo = _mm_load_si128((const __m128i*) params->sse2.a_multiplier_lo);
  const __m128i va_multiplier_hi = _mm_load_si128((const __m128i*) params->sse2.a_multiplier_hi);
  const __m128i vb_multiplier_lo = _mm_load_si128((const __m128i*) params->sse2.b_multiplier_lo);
  const __m128i vb_multiplier_hi = _mm_load_si128((const __m128i*) params->sse2.b_multiplier_hi);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->sse2.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);

  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    input_a += 8;
    input_b += 8;


    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
    const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
    vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));


    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

    vacc0123 = _mm_sra_epi32(_mm_add_epi32(vacc0123, vrounding), vshift);
    vacc4567 = _mm_sra_epi32(_mm_add_epi32(vacc4567, vrounding), vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));


      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
      const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
      vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));


      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

      vacc0123 = _mm_sra_epi32(_mm_add_epi32(vacc0123, vrounding), vshift);
      vacc4567 = _mm_sra_epi32(_mm_add_epi32(vacc4567, vrounding), vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (n & (4 * sizeof(uint8_t))) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (n & (2 * sizeof(uint8_t))) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (n & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}
