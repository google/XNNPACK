// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/sse-mul32-ld32.c.in
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


void xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_x32(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const __m128i vzero_point_product = _mm_load_si128((const __m128i*) params->sse2.zero_point_product);
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse2.x_multiplier);
  const __m128i vb_multiplier = _mm_load_si128((const __m128i*) params->sse2.y_multiplier);
  const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->sse2.remainder_mask);
  const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->sse2.remainder_threshold);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->sse2.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);

  for (; n >= 32 * sizeof(int8_t); n -= 32 * sizeof(int8_t)) {
    const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
    const __m128i vb0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b));
    const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));
    const __m128i vb4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 4));
    const __m128i va89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 8));
    const __m128i vb89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 8));
    const __m128i vaCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 12));
    const __m128i vbCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 12));
    const __m128i vaGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 16));
    const __m128i vbGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 16));
    const __m128i vaKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 20));
    const __m128i vbKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 20));
    const __m128i vaOPQR = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 24));
    const __m128i vbOPQR = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 24));
    const __m128i vaSTUV = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 28));
    const __m128i vbSTUV = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 28));
    input_a += 32;
    input_b += 32;

    __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(va0123, va_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(va4567, va_multiplier));
    __m128i vacc89AB = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(va89AB, va_multiplier));
    __m128i vaccCDEF = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vaCDEF, va_multiplier));
    __m128i vaccGHIJ = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vaGHIJ, va_multiplier));
    __m128i vaccKLMN = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vaKLMN, va_multiplier));
    __m128i vaccOPQR = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vaOPQR, va_multiplier));
    __m128i vaccSTUV = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vaSTUV, va_multiplier));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));
    vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vb89AB, vb_multiplier));
    vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vbCDEF, vb_multiplier));
    vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vbGHIJ, vb_multiplier));
    vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vbKLMN, vb_multiplier));
    vaccOPQR = _mm_add_epi32(vaccOPQR, _mm_mullo_epi32(vbOPQR, vb_multiplier));
    vaccSTUV = _mm_add_epi32(vaccSTUV, _mm_mullo_epi32(vbSTUV, vb_multiplier));

    const __m128i vrem0123 = _mm_add_epi32(_mm_and_si128(vacc0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123));
    const __m128i vrem4567 = _mm_add_epi32(_mm_and_si128(vacc4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567));
    const __m128i vrem89AB = _mm_add_epi32(_mm_and_si128(vacc89AB, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc89AB));
    const __m128i vremCDEF = _mm_add_epi32(_mm_and_si128(vaccCDEF, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccCDEF));
    const __m128i vremGHIJ = _mm_add_epi32(_mm_and_si128(vaccGHIJ, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccGHIJ));
    const __m128i vremKLMN = _mm_add_epi32(_mm_and_si128(vaccKLMN, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccKLMN));
    const __m128i vremOPQR = _mm_add_epi32(_mm_and_si128(vaccOPQR, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccOPQR));
    const __m128i vremSTUV = _mm_add_epi32(_mm_and_si128(vaccSTUV, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccSTUV));

    vacc0123 = _mm_sub_epi32(_mm_sra_epi32(vacc0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
    vacc4567 = _mm_sub_epi32(_mm_sra_epi32(vacc4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));
    vacc89AB = _mm_sub_epi32(_mm_sra_epi32(vacc89AB, vshift), _mm_cmpgt_epi32(vrem89AB, vremainder_threshold));
    vaccCDEF = _mm_sub_epi32(_mm_sra_epi32(vaccCDEF, vshift), _mm_cmpgt_epi32(vremCDEF, vremainder_threshold));
    vaccGHIJ = _mm_sub_epi32(_mm_sra_epi32(vaccGHIJ, vshift), _mm_cmpgt_epi32(vremGHIJ, vremainder_threshold));
    vaccKLMN = _mm_sub_epi32(_mm_sra_epi32(vaccKLMN, vshift), _mm_cmpgt_epi32(vremKLMN, vremainder_threshold));
    vaccOPQR = _mm_sub_epi32(_mm_sra_epi32(vaccOPQR, vshift), _mm_cmpgt_epi32(vremOPQR, vremainder_threshold));
    vaccSTUV = _mm_sub_epi32(_mm_sra_epi32(vaccSTUV, vshift), _mm_cmpgt_epi32(vremSTUV, vremainder_threshold));

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);
    __m128i voutGHIJKLMN = _mm_adds_epi16(_mm_packs_epi32(vaccGHIJ, vaccKLMN), voutput_zero_point);
    __m128i voutOPQRSTUV = _mm_adds_epi16(_mm_packs_epi32(vaccOPQR, vaccSTUV), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
    vout89ABCDEF = _mm_max_epi16(vout89ABCDEF, voutput_min);
    voutGHIJKLMN = _mm_max_epi16(voutGHIJKLMN, voutput_min);
    voutOPQRSTUV = _mm_max_epi16(voutOPQRSTUV, voutput_min);

    vout01234567 = _mm_min_epi16(vout01234567, voutput_max);
    vout89ABCDEF = _mm_min_epi16(vout89ABCDEF, voutput_max);
    voutGHIJKLMN = _mm_min_epi16(voutGHIJKLMN, voutput_max);
    voutOPQRSTUV = _mm_min_epi16(voutOPQRSTUV, voutput_max);

    const __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
    const __m128i voutGHIJKLMNOPQRSTUV = _mm_packs_epi16(voutGHIJKLMN, voutOPQRSTUV);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
    output += 32;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
      const __m128i vb0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b));
      const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));
      const __m128i vb4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 4));
      input_a += 8;
      input_b += 8;

      __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(va0123, va_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(va4567, va_multiplier));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));

      const __m128i vrem0123 = _mm_add_epi32(_mm_and_si128(vacc0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123));
      const __m128i vrem4567 = _mm_add_epi32(_mm_and_si128(vacc4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567));

      vacc0123 = _mm_sub_epi32(_mm_sra_epi32(vacc0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
      vacc4567 = _mm_sub_epi32(_mm_sra_epi32(vacc4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

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
