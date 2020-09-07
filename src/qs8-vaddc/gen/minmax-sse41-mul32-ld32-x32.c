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


void xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32(
    size_t n,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_add_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const __m128i vx_multiplier = _mm_load_si128((const __m128i*) params->sse2.x_multiplier);
  const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->sse2.remainder_mask);
  const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->sse2.remainder_threshold);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->sse2.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);

  __m128i vzero_point_product = _mm_cvtsi32_si128(params->sse2.y_multiplier[0] * (int32_t) *input_y);
  vzero_point_product = _mm_shuffle_epi32(vzero_point_product, _MM_SHUFFLE(0, 0, 0, 0));
  vzero_point_product = _mm_add_epi32(vzero_point_product, _mm_load_si128((const __m128i*) params->sse2.zero_point_product));
  for (; n >= 32 * sizeof(int8_t); n -= 32 * sizeof(int8_t)) {
    const __m128i vx0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x));
    const __m128i vx4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 4));
    const __m128i vx89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 8));
    const __m128i vxCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 12));
    const __m128i vxGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 16));
    const __m128i vxKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 20));
    const __m128i vxOPQR = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 24));
    const __m128i vxSTUV = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 28));
    input_x += 32;
    input_y += 32;

    __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vx0123, vx_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vx4567, vx_multiplier));
    __m128i vacc89AB = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vx89AB, vx_multiplier));
    __m128i vaccCDEF = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vxCDEF, vx_multiplier));
    __m128i vaccGHIJ = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vxGHIJ, vx_multiplier));
    __m128i vaccKLMN = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vxKLMN, vx_multiplier));
    __m128i vaccOPQR = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vxOPQR, vx_multiplier));
    __m128i vaccSTUV = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vxSTUV, vx_multiplier));

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
      const __m128i vx0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x));
      const __m128i vx4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_x + 4));
      input_x += 8;

      __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vx0123, vx_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_mullo_epi32(vx4567, vx_multiplier));

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
