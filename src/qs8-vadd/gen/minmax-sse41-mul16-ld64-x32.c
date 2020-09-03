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


void xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32(
    size_t n,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_add_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const __m128i vzero_point_product = _mm_load_si128((const __m128i*) params->sse2.zero_point_product);
  const __m128i vx_multiplier_lo = _mm_load_si128((const __m128i*) params->sse2.x_multiplier_lo);
  const __m128i vx_multiplier_hi = _mm_load_si128((const __m128i*) params->sse2.x_multiplier_hi);
  const __m128i vy_multiplier_lo = _mm_load_si128((const __m128i*) params->sse2.y_multiplier_lo);
  const __m128i vy_multiplier_hi = _mm_load_si128((const __m128i*) params->sse2.y_multiplier_hi);
  const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->sse2.remainder_mask);
  const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->sse2.remainder_threshold);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->sse2.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);

  for (; n >= 32 * sizeof(int8_t); n -= 32 * sizeof(int8_t)) {
    const __m128i vx01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_x));
    const __m128i vy01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_y));
    const __m128i vx89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_x + 8)));
    const __m128i vy89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_y + 8)));
    const __m128i vxGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_x + 16)));
    const __m128i vyGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_y + 16)));
    const __m128i vxOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_x + 24)));
    const __m128i vyOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_y + 24)));
    input_x += 32;
    input_y += 32;


    __m128i vxprod01234567hi = _mm_mulhi_epu16(vx01234567, vx_multiplier_lo);
    __m128i vyprod01234567hi = _mm_mulhi_epu16(vy01234567, vy_multiplier_lo);
    const __m128i vxprod01234567lo = _mm_mullo_epi16(vx01234567, vx_multiplier_lo);
    const __m128i vyprod01234567lo = _mm_mullo_epi16(vy01234567, vy_multiplier_lo);
    __m128i vxprod89ABCDEFhi = _mm_mulhi_epu16(vx89ABCDEF, vx_multiplier_lo);
    __m128i vyprod89ABCDEFhi = _mm_mulhi_epu16(vy89ABCDEF, vy_multiplier_lo);
    const __m128i vxprod89ABCDEFlo = _mm_mullo_epi16(vx89ABCDEF, vx_multiplier_lo);
    const __m128i vyprod89ABCDEFlo = _mm_mullo_epi16(vy89ABCDEF, vy_multiplier_lo);
    __m128i vxprodGHIJKLMNhi = _mm_mulhi_epu16(vxGHIJKLMN, vx_multiplier_lo);
    __m128i vyprodGHIJKLMNhi = _mm_mulhi_epu16(vyGHIJKLMN, vy_multiplier_lo);
    const __m128i vxprodGHIJKLMNlo = _mm_mullo_epi16(vxGHIJKLMN, vx_multiplier_lo);
    const __m128i vyprodGHIJKLMNlo = _mm_mullo_epi16(vyGHIJKLMN, vy_multiplier_lo);
    __m128i vxprodOPQRSTUVhi = _mm_mulhi_epu16(vxOPQRSTUV, vx_multiplier_lo);
    __m128i vyprodOPQRSTUVhi = _mm_mulhi_epu16(vyOPQRSTUV, vy_multiplier_lo);
    const __m128i vxprodOPQRSTUVlo = _mm_mullo_epi16(vxOPQRSTUV, vx_multiplier_lo);
    const __m128i vyprodOPQRSTUVlo = _mm_mullo_epi16(vyOPQRSTUV, vy_multiplier_lo);

    vxprod01234567hi = _mm_add_epi16(vxprod01234567hi, _mm_mullo_epi16(vx01234567, vx_multiplier_hi));
    vyprod01234567hi = _mm_add_epi16(vyprod01234567hi, _mm_mullo_epi16(vy01234567, vy_multiplier_hi));
    vxprod89ABCDEFhi = _mm_add_epi16(vxprod89ABCDEFhi, _mm_mullo_epi16(vx89ABCDEF, vx_multiplier_hi));
    vyprod89ABCDEFhi = _mm_add_epi16(vyprod89ABCDEFhi, _mm_mullo_epi16(vy89ABCDEF, vy_multiplier_hi));
    vxprodGHIJKLMNhi = _mm_add_epi16(vxprodGHIJKLMNhi, _mm_mullo_epi16(vxGHIJKLMN, vx_multiplier_hi));
    vyprodGHIJKLMNhi = _mm_add_epi16(vyprodGHIJKLMNhi, _mm_mullo_epi16(vyGHIJKLMN, vy_multiplier_hi));
    vxprodOPQRSTUVhi = _mm_add_epi16(vxprodOPQRSTUVhi, _mm_mullo_epi16(vxOPQRSTUV, vx_multiplier_hi));
    vyprodOPQRSTUVhi = _mm_add_epi16(vyprodOPQRSTUVhi, _mm_mullo_epi16(vyOPQRSTUV, vy_multiplier_hi));

    vxprod01234567hi = _mm_sub_epi16(vxprod01234567hi, _mm_and_si128(_mm_srai_epi16(vx01234567, 15), vx_multiplier_lo));
    vyprod01234567hi = _mm_sub_epi16(vyprod01234567hi, _mm_and_si128(_mm_srai_epi16(vy01234567, 15), vy_multiplier_lo));
    vxprod89ABCDEFhi = _mm_sub_epi16(vxprod89ABCDEFhi, _mm_and_si128(_mm_srai_epi16(vx89ABCDEF, 15), vx_multiplier_lo));
    vyprod89ABCDEFhi = _mm_sub_epi16(vyprod89ABCDEFhi, _mm_and_si128(_mm_srai_epi16(vy89ABCDEF, 15), vy_multiplier_lo));
    vxprodGHIJKLMNhi = _mm_sub_epi16(vxprodGHIJKLMNhi, _mm_and_si128(_mm_srai_epi16(vxGHIJKLMN, 15), vx_multiplier_lo));
    vyprodGHIJKLMNhi = _mm_sub_epi16(vyprodGHIJKLMNhi, _mm_and_si128(_mm_srai_epi16(vyGHIJKLMN, 15), vy_multiplier_lo));
    vxprodOPQRSTUVhi = _mm_sub_epi16(vxprodOPQRSTUVhi, _mm_and_si128(_mm_srai_epi16(vxOPQRSTUV, 15), vx_multiplier_lo));
    vyprodOPQRSTUVhi = _mm_sub_epi16(vyprodOPQRSTUVhi, _mm_and_si128(_mm_srai_epi16(vyOPQRSTUV, 15), vy_multiplier_lo));

    __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vxprod01234567lo, vxprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vxprod01234567lo, vxprod01234567hi));
    __m128i vacc89AB = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vxprod89ABCDEFlo, vxprod89ABCDEFhi));
    __m128i vaccCDEF = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vxprod89ABCDEFlo, vxprod89ABCDEFhi));
    __m128i vaccGHIJ = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vxprodGHIJKLMNlo, vxprodGHIJKLMNhi));
    __m128i vaccKLMN = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vxprodGHIJKLMNlo, vxprodGHIJKLMNhi));
    __m128i vaccOPQR = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vxprodOPQRSTUVlo, vxprodOPQRSTUVhi));
    __m128i vaccSTUV = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vxprodOPQRSTUVlo, vxprodOPQRSTUVhi));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vyprod01234567lo, vyprod01234567hi));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vyprod01234567lo, vyprod01234567hi));
    vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vyprod89ABCDEFlo, vyprod89ABCDEFhi));
    vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vyprod89ABCDEFlo, vyprod89ABCDEFhi));
    vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vyprodGHIJKLMNlo, vyprodGHIJKLMNhi));
    vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vyprodGHIJKLMNlo, vyprodGHIJKLMNhi));
    vaccOPQR = _mm_add_epi32(vaccOPQR, _mm_unpacklo_epi16(vyprodOPQRSTUVlo, vyprodOPQRSTUVhi));
    vaccSTUV = _mm_add_epi32(vaccSTUV, _mm_unpackhi_epi16(vyprodOPQRSTUVlo, vyprodOPQRSTUVhi));

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
      const __m128i vx01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_x));
      const __m128i vy01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_y));
      input_x += 8;
      input_y += 8;


      __m128i vxprod01234567hi = _mm_mulhi_epu16(vx01234567, vx_multiplier_lo);
      __m128i vyprod01234567hi = _mm_mulhi_epu16(vy01234567, vy_multiplier_lo);
      const __m128i vxprod01234567lo = _mm_mullo_epi16(vx01234567, vx_multiplier_lo);
      const __m128i vyprod01234567lo = _mm_mullo_epi16(vy01234567, vy_multiplier_lo);

      vxprod01234567hi = _mm_add_epi16(vxprod01234567hi, _mm_mullo_epi16(vx01234567, vx_multiplier_hi));
      vyprod01234567hi = _mm_add_epi16(vyprod01234567hi, _mm_mullo_epi16(vy01234567, vy_multiplier_hi));

      vxprod01234567hi = _mm_sub_epi16(vxprod01234567hi, _mm_and_si128(_mm_srai_epi16(vx01234567, 15), vx_multiplier_lo));
      vyprod01234567hi = _mm_sub_epi16(vyprod01234567hi, _mm_and_si128(_mm_srai_epi16(vy01234567, 15), vy_multiplier_lo));

      __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vxprod01234567lo, vxprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vxprod01234567lo, vxprod01234567hi));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vyprod01234567lo, vyprod01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vyprod01234567lo, vyprod01234567hi));

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
