// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/avx2-mul32-ld64.c.in
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


void xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_x32(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const __m256i vzero_point_product = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.zero_point_product));
  const __m256i va_multiplier = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.x_multiplier));
  const __m256i vb_multiplier = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.y_multiplier));
  const __m256i vremainder_mask = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.remainder_mask));
  const __m256i vremainder_threshold = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.remainder_threshold));
  const __m128i vshift = _mm_cvtsi32_si128((int) params->sse2.shift);
  const __m256i voutput_zero_point = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.output_zero_point));
  const __m256i voutput_min = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.output_min));
  const __m256i voutput_max = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.output_max));

  for (; n >= 32 * sizeof(int8_t); n -= 32 * sizeof(int8_t)) {
    const __m256i va01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
    const __m256i vb01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
    const __m256i va89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m256i vb89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    const __m256i vaGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 16)));
    const __m256i vbGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_b + 16)));
    const __m256i vaOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 24)));
    const __m256i vbOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_b + 24)));
    input_a += 32;
    input_b += 32;

    __m256i vacc01234567 = _mm256_add_epi32(vzero_point_product, _mm256_mullo_epi32(va01234567, va_multiplier));
    __m256i vacc89ABCDEF = _mm256_add_epi32(vzero_point_product, _mm256_mullo_epi32(va89ABCDEF, va_multiplier));
    __m256i vaccGHIJKLMN = _mm256_add_epi32(vzero_point_product, _mm256_mullo_epi32(vaGHIJKLMN, va_multiplier));
    __m256i vaccOPQRSTUV = _mm256_add_epi32(vzero_point_product, _mm256_mullo_epi32(vaOPQRSTUV, va_multiplier));

    vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));
    vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vb89ABCDEF, vb_multiplier));
    vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vbGHIJKLMN, vb_multiplier));
    vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vbOPQRSTUV, vb_multiplier));

    const __m256i vrem01234567 = _mm256_add_epi32(_mm256_and_si256(vacc01234567, vremainder_mask), _mm256_srai_epi32(vacc01234567, 31));
    const __m256i vrem89ABCDEF = _mm256_add_epi32(_mm256_and_si256(vacc89ABCDEF, vremainder_mask), _mm256_srai_epi32(vacc89ABCDEF, 31));
    const __m256i vremGHIJKLMN = _mm256_add_epi32(_mm256_and_si256(vaccGHIJKLMN, vremainder_mask), _mm256_srai_epi32(vaccGHIJKLMN, 31));
    const __m256i vremOPQRSTUV = _mm256_add_epi32(_mm256_and_si256(vaccOPQRSTUV, vremainder_mask), _mm256_srai_epi32(vaccOPQRSTUV, 31));

    vacc01234567 = _mm256_sub_epi32(_mm256_sra_epi32(vacc01234567, vshift), _mm256_cmpgt_epi32(vrem01234567, vremainder_threshold));
    vacc89ABCDEF = _mm256_sub_epi32(_mm256_sra_epi32(vacc89ABCDEF, vshift), _mm256_cmpgt_epi32(vrem89ABCDEF, vremainder_threshold));
    vaccGHIJKLMN = _mm256_sub_epi32(_mm256_sra_epi32(vaccGHIJKLMN, vshift), _mm256_cmpgt_epi32(vremGHIJKLMN, vremainder_threshold));
    vaccOPQRSTUV = _mm256_sub_epi32(_mm256_sra_epi32(vaccOPQRSTUV, vshift), _mm256_cmpgt_epi32(vremOPQRSTUV, vremainder_threshold));

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);
    __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(vaccGHIJKLMN, vaccOPQRSTUV), voutput_zero_point);

    vout012389AB4567CDEF = _mm256_min_epi16(_mm256_max_epi16(vout012389AB4567CDEF, voutput_min), voutput_max);
    voutGHIJOPQRKLMNSTUV = _mm256_min_epi16(_mm256_max_epi16(voutGHIJOPQRKLMNSTUV, voutput_min), voutput_max);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
    __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV), _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
    output += 32;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m256i va01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
      const __m256i vb01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;

      __m256i vacc01234567 = _mm256_add_epi32(vzero_point_product, _mm256_mullo_epi32(va01234567, va_multiplier));

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));

      const __m256i vrem01234567 = _mm256_add_epi32(_mm256_and_si256(vacc01234567, vremainder_mask), _mm256_srai_epi32(vacc01234567, 31));

      vacc01234567 = _mm256_sub_epi32(_mm256_sra_epi32(vacc01234567, vshift), _mm256_cmpgt_epi32(vrem01234567, vremainder_threshold));

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), _mm256_castsi256_si128(voutput_zero_point));
      vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, _mm256_castsi256_si128(voutput_min)), _mm256_castsi256_si128(voutput_max));
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
