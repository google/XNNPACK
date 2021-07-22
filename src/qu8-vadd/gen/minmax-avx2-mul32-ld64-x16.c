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


void xnn_qu8_vadd_minmax_ukernel__avx2_mul32_ld64_x16(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  const __m256i vbias = _mm256_load_si256((const __m256i*) params->avx2.bias);
  const __m256i va_multiplier = _mm256_load_si256((const __m256i*) params->avx2.a_multiplier);
  const __m256i vb_multiplier = _mm256_load_si256((const __m256i*) params->avx2.b_multiplier);
  const __m256i vrounding = _mm256_load_si256((const __m256i*) params->avx2.rounding);
  const __m128i vshift = _mm_loadu_si32(params->avx2.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx2.output_max);

  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    const __m256i va01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
    const __m256i vb01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
    const __m256i va89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m256i vb89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;

    __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));
    __m256i vacc89ABCDEF = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va89ABCDEF, va_multiplier));

    vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));
    vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vb89ABCDEF, vb_multiplier));

    vacc01234567 = _mm256_sra_epi32(_mm256_add_epi32(vacc01234567, vrounding), vshift);
    vacc89ABCDEF = _mm256_sra_epi32(_mm256_add_epi32(vacc89ABCDEF, vrounding), vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m256i va01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
      const __m256i vb01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;

      __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));

      vacc01234567 = _mm256_sra_epi32(_mm256_add_epi32(vacc01234567, vrounding), vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), _mm256_castsi256_si128(voutput_zero_point));
      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(n >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        n -= 8 * sizeof(uint8_t);
      } else {
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
        n = 0;
      }
    } while (n != 0);
  }
}
