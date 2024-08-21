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

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vbinary.h"


void xnn_qu8_vadd_minmax_ukernel__sse41_mul32_ld32_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.bias);
  const __m128i va_multiplier = _mm_set1_epi32(params->scalar.a_multiplier);
  const __m128i vb_multiplier = _mm_set1_epi32(params->scalar.b_multiplier);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier);
  XNN_FORCE_REALIZATION(vb_multiplier);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i va0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
    const __m128i vb0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b)));
    const __m128i va4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
    const __m128i vb4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 4)));
    const __m128i va89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 8)));
    const __m128i vb89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 8)));
    const __m128i vaCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 12)));
    const __m128i vbCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 12)));
    input_a += 16;
    input_b += 16;

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));
    __m128i vacc89AB = _mm_add_epi32(vbias, _mm_mullo_epi32(va89AB, va_multiplier));
    __m128i vaccCDEF = _mm_add_epi32(vbias, _mm_mullo_epi32(vaCDEF, va_multiplier));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));
    vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vb89AB, vb_multiplier));
    vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vbCDEF, vb_multiplier));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);
    vacc89AB = _mm_sra_epi32(vacc89AB, vshift);
    vaccCDEF = _mm_sra_epi32(vaccCDEF, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    const __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128i va0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
      const __m128i vb0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b)));
      const __m128i va4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
      const __m128i vb4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 4)));
      input_a += 8;
      input_b += 8;

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(uint8_t);
      } else {
        if (batch & (4 * sizeof(uint8_t))) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(uint8_t))) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(uint8_t))) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}
