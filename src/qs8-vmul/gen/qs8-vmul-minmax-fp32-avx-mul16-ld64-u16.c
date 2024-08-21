// Auto-generated file. Do not edit!
//   Template: src/qs8-vmul/sse-mul16-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vbinary.h"


void xnn_qs8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_u16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128i vb_zero_point = _mm_set1_epi16(params->scalar.b_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vb_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    const __m128i va89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m128i vb89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);
    const __m128i vxb89ABCDEF = _mm_sub_epi16(vb89ABCDEF, vb_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb89ABCDEF);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb89ABCDEF);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
      const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(int8_t);
      } else {
        if (batch & (4 * sizeof(int8_t))) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(int8_t))) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}
