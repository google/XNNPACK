// Auto-generated file. Do not edit!
//   Template: src/qs8-vmul/sse-mul16-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vbinary.h"


void xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
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

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);
    input_a += 8;
    input_b += 8;

    const __m128i vzero = _mm_setzero_si128();
    va01234567 = _mm_unpacklo_epi8(va01234567, vzero);
    vb01234567 = _mm_unpacklo_epi8(vb01234567, vzero);

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


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
      __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);

      const __m128i vzero = _mm_setzero_si128();
      va01234567 = _mm_unpacklo_epi8(va01234567, vzero);
      vb01234567 = _mm_unpacklo_epi8(vb01234567, vzero);

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

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}
