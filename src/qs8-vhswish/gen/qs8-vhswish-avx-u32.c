// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/sse4.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vhswish.h"


void xnn_qs8_vhswish_ukernel__avx_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->sse2.input_zero_point);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->sse2.output_zero_point);
  const __m128i vinput_scale_div = _mm_set1_epi16(params->sse2.input_scale_div);
  const __m128i vscale_ratio = _mm_set1_epi16(params->sse2.scale_ratio);
  const __m128i vhalf = _mm_set1_epi32(0x4000);
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(vinput_scale_div);
  XNN_FORCE_REALIZATION(vscale_ratio);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m128i vextx0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vextx1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 8)));
    __m128i vextx2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 16)));
    __m128i vextx3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 24)));
    input += 32;

    vextx0 = _mm_sub_epi16(vextx0, vinput_zero_point);
    vextx1 = _mm_sub_epi16(vextx1, vinput_zero_point);
    vextx2 = _mm_sub_epi16(vextx2, vinput_zero_point);
    vextx3 = _mm_sub_epi16(vextx3, vinput_zero_point);

    vextx0 = _mm_slli_epi16(vextx0, 7);
    vextx1 = _mm_slli_epi16(vextx1, 7);
    vextx2 = _mm_slli_epi16(vextx2, 7);
    vextx3 = _mm_slli_epi16(vextx3, 7);

    const __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vinput_scale_div);
    const __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vinput_scale_div);
    const __m128i vprodlo2 = _mm_mullo_epi16(vextx2, vinput_scale_div);
    const __m128i vprodlo3 = _mm_mullo_epi16(vextx3, vinput_scale_div);

    const __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vinput_scale_div);
    const __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vinput_scale_div);
    const __m128i vprodhi2 = _mm_mulhi_epi16(vextx2, vinput_scale_div);
    const __m128i vprodhi3 = _mm_mulhi_epi16(vextx3, vinput_scale_div);

    __m128i vprod32firstfour0 = _mm_unpacklo_epi16(vprodlo0, vprodhi0);
    vprod32firstfour0 = _mm_sub_epi32(vprod32firstfour0, vhalf);
    __m128i vprod32firstfour1 = _mm_unpacklo_epi16(vprodlo1, vprodhi1);
    vprod32firstfour1 = _mm_sub_epi32(vprod32firstfour1, vhalf);
    __m128i vprod32firstfour2 = _mm_unpacklo_epi16(vprodlo2, vprodhi2);
    vprod32firstfour2 = _mm_sub_epi32(vprod32firstfour2, vhalf);
    __m128i vprod32firstfour3 = _mm_unpacklo_epi16(vprodlo3, vprodhi3);
    vprod32firstfour3 = _mm_sub_epi32(vprod32firstfour3, vhalf);

    __m128i vprod32lastfour0 = _mm_unpackhi_epi16(vprodlo0, vprodhi0);
    vprod32lastfour0 = _mm_sub_epi32(vprod32lastfour0, vhalf);
    __m128i vprod32lastfour1 = _mm_unpackhi_epi16(vprodlo1, vprodhi1);
    vprod32lastfour1 = _mm_sub_epi32(vprod32lastfour1, vhalf);
    __m128i vprod32lastfour2 = _mm_unpackhi_epi16(vprodlo2, vprodhi2);
    vprod32lastfour2 = _mm_sub_epi32(vprod32lastfour2, vhalf);
    __m128i vprod32lastfour3 = _mm_unpackhi_epi16(vprodlo3, vprodhi3);
    vprod32lastfour3 = _mm_sub_epi32(vprod32lastfour3, vhalf);

    __m128i vin0 = _mm_packs_epi32(vprod32firstfour0, vprod32lastfour0);
    __m128i vin1 = _mm_packs_epi32(vprod32firstfour1, vprod32lastfour1);
    __m128i vin2 = _mm_packs_epi32(vprod32firstfour2, vprod32lastfour2);
    __m128i vin3 = _mm_packs_epi32(vprod32firstfour3, vprod32lastfour3);

    vin0 = _mm_min_epi16(vin0, vzero);
    vin1 = _mm_min_epi16(vin1, vzero);
    vin2 = _mm_min_epi16(vin2, vzero);
    vin3 = _mm_min_epi16(vin3, vzero);

    const __m128i vout0 = _mm_mulhrs_epi16(vextx0, vscale_ratio);
    const __m128i vout1 = _mm_mulhrs_epi16(vextx1, vscale_ratio);
    const __m128i vout2 = _mm_mulhrs_epi16(vextx2, vscale_ratio);
    const __m128i vout3 = _mm_mulhrs_epi16(vextx3, vscale_ratio);

    __m128i vacc0 = _mm_mulhrs_epi16(vout0, vin0);
    __m128i vacc1 = _mm_mulhrs_epi16(vout1, vin1);
    __m128i vacc2 = _mm_mulhrs_epi16(vout2, vin2);
    __m128i vacc3 = _mm_mulhrs_epi16(vout3, vin3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i vextx = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    vextx = _mm_sub_epi16(vextx, vinput_zero_point);
    vextx = _mm_slli_epi16(vextx, 7);
    const __m128i vprodlo = _mm_mullo_epi16(vextx, vinput_scale_div);
    const __m128i vprodhi = _mm_mulhi_epi16(vextx, vinput_scale_div);
    __m128i vprod32firstfour = _mm_unpacklo_epi16(vprodlo, vprodhi);
    vprod32firstfour = _mm_sub_epi32(vprod32firstfour, vhalf);
    __m128i vprod32lastfour = _mm_unpackhi_epi16(vprodlo, vprodhi);
    vprod32lastfour = _mm_sub_epi32(vprod32lastfour, vhalf);
    __m128i vin = _mm_packs_epi32(vprod32firstfour, vprod32lastfour);
    vin = _mm_min_epi16(vin, vzero);
    const __m128i vout = _mm_mulhrs_epi16(vextx, vscale_ratio);
    __m128i vacc = _mm_mulhrs_epi16(vout, vin);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);
    input += 8;

    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m128i vextx = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));

    vextx = _mm_sub_epi16(vextx, vinput_zero_point);
    vextx = _mm_slli_epi16(vextx, 7);
    const __m128i vprodlo = _mm_mullo_epi16(vextx, vinput_scale_div);
    const __m128i vprodhi = _mm_mulhi_epi16(vextx, vinput_scale_div);
    __m128i vprod32firstfour = _mm_unpacklo_epi16(vprodlo, vprodhi);
    vprod32firstfour = _mm_sub_epi32(vprod32firstfour, vhalf);
    __m128i vprod32lastfour = _mm_unpackhi_epi16(vprodlo, vprodhi);
    vprod32lastfour = _mm_sub_epi32(vprod32lastfour, vhalf);
    __m128i vin = _mm_packs_epi32(vprod32firstfour, vprod32lastfour);
    vin = _mm_min_epi16(vin, vzero);
    const __m128i vout = _mm_mulhrs_epi16(vextx, vscale_ratio);
    __m128i vacc = _mm_mulhrs_epi16(vout, vin);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
