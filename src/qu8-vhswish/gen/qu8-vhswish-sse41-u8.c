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


void xnn_qu8_vhswish_ukernel__sse41_u8(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
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
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i vextx = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));
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

    const __m128i vy = _mm_packus_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    __m128i vextx = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));

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

    __m128i vy = _mm_packus_epi16(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
