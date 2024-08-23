// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <tmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vhswish.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_vhswish_ukernel__ssse3_u32(
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
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    __m128i vextx0 = _mm_unpacklo_epi8(vx0, vzero);
    __m128i vextx1 = _mm_unpackhi_epi8(vx0, vzero);
    __m128i vextx2 = _mm_unpacklo_epi8(vx1, vzero);
    __m128i vextx3 = _mm_unpackhi_epi8(vx1, vzero);

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

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    __m128i vextx0 = _mm_unpacklo_epi8(vx, vzero);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vzero);

    vextx0 = _mm_sub_epi16(vextx0, vinput_zero_point);
    vextx1 = _mm_sub_epi16(vextx1, vinput_zero_point);

    vextx0 = _mm_slli_epi16(vextx0, 7);
    vextx1 = _mm_slli_epi16(vextx1, 7);

    const __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vinput_scale_div);
    const __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vinput_scale_div);
    const __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vinput_scale_div);
    const __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vinput_scale_div);

    __m128i vprod32firstfour0 = _mm_unpacklo_epi16(vprodlo0, vprodhi0);
    __m128i vprod32firstfour1 = _mm_unpacklo_epi16(vprodlo1, vprodhi1);
    vprod32firstfour0 = _mm_sub_epi32(vprod32firstfour0, vhalf);
    vprod32firstfour1 = _mm_sub_epi32(vprod32firstfour1, vhalf);

    __m128i vprod32lastfour0 = _mm_unpackhi_epi16(vprodlo0, vprodhi0);
    __m128i vprod32lastfour1 = _mm_unpackhi_epi16(vprodlo1, vprodhi1);
    vprod32lastfour0 = _mm_sub_epi32(vprod32lastfour0, vhalf);
    vprod32lastfour1 = _mm_sub_epi32(vprod32lastfour1, vhalf);

    __m128i vin0 = _mm_packs_epi32(vprod32firstfour0, vprod32lastfour0);
    __m128i vin1 = _mm_packs_epi32(vprod32firstfour1, vprod32lastfour1);

    vin0 = _mm_min_epi16(vin0, vzero);
    vin1 = _mm_min_epi16(vin1, vzero);

    const __m128i vout0 = _mm_mulhrs_epi16(vextx0, vscale_ratio);
    const __m128i vout1 = _mm_mulhrs_epi16(vextx1, vscale_ratio);

    __m128i vacc0 = _mm_mulhrs_epi16(vout0, vin0);
    __m128i vacc1 = _mm_mulhrs_epi16(vout1, vin1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    const __m128i vy = _mm_packus_epi16(vacc0, vacc1);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    __m128i vextx0 = _mm_unpacklo_epi8(vx, vzero);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vzero);

    vextx0 = _mm_sub_epi16(vextx0, vinput_zero_point);
    vextx1 = _mm_sub_epi16(vextx1, vinput_zero_point);

    vextx0 = _mm_slli_epi16(vextx0, 7);
    vextx1 = _mm_slli_epi16(vextx1, 7);

    const __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vinput_scale_div);
    const __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vinput_scale_div);
    const __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vinput_scale_div);
    const __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vinput_scale_div);

    __m128i vprod32firstfour0 = _mm_unpacklo_epi16(vprodlo0, vprodhi0);
    __m128i vprod32firstfour1 = _mm_unpacklo_epi16(vprodlo1, vprodhi1);
    vprod32firstfour0 = _mm_sub_epi32(vprod32firstfour0, vhalf);
    vprod32firstfour1 = _mm_sub_epi32(vprod32firstfour1, vhalf);

    __m128i vprod32lastfour0 = _mm_unpackhi_epi16(vprodlo0, vprodhi0);
    __m128i vprod32lastfour1 = _mm_unpackhi_epi16(vprodlo1, vprodhi1);
    vprod32lastfour0 = _mm_sub_epi32(vprod32lastfour0, vhalf);
    vprod32lastfour1 = _mm_sub_epi32(vprod32lastfour1, vhalf);

    __m128i vin0 = _mm_packs_epi32(vprod32firstfour0, vprod32lastfour0);
    __m128i vin1 = _mm_packs_epi32(vprod32firstfour1, vprod32lastfour1);

    vin0 = _mm_min_epi16(vin0, vzero);
    vin1 = _mm_min_epi16(vin1, vzero);

    const __m128i vout0 = _mm_mulhrs_epi16(vextx0, vscale_ratio);
    const __m128i vout1 = _mm_mulhrs_epi16(vextx1, vscale_ratio);

    __m128i vacc0 = _mm_mulhrs_epi16(vout0, vin0);
    __m128i vacc1 = _mm_mulhrs_epi16(vout1, vin1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc0, vacc1);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy0 = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      unaligned_store_u16(output, (uint16_t) vy0);
      vy0 >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) vy0;
    }
  }
}
