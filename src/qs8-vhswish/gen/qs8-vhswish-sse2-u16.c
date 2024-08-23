// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vhswish.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_vhswish_ukernel__sse2_u16(
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
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vextx0 = _mm_unpacklo_epi8(vx, vm);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vm);

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

    const __m128i voutlo0 =  _mm_mullo_epi16(vextx0, vscale_ratio);
    const __m128i voutlo1 =  _mm_mullo_epi16(vextx1, vscale_ratio);

    const __m128i vouthi0 =  _mm_mulhi_epi16(vextx0, vscale_ratio);
    const __m128i vouthi1 =  _mm_mulhi_epi16(vextx1, vscale_ratio);

    __m128i voutfirstfour0 = _mm_unpacklo_epi16(voutlo0, vouthi0);
    __m128i voutlastfour0 = _mm_unpackhi_epi16(voutlo0, vouthi0);
    __m128i voutfirstfour1 = _mm_unpacklo_epi16(voutlo1, vouthi1);
    __m128i voutlastfour1 = _mm_unpackhi_epi16(voutlo1, vouthi1);

    voutfirstfour0 = _mm_add_epi32(voutfirstfour0, vhalf);
    voutlastfour0 = _mm_add_epi32(voutlastfour0, vhalf);
    voutfirstfour1 = _mm_add_epi32(voutfirstfour1, vhalf);
    voutlastfour1 = _mm_add_epi32(voutlastfour1, vhalf);

    voutfirstfour0 = _mm_srai_epi32(voutfirstfour0, 15);
    voutlastfour0 = _mm_srai_epi32(voutlastfour0, 15);
    voutfirstfour1 = _mm_srai_epi32(voutfirstfour1, 15);
    voutlastfour1 = _mm_srai_epi32(voutlastfour1, 15);

    const __m128i vout0 = _mm_packs_epi32(voutfirstfour0, voutlastfour0);
    const __m128i vout1 = _mm_packs_epi32(voutfirstfour1, voutlastfour1);

    const __m128i vacclo0 = _mm_mullo_epi16(vout0, vin0);
    const __m128i vacclo1 = _mm_mullo_epi16(vout1, vin1);
    const __m128i vacchi0 = _mm_mulhi_epi16(vout0, vin0);
    const __m128i vacchi1 = _mm_mulhi_epi16(vout1, vin1);

    __m128i vaccfirst0 = _mm_unpacklo_epi16(vacclo0, vacchi0);
    __m128i vaccfirst1 = _mm_unpacklo_epi16(vacclo1, vacchi1);
    vaccfirst0 = _mm_add_epi32(vaccfirst0, vhalf);
    vaccfirst1 = _mm_add_epi32(vaccfirst1, vhalf);
    vaccfirst0 = _mm_srai_epi32(vaccfirst0, 15);
    vaccfirst1 = _mm_srai_epi32(vaccfirst1, 15);

    __m128i vaccsec0 = _mm_unpackhi_epi16(vacclo0, vacchi0);
    __m128i vaccsec1 = _mm_unpackhi_epi16(vacclo1, vacchi1);
    vaccsec0 = _mm_add_epi32(vaccsec0, vhalf);
    vaccsec1 = _mm_add_epi32(vaccsec1, vhalf);
    vaccsec0 = _mm_srai_epi32(vaccsec0, 15);
    vaccsec1 = _mm_srai_epi32(vaccsec1, 15);

    __m128i vacc0 = _mm_packs_epi32(vaccfirst0, vaccsec0);
    __m128i vacc1 = _mm_packs_epi32(vaccfirst1, vaccsec1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    const __m128i vy = _mm_packs_epi16(vacc0, vacc1);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vextx0 = _mm_unpacklo_epi8(vx, vm);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vm);

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

    const __m128i voutlo0 =  _mm_mullo_epi16(vextx0, vscale_ratio);
    const __m128i voutlo1 =  _mm_mullo_epi16(vextx1, vscale_ratio);

    const __m128i vouthi0 =  _mm_mulhi_epi16(vextx0, vscale_ratio);
    const __m128i vouthi1 =  _mm_mulhi_epi16(vextx1, vscale_ratio);

    __m128i voutfirstfour0 = _mm_unpacklo_epi16(voutlo0, vouthi0);
    __m128i voutlastfour0 = _mm_unpackhi_epi16(voutlo0, vouthi0);
    __m128i voutfirstfour1 = _mm_unpacklo_epi16(voutlo1, vouthi1);
    __m128i voutlastfour1 = _mm_unpackhi_epi16(voutlo1, vouthi1);

    voutfirstfour0 = _mm_add_epi32(voutfirstfour0, vhalf);
    voutlastfour0 = _mm_add_epi32(voutlastfour0, vhalf);
    voutfirstfour1 = _mm_add_epi32(voutfirstfour1, vhalf);
    voutlastfour1 = _mm_add_epi32(voutlastfour1, vhalf);

    voutfirstfour0 = _mm_srai_epi32(voutfirstfour0, 15);
    voutlastfour0 = _mm_srai_epi32(voutlastfour0, 15);
    voutfirstfour1 = _mm_srai_epi32(voutfirstfour1, 15);
    voutlastfour1 = _mm_srai_epi32(voutlastfour1, 15);

    const __m128i vout0 = _mm_packs_epi32(voutfirstfour0, voutlastfour0);
    const __m128i vout1 = _mm_packs_epi32(voutfirstfour1, voutlastfour1);

    const __m128i vacclo0 = _mm_mullo_epi16(vout0, vin0);
    const __m128i vacclo1 = _mm_mullo_epi16(vout1, vin1);
    const __m128i vacchi0 = _mm_mulhi_epi16(vout0, vin0);
    const __m128i vacchi1 = _mm_mulhi_epi16(vout1, vin1);

    __m128i vaccfirst0 = _mm_unpacklo_epi16(vacclo0, vacchi0);
    __m128i vaccfirst1 = _mm_unpacklo_epi16(vacclo1, vacchi1);
    vaccfirst0 = _mm_srai_epi32(vaccfirst0, 15);
    vaccfirst1 = _mm_srai_epi32(vaccfirst1, 15);

    __m128i vaccsec0 = _mm_unpackhi_epi16(vacclo0, vacchi0);
    __m128i vaccsec1 = _mm_unpackhi_epi16(vacclo1, vacchi1);
    vaccsec0 = _mm_srai_epi32(vaccsec0, 15);
    vaccsec1 = _mm_srai_epi32(vaccsec1, 15);

    __m128i vacc0 = _mm_packs_epi32(vaccfirst0, vaccsec0);
    __m128i vacc1 = _mm_packs_epi32(vaccfirst1, vaccsec1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc0, vacc1);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy0 = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      unaligned_store_u16(output, (uint16_t) vy0);
      vy0 >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) vy0;
    }
  }
}
