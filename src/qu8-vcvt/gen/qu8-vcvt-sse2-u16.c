// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <tmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_vcvt_ukernel__sse2_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmultiplier = _mm_set1_epi16(params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi32(
      ((int32_t) params->scalar.output_zero_point << 8) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point + 
      INT32_C(0x80));
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  const __m128i vzero = _mm_setzero_si128();
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vextx0 = _mm_unpacklo_epi8(vx0, vzero);
    const __m128i vextx1 = _mm_unpackhi_epi8(vx0, vzero);

    const __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier);
    const __m128i vprodhi0 = _mm_mulhi_epu16(vextx0, vmultiplier);
    const __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier);
    const __m128i vprodhi1 = _mm_mulhi_epu16(vextx1, vmultiplier);

    __m128i vacc0 = _mm_unpacklo_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_unpackhi_epi16(vprodlo0, vprodhi0);
    __m128i vacc2 = _mm_unpacklo_epi16(vprodlo1, vprodhi1);
    __m128i vacc3 = _mm_unpackhi_epi16(vprodlo1, vprodhi1);

    vacc0 = _mm_add_epi32(vacc0, vbias);
    vacc1 = _mm_add_epi32(vacc1, vbias);
    vacc2 = _mm_add_epi32(vacc2, vbias);
    vacc3 = _mm_add_epi32(vacc3, vbias);

    vacc0 = _mm_srai_epi32(vacc0, 8);
    vacc1 = _mm_srai_epi32(vacc1, 8);
    vacc2 = _mm_srai_epi32(vacc2, 8);
    vacc3 = _mm_srai_epi32(vacc3, 8);

    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc1 = _mm_packs_epi32(vacc2, vacc3);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);

    _mm_storeu_si128((__m128i*) output, vy0);
    output += 16;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vextx_lo = _mm_unpacklo_epi8(vx, vzero);
    const __m128i vextx_hi = _mm_unpackhi_epi8(vx, vzero);

    const __m128i vprodlo_lo = _mm_mullo_epi16(vextx_lo, vmultiplier);
    const __m128i vprodlo_hi = _mm_mullo_epi16(vextx_hi, vmultiplier);
    const __m128i vprodhi_lo = _mm_mulhi_epu16(vextx_lo, vmultiplier);
    const __m128i vprodhi_hi = _mm_mulhi_epu16(vextx_hi, vmultiplier);

    __m128i vacc_ll = _mm_unpacklo_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_lh = _mm_unpackhi_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_hl = _mm_unpacklo_epi16(vprodlo_hi, vprodhi_hi);
    __m128i vacc_hh = _mm_unpackhi_epi16(vprodlo_hi, vprodhi_hi);

    vacc_ll = _mm_add_epi32(vacc_ll, vbias);
    vacc_lh = _mm_add_epi32(vacc_lh, vbias);
    vacc_hl = _mm_add_epi32(vacc_hl, vbias);
    vacc_hh = _mm_add_epi32(vacc_hh, vbias);

    vacc_ll = _mm_srai_epi32(vacc_ll, 8);
    vacc_lh = _mm_srai_epi32(vacc_lh, 8);
    vacc_hl = _mm_srai_epi32(vacc_hl, 8);
    vacc_hh = _mm_srai_epi32(vacc_hh, 8);

    const __m128i vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    const __m128i vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    const __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vextx_lo = _mm_unpacklo_epi8(vx, vzero);
    const __m128i vextx_hi = _mm_unpackhi_epi8(vx, vzero);

    const __m128i vprodlo_lo = _mm_mullo_epi16(vextx_lo, vmultiplier);
    const __m128i vprodlo_hi = _mm_mullo_epi16(vextx_hi, vmultiplier);
    const __m128i vprodhi_lo = _mm_mulhi_epu16(vextx_lo, vmultiplier);
    const __m128i vprodhi_hi = _mm_mulhi_epu16(vextx_hi, vmultiplier);

    __m128i vacc_ll = _mm_unpacklo_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_lh = _mm_unpackhi_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_hl = _mm_unpacklo_epi16(vprodlo_hi, vprodhi_hi);
    __m128i vacc_hh = _mm_unpackhi_epi16(vprodlo_hi, vprodhi_hi);

    vacc_ll = _mm_add_epi32(vacc_ll, vbias);
    vacc_lh = _mm_add_epi32(vacc_lh, vbias);
    vacc_hl = _mm_add_epi32(vacc_hl, vbias);
    vacc_hh = _mm_add_epi32(vacc_hh, vbias);

    vacc_ll = _mm_srai_epi32(vacc_ll, 8);
    vacc_lh = _mm_srai_epi32(vacc_lh, 8);
    vacc_hl = _mm_srai_epi32(vacc_hl, 8);
    vacc_hh = _mm_srai_epi32(vacc_hh, 8);

    const __m128i vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    const __m128i vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
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
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) vy_lo;
    }
  }
}
