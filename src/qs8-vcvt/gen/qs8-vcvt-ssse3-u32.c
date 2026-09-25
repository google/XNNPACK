// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <tmmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_vcvt_ukernel__ssse3_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t multiplier = params->scalar.multiplier;
  const int16_t mult_hi = (int16_t) (multiplier >> 8 <= 32767 ? (multiplier >> 8) : 32767);
  const int16_t mult_lo = (int16_t) (multiplier - ((int32_t) mult_hi << 8));
  const __m128i vmultiplier = _mm_set1_epi32((int32_t) (((uint32_t) (uint16_t) mult_lo) | (((uint32_t) (uint16_t) mult_hi) << 16)));
  const __m128i vbias = _mm_set1_epi32(
      (int32_t) (((uint32_t) (int32_t) params->scalar.output_zero_point) << 16) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point +
      INT32_C(0x8000));
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    const __m128i vm0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx0);
    __m128i vacc0 = _mm_unpacklo_epi8(vx0, vm0);
    __m128i vacc1 = _mm_unpackhi_epi8(vx0, vm0);
    const __m128i vm1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx1);
    __m128i vacc2 = _mm_unpacklo_epi8(vx1, vm1);
    __m128i vacc3 = _mm_unpackhi_epi8(vx1, vm1);

    const __m128i vshift0 = _mm_slli_epi16(vacc0, 8);
    const __m128i va_lo0 = _mm_unpacklo_epi16(vacc0, vshift0);
    const __m128i va_hi0 = _mm_unpackhi_epi16(vacc0, vshift0);
    const __m128i vshift1 = _mm_slli_epi16(vacc1, 8);
    const __m128i va_lo1 = _mm_unpacklo_epi16(vacc1, vshift1);
    const __m128i va_hi1 = _mm_unpackhi_epi16(vacc1, vshift1);
    const __m128i vshift2 = _mm_slli_epi16(vacc2, 8);
    const __m128i va_lo2 = _mm_unpacklo_epi16(vacc2, vshift2);
    const __m128i va_hi2 = _mm_unpackhi_epi16(vacc2, vshift2);
    const __m128i vshift3 = _mm_slli_epi16(vacc3, 8);
    const __m128i va_lo3 = _mm_unpacklo_epi16(vacc3, vshift3);
    const __m128i va_hi3 = _mm_unpackhi_epi16(vacc3, vshift3);

    __m128i vacc_lo0 = _mm_add_epi32(_mm_madd_epi16(va_lo0, vmultiplier), vbias);
    __m128i vacc_hi0 = _mm_add_epi32(_mm_madd_epi16(va_hi0, vmultiplier), vbias);
    __m128i vacc_lo1 = _mm_add_epi32(_mm_madd_epi16(va_lo1, vmultiplier), vbias);
    __m128i vacc_hi1 = _mm_add_epi32(_mm_madd_epi16(va_hi1, vmultiplier), vbias);
    __m128i vacc_lo2 = _mm_add_epi32(_mm_madd_epi16(va_lo2, vmultiplier), vbias);
    __m128i vacc_hi2 = _mm_add_epi32(_mm_madd_epi16(va_hi2, vmultiplier), vbias);
    __m128i vacc_lo3 = _mm_add_epi32(_mm_madd_epi16(va_lo3, vmultiplier), vbias);
    __m128i vacc_hi3 = _mm_add_epi32(_mm_madd_epi16(va_hi3, vmultiplier), vbias);

    vacc_lo0 = _mm_srai_epi32(vacc_lo0, 16);
    vacc_hi0 = _mm_srai_epi32(vacc_hi0, 16);
    vacc_lo1 = _mm_srai_epi32(vacc_lo1, 16);
    vacc_hi1 = _mm_srai_epi32(vacc_hi1, 16);
    vacc_lo2 = _mm_srai_epi32(vacc_lo2, 16);
    vacc_hi2 = _mm_srai_epi32(vacc_hi2, 16);
    vacc_lo3 = _mm_srai_epi32(vacc_lo3, 16);
    vacc_hi3 = _mm_srai_epi32(vacc_hi3, 16);

    vacc0 = _mm_packs_epi32(vacc_lo0, vacc_hi0);
    vacc1 = _mm_packs_epi32(vacc_lo1, vacc_hi1);
    vacc2 = _mm_packs_epi32(vacc_lo2, vacc_hi2);
    vacc3 = _mm_packs_epi32(vacc_lo3, vacc_hi3);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vm);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vm);

    const __m128i vshift_lo = _mm_slli_epi16(vacc_lo, 8);
    const __m128i vshift_hi = _mm_slli_epi16(vacc_hi, 8);
    const __m128i va_ll = _mm_unpacklo_epi16(vacc_lo, vshift_lo);
    const __m128i va_lh = _mm_unpackhi_epi16(vacc_lo, vshift_lo);
    const __m128i va_hl = _mm_unpacklo_epi16(vacc_hi, vshift_hi);
    const __m128i va_hh = _mm_unpackhi_epi16(vacc_hi, vshift_hi);

    __m128i vacc_ll = _mm_add_epi32(_mm_madd_epi16(va_ll, vmultiplier), vbias);
    __m128i vacc_lh = _mm_add_epi32(_mm_madd_epi16(va_lh, vmultiplier), vbias);
    __m128i vacc_hl = _mm_add_epi32(_mm_madd_epi16(va_hl, vmultiplier), vbias);
    __m128i vacc_hh = _mm_add_epi32(_mm_madd_epi16(va_hh, vmultiplier), vbias);

    vacc_ll = _mm_srai_epi32(vacc_ll, 16);
    vacc_lh = _mm_srai_epi32(vacc_lh, 16);
    vacc_hl = _mm_srai_epi32(vacc_hl, 16);
    vacc_hh = _mm_srai_epi32(vacc_hh, 16);

    vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    const __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vm);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vm);

    const __m128i vshift_lo = _mm_slli_epi16(vacc_lo, 8);
    const __m128i vshift_hi = _mm_slli_epi16(vacc_hi, 8);
    const __m128i va_ll = _mm_unpacklo_epi16(vacc_lo, vshift_lo);
    const __m128i va_lh = _mm_unpackhi_epi16(vacc_lo, vshift_lo);
    const __m128i va_hl = _mm_unpacklo_epi16(vacc_hi, vshift_hi);
    const __m128i va_hh = _mm_unpackhi_epi16(vacc_hi, vshift_hi);

    __m128i vacc_ll = _mm_add_epi32(_mm_madd_epi16(va_ll, vmultiplier), vbias);
    __m128i vacc_lh = _mm_add_epi32(_mm_madd_epi16(va_lh, vmultiplier), vbias);
    __m128i vacc_hl = _mm_add_epi32(_mm_madd_epi16(va_hl, vmultiplier), vbias);
    __m128i vacc_hh = _mm_add_epi32(_mm_madd_epi16(va_hh, vmultiplier), vbias);

    vacc_ll = _mm_srai_epi32(vacc_ll, 16);
    vacc_lh = _mm_srai_epi32(vacc_lh, 16);
    vacc_hl = _mm_srai_epi32(vacc_hl, 16);
    vacc_hh = _mm_srai_epi32(vacc_hh, 16);

    vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
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
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) vy_lo;
    }
  }
}
