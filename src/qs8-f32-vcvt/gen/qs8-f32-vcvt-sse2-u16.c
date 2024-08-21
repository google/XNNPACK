// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_qs8_f32_vcvt_ukernel__sse2_u16(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_set1_epi8(UINT8_C(0x80));
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m128i vmagic_exp = _mm_set1_epi16(UINT16_C(0x4B00));
  const __m128 vmagic_bias = _mm_set1_ps((float) (INT32_C(0x00800080) + (int32_t) params->scalar.zero_point));
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vmagic_exp);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m128i vx01234567 = _mm_loadl_epi64((const __m128i*) input);
    __m128i vx89ABCDEF = _mm_loadl_epi64((const __m128i*) (input + 8));
    input += 16;

    vx01234567 = _mm_xor_si128(vx01234567, vsign_mask);
    vx89ABCDEF = _mm_xor_si128(vx89ABCDEF, vsign_mask);

    vx01234567 = _mm_unpacklo_epi8(vx01234567, vzero);
    vx89ABCDEF = _mm_unpacklo_epi8(vx89ABCDEF, vzero);

    __m128 vy0123 = _mm_castsi128_ps(_mm_unpacklo_epi16(vx01234567, vmagic_exp));
    __m128 vy4567 = _mm_castsi128_ps(_mm_unpackhi_epi16(vx01234567, vmagic_exp));
    __m128 vy89AB = _mm_castsi128_ps(_mm_unpacklo_epi16(vx89ABCDEF, vmagic_exp));
    __m128 vyCDEF = _mm_castsi128_ps(_mm_unpackhi_epi16(vx89ABCDEF, vmagic_exp));

    vy0123 = _mm_sub_ps(vy0123, vmagic_bias);
    vy4567 = _mm_sub_ps(vy4567, vmagic_bias);
    vy89AB = _mm_sub_ps(vy89AB, vmagic_bias);
    vyCDEF = _mm_sub_ps(vyCDEF, vmagic_bias);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);
    vy89AB = _mm_mul_ps(vy89AB, vscale);
    vyCDEF = _mm_mul_ps(vyCDEF, vscale);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_xor_si128(vx, vsign_mask);
    vx = _mm_unpacklo_epi8(vx, vzero);
    input += 8;

    __m128 vy_lo = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    __m128 vy_hi = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));

    vy_lo = _mm_sub_ps(vy_lo, vmagic_bias);
    vy_hi = _mm_sub_ps(vy_hi, vmagic_bias);

    vy_lo = _mm_mul_ps(vy_lo, vscale);
    vy_hi = _mm_mul_ps(vy_hi, vscale);

    _mm_storeu_ps(output, vy_lo);
    _mm_storeu_ps(output + 4, vy_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_xor_si128(vx, vsign_mask);
    vx = _mm_unpacklo_epi8(vx, vzero);

    __m128 vy = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    vy = _mm_sub_ps(vy, vmagic_bias);
    vy = _mm_mul_ps(vy, vscale);

    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_ps(output, vy);
      vy = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));
      vy = _mm_sub_ps(vy, vmagic_bias);
      vy = _mm_mul_ps(vy, vscale);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      _mm_store_ss(output, vy);
    }
  }
}
