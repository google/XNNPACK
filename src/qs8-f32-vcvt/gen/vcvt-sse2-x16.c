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

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>


void xnn_qs8_f32_vcvt_ukernel__sse2_x16(
    size_t n,
    const int8_t* x,
    float* y,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(int8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vsign_mask = _mm_load_si128((const __m128i*) params->sse2.sign_mask);
  const __m128i vmagic_exp = _mm_load_si128((const __m128i*) params->sse2.magic_exp);
  const __m128 vmagic_bias = _mm_load_ps(params->sse2.magic_bias);
  const __m128 vscale = _mm_load_ps(params->sse2.scale);
  const __m128i vzero = _mm_setzero_si128();
  for (; n >= 16 * sizeof(int8_t); n -= 16 * sizeof(int8_t)) {
    __m128i vx01234567 = _mm_loadl_epi64((const __m128i*) x);
    __m128i vx89ABCDEF = _mm_loadl_epi64((const __m128i*) (x + 8));
    x += 16;

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

    _mm_storeu_ps(y, vy0123);
    _mm_storeu_ps(y + 4, vy4567);
    _mm_storeu_ps(y + 8, vy89AB);
    _mm_storeu_ps(y + 12, vyCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) x);
    vx = _mm_xor_si128(vx, vsign_mask);
    vx = _mm_unpacklo_epi8(vx, vzero);
    x += 8;

    __m128 vy_lo = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    __m128 vy_hi = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));

    vy_lo = _mm_sub_ps(vy_lo, vmagic_bias);
    vy_hi = _mm_sub_ps(vy_hi, vmagic_bias);

    vy_lo = _mm_mul_ps(vy_lo, vscale);
    vy_hi = _mm_mul_ps(vy_hi, vscale);

    _mm_storeu_ps(y, vy_lo);
    _mm_storeu_ps(y + 4, vy_hi);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(int8_t));
    assert(n <= 7 * sizeof(int8_t));

    __m128i vx = _mm_loadl_epi64((const __m128i*) x);
    vx = _mm_xor_si128(vx, vsign_mask);
    vx = _mm_unpacklo_epi8(vx, vzero);

    __m128 vy = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    vy = _mm_sub_ps(vy, vmagic_bias);
    vy = _mm_mul_ps(vy, vscale);

    if (n & (4 * sizeof(int8_t))) {
      _mm_storeu_ps(y, vy);
      vy = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));
      vy = _mm_sub_ps(vy, vmagic_bias);
      vy = _mm_mul_ps(vy, vscale);
      y += 4;
    }
    if (n & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & (1 * sizeof(int8_t))) {
      _mm_store_ss(y, vy);
    }
  }
}
