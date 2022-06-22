// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/sse4.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vlrelu.h>


void xnn_qu8_vlrelu_ukernel__sse41_x32(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vinput_zero_point = _mm_load_si128((const __m128i*) params->sse2.input_zero_point);
  const __m128i vmultiplier_diff = _mm_load_si128((const __m128i*) params->sse2.multiplier_diff);
  const __m128i vmultiplier_base = _mm_load_si128((const __m128i*) params->sse2.multiplier_base);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
  for (; n >= 32 * sizeof(uint8_t); n -= 32 * sizeof(uint8_t)) {
    __m128i vacc0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) x));
    __m128i vacc1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (x + 8)));
    __m128i vacc2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (x + 16)));
    __m128i vacc3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (x + 24)));
    x += 32;

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vacc0, vinput_zero_point);
    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vacc1, vinput_zero_point);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    __m128i vmultiplier2 = _mm_cmpgt_epi16(vacc2, vinput_zero_point);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    __m128i vmultiplier3 = _mm_cmpgt_epi16(vacc3, vinput_zero_point);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vacc0 = _mm_slli_epi16(vacc0, 7);
    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);
    vmultiplier2 = _mm_and_si128(vmultiplier2, vmultiplier_diff);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vmultiplier2 = _mm_xor_si128(vmultiplier2, vmultiplier_base);
    vmultiplier3 = _mm_and_si128(vmultiplier3, vmultiplier_diff);
    vacc3 = _mm_slli_epi16(vacc3, 7);
    vmultiplier3 = _mm_xor_si128(vmultiplier3, vmultiplier_base);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier0);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier1);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier2);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) y, vy0);
    _mm_storeu_si128((__m128i*) (y + 16), vy1);
    y += 32;
  }
  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    __m128i vacc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) x));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);
    x += 8;

    const __m128i vy = _mm_packus_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 7 * sizeof(uint8_t));

    __m128i vacc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) x));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc, vacc);
    if (n & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(y, vy);
      vy = _mm_srli_epi64(vy, 32);
      y += 4;
    }
    if (n & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(y, vy);
      vy = _mm_srli_epi32(vy, 16);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      *y = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
