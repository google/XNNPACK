// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/sse4.c.in
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
#include <xnnpack/vcvt.h>


void xnn_qu8_vcvt_ukernel__sse41_x8(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vinput_zero_point = _mm_load_si128((const __m128i*) params->ssse3.input_zero_point);
  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->ssse3.multiplier);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->ssse3.output_zero_point);
  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    __m128i vacc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) x));
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm_slli_epi16(vacc, 7);
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
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm_slli_epi16(vacc, 7);
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
