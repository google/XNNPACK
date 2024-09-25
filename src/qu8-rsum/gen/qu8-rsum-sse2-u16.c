// Auto-generated file. Do not edit!
//   Template: src/qu8-rsum/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

void xnn_qu8_rsum_ukernel__sse2_u16(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(16) static const int8_t mask_table[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m128i vzero = _mm_setzero_si128();
  __m128i vacc0 = _mm_setzero_si128();


  for (; batch >= 16; batch -= 16) {
    const __m128i vin = _mm_loadu_si128((const __m128i*) input);
    input += 16;
    const __m128i vt = _mm_sad_epu8(vin, vzero);
    vacc0 = _mm_add_epi32(vacc0, vt);
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch <= 15);
    const __m128i vmask = _mm_loadu_si128((const __m128i*) &mask_table[16 - batch]);
    const __m128i vt = _mm_sad_epu8(_mm_and_si128(_mm_loadu_si128((const __m128i*) input), vmask), vzero);
    vacc0 = _mm_add_epi32(vacc0, vt);
  }

  vacc0 = _mm_add_epi32(vacc0, _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(vacc0), _mm_castsi128_ps(vacc0))));
  vacc0 = _mm_add_epi32(vacc0, _mm_shuffle_epi32(vacc0, _MM_SHUFFLE(1, 1, 1, 1)));

  *output += (uint32_t)_mm_cvtsi128_si32(vacc0);
}
