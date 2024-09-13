// Auto-generated file. Do not edit!
//   Template: src/qu8-rsum/ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

void xnn_qu8_rsum_ukernel__ssse3_u32(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(16) static const int8_t onemask_table[32] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m128i vzero = _mm_setzero_si128();
  __m128i vacc0 = _mm_setzero_si128();

  // 256 uint8s may be summed into an uint16 before overflowing.
  // Each register has 8 lanes and there are 1 accumulators so batch size is 2048
  for (; batch >= 2048; batch -= 2048) {
    for (size_t current_batch = 2048; current_batch > 0; current_batch -= 32) {
      const __m128i vt0 = _mm_sad_epu8(_mm_loadu_si128((const __m128i*) input), vzero); input += 16;
      const __m128i vt1 = _mm_sad_epu8(_mm_loadu_si128((const __m128i*) input), vzero); input += 16;
      vacc0 = _mm_add_epi32(vacc0, vt0);
      vacc0 = _mm_add_epi32(vacc0, vt1);
    }
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 2048);
    for (; batch >= 32; batch -= 32) {
      const __m128i vt0 = _mm_sad_epu8(_mm_loadu_si128((const __m128i*) input), vzero); input += 16;
      const __m128i vt1 = _mm_sad_epu8(_mm_loadu_si128((const __m128i*) input), vzero); input += 16;
      vacc0 = _mm_add_epi32(vacc0, vt0);
      vacc0 = _mm_add_epi32(vacc0, vt1);
    }
  }
  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 2048);
    for (; batch >= 16; batch -= 16) {
      const __m128i vt = _mm_sad_epu8(_mm_loadu_si128((const __m128i*) input), vzero); input += 16;
      vacc0 = _mm_add_epi32(vacc0, vt);
    }
    __m128i vacc16 = _mm_setzero_si128();
    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 15);
      const __m128i vonemask = _mm_loadu_si128((const __m128i*) &onemask_table[16 - batch]);
      const __m128i vt = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*) input), vonemask);
      vacc16 = _mm_add_epi16(vacc16, vt);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_unpacklo_epi16(vacc16, _mm_setzero_si128()));
    vacc0 = _mm_add_epi32(vacc0, _mm_unpackhi_epi16(vacc16, _mm_setzero_si128()));
  }

  __m128i vacc_lo = _mm_hadd_epi32(vacc0, vacc0);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);

  *output += (uint32_t)_mm_cvtsi128_si32(vacc_lo);
}
