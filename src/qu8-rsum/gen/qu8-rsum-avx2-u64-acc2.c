// Auto-generated file. Do not edit!
//   Template: src/qu8-rsum/avx2.c.in
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


void xnn_qu8_rsum_ukernel__avx2_u64_acc2(
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

  const __m256i vzero = _mm256_setzero_si256();
  __m256i vacc0 = _mm256_setzero_si256();
  __m256i vacc1 = _mm256_setzero_si256();

  for (; batch >= 64; batch -= 64) {
    const __m256i vin0 = _mm256_loadu_si256((const __m256i*) (input + 0));
    const __m256i vin1 = _mm256_loadu_si256((const __m256i*) (input + 32));
    input += 64;
    const __m256i vt0 = _mm256_sad_epu8(vin0, vzero);
    const __m256i vt1 = _mm256_sad_epu8(vin1, vzero);
     vacc0 = _mm256_add_epi32(vacc0, vt0);
     vacc1 = _mm256_add_epi32(vacc1, vt1);
  }
  vacc0 = _mm256_add_epi32(vacc0, vacc1);

  __m128i vacc = _mm_add_epi32(_mm256_castsi256_si128(vacc0), _mm256_extractf128_si256(vacc0, 1));

  for (; batch >= 16; batch -= 16) {
    const __m128i vin = _mm_loadu_si128((const __m128i*) input);
    input += 16;
    const __m128i vt = _mm_sad_epu8(vin, _mm_setzero_si128());
    vacc = _mm_add_epi32(vacc, vt);
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch <= 15);
    const __m128i vmask = _mm_loadu_si128((const __m128i*) &mask_table[16 - batch]);
    const __m128i vt = _mm_sad_epu8(_mm_and_si128(_mm_loadu_si128((const __m128i*) input), vmask), _mm_setzero_si128());
    vacc = _mm_add_epi32(vacc, vt);
  }

  vacc = _mm_hadd_epi32(vacc, vacc);
  vacc = _mm_hadd_epi32(vacc, vacc);
  *output += (uint32_t)_mm_cvtsi128_si32(vacc);
}



