// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_qu8_vcvt_ukernel__avx2_u64(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vinput_zero_point = _mm256_set1_epi16(params->scalar.input_zero_point);
  const __m256i vmultiplier = _mm256_set1_epi16(-params->scalar.multiplier);
  const __m256i voutput_zero_point = _mm256_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 64 * sizeof(uint8_t); batch -= 64 * sizeof(uint8_t)) {
    __m256i vacc0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vacc1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (input + 16)));
    __m256i vacc2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (input + 32)));
    __m256i vacc3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (input + 48)));
    input += 64;

    vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);
    vacc2 = _mm256_sub_epi16(vinput_zero_point, vacc2);
    vacc3 = _mm256_sub_epi16(vinput_zero_point, vacc3);

    vacc0 = _mm256_slli_epi16(vacc0, 7);
    vacc1 = _mm256_slli_epi16(vacc1, 7);
    vacc2 = _mm256_slli_epi16(vacc2, 7);
    vacc3 = _mm256_slli_epi16(vacc3, 7);

    vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier);
    vacc2 = _mm256_mulhrs_epi16(vacc2, vmultiplier);
    vacc3 = _mm256_mulhrs_epi16(vacc3, vmultiplier);

    vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm256_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm256_adds_epi16(vacc3, voutput_zero_point);

    __m256i vy0 = _mm256_packus_epi16(vacc0, vacc1);
    __m256i vy1 = _mm256_packus_epi16(vacc2, vacc3);

    vy0 = _mm256_permute4x64_epi64(vy0, _MM_SHUFFLE(3, 1, 2, 0));
    vy1 = _mm256_permute4x64_epi64(vy1, _MM_SHUFFLE(3, 1, 2, 0));

    _mm256_storeu_si256((__m256i*) output, vy0);
    _mm256_storeu_si256((__m256i*) (output + 32), vy1);
    output += 64;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);
    input += 16;

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    const __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
