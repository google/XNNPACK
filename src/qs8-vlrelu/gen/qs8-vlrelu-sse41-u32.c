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

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vlrelu.h"


void xnn_qs8_vlrelu_ukernel__sse41_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m128i vacc0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 8)));
    __m128i vacc2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 16)));
    __m128i vacc3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 24)));
    input += 32;

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

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);
    input += 8;

    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
