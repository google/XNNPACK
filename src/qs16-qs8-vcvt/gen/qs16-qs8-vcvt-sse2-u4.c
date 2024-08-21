// Auto-generated file. Do not edit!
//   Template: src/qs16-qs8-vcvt/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"

void xnn_qs16_qs8_vcvt_ukernel__sse2_u4(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_bias = _mm_set1_epi16(UINT16_C(0x8000));
  const __m128i vmultiplier = _mm_set1_epi32(params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi64x(
      (int64_t) ((uint64_t) params->scalar.output_zero_point << 32) + 
      INT64_C(0x80000000) -
      (INT64_C(0x80000000) * (int64_t) params->scalar.multiplier));
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vinput_bias);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) input); input += 4;
    vx = _mm_xor_si128(vx, vinput_bias);
    __m128i vacce = _mm_unpacklo_epi16(vzero, vx);
    __m128i vacco = _mm_shuffle_epi32(vacce, _MM_SHUFFLE(3, 3, 1, 1));
    vacce = _mm_mul_epu32(vacce, vmultiplier);
    vacco = _mm_mul_epu32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce), _mm_castsi128_ps(vacco),
                                                   _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(3, 1, 2, 0));
    vacc = _mm_packs_epi32(vacc, vacc);
    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    _mm_storeu_si32(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 3 * sizeof(int16_t));

    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_xor_si128(vx, vinput_bias);
    __m128i vacce = _mm_unpacklo_epi16(vzero, vx);
    __m128i vacco = _mm_shuffle_epi32(vacce, _MM_SHUFFLE(3, 3, 1, 1));
    vacce = _mm_mul_epu32(vacce, vmultiplier);
    vacco = _mm_mul_epu32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce), _mm_castsi128_ps(vacco),
                                                   _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(3, 1, 2, 0));
    vacc = _mm_packs_epi32(vacc, vacc);
    __m128i vy = _mm_packs_epi16(vacc, vacc);

    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int16_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int16_t))) {
      *output = (int8_t) vy_lo;
    }
  }
}
