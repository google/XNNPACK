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

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>

void xnn_qs16_qs8_vcvt_ukernel__sse2_x4(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_bias = _mm_load_si128((const __m128i*) params->sse2.input_bias);
  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128();


  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) input); input += 4;
    vx = _mm_xor_si128(vx, vinput_bias);  // Convert signed inputs to unsigned.
    const __m128i vu = _mm_unpacklo_epi16(vx, vzero);
    __m128i vacclo = _mm_unpacklo_epi32(vu, vzero);
    __m128i vacchi = _mm_unpackhi_epi32(vu, vzero);
    vacclo = _mm_mul_epu32(vacclo, vmultiplier);
    vacchi = _mm_mul_epu32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    vacclo = _mm_srli_epi64(vacclo, 16);
    vacchi = _mm_srli_epi64(vacchi, 16);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo),
                                                   _mm_castsi128_ps(vacchi), _MM_SHUFFLE(2, 0, 2, 0)));
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
    const __m128i vu = _mm_unpacklo_epi16(vx, vzero);
    __m128i vacclo = _mm_unpacklo_epi32(vu, vzero);
    __m128i vacchi = _mm_unpackhi_epi32(vu, vzero);
    vacclo = _mm_mul_epu32(vacclo, vmultiplier);
    vacchi = _mm_mul_epu32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    vacclo = _mm_srli_epi64(vacclo, 16);
    vacchi = _mm_srli_epi64(vacchi, 16);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo),
                                                   _mm_castsi128_ps(vacchi), _MM_SHUFFLE(2, 0, 2, 0)));
    vacc = _mm_packs_epi32(vacc, vacc);
    __m128i vy = _mm_packs_epi16(vacc, vacc);

    if (batch & (2 * sizeof(int16_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int16_t))) {
      *output = (int8_t) _mm_cvtsi128_si32(vy);
    }
  }
}
