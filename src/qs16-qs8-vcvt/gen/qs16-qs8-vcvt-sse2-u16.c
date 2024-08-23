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

void xnn_qs16_qs8_vcvt_ukernel__sse2_u16(
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
  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    __m128i vx0 = _mm_loadu_si128((const __m128i*) input); input += 8;
    __m128i vx2 = _mm_loadu_si128((const __m128i*) input); input += 8;

    // Add 0x8000 to convert signed inputs to unsigned.
    vx0 = _mm_xor_si128(vx0, vinput_bias);
    vx2 = _mm_xor_si128(vx2, vinput_bias);

    // Move int16 to upper part of int32
    __m128i vacce0 = _mm_unpacklo_epi16(vzero, vx0);
    __m128i vacce1 = _mm_unpackhi_epi16(vzero, vx0);
    __m128i vacce2 = _mm_unpacklo_epi16(vzero, vx2);
    __m128i vacce3 = _mm_unpackhi_epi16(vzero, vx2);

    __m128i vacco0 = _mm_shuffle_epi32(vacce0, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco1 = _mm_shuffle_epi32(vacce1, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco2 = _mm_shuffle_epi32(vacce2, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco3 = _mm_shuffle_epi32(vacce3, _MM_SHUFFLE(3, 3, 1, 1));

    vacce0 = _mm_mul_epu32(vacce0, vmultiplier);
    vacco0 = _mm_mul_epu32(vacco0, vmultiplier);
    vacce1 = _mm_mul_epu32(vacce1, vmultiplier);
    vacco1 = _mm_mul_epu32(vacco1, vmultiplier);
    vacce2 = _mm_mul_epu32(vacce2, vmultiplier);
    vacco2 = _mm_mul_epu32(vacco2, vmultiplier);
    vacce3 = _mm_mul_epu32(vacce3, vmultiplier);
    vacco3 = _mm_mul_epu32(vacco3, vmultiplier);

    vacce0 = _mm_add_epi64(vacce0, vbias);
    vacco0 = _mm_add_epi64(vacco0, vbias);
    vacce1 = _mm_add_epi64(vacce1, vbias);
    vacco1 = _mm_add_epi64(vacco1, vbias);
    vacce2 = _mm_add_epi64(vacce2, vbias);
    vacco2 = _mm_add_epi64(vacco2, vbias);
    vacce3 = _mm_add_epi64(vacce3, vbias);
    vacco3 = _mm_add_epi64(vacco3, vbias);

    __m128i vacc0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce0),
                                                            _mm_castsi128_ps(vacco0),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce1),
                                                            _mm_castsi128_ps(vacco1),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce2),
                                                            _mm_castsi128_ps(vacco2),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce3),
                                                            _mm_castsi128_ps(vacco3),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));

    // Shuffle order from 3,1,2,0 to 3,2,1,0
    vacc0 = _mm_shuffle_epi32(vacc0, _MM_SHUFFLE(3, 1, 2, 0));
    vacc1 = _mm_shuffle_epi32(vacc1, _MM_SHUFFLE(3, 1, 2, 0));
    vacc2 = _mm_shuffle_epi32(vacc2, _MM_SHUFFLE(3, 1, 2, 0));
    vacc3 = _mm_shuffle_epi32(vacc3, _MM_SHUFFLE(3, 1, 2, 0));

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc2 = _mm_packs_epi32(vacc2, vacc3);

    // Pack 16 shorts into 16 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc2);

    _mm_storeu_si128((__m128i*) output, vy0); output += 16;
  }

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
