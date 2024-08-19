// Auto-generated file. Do not edit!
//   Template: src/qs16-qs8-vcvt/sse4.c.in
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

void xnn_qs16_qs8_vcvt_ukernel__sse41_u16(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  XNN_ALIGN(16) static const uint8_t shuffle01[16] = {
    0x80, 0x80, 0, 1, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 2, 3, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle23[16] = {
    0x80, 0x80, 4, 5, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 6, 7, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle45[16] = {
    0x80, 0x80, 8, 9, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 10, 11, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle67[16] = {
    0x80, 0x80, 12, 13, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 14, 15, 0x80, 0x80, 0x80, 0x80,
  };

  const __m128i vmultiplier = _mm_set1_epi32(params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi64x((int64_t) ((uint64_t) params->scalar.output_zero_point << 32) + INT64_C(0x80000000));
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  const __m128i vshuffle01 = _mm_load_si128((const __m128i*) shuffle01);
  const __m128i vshuffle23 = _mm_load_si128((const __m128i*) shuffle23);

  const __m128i vshuffle45 = _mm_load_si128((const __m128i*) shuffle45);
  const __m128i vshuffle67 = _mm_load_si128((const __m128i*) shuffle67);
  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input); input += 8;
    const __m128i vx2 = _mm_loadu_si128((const __m128i*) input); input += 8;

    // Move int16 to upper part of int32
    __m128i vacc0lo   = _mm_shuffle_epi8(vx0, vshuffle01);
    __m128i vacc0hi   = _mm_shuffle_epi8(vx0, vshuffle23);
    __m128i vacc1lo = _mm_shuffle_epi8(vx0, vshuffle45);
    __m128i vacc1hi = _mm_shuffle_epi8(vx0, vshuffle67);
    __m128i vacc2lo   = _mm_shuffle_epi8(vx2, vshuffle01);
    __m128i vacc2hi   = _mm_shuffle_epi8(vx2, vshuffle23);
    __m128i vacc3lo = _mm_shuffle_epi8(vx2, vshuffle45);
    __m128i vacc3hi = _mm_shuffle_epi8(vx2, vshuffle67);

    vacc0lo = _mm_mul_epi32(vacc0lo, vmultiplier);
    vacc0hi = _mm_mul_epi32(vacc0hi, vmultiplier);
    vacc1lo = _mm_mul_epi32(vacc1lo, vmultiplier);
    vacc1hi = _mm_mul_epi32(vacc1hi, vmultiplier);
    vacc2lo = _mm_mul_epi32(vacc2lo, vmultiplier);
    vacc2hi = _mm_mul_epi32(vacc2hi, vmultiplier);
    vacc3lo = _mm_mul_epi32(vacc3lo, vmultiplier);
    vacc3hi = _mm_mul_epi32(vacc3hi, vmultiplier);

    vacc0lo = _mm_add_epi64(vacc0lo, vbias);
    vacc0hi = _mm_add_epi64(vacc0hi, vbias);
    vacc1lo = _mm_add_epi64(vacc1lo, vbias);
    vacc1hi = _mm_add_epi64(vacc1hi, vbias);
    vacc2lo = _mm_add_epi64(vacc2lo, vbias);
    vacc2hi = _mm_add_epi64(vacc2hi, vbias);
    vacc3lo = _mm_add_epi64(vacc3lo, vbias);
    vacc3hi = _mm_add_epi64(vacc3hi, vbias);

    __m128i vacc0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc0lo), _mm_castsi128_ps(vacc0hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc1lo), _mm_castsi128_ps(vacc1hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc2lo), _mm_castsi128_ps(vacc2hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc3lo), _mm_castsi128_ps(vacc3hi), _MM_SHUFFLE(3, 1, 3, 1)));

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc2 = _mm_packs_epi32(vacc2, vacc3);

    // Pack 16 shorts into 16 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc2);

    _mm_storeu_si128((__m128i*) output, vy0); output += 16;
  }

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input); input += 4;
    __m128i vacclo = _mm_shuffle_epi8(vx, vshuffle01);
    __m128i vacchi = _mm_shuffle_epi8(vx, vshuffle23);
    vacclo = _mm_mul_epi32(vacclo, vmultiplier);
    vacchi = _mm_mul_epi32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo), _mm_castsi128_ps(vacchi), _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_packs_epi32(vacc, vacc);
    const __m128i vy = _mm_packs_epi16(vacc, vacc);

    _mm_storeu_si32(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 3 * sizeof(int16_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    __m128i vacclo = _mm_shuffle_epi8(vx, vshuffle01);
    __m128i vacchi = _mm_shuffle_epi8(vx, vshuffle23);
    vacclo = _mm_mul_epi32(vacclo, vmultiplier);
    vacchi = _mm_mul_epi32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo), _mm_castsi128_ps(vacchi), _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_packs_epi32(vacc, vacc);
    __m128i vy = _mm_packs_epi16(vacc, vacc);

    if (batch & (2 * sizeof(int16_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int16_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
