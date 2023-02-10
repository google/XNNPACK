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
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>

void xnn_qs16_qs8_vcvt_ukernel__sse2_x8(
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

  for (; batch >= 8 * sizeof(int16_t); batch -= 8 * sizeof(int16_t)) {
    __m128i vx0 = _mm_loadu_si128((const __m128i*) input); input += 8;

    // Add 0x8000 to convert signed inputs to unsigned.
    vx0 = _mm_xor_si128(vx0, vinput_bias);

    const __m128i vu0 = _mm_unpacklo_epi16(vx0, vzero);
    const __m128i vu1 = _mm_unpackhi_epi16(vx0, vzero);

    __m128i vacc0lo = _mm_unpacklo_epi32(vu0, vzero);  // low
    __m128i vacc0hi = _mm_unpackhi_epi32(vu0, vzero);  // high
    __m128i vacc1lo = _mm_unpacklo_epi32(vu1, vzero);  // low
    __m128i vacc1hi = _mm_unpackhi_epi32(vu1, vzero);  // high

    vacc0lo = _mm_mul_epu32(vacc0lo, vmultiplier);
    vacc0hi = _mm_mul_epu32(vacc0hi, vmultiplier);
    vacc1lo = _mm_mul_epu32(vacc1lo, vmultiplier);
    vacc1hi = _mm_mul_epu32(vacc1hi, vmultiplier);

    vacc0lo = _mm_add_epi64(vacc0lo, vbias);
    vacc0hi = _mm_add_epi64(vacc0hi, vbias);
    vacc1lo = _mm_add_epi64(vacc1lo, vbias);
    vacc1hi = _mm_add_epi64(vacc1hi, vbias);

    vacc0lo = _mm_srli_epi64(vacc0lo, 16);
    vacc0hi = _mm_srli_epi64(vacc0hi, 16);
    vacc1lo = _mm_srli_epi64(vacc1lo, 16);
    vacc1hi = _mm_srli_epi64(vacc1hi, 16);

    __m128i vacc0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc0lo),
                                                            _mm_castsi128_ps(vacc0hi),
                                                            _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i vacc1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc1lo),
                                                            _mm_castsi128_ps(vacc1hi),
                                                            _MM_SHUFFLE(2, 0, 2, 0)));

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);

    // Pack 8 shorts into 8 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc0);

    _mm_storel_epi64((__m128i*) output, vy0); output += 8;
  }

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
    unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
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
    const __m128i vy = _mm_packs_epi16(vacc, vacc);

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
