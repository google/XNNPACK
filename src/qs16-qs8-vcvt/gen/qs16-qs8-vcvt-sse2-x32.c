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

void xnn_qs16_qs8_vcvt_ukernel__sse2_x32(
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

  for (; batch >= 32 * sizeof(int16_t); batch -= 32 * sizeof(int16_t)) {
    __m128i vx0 = _mm_loadu_si128((const __m128i*) input); input += 8;
    __m128i vx2 = _mm_loadu_si128((const __m128i*) input); input += 8;
    __m128i vx4 = _mm_loadu_si128((const __m128i*) input); input += 8;
    __m128i vx6 = _mm_loadu_si128((const __m128i*) input); input += 8;

    // Add 0x8000 to convert signed inputs to unsigned.
    vx0 = _mm_xor_si128(vx0, vinput_bias);
    vx2 = _mm_xor_si128(vx2, vinput_bias);
    vx4 = _mm_xor_si128(vx4, vinput_bias);
    vx6 = _mm_xor_si128(vx6, vinput_bias);

    const __m128i vu0 = _mm_unpacklo_epi16(vx0, vzero);
    const __m128i vu1 = _mm_unpackhi_epi16(vx0, vzero);
    const __m128i vu2 = _mm_unpacklo_epi16(vx2, vzero);
    const __m128i vu3 = _mm_unpackhi_epi16(vx2, vzero);
    const __m128i vu4 = _mm_unpacklo_epi16(vx4, vzero);
    const __m128i vu5 = _mm_unpackhi_epi16(vx4, vzero);
    const __m128i vu6 = _mm_unpacklo_epi16(vx6, vzero);
    const __m128i vu7 = _mm_unpackhi_epi16(vx6, vzero);

    __m128i vacco0 = _mm_shuffle_epi32(vu0, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce0 = _mm_shuffle_epi32(vu0, _MM_SHUFFLE(3, 1, 2, 0));  // low
    __m128i vacco1 = _mm_shuffle_epi32(vu1, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce1 = _mm_shuffle_epi32(vu1, _MM_SHUFFLE(3, 1, 2, 0));  // low
    __m128i vacco2 = _mm_shuffle_epi32(vu2, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce2 = _mm_shuffle_epi32(vu2, _MM_SHUFFLE(3, 1, 2, 0));  // low
    __m128i vacco3 = _mm_shuffle_epi32(vu3, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce3 = _mm_shuffle_epi32(vu3, _MM_SHUFFLE(3, 1, 2, 0));  // low
    __m128i vacco4 = _mm_shuffle_epi32(vu4, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce4 = _mm_shuffle_epi32(vu4, _MM_SHUFFLE(3, 1, 2, 0));  // low
    __m128i vacco5 = _mm_shuffle_epi32(vu5, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce5 = _mm_shuffle_epi32(vu5, _MM_SHUFFLE(3, 1, 2, 0));  // low
    __m128i vacco6 = _mm_shuffle_epi32(vu6, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce6 = _mm_shuffle_epi32(vu6, _MM_SHUFFLE(3, 1, 2, 0));  // low
    __m128i vacco7 = _mm_shuffle_epi32(vu7, _MM_SHUFFLE(3, 3, 2, 2));  // high
    __m128i vacce7 = _mm_shuffle_epi32(vu7, _MM_SHUFFLE(3, 1, 2, 0));  // low

    vacce0 = _mm_mul_epu32(vacce0, vmultiplier);
    vacco0 = _mm_mul_epu32(vacco0, vmultiplier);
    vacce1 = _mm_mul_epu32(vacce1, vmultiplier);
    vacco1 = _mm_mul_epu32(vacco1, vmultiplier);
    vacce2 = _mm_mul_epu32(vacce2, vmultiplier);
    vacco2 = _mm_mul_epu32(vacco2, vmultiplier);
    vacce3 = _mm_mul_epu32(vacce3, vmultiplier);
    vacco3 = _mm_mul_epu32(vacco3, vmultiplier);
    vacce4 = _mm_mul_epu32(vacce4, vmultiplier);
    vacco4 = _mm_mul_epu32(vacco4, vmultiplier);
    vacce5 = _mm_mul_epu32(vacce5, vmultiplier);
    vacco5 = _mm_mul_epu32(vacco5, vmultiplier);
    vacce6 = _mm_mul_epu32(vacce6, vmultiplier);
    vacco6 = _mm_mul_epu32(vacco6, vmultiplier);
    vacce7 = _mm_mul_epu32(vacce7, vmultiplier);
    vacco7 = _mm_mul_epu32(vacco7, vmultiplier);

    vacce0 = _mm_add_epi64(vacce0, vbias);
    vacco0 = _mm_add_epi64(vacco0, vbias);
    vacce1 = _mm_add_epi64(vacce1, vbias);
    vacco1 = _mm_add_epi64(vacco1, vbias);
    vacce2 = _mm_add_epi64(vacce2, vbias);
    vacco2 = _mm_add_epi64(vacco2, vbias);
    vacce3 = _mm_add_epi64(vacce3, vbias);
    vacco3 = _mm_add_epi64(vacco3, vbias);
    vacce4 = _mm_add_epi64(vacce4, vbias);
    vacco4 = _mm_add_epi64(vacco4, vbias);
    vacce5 = _mm_add_epi64(vacce5, vbias);
    vacco5 = _mm_add_epi64(vacco5, vbias);
    vacce6 = _mm_add_epi64(vacce6, vbias);
    vacco6 = _mm_add_epi64(vacco6, vbias);
    vacce7 = _mm_add_epi64(vacce7, vbias);
    vacco7 = _mm_add_epi64(vacco7, vbias);

    vacce0 = _mm_srli_epi64(vacce0, 16);
    vacco0 = _mm_slli_epi64(vacco0, 16);
    vacce1 = _mm_srli_epi64(vacce1, 16);
    vacco1 = _mm_slli_epi64(vacco1, 16);
    vacce2 = _mm_srli_epi64(vacce2, 16);
    vacco2 = _mm_slli_epi64(vacco2, 16);
    vacce3 = _mm_srli_epi64(vacce3, 16);
    vacco3 = _mm_slli_epi64(vacco3, 16);
    vacce4 = _mm_srli_epi64(vacce4, 16);
    vacco4 = _mm_slli_epi64(vacco4, 16);
    vacce5 = _mm_srli_epi64(vacce5, 16);
    vacco5 = _mm_slli_epi64(vacco5, 16);
    vacce6 = _mm_srli_epi64(vacce6, 16);
    vacco6 = _mm_slli_epi64(vacco6, 16);
    vacce7 = _mm_srli_epi64(vacce7, 16);
    vacco7 = _mm_slli_epi64(vacco7, 16);

    __m128i vacc0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce0),
                                                            _mm_castsi128_ps(vacco0), 0xd8));
    __m128i vacc1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce1),
                                                            _mm_castsi128_ps(vacco1), 0xd8));
    __m128i vacc2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce2),
                                                            _mm_castsi128_ps(vacco2), 0xd8));
    __m128i vacc3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce3),
                                                            _mm_castsi128_ps(vacco3), 0xd8));
    __m128i vacc4 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce4),
                                                            _mm_castsi128_ps(vacco4), 0xd8));
    __m128i vacc5 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce5),
                                                            _mm_castsi128_ps(vacco5), 0xd8));
    __m128i vacc6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce6),
                                                            _mm_castsi128_ps(vacco6), 0xd8));
    __m128i vacc7 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce7),
                                                            _mm_castsi128_ps(vacco7), 0xd8));

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc2 = _mm_packs_epi32(vacc2, vacc3);
    vacc4 = _mm_packs_epi32(vacc4, vacc5);
    vacc6 = _mm_packs_epi32(vacc6, vacc7);

    // Pack 16 shorts into 16 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc2);
    const __m128i vy4 = _mm_packs_epi16(vacc4, vacc6);

    _mm_storeu_si128((__m128i*) output, vy0); output += 16;
    _mm_storeu_si128((__m128i*) output, vy4); output += 16;
  }

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) input); input += 4;
    vx = _mm_xor_si128(vx, vinput_bias);  // Convert signed inputs to unsigned.
    const __m128i vu = _mm_unpacklo_epi16(vx, vzero);
    __m128i vacco = _mm_shuffle_epi32(vu, _MM_SHUFFLE(3, 3, 2, 2));
    __m128i vacce = _mm_shuffle_epi32(vu, _MM_SHUFFLE(3, 1, 2, 0));
    vacce = _mm_mul_epu32(vacce, vmultiplier);
    vacco = _mm_mul_epu32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    vacce = _mm_srli_epi64(vacce, 16);
    vacco = _mm_slli_epi64(vacco, 16);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce),
                                                   _mm_castsi128_ps(vacco), 0xd8));
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
    __m128i vacco = _mm_shuffle_epi32(vu, _MM_SHUFFLE(3, 3, 2, 2));
    __m128i vacce = _mm_shuffle_epi32(vu, _MM_SHUFFLE(3, 1, 2, 0));
    vacce = _mm_mul_epu32(vacce, vmultiplier);
    vacco = _mm_mul_epu32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    vacce = _mm_srli_epi64(vacce, 16);
    vacco = _mm_slli_epi64(vacco, 16);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce),
                                                   _mm_castsi128_ps(vacco), 0xd8));
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
