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

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>

void xnn_qs16_qs8_vcvt_ukernel__sse41_x16(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse4.multiplier);
  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse4.bias);

  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    __m128i vacce0 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i*) input)); input += 4;
    __m128i vacce1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i*) input)); input += 4;
    __m128i vacce2 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i*) input)); input += 4;
    __m128i vacce3 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i*) input)); input += 4;

    __m128i vacco0 = _mm_shuffle_epi32(vacce0, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco1 = _mm_shuffle_epi32(vacce1, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco2 = _mm_shuffle_epi32(vacce2, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco3 = _mm_shuffle_epi32(vacce3, _MM_SHUFFLE(3, 3, 1, 1));

    vacce0 = _mm_mul_epi32(vacce0, vmultiplier);
    vacco0 = _mm_mul_epi32(vacco0, vmultiplier);
    vacce1 = _mm_mul_epi32(vacce1, vmultiplier);
    vacco1 = _mm_mul_epi32(vacco1, vmultiplier);
    vacce2 = _mm_mul_epi32(vacce2, vmultiplier);
    vacco2 = _mm_mul_epi32(vacco2, vmultiplier);
    vacce3 = _mm_mul_epi32(vacce3, vmultiplier);
    vacco3 = _mm_mul_epi32(vacco3, vmultiplier);

    vacce0 = _mm_add_epi64(vacce0, vbias);
    vacco0 = _mm_add_epi64(vacco0, vbias);
    vacce1 = _mm_add_epi64(vacce1, vbias);
    vacco1 = _mm_add_epi64(vacco1, vbias);
    vacce2 = _mm_add_epi64(vacce2, vbias);
    vacco2 = _mm_add_epi64(vacco2, vbias);
    vacce3 = _mm_add_epi64(vacce3, vbias);
    vacco3 = _mm_add_epi64(vacco3, vbias);

    vacce0 = _mm_srli_epi64(vacce0, 16);
    vacco0 = _mm_slli_epi64(vacco0, 16);
    vacce1 = _mm_srli_epi64(vacce1, 16);
    vacco1 = _mm_slli_epi64(vacco1, 16);
    vacce2 = _mm_srli_epi64(vacce2, 16);
    vacco2 = _mm_slli_epi64(vacco2, 16);
    vacce3 = _mm_srli_epi64(vacce3, 16);
    vacco3 = _mm_slli_epi64(vacco3, 16);

    __m128i vacc0 = _mm_blend_epi16(vacce0, vacco0, 0xcc);
    __m128i vacc1 = _mm_blend_epi16(vacce1, vacco1, 0xcc);
    __m128i vacc2 = _mm_blend_epi16(vacce2, vacco2, 0xcc);
    __m128i vacc3 = _mm_blend_epi16(vacce3, vacco3, 0xcc);

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc2 = _mm_packs_epi32(vacc2, vacc3);

    // Pack 16 shorts into 16 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc2);

    _mm_storeu_si128((__m128i*) output, vy0); output += 16;
  }

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    __m128i vacce = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i*) input)); input += 4;
    __m128i vacco = _mm_shuffle_epi32(vacce, _MM_SHUFFLE(3, 3, 1, 1));
    vacce = _mm_mul_epi32(vacce, vmultiplier);
    vacco = _mm_mul_epi32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    vacce = _mm_srli_epi64(vacce, 16);
    vacco = _mm_slli_epi64(vacco, 16);
    __m128i vacc = _mm_blend_epi16(vacce, vacco, 0xcc);
    vacc = _mm_packs_epi32(vacc, vacc);
    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 3 * sizeof(int16_t));

    __m128i vacce = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacco = _mm_shuffle_epi32(vacce, _MM_SHUFFLE(3, 3, 1, 1));
    vacce = _mm_mul_epi32(vacce, vmultiplier);
    vacco = _mm_mul_epi32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    vacce = _mm_srli_epi64(vacce, 16);
    vacco = _mm_slli_epi64(vacco, 16);
    __m128i vacc = _mm_blend_epi16(vacce, vacco, 0xcc);
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
