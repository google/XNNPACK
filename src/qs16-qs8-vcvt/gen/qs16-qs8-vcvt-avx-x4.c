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

void xnn_qs16_qs8_vcvt_ukernel__avx_x4(
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
