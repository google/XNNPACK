// Auto-generated file. Do not edit!
//   Template: src/f16-vhswish/f16c.c.in
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
#include "xnnpack/vunary.h"


void xnn_f16_vhswish_ukernel__f16c_u8(
    size_t batch,
    const void* restrict input,
    void* restrict output,
    const union xnn_f16_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const __m256 vsixth = _mm256_set1_ps(0x1.554000p-3f);
  const __m256 vthree = _mm256_set1_ps(3.0f);
  const __m128i vsix = _mm_set1_epi16(UINT16_C(0x4600));
  const __m128i vzero = _mm_setzero_si128();

  XNN_FORCE_REALIZATION(vsixth);
  XNN_FORCE_REALIZATION(vthree);
  XNN_FORCE_REALIZATION(vsix);
  // XNN_FORCE_REALIZATION(vzero);

  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    __m128i vacc = _mm256_cvtps_ph(_mm256_add_ps(vx, vthree), _MM_FROUND_TO_NEAREST_INT);
    vx = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx, vsixth), _MM_FROUND_TO_NEAREST_INT));
    vacc = _mm_max_epi16(vacc, vzero);
    vacc = _mm_min_epi16(vacc, vsix);
    vacc = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc), vx), _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*) o, vacc);
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m128i vacc = _mm256_cvtps_ph(_mm256_add_ps(vx, vthree), _MM_FROUND_TO_NEAREST_INT);
    vx = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx, vsixth), _MM_FROUND_TO_NEAREST_INT));
    vacc = _mm_max_epi16(vacc, vzero);
    vacc = _mm_min_epi16(vacc, vsix);
    vacc = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc), vx), _MM_FROUND_TO_NEAREST_INT);

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vacc);
      vacc = _mm_unpackhi_epi64(vacc, vacc);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vacc);
      vacc = _mm_srli_epi64(vacc, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vacc, 0);
    }
  }
}
