// Auto-generated file. Do not edit!
//   Template: src/f16-rsum/avx512fp16.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f16_rsum_ukernel__avx512fp16_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  __m512h vacc0 = _mm512_setzero_ph();
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h vt = _mm512_loadu_ph(i);
    i += 32;

    vacc0 = _mm512_add_ph(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h vt = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));

    vacc0 = _mm512_add_ph(vacc0, vt);
  }

  const __m256h vacc256 = _mm256_add_ph(_mm512_castph512_ph256(vacc0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vacc0), 1)));
  __m128h vacc = _mm_add_ph(_mm256_castph256_ph128(vacc256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vacc256), 1)));
  vacc = _mm_add_ph(vacc, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vacc), _mm_castph_ps(vacc))));
  vacc = _mm_add_ph(vacc, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vacc))));
  vacc = _mm_add_sh(vacc, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vacc), 16)));

  const __m128h vscale = _mm_castsi128_ph(_mm_set1_epi16(params->scale));

  vacc = _mm_mul_sh(vacc, vscale);
  *((uint16_t*) o) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vacc), 0);
}
