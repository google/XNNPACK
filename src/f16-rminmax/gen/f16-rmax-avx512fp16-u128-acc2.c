// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/avx512fp16.c.in
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


void xnn_f16_rmax_ukernel__avx512fp16_u128_acc2(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* i = (const uint16_t*) input;
  __m512h vmax0 = _mm512_castsi512_ph(_mm512_set1_epi16(*i));
  __m512h vmax1 = vmax0;
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    const __m512h vt0 = _mm512_loadu_ph(i);
    const __m512h vt1 = _mm512_loadu_ph((i + 32));
    const __m512h vt2 = _mm512_loadu_ph((i + 64));
    const __m512h vt3 = _mm512_loadu_ph((i + 96));
    i += 128;

    vmax0 = _mm512_max_ph(vmax0, vt0);
    vmax1 = _mm512_max_ph(vmax1, vt1);
    vmax0 = _mm512_max_ph(vmax0, vt2);
    vmax1 = _mm512_max_ph(vmax1, vt3);
  }
  vmax0 = _mm512_max_ph(vmax0, vmax1);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h vt = _mm512_loadu_ph(i);
    i += 32;

    vmax0 = _mm512_max_ph(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h vt = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));

    vmax0 = _mm512_mask_max_ph(vmax0, vmask, vmax0, vt);
  }
  __m256h vmax256 = _mm256_max_ph(_mm512_castph512_ph256(vmax0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmax0), 1)));
  __m128h vmax = _mm_max_ph(_mm256_castph256_ph128(vmax256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmax256), 1)));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmax), _mm_castph_ps(vmax))));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmax))));
  vmax = _mm_max_sh(vmax, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmax), 16)));

  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmax), 0);
#endif  // defined(__AVX512FP16__)
}
