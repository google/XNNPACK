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


void xnn_f16_rmin_ukernel__avx512fp16_u32(
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
  __m512h vmin0 = _mm512_castsi512_ph(_mm512_set1_epi16(*i));
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h vt = _mm512_loadu_ph(i);
    i += 32;

    vmin0 = _mm512_min_ph(vmin0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h vt = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));

    vmin0 = _mm512_mask_min_ph(vmin0, vmask, vmin0, vt);
  }
  __m256h vmin256 = _mm256_min_ph(_mm512_castph512_ph256(vmin0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmin0), 1)));
  __m128h vmin = _mm_min_ph(_mm256_castph256_ph128(vmin256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmin256), 1)));
  vmin = _mm_min_ph(vmin, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmin), _mm_castph_ps(vmin))));
  vmin = _mm_min_ph(vmin, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmin))));
  vmin = _mm_min_sh(vmin, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmin), 16)));

  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmin), 0);
#endif  // defined(__AVX512FP16__)
}
