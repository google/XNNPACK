// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/avx512skx.c.in
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


void xnn_f16_rmin_ukernel__avx512skx_u64_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vmin0 = _mm512_cvtph_ps(_mm256_set1_epi16(*i));
  __m512 vmin1 = vmin0;
  __m512 vmin2 = vmin0;
  __m512 vmin3 = vmin0;
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    const __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    const __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    const __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    const __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    i += 64;

    vmin0 = _mm512_min_ps(vmin0, vt0);
    vmin1 = _mm512_min_ps(vmin1, vt1);
    vmin2 = _mm512_min_ps(vmin2, vt2);
    vmin3 = _mm512_min_ps(vmin3, vt3);
  }
  vmin0 = _mm512_min_ps(vmin0, vmin1);
  vmin2 = _mm512_min_ps(vmin2, vmin3);
  vmin0 = _mm512_min_ps(vmin0, vmin2);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;

    vmin0 = _mm512_min_ps(vmin0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vmin0 = _mm512_mask_min_ps(vmin0, vmask, vmin0, vt);
  }
  __m256 vmin256 = _mm256_min_ps(_mm512_castps512_ps256(vmin0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmin0), 1)));
  __m128 vmin = _mm_min_ps(_mm256_castps256_ps128(vmin256), _mm256_extractf128_ps(vmin256, 1));
  vmin = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
  vmin = _mm_min_ss(vmin, _mm_movehdup_ps(vmin));
  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(vmin, _MM_FROUND_TO_NEAREST_INT), 0);
}
