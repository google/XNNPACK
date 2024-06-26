// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/avx512skx-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512 vacc0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vacc1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    i += 32;

    vacc0 = _mm512_sqrt_ps(vacc0);
    vacc1 = _mm512_sqrt_ps(vacc1);

    _mm256_storeu_si256((__m256i*) o, _mm512_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm256_storeu_si256((__m256i*) (o + 16), _mm512_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m512 vacc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    vacc = _mm512_sqrt_ps(vacc);
    _mm256_storeu_si256((__m256i*) o, _mm512_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));
    vacc = _mm512_sqrt_ps(vacc);
    _mm256_mask_storeu_epi16(o, vmask, _mm512_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
  }
}
