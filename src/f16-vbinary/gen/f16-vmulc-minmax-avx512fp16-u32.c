// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vopc-avx512fp16.c.in
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
#include "xnnpack/vbinary.h"


void xnn_f16_vmulc_minmax_ukernel__avx512fp16_u32(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    a += 32;

    vacc0 = _mm512_mul_ph(vacc0, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);

    _mm512_storeu_ph(o, vacc0);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_mul_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}
