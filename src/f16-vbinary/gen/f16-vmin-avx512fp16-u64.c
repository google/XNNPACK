// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vop-avx512fp16.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"


void xnn_f16_vmin_ukernel__avx512fp16_u64(
    size_t batch,
    const xnn_float16* restrict input_a,
    const xnn_float16* restrict input_b,
    xnn_float16* restrict output,
    const struct xnn_f16_default_params* restrict params)
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


  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    const __m512h va0 = _mm512_loadu_ph(a);
    const __m512h va1 = _mm512_loadu_ph(a + 32);
    a += 64;

    __m512h vacc0 = _mm512_min_ph(va0, _mm512_loadu_ph(b));
    __m512h vacc1 = _mm512_min_ph(va1, _mm512_loadu_ph(b + 32));
    b += 64;


    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h va = _mm512_loadu_ph(a);
    a += 32;

    __m512h vacc = _mm512_min_ph(va, _mm512_loadu_ph(b));
    b += 32;


    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h va = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    __m512h vacc = _mm512_maskz_min_ph(vmask, va, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));


    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}
