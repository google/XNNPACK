// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vbinary.h"


void xnn_f32_vpreluc_ukernel__avx512f_u32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 vb = _mm512_set1_ps(*input_b);
  const __m512 vzero = _mm512_setzero_ps();

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m512 va0 = _mm512_loadu_ps(input_a);
    __m512 va1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    const __mmask16 vsign0 = _mm512_cmp_ps_mask(va0, vzero, _CMP_LT_OQ);
    __m512 vacc0 = _mm512_mask_mul_ps(va0, vsign0, va0, vb);
    const __mmask16 vsign1 = _mm512_cmp_ps_mask(va1, vzero, _CMP_LT_OQ);
    __m512 vacc1 = _mm512_mask_mul_ps(va1, vsign1, va1, vb);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(input_a);
    input_a += 16;

    const __mmask16 vsign = _mm512_cmp_ps_mask(va, vzero, _CMP_LT_OQ);
    __m512 vacc = _mm512_mask_mul_ps(va, vsign, va, vb);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 va = _mm512_maskz_loadu_ps(vmask, input_a);
    const __mmask16 vsign = _mm512_cmp_ps_mask(va, vzero, _CMP_LT_OQ);
    __m512 vacc = _mm512_mask_mul_ps(va, vsign, va, vb);

    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}
