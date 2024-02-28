// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f32_vrelu_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vzero = _mm512_setzero_ps();

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEF, vzero);

    _mm512_storeu_ps(output, vacc0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input);
    vacc = _mm512_max_ps(vacc, vzero);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}
