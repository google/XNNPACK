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


void xnn_f32_vrelu_ukernel__avx512f_x32(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m512 vzero = _mm512_setzero_ps();

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(x);
    __m512 vaccGHIJKLMNOPQRSTUV = _mm512_loadu_ps(x + 16);
    x += 32;

    vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEF, vzero);
    vaccGHIJKLMNOPQRSTUV = _mm512_max_ps(vaccGHIJKLMNOPQRSTUV, vzero);

    _mm512_storeu_ps(y, vacc0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vaccGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(x);
    x += 16;

    vacc = _mm512_max_ps(vacc, vzero);

    _mm512_storeu_ps(y, vacc);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, x);
    vacc = _mm512_max_ps(vacc, vzero);
    _mm512_mask_storeu_ps(y, vmask, vacc);
  }
}
