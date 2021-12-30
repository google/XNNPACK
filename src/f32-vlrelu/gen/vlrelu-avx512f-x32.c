// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vlrelu_ukernel__avx512f_x32(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 vslope = _mm512_set1_ps(params->scalar.slope);
  const __m512 vzero = _mm512_setzero_ps();

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(x);
    __m512 vaccGHIJKLMNOPQRSTUV = _mm512_loadu_ps(x + 16);
    x += 32;

    const __mmask16 vsign0123456789ABCDEF = _mm512_cmp_ps_mask(vacc0123456789ABCDEF, vzero, _CMP_LT_OQ);
    const __mmask16 vsignGHIJKLMNOPQRSTUV = _mm512_cmp_ps_mask(vaccGHIJKLMNOPQRSTUV, vzero, _CMP_LT_OQ);

    vacc0123456789ABCDEF = _mm512_mask_mul_ps(vacc0123456789ABCDEF, vsign0123456789ABCDEF, vacc0123456789ABCDEF, vslope);
    vaccGHIJKLMNOPQRSTUV = _mm512_mask_mul_ps(vaccGHIJKLMNOPQRSTUV, vsignGHIJKLMNOPQRSTUV, vaccGHIJKLMNOPQRSTUV, vslope);

    _mm512_storeu_ps(y, vacc0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vaccGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(x);
    x += 16;
    const __mmask16 vsign = _mm512_cmp_ps_mask(vacc, vzero, _CMP_LT_OQ);
    vacc = _mm512_mask_mul_ps(vacc, vsign, vacc, vslope);
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
    const __mmask16 vsign = _mm512_mask_cmp_ps_mask(vmask, vacc, vzero, _CMP_LT_OQ);
    vacc = _mm512_mask_mul_ps(vacc, vsign, vacc, vslope);
    _mm512_mask_storeu_ps(y, vmask, vacc);
  }
}
