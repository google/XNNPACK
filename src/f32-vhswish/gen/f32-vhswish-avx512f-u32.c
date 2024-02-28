// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vhswish_ukernel__avx512f_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vsixth = _mm512_set1_ps(params->avx512.sixth);
  const __m512 vhalf = _mm512_set1_ps(params->avx512.half);
  const __m512 vone = _mm512_set1_ps(params->avx512.one);
  const __m512 vzero = _mm512_setzero_ps();

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    const __m512 vxGHIJKLMNOPQRSTUV = _mm512_loadu_ps(input + 16);
    input += 32;

    __m512 vacc0123456789ABCDEF = _mm512_fmadd_ps(vx0123456789ABCDEF, vsixth, vhalf);
    __m512 vaccGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vxGHIJKLMNOPQRSTUV, vsixth, vhalf);

    vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEF, vzero);
    vaccGHIJKLMNOPQRSTUV = _mm512_max_ps(vaccGHIJKLMNOPQRSTUV, vzero);

    vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vone);
    vaccGHIJKLMNOPQRSTUV = _mm512_min_ps(vaccGHIJKLMNOPQRSTUV, vone);

    vacc0123456789ABCDEF = _mm512_mul_ps(vacc0123456789ABCDEF, vx0123456789ABCDEF);
    vaccGHIJKLMNOPQRSTUV = _mm512_mul_ps(vaccGHIJKLMNOPQRSTUV, vxGHIJKLMNOPQRSTUV);

    _mm512_storeu_ps(output, vacc0123456789ABCDEF);
    _mm512_storeu_ps(output + 16, vaccGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}
