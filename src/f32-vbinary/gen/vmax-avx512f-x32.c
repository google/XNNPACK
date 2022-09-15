// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-avx512f.c.in
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
#include <xnnpack/vbinary.h>


void xnn_f32_vmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(input_a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(input_b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(input_b + 16);
    input_b += 32;

    __m512 vy0123456789ABCDEF = _mm512_max_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);



    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    _mm512_storeu_ps(output + 16, vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(input_a);
    input_a += 16;

    const __m512 vb = _mm512_loadu_ps(input_b);
    input_b += 16;

    __m512 vy = _mm512_max_ps(va, vb);
    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, input_a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, input_b);

    __m512 vy = _mm512_max_ps(va, vb);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
