// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/avx512f-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


// In the following, we first compute the _reciprocal_ square root of an input
// `a` and then multiply it with `a` to produce the square root.
//
// We compute the reciprocal square root using a single Newton-Raphson step on
// the equation $x^{-2} - a$, which expands to:
//
//  $$x_{k+1} = 0.5 * x_k * (3.0 - a * x_k^2)$$
//
// So we do the following steps:
//
//  1. t0 = x_k
//  2. t1 = t0 * t0       (x_k^2)
//  3. t3 = a * t1 - 3.0  (a * x_k^2 - 3.0)
//  4. t4 = 0.5 * t0      (-0.5 * x_k)
//  5. t5  = t3 * t4      ((-0.5 * x_k) * (a * x_k^2 - 3.0))
//  6. y = a * t5         (a * a^{-1/2})
//
// Where $x_k$ is the original 14-bit approximation and `t5` contains the final
// 24-bit approximation $x_{k+1}$.

void xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16(
    size_t batch, const float* input, float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m512 vneg_three = _mm512_set1_ps(-3.0f);
  const __m512 vneg_half = _mm512_set1_ps(-0.5f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __mmask16 vinf_mask = 
        _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 14-bit approximation.
    const __m512 vt0 = _mm512_rsqrt14_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m512 vt1 = _mm512_mul_ps(vt0, vt0);
    const __m512 vt3 = _mm512_fmadd_ps(vx, vt1, vneg_three);
    const __m512 vt4 = _mm512_mul_ps(vneg_half, vt0);
    const __m512 vt5 = _mm512_mul_ps(vt3, vt4);
    const __m512 vt6 = _mm512_mask_blend_ps(vinf_mask, vt5, _mm512_setzero_ps());
    const __m512 vy = _mm512_mul_ps(vx, vt6);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __mmask16 vinf_mask = 
        _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 14-bit approximation.
    const __m512 vt0 = _mm512_rsqrt14_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m512 vt1 = _mm512_mul_ps(vt0, vt0);
    const __m512 vt3 = _mm512_fmadd_ps(vx, vt1, vneg_three);
    const __m512 vt4 = _mm512_mul_ps(vneg_half, vt0);
    const __m512 vt5 = _mm512_mul_ps(vt3, vt4);
    const __m512 vt6 = _mm512_mask_blend_ps(vinf_mask, vt5, _mm512_setzero_ps());
    const __m512 vy = _mm512_mul_ps(vx, vt6);


    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
