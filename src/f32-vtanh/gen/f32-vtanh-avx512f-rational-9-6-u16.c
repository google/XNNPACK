// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/avx512f-rational-9-6.c.in
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

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

void xnn_f32_vtanh_ukernel__avx512f_rational_9_6_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  const __m512 vmax_x = _mm512_set1_ps(7.646893501282f);
  const __m512 vmin_x = _mm512_set1_ps(-7.646893501282f);
  
  // The monomial coefficients of the numerator polynomial (odd).
  const __m512 valpha_1 = _mm512_set1_ps(-9.022999554873e-03f);
  const __m512 valpha_3 = _mm512_set1_ps(-1.146968104877e-03f);
  const __m512 valpha_5 = _mm512_set1_ps(-2.432360815874e-05f);
  const __m512 valpha_7 = _mm512_set1_ps(-6.458659385089e-08f);
  const __m512 valpha_9 = _mm512_set1_ps(5.535878699892e-11f);

  // The monomial coefficients of the denominator polynomial (even).
  const __m512 vbeta_0 = _mm512_set1_ps(-9.023001417518e-03f);
  const __m512 vbeta_2 = _mm512_set1_ps(-4.154618829489e-03f);
  const __m512 vbeta_4 = _mm512_set1_ps(-2.061512641376e-04f);
  const __m512 vbeta_6 = _mm512_set1_ps(-1.774490101525e-06f);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vx_0 = _mm512_loadu_ps(input);
    __m512 vx_1 = _mm512_loadu_ps(input + 16);
    input += 32;

    // Clamp the inputs to the interpolation range.
    vx_0 = _mm512_min_ps(vmax_x, vx_0);
    vx_1 = _mm512_min_ps(vmax_x, vx_1);
    vx_0 = _mm512_max_ps(vmin_x, vx_0);
    vx_1 = _mm512_max_ps(vmin_x, vx_1);

    // Since the polynomials are odd/even, we need x^2.
    const __m512 vx2_0 = _mm512_mul_ps(vx_0, vx_0);
    const __m512 vx2_1 = _mm512_mul_ps(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    __m512 vp_0 = _mm512_fmadd_ps(vx2_0, valpha_9, valpha_7);
    __m512 vp_1 = _mm512_fmadd_ps(vx2_1, valpha_9, valpha_7);
    vp_0 = _mm512_fmadd_ps(vx2_0, vp_0, valpha_5);
    vp_1 = _mm512_fmadd_ps(vx2_1, vp_1, valpha_5);
    vp_0 = _mm512_fmadd_ps(vx2_0, vp_0, valpha_3);
    vp_1 = _mm512_fmadd_ps(vx2_1, vp_1, valpha_3);
    vp_0 = _mm512_fmadd_ps(vx2_0, vp_0, valpha_1);
    vp_1 = _mm512_fmadd_ps(vx2_1, vp_1, valpha_1);
    vp_0 = _mm512_mul_ps(vx_0, vp_0);
    vp_1 = _mm512_mul_ps(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    __m512 vq_0 = _mm512_fmadd_ps(vx2_0, vbeta_6, vbeta_4);
    __m512 vq_1 = _mm512_fmadd_ps(vx2_1, vbeta_6, vbeta_4);
    vq_0 = _mm512_fmadd_ps(vx2_0, vq_0, vbeta_2);
    vq_1 = _mm512_fmadd_ps(vx2_1, vq_1, vbeta_2);
    vq_0 = _mm512_fmadd_ps(vx2_0, vq_0, vbeta_0);
    vq_1 = _mm512_fmadd_ps(vx2_1, vq_1, vbeta_0);

    // Divide the numerator by the denominator.
    const __m512 vy_0 =  _mm512_div_ps(vp_0, vq_0);
    const __m512 vy_1 =  _mm512_div_ps(vp_1, vq_1);

    _mm512_storeu_ps(output, vy_0);
    _mm512_storeu_ps(output + 16, vy_1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    // Clamp the inputs to the interpolation range.
    vx = _mm512_min_ps(vmax_x, vx);
    vx = _mm512_max_ps(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const __m512 vx2 = _mm512_mul_ps(vx, vx);

    // Evaluate the numerator polynomial p.
    __m512 vp = _mm512_fmadd_ps(vx2, valpha_9, valpha_7);
    vp = _mm512_fmadd_ps(vx2, vp, valpha_5);
    vp = _mm512_fmadd_ps(vx2, vp, valpha_3);
    vp = _mm512_fmadd_ps(vx2, vp, valpha_1);
    vp = _mm512_mul_ps(vx, vp);

    // Evaluate the denominator polynomial q.
    __m512 vq = _mm512_fmadd_ps(vx2, vbeta_6, vbeta_4);
    vq = _mm512_fmadd_ps(vx2, vq, vbeta_2);
    vq = _mm512_fmadd_ps(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const __m512 vy =  _mm512_div_ps(vp, vq);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    // Clamp the inputs to the interpolation range.
    vx = _mm512_min_ps(vmax_x, vx);
    vx = _mm512_max_ps(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const __m512 vx2 = _mm512_mul_ps(vx, vx);

    // Evaluate the numerator polynomial p.
    __m512 vp = _mm512_fmadd_ps(vx2, valpha_9, valpha_7);
    vp = _mm512_fmadd_ps(vx2, vp, valpha_5);
    vp = _mm512_fmadd_ps(vx2, vp, valpha_3);
    vp = _mm512_fmadd_ps(vx2, vp, valpha_1);
    vp = _mm512_mul_ps(vx, vp);

    // Evaluate the denominator polynomial q.
    __m512 vq = _mm512_fmadd_ps(vx2, vbeta_6, vbeta_4);
    vq = _mm512_fmadd_ps(vx2, vq, vbeta_2);
    vq = _mm512_fmadd_ps(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const __m512 vy =  _mm512_div_ps(vp, vq);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
