// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/sse-rational-9-6.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xmmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

void xnn_f32_vtanh_ukernel__sse2_rational_9_6_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
#if XNN_ARCH_X86
  const __m128 vmax_x = _mm_load_ps(params->sse_rational_9_6.max_x);
  const __m128 vmin_x = _mm_load_ps(params->sse_rational_9_6.min_x);
#else
  const __m128 vmax_x = _mm_set1_ps(7.623543739319f);
  const __m128 vmin_x = _mm_set1_ps(-7.623543739319f);
#endif  // XNN_ARCH_X86
  
  // The monomial coefficients of the numerator polynomial (odd).
#if XNN_ARCH_X86
  const __m128 valpha_1 = _mm_load_ps(params->sse_rational_9_6.alpha_1);
  const __m128 valpha_3 = _mm_load_ps(params->sse_rational_9_6.alpha_3);
  const __m128 valpha_5 = _mm_load_ps(params->sse_rational_9_6.alpha_5);
  const __m128 valpha_7 = _mm_load_ps(params->sse_rational_9_6.alpha_7);
  const __m128 valpha_9 = _mm_load_ps(params->sse_rational_9_6.alpha_9);
#else
  const __m128 valpha_1 = _mm_set1_ps(-9.022999554873e-03f);
  const __m128 valpha_3 = _mm_set1_ps(-1.146968104877e-03f);
  const __m128 valpha_5 = _mm_set1_ps(-2.432360815874e-05f);
  const __m128 valpha_7 = _mm_set1_ps(-6.458659385089e-08f);
  const __m128 valpha_9 = _mm_set1_ps(5.535878699892e-11f);
#endif  // XNN_ARCH_X86

  // The monomial coefficients of the denominator polynomial (even).
#if XNN_ARCH_X86
  const __m128 vbeta_0 = _mm_load_ps(params->sse_rational_9_6.beta_0);
  const __m128 vbeta_2 = _mm_load_ps(params->sse_rational_9_6.beta_2);
  const __m128 vbeta_4 = _mm_load_ps(params->sse_rational_9_6.beta_4);
  const __m128 vbeta_6 = _mm_load_ps(params->sse_rational_9_6.beta_6);
#else
  const __m128 vbeta_0 = _mm_set1_ps(-9.023001417518e-03f);
  const __m128 vbeta_2 = _mm_set1_ps(-4.154618829489e-03f);
  const __m128 vbeta_4 = _mm_set1_ps(-2.061512641376e-04f);
  const __m128 vbeta_6 = _mm_set1_ps(-1.774490101525e-06f);
#endif  // XNN_ARCH_X86
  

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m128 vx_0 = _mm_loadu_ps(input);
    __m128 vx_1 = _mm_loadu_ps(input + 4);
    __m128 vx_2 = _mm_loadu_ps(input + 8);
    __m128 vx_3 = _mm_loadu_ps(input + 12);
    input += 16;

    // Clamp the inputs to the interpolation range.
    vx_0 = _mm_min_ps(vmax_x, vx_0);
    vx_1 = _mm_min_ps(vmax_x, vx_1);
    vx_2 = _mm_min_ps(vmax_x, vx_2);
    vx_3 = _mm_min_ps(vmax_x, vx_3);
    vx_0 = _mm_max_ps(vmin_x, vx_0);
    vx_1 = _mm_max_ps(vmin_x, vx_1);
    vx_2 = _mm_max_ps(vmin_x, vx_2);
    vx_3 = _mm_max_ps(vmin_x, vx_3);

    // Since the polynomials are odd/even, we need x^2.
    const __m128 vx2_0 = _mm_mul_ps(vx_0, vx_0);
    const __m128 vx2_1 = _mm_mul_ps(vx_1, vx_1);
    const __m128 vx2_2 = _mm_mul_ps(vx_2, vx_2);
    const __m128 vx2_3 = _mm_mul_ps(vx_3, vx_3);

    // Evaluate the numerator polynomial p.
    __m128 vp_0 = _mm_add_ps(_mm_mul_ps(vx2_0, valpha_9), valpha_7);
    __m128 vp_1 = _mm_add_ps(_mm_mul_ps(vx2_1, valpha_9), valpha_7);
    __m128 vp_2 = _mm_add_ps(_mm_mul_ps(vx2_2, valpha_9), valpha_7);
    __m128 vp_3 = _mm_add_ps(_mm_mul_ps(vx2_3, valpha_9), valpha_7);
    vp_0 = _mm_add_ps(_mm_mul_ps(vx2_0, vp_0), valpha_5);
    vp_1 = _mm_add_ps(_mm_mul_ps(vx2_1, vp_1), valpha_5);
    vp_2 = _mm_add_ps(_mm_mul_ps(vx2_2, vp_2), valpha_5);
    vp_3 = _mm_add_ps(_mm_mul_ps(vx2_3, vp_3), valpha_5);
    vp_0 = _mm_add_ps(_mm_mul_ps(vx2_0, vp_0), valpha_3);
    vp_1 = _mm_add_ps(_mm_mul_ps(vx2_1, vp_1), valpha_3);
    vp_2 = _mm_add_ps(_mm_mul_ps(vx2_2, vp_2), valpha_3);
    vp_3 = _mm_add_ps(_mm_mul_ps(vx2_3, vp_3), valpha_3);
    vp_0 = _mm_add_ps(_mm_mul_ps(vx2_0, vp_0), valpha_1);
    vp_1 = _mm_add_ps(_mm_mul_ps(vx2_1, vp_1), valpha_1);
    vp_2 = _mm_add_ps(_mm_mul_ps(vx2_2, vp_2), valpha_1);
    vp_3 = _mm_add_ps(_mm_mul_ps(vx2_3, vp_3), valpha_1);
    vp_0 = _mm_mul_ps(vx_0, vp_0);
    vp_1 = _mm_mul_ps(vx_1, vp_1);
    vp_2 = _mm_mul_ps(vx_2, vp_2);
    vp_3 = _mm_mul_ps(vx_3, vp_3);

    // Evaluate the denominator polynomial q.
    __m128 vq_0 = _mm_add_ps(_mm_mul_ps(vx2_0, vbeta_6), vbeta_4);
    __m128 vq_1 = _mm_add_ps(_mm_mul_ps(vx2_1, vbeta_6), vbeta_4);
    __m128 vq_2 = _mm_add_ps(_mm_mul_ps(vx2_2, vbeta_6), vbeta_4);
    __m128 vq_3 = _mm_add_ps(_mm_mul_ps(vx2_3, vbeta_6), vbeta_4);
    vq_0 = _mm_add_ps(_mm_mul_ps(vx2_0, vq_0), vbeta_2);
    vq_1 = _mm_add_ps(_mm_mul_ps(vx2_1, vq_1), vbeta_2);
    vq_2 = _mm_add_ps(_mm_mul_ps(vx2_2, vq_2), vbeta_2);
    vq_3 = _mm_add_ps(_mm_mul_ps(vx2_3, vq_3), vbeta_2);
    vq_0 = _mm_add_ps(_mm_mul_ps(vx2_0, vq_0), vbeta_0);
    vq_1 = _mm_add_ps(_mm_mul_ps(vx2_1, vq_1), vbeta_0);
    vq_2 = _mm_add_ps(_mm_mul_ps(vx2_2, vq_2), vbeta_0);
    vq_3 = _mm_add_ps(_mm_mul_ps(vx2_3, vq_3), vbeta_0);

    // Divide the numerator by the denominator.
    const __m128 vy_0 =  _mm_div_ps(vp_0, vq_0);
    const __m128 vy_1 =  _mm_div_ps(vp_1, vq_1);
    const __m128 vy_2 =  _mm_div_ps(vp_2, vq_2);
    const __m128 vy_3 =  _mm_div_ps(vp_3, vq_3);

    _mm_storeu_ps(output, vy_0);
    _mm_storeu_ps(output + 4, vy_1);
    _mm_storeu_ps(output + 8, vy_2);
    _mm_storeu_ps(output + 12, vy_3);
    output += 16;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vx = _mm_loadu_ps(input);
    input += 4;

    // Clamp the inputs to the interpolation range.
    vx = _mm_min_ps(vmax_x, vx);
    vx = _mm_max_ps(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const __m128 vx2 = _mm_mul_ps(vx, vx);

    // Evaluate the numerator polynomial p.
    __m128 vp = _mm_add_ps(_mm_mul_ps(vx2, valpha_9), valpha_7);
    vp = _mm_add_ps(_mm_mul_ps(vx2, vp), valpha_5);
    vp = _mm_add_ps(_mm_mul_ps(vx2, vp), valpha_3);
    vp = _mm_add_ps(_mm_mul_ps(vx2, vp), valpha_1);
    vp = _mm_mul_ps(vx, vp);

    // Evaluate the denominator polynomial q.
    __m128 vq = _mm_add_ps(_mm_mul_ps(vx2, vbeta_6), vbeta_4);
    vq = _mm_add_ps(_mm_mul_ps(vx2, vq), vbeta_2);
    vq = _mm_add_ps(_mm_mul_ps(vx2, vq), vbeta_0);

    // Divide the numerator by the denominator.
    const __m128 vy =  _mm_div_ps(vp, vq);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx = _mm_loadu_ps(input);

    // Clamp the inputs to the interpolation range.
    vx = _mm_min_ps(vmax_x, vx);
    vx = _mm_max_ps(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const __m128 vx2 = _mm_mul_ps(vx, vx);

    // Evaluate the numerator polynomial p.
    __m128 vp = _mm_add_ps(_mm_mul_ps(vx2, valpha_9), valpha_7);
    vp = _mm_add_ps(_mm_mul_ps(vx2, vp), valpha_5);
    vp = _mm_add_ps(_mm_mul_ps(vx2, vp), valpha_3);
    vp = _mm_add_ps(_mm_mul_ps(vx2, vp), valpha_1);
    vp = _mm_mul_ps(vx, vp);

    // Evaluate the denominator polynomial q.
    __m128 vq = _mm_add_ps(_mm_mul_ps(vx2, vbeta_6), vbeta_4);
    vq = _mm_add_ps(_mm_mul_ps(vx2, vq), vbeta_2);
    vq = _mm_add_ps(_mm_mul_ps(vx2, vq), vbeta_0);

    // Divide the numerator by the denominator.
    __m128 vy =  _mm_div_ps(vp, vq);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}
