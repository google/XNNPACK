// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/avx-rational-9-6.c.in
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

void xnn_f32_vtanh_ukernel__avx_rational_9_6_div_u32(
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
  const __m256 vmax_x = _mm256_set1_ps(params->avx_rational_9_6.max_abs_x);
  const __m256 vmin_x = _mm256_set1_ps(-params->avx_rational_9_6.max_abs_x);
  
  // The monomial coefficients of the numerator polynomial (odd).
  const __m256 valpha_1 = _mm256_set1_ps(params->avx_rational_9_6.alpha_1);
  const __m256 valpha_3 = _mm256_set1_ps(params->avx_rational_9_6.alpha_3);
  const __m256 valpha_5 = _mm256_set1_ps(params->avx_rational_9_6.alpha_5);
  const __m256 valpha_7 = _mm256_set1_ps(params->avx_rational_9_6.alpha_7);
  const __m256 valpha_9 = _mm256_set1_ps(params->avx_rational_9_6.alpha_9);

  // The monomial coefficients of the denominator polynomial (even).
  const __m256 vbeta_0 = _mm256_set1_ps(params->avx_rational_9_6.beta_0);
  const __m256 vbeta_2 = _mm256_set1_ps(params->avx_rational_9_6.beta_2);
  const __m256 vbeta_4 = _mm256_set1_ps(params->avx_rational_9_6.beta_4);
  const __m256 vbeta_6 = _mm256_set1_ps(params->avx_rational_9_6.beta_6);


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m256 vx_0 = _mm256_loadu_ps(input);
    __m256 vx_1 = _mm256_loadu_ps(input + 8);
    __m256 vx_2 = _mm256_loadu_ps(input + 16);
    __m256 vx_3 = _mm256_loadu_ps(input + 24);
    input += 32;

    // Clamp the inputs to the interpolation range.
    vx_0 = _mm256_min_ps(vmax_x, vx_0);
    vx_1 = _mm256_min_ps(vmax_x, vx_1);
    vx_2 = _mm256_min_ps(vmax_x, vx_2);
    vx_3 = _mm256_min_ps(vmax_x, vx_3);
    vx_0 = _mm256_max_ps(vmin_x, vx_0);
    vx_1 = _mm256_max_ps(vmin_x, vx_1);
    vx_2 = _mm256_max_ps(vmin_x, vx_2);
    vx_3 = _mm256_max_ps(vmin_x, vx_3);

    // Since the polynomials are odd/even, we need x^2.
    const __m256 vx2_0 = _mm256_mul_ps(vx_0, vx_0);
    const __m256 vx2_1 = _mm256_mul_ps(vx_1, vx_1);
    const __m256 vx2_2 = _mm256_mul_ps(vx_2, vx_2);
    const __m256 vx2_3 = _mm256_mul_ps(vx_3, vx_3);

    // Evaluate the numerator polynomial p.
    __m256 vp_0 = _mm256_add_ps(_mm256_mul_ps(vx2_0, valpha_9), valpha_7);
    __m256 vp_1 = _mm256_add_ps(_mm256_mul_ps(vx2_1, valpha_9), valpha_7);
    __m256 vp_2 = _mm256_add_ps(_mm256_mul_ps(vx2_2, valpha_9), valpha_7);
    __m256 vp_3 = _mm256_add_ps(_mm256_mul_ps(vx2_3, valpha_9), valpha_7);
    vp_0 = _mm256_add_ps(_mm256_mul_ps(vx2_0, vp_0), valpha_5);
    vp_1 = _mm256_add_ps(_mm256_mul_ps(vx2_1, vp_1), valpha_5);
    vp_2 = _mm256_add_ps(_mm256_mul_ps(vx2_2, vp_2), valpha_5);
    vp_3 = _mm256_add_ps(_mm256_mul_ps(vx2_3, vp_3), valpha_5);
    vp_0 = _mm256_add_ps(_mm256_mul_ps(vx2_0, vp_0), valpha_3);
    vp_1 = _mm256_add_ps(_mm256_mul_ps(vx2_1, vp_1), valpha_3);
    vp_2 = _mm256_add_ps(_mm256_mul_ps(vx2_2, vp_2), valpha_3);
    vp_3 = _mm256_add_ps(_mm256_mul_ps(vx2_3, vp_3), valpha_3);
    vp_0 = _mm256_add_ps(_mm256_mul_ps(vx2_0, vp_0), valpha_1);
    vp_1 = _mm256_add_ps(_mm256_mul_ps(vx2_1, vp_1), valpha_1);
    vp_2 = _mm256_add_ps(_mm256_mul_ps(vx2_2, vp_2), valpha_1);
    vp_3 = _mm256_add_ps(_mm256_mul_ps(vx2_3, vp_3), valpha_1);
    vp_0 = _mm256_mul_ps(vx_0, vp_0);
    vp_1 = _mm256_mul_ps(vx_1, vp_1);
    vp_2 = _mm256_mul_ps(vx_2, vp_2);
    vp_3 = _mm256_mul_ps(vx_3, vp_3);

    // Evaluate the denominator polynomial q.
    __m256 vq_0 = _mm256_add_ps(_mm256_mul_ps(vx2_0, vbeta_6), vbeta_4);
    __m256 vq_1 = _mm256_add_ps(_mm256_mul_ps(vx2_1, vbeta_6), vbeta_4);
    __m256 vq_2 = _mm256_add_ps(_mm256_mul_ps(vx2_2, vbeta_6), vbeta_4);
    __m256 vq_3 = _mm256_add_ps(_mm256_mul_ps(vx2_3, vbeta_6), vbeta_4);
    vq_0 = _mm256_add_ps(_mm256_mul_ps(vx2_0, vq_0), vbeta_2);
    vq_1 = _mm256_add_ps(_mm256_mul_ps(vx2_1, vq_1), vbeta_2);
    vq_2 = _mm256_add_ps(_mm256_mul_ps(vx2_2, vq_2), vbeta_2);
    vq_3 = _mm256_add_ps(_mm256_mul_ps(vx2_3, vq_3), vbeta_2);
    vq_0 = _mm256_add_ps(_mm256_mul_ps(vx2_0, vq_0), vbeta_0);
    vq_1 = _mm256_add_ps(_mm256_mul_ps(vx2_1, vq_1), vbeta_0);
    vq_2 = _mm256_add_ps(_mm256_mul_ps(vx2_2, vq_2), vbeta_0);
    vq_3 = _mm256_add_ps(_mm256_mul_ps(vx2_3, vq_3), vbeta_0);

    // Divide the numerator by the denominator.
    const __m256 vy_0 =  _mm256_div_ps(vp_0, vq_0);
    const __m256 vy_1 =  _mm256_div_ps(vp_1, vq_1);
    const __m256 vy_2 =  _mm256_div_ps(vp_2, vq_2);
    const __m256 vy_3 =  _mm256_div_ps(vp_3, vq_3);


    _mm256_storeu_ps(output, vy_0);
    _mm256_storeu_ps(output + 8, vy_1);
    _mm256_storeu_ps(output + 16, vy_2);
    _mm256_storeu_ps(output + 24, vy_3);
    output += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    // Clamp the inputs to the interpolation range.
    vx = _mm256_min_ps(vmax_x, vx);
    vx = _mm256_max_ps(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const __m256 vx2 = _mm256_mul_ps(vx, vx);

    // Evaluate the numerator polynomial p.
    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vx2, valpha_9), valpha_7);
    vp = _mm256_add_ps(_mm256_mul_ps(vx2, vp), valpha_5);
    vp = _mm256_add_ps(_mm256_mul_ps(vx2, vp), valpha_3);
    vp = _mm256_add_ps(_mm256_mul_ps(vx2, vp), valpha_1);
    vp = _mm256_mul_ps(vx, vp);

    // Evaluate the denominator polynomial q.
    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx2, vbeta_6), vbeta_4);
    vq = _mm256_add_ps(_mm256_mul_ps(vx2, vq), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx2, vq), vbeta_0);

    // Divide the numerator by the denominator.
    const __m256 vy =  _mm256_div_ps(vp, vq);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256(
        (const __m256i*) ((uintptr_t) &params->avx_rational_9_6.mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);

    // Clamp the inputs to the interpolation range.
    vx = _mm256_min_ps(vmax_x, vx);
    vx = _mm256_max_ps(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const __m256 vx2 = _mm256_mul_ps(vx, vx);

    // Evaluate the numerator polynomial p.
    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vx2, valpha_9), valpha_7);
    vp = _mm256_add_ps(_mm256_mul_ps(vx2, vp), valpha_5);
    vp = _mm256_add_ps(_mm256_mul_ps(vx2, vp), valpha_3);
    vp = _mm256_add_ps(_mm256_mul_ps(vx2, vp), valpha_1);
    vp = _mm256_mul_ps(vx, vp);

    // Evaluate the denominator polynomial q.
    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx2, vbeta_6), vbeta_4);
    vq = _mm256_add_ps(_mm256_mul_ps(vx2, vq), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx2, vq), vbeta_0);

    // Divide the numerator by the denominator.
    const __m256 vy =  _mm256_div_ps(vp, vq);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}
