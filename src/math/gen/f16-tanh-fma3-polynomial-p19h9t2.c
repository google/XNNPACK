// Auto-generated file. Do not edit!
//   Template: src/math/f16-tanh-avx-polynomial.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f16_tanh__fma3_polynomial_p19h9t2(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // The smallest number x above -0x1.208p+2h (the largest number z for which tanhh(z) is saturated at -1.0h) for which
  // this implementation of tanh(x) produce -1.0h output.
  const __m256 vneg_sat_cutoff = _mm256_set1_ps(-0x1.1F0000p+2f);
  // The largest number x below 0x1.208p+2h (the smallest number z for which tanhh(z) is saturated at 1.0h) for which
  // this implementation of tanh(x) produce 1.0h output.
  const __m256 vpos_sat_cutoff = _mm256_set1_ps(0x1.1F0000p+2f);
  // Coefficient of polynomial approximation
  //   tanh(x) ~ x * (1 + t * (c3 + t * (c5 + t * (c7 + t * (c9 + t * (c11 + t * (c13 + t * (c15 + t * (c17 + t * c19)))))))))
  // on [-0x1.208p+2h, 0x1.208p+2] where t = x * x
  const __m256 vc19 = _mm256_set1_ps(-0x1.1D841Cp-32f);
  const __m256 vc17 = _mm256_set1_ps(0x1.C4FC88p-26f);
  const __m256 vc15 = _mm256_set1_ps(-0x1.332066p-20f);
  const __m256 vc13 = _mm256_set1_ps(0x1.D1AEA2p-16f);
  const __m256 vc11 = _mm256_set1_ps(-0x1.B2782Ep-12f);
  const __m256 vc9 = _mm256_set1_ps(0x1.03CAEAp-8f);
  const __m256 vc7 = _mm256_set1_ps(-0x1.967628p-6f);
  const __m256 vc5 = _mm256_set1_ps(0x1.ABC35Cp-4f);
  const __m256 vc3 = _mm256_set1_ps(-0x1.499D08p-2f);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) i));
    i += 8;

    // tanhh(x) saturates at -1 for large negative inputs and at +1 for large positive inputs: tanhh(x) == -1.0h for
    // x <= -0x1.208p+2 ~= -4.5078125 and tanhh(x) == 1.0h for x >= 0x1.208p+2 ~= 4.5078125. To guarantee this
    // behaviour, we clip input x on [neg_sat_cutoff, pos_sat_cutoff] containing [-0x1.208p+2, 0x1.208p+2], and
    // leverage the fact that for our implementation tanhh(neg_sat_cutoff) == -1.0h and tanhh(pos_sat_cutoff) == 1.0h.
    // NaN inputs are passed unchanged.
    vx = _mm256_max_ps(vneg_sat_cutoff, vx);
    vx = _mm256_min_ps(vpos_sat_cutoff, vx);

    // Compute t = x * x to use for polynomial evaluation
    const __m256 vt = _mm256_mul_ps(vx, vx);

    // Compute degree-19 polynomial approximation for tanh(x) on [-0x1.208p+2, 0x1.208p+2].
    //   P(t) = c3 + t * (c5 + t * (c7 + t * (c9 + t * (c11 + t * (c13 + t * (c15 + t * (c17 + t * c19)))))))
    __m256 vp = vc19;
    vp = _mm256_fmadd_ps(vp, vt, vc17);
    vp = _mm256_fmadd_ps(vp, vt, vc15);
    vp = _mm256_fmadd_ps(vp, vt, vc13);
    vp = _mm256_fmadd_ps(vp, vt, vc11);
    vp = _mm256_fmadd_ps(vp, vt, vc9);
    vp = _mm256_fmadd_ps(vp, vt, vc7);
    vp = _mm256_fmadd_ps(vp, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc3);

    // Reconstruct the tanh(x) value:
    //   tanh(x) ~ x * (1 + t * P(t))
    //           = x + (x * t) * P(t)
    const __m256 vxt = _mm256_mul_ps(vx, vt);
    const __m256 vy = _mm256_fmadd_ps(vp, vxt, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
}
