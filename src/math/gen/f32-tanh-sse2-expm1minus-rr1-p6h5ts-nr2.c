// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-sse-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr2(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m128) == 0);

  // Mask for the sign bit.
  const __m128 vsign_mask = _mm_set1_ps(-0.0f);
  // The largest z for which tanhf(z) is saturated at -1.0f.
  const __m128 vsat_cutoff = _mm_set1_ps(-0x1.205968p+3f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const __m128 vmagic_bias = _mm_set1_ps(0x1.8000FEp+22f);
  const __m128 vminus_ln2 = _mm_set1_ps(-0x1.62E430p-1f);
  // Coefficients of polynomial approximation
  //   exp(2t) - 1 ~ t * (2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/4, log(2)/4]
  const __m128 vc6 = _mm_set1_ps(0x1.6B7338p-4f);
  const __m128 vc5 = _mm_set1_ps(0x1.12278Ep-2f);
  const __m128 vc4 = _mm_set1_ps(0x1.555716p-1f);
  const __m128 vc3 = _mm_set1_ps(0x1.5554B0p+0f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFFFEp+0f);
  const __m128 vminus_two = _mm_set1_ps(-2.0f);
  const __m128 vminus_one = _mm_set1_ps(-1.0f);

  for (; n != 0; n -= sizeof(__m128)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f(x) :=
    //           \ -f(-x) if x >= 0
    //
    // First we compute f(z) := expm1(2z) / (2 + expm1(2z)) where z = -abs(x), then negate the result if x >= 0.
    __m128 vz = _mm_or_ps(vx, vsign_mask);

    // Inverted mask for the sign of input: 0x00000000 for negative x, 0x80000000 for positive x.
    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    // The function saturates at -1 for large negative inputs: tanhf(z) == -1.0f for z <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we compute the saturation mask here, and later use it to replace computed outputs
    // with the saturation value (-1). Note that for NaN inputs the saturation mask is inactive.
    const __m128 vm = _mm_cmple_ps(vz, vsat_cutoff);

    // Compute reduced argument n := round(z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 1 fractional bit,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [-9.010913, 0]) saturate tanhf(x).
    // Additionally, we fuse addition of the floating-point exponent bias (127) into the magic bias.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // -9.010913 <= z <= 0, and -13 <= n <= 0 accordingly.
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    // Subtract the large number back to get final n := round(z / log(2), 1) as a floating-point number.
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    const __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2), vz);

    // Compute degree-6 polynomial approximation for exp(2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(t) = t * (2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t * p
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc6, vt), vc5);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    // Reconstruct the exp(2z) - 1 value:
    //   exp(2z) - 1 = s * (t * (2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) + 1) - 1
    //               = s * t * p + (s - 1)
    //               = (s - 1) + (t * s) * p
    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vp, vts), vsmo);

    // Denominator of the tanh fraction: exp(2z) + 1 = expm1(2z) + 2
    const __m128 vepo = _mm_sub_ps(vminus_two, vemo);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of the denominator.
    // Note: 2 < exp(2z) + 1 <= 3, because z <= 0 and 0 < exp(2z) <= 1.
    // Thus the reciprocal of the denominator never overflows.
    __m128 vrepo = _mm_rcp_ps(vepo);
    vrepo = _mm_mul_ps(vrepo, _mm_add_ps(_mm_mul_ps(vrepo, vepo), vminus_two));
    vrepo = _mm_mul_ps(vrepo, _mm_sub_ps(_mm_mul_ps(vrepo, vepo), vminus_two));

    // Reconstruct tanh(z) := expm1(2z) / (2 + expm1(2z))
    __m128 vy = _mm_mul_ps(vemo, vrepo);

    // Saturate tanh(z) at -1 for large inputs.
    vy = _mm_or_ps(_mm_andnot_ps(vm, vy), _mm_and_ps(vminus_one, vm));

    // Reconstruct tanh(x):
    //
    //             / tanh(z) if x <= 0
    //   tanh(x) =
    //             \ -tanh(z) if x >= 0
    vy = _mm_xor_ps(vy, vinvsignx);

    _mm_store_ps(output, vy);
    output += 4;
  }
}
