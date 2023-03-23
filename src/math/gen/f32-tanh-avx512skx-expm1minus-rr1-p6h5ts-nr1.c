// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-avx512skx-expm1minus.c.in
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

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m512) == 0);

  // The smallest z for which tanhf(-z) is saturated at -1.0f.
  const __m512 vsat_cutoff = _mm512_set1_ps(0x1.205968p+3f);
  const __m512 vminus_log2e = _mm512_set1_ps(-0x1.715476p+0f);
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.8000FEp+22f);
  const __m512 vln2 = _mm512_set1_ps(0x1.62E430p-1f);
  // Coefficients of polynomial approximation
  //   exp(2t) - 1 ~ t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/4, log(2)/4]
  const __m512 vc6 = _mm512_set1_ps(0x1.6B7338p-4f);
  const __m512 vc5 = _mm512_set1_ps(-0x1.12278Ep-2f);
  const __m512 vc4 = _mm512_set1_ps(0x1.555716p-1f);
  const __m512 vc3 = _mm512_set1_ps(-0x1.5554B0p+0f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFFFEp+0f);
  const __m512 vminus_two = _mm512_set1_ps(-2.0f);
  const __m512 vone = _mm512_set1_ps(1.0f);
  // Mask for the sign bit.
  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);

  for (; n != 0; n -= sizeof(__m512)) {
    const __m512 vx = _mm512_load_ps(input);
    input += 16;

    // General structure of the algorithm:
    //
    //           / -expm1(-2x) / (2 + expm1(-2x)) if x >= 0
    //   f(x) :=
    //           \ -f(-x) if x <= 0
    //
    // First we compute y := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
    // then set its sign according to the sign of x: f(x) := sign(x) * abs(y).
    //
    // The function saturates at -1 for large positive inputs: tanhf(-z) == -1.0f for z >= sat_cutoff ~= 9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    const __m512 vz = _mm512_range_ps(vsat_cutoff, vx, 0xA);

    // Compute reduced argument n := round(-z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 1 fractional bit,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x).
    // Additionally, we fuse addition of the floating-point exponent bias (127) into the magic bias.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // 0 <= z <= 9.010913, and -13 <= n <= 0 accordingly.
    const __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));

    // Subtract the large number back to get final n := round(-z / log(2), 1) as a floating-point number.
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    // Compute degree-6 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(t) = t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t * p
    __m512 vp = vc6;
    vp = _mm512_fmadd_ps(vp, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vminus_two);

    // Reconstruct the exp(-2z) - 1 value:
    //   exp(-2z) - 1 = s * (t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) + 1) - 1
    //                = s * t * p + (s - 1)
    //                = (s - 1) + (t * s) * p
    const __m512 vts = _mm512_mul_ps(vt, vs);
    const __m512 vsmo = _mm512_sub_ps(vs, vone);
    const __m512 vemo = _mm512_fmadd_ps(vp, vts, vsmo);

    // Denominator of the tanh fraction: exp(-2z) + 1 = expm1(-2z) + 2
    const __m512 vepo = _mm512_sub_ps(vemo, vminus_two);

    // Use Newton-Raphson method (1 iteration) to compute reciprocal of the denominator.
    // Note: 2 < exp(-2z) + 1 <= 3, because z <= 0 and 0 < exp(2z) <= 1.
    // Thus the reciprocal of the denominator never overflows.
    __m512 vrepo = _mm512_rcp14_ps(vepo);
    const __m512 verepo = _mm512_fnmadd_ps(vrepo, vepo, vone);
    vrepo = _mm512_fmadd_ps(verepo, vrepo, vrepo);

    // Reconstruct y = expm1(-2z) / (expm1(-2z) + 2)
    __m512 vy = _mm512_mul_ps(vemo, vrepo);


    // Reconstruct tanh(x) = copysign(y, x)
    vy = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy), _mm512_castps_si512(vx), vsign_mask, 0xD8));

    _mm512_store_ps(output, vy);
    output += 16;
  }
}
