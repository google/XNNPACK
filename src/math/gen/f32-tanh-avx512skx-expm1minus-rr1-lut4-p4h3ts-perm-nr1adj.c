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


void xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m512) == 0);

  // The smallest z for which tanhf(-z) is saturated at -1.0f.
  const __m512 vsat_cutoff = _mm512_set1_ps(0x1.205968p+3f);
  const __m512 vminus_log2e = _mm512_set1_ps(-0x1.715476p+0f);
  // Large number such that ulp(magic bias) == exp2(-3)
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p+20f);
  // Table of exp2(k / 4) values decremented (as integer) by (k << 21), k = 0..3
  const __m512 vtable = _mm512_set_ps(
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f,
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f,
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f,
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f);
  const __m512 vln2 = _mm512_set1_ps(0x1.62E430p-1f);
  // Coefficients of polynomial approximation
  //   exp(2t) - 1 ~ t * (-2 + t * (c2 + t * (c3 + t * c4)))
  // on [-log(2)/16, log(2)/16]
  const __m512 vc4 = _mm512_set1_ps(0x1.554F9Ap-1f);
  const __m512 vc3 = _mm512_set1_ps(-0x1.557082p+0f);
  const __m512 vc2 = _mm512_set1_ps(0x1.000002p+1f);
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

    // Compute reduced argument n := round(-z / log(2), 3).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 3 fractional bits,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**19, i.e. |z| <= 0x1.62E43p+18 = 363408.75), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x).
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. 0 <= z <= 9.010913. As
    // n has 3 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 2 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fetched value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 9.010913 we have -13 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -13.
    //
    // Shift bits 2:10 into 23:31 (position of floating-point exponent).
    const __m512i ve = _mm512_slli_epi32(_mm512_castps_si512(vn), 21);

    // Use bits 0:2 bits of n, as integer, as an index for table lookup of l := 2**frac(2n).
    const __m512i vl = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn)));

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ve));

    // Subtract the large number back to get final n := round(-z / log(2), 3) as a floating-point number.
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    // Compute degree-4 polynomial approximation for exp(-2t) - 1 on [-log(2)/16, log(2)/16].
    //   P(t) = t * (-2 + t * (c2 + t * (c3 + t * c4)))
    //        = t * p
    __m512 vp = vc4;
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vminus_two);

    // Reconstruct the exp(-2z) - 1 value:
    //   exp(-2z) - 1 = s * (t * (-2 + t * (c2 + t * (c3 + t * c4))) + 1) - 1
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

    // Adjust reconstructred expm1(-2z) / (2 + expm1(-2z)) to match the correctly rounded division result
    const __m512 vey = _mm512_fnmadd_ps(vy, vepo, vemo);
    vy = _mm512_fmadd_ps(vey, vrepo, vy);

    // Reconstruct tanh(x) = copysign(y, x)
    vy = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy), _mm512_castps_si512(vx), vsign_mask, 0xD8));

    _mm512_store_ps(output, vy);
    output += 16;
  }
}
