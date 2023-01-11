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


void xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4_perm_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m512) == 0);

  // Mask for the sign bit.
  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);
  // The largest z for which tanhf(-z) is not saturated at -1.0f.
  const __m512 vsat_cutoff = _mm512_set1_ps(0x1.205966p+3f);
  // Large number such that ulp(magic bias) == exp2(-3).
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p20f);
  const __m512 vminus_log2e = _mm512_set1_ps(-0x1.715476p+0f);
  // Table of exp2(k / 4) values decremented (as integer) by (k << 21), k = 0..3
  const __m512 vtable = _mm512_set_ps(
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f,
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f,
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f,
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f);
  const __m512 vln2 = _mm512_set1_ps(0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(-2t) - 1 ~ -2 * (t * (1 + t * (c2 + t * (c3 + t * c4))))
  // on [-log(2)/16, log(2)/16]
  const __m512 vc4 = _mm512_set1_ps(-0x1.554F9Ap-2f);
  const __m512 vc3 = _mm512_set1_ps(0x1.557082p-1f);
  const __m512 vc2 = _mm512_set1_ps(-0x1.000002p+0f);
  const __m512 vone = _mm512_set1_ps(1.0f);
  const __m512 vminus_two = _mm512_set1_ps(-2.0f);

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_load_ps(input);
    input += 16;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f[x] :=
    //           \ -f[-x] if x >= 0
    //
    // First we compute f[-z] := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
    // then replace result with -f[-z] if x >= 0.
    __m512 vz = _mm512_castsi512_ps(_mm512_andnot_epi32(vsign_mask, _mm512_castps_si512(vx)));

    // The function f[z] saturates at -1 for large inputs: tanhf(x) == -1.0f for x <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. The order of operands in the VMINPS instruction matters: it ensures that NaN
    // inputs are passed unchanged.
    vz = _mm512_min_ps(vsat_cutoff, vz);

    // Compute reduced argument n := round(-z / log(2), 3).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**19, i.e. |z| <= 0x1.62E43p+18 = 363408.75), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x). We fixup the result for such
    // inputs at the very end of the algorithm.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. -17.328680 <= x <= 0.0. As
    // n has 3 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 2 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 9.010913 we have -13 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -13.
    //
    // Shift bits 2:10 into 23:31 (position of floating-point exponent).
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 21);

    // Use bits 0:2 bits of n, as integer, as an index for table lookup of l := 2**frac(2n).
    const __m512i vl = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn)));

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));

    // Subtract the large number back to get final n := round(-z / log(2), 3) as a floating-point number.
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    // Compute degree-4 polynomial approximation for exp(-2t) - 1 on [-log(2)/16, log(2)/16].
    //   P(-2t) = t * (1 + t * (c2 + t * (c3 + t * c4)))
    //          = t + t * (t * (c2 + t * (c3 + t * c4)))
    //          = -2 * (t + t * p)
    __m512 vp = _mm512_fmadd_ps(vc4, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 - 2t * (1 + t * (c2 + t * (c3 + t * c4)))) - 1
    //              = (s - 1) + s * (-2t) * (t + t * p)
    //              = (s - 1) - 2 * ((t * s) + (t * s) * p)
    const __m512 vts = _mm512_mul_ps(vt, vs);
    const __m512 vsm1 = _mm512_sub_ps(vs, vone);
    vp = _mm512_fmadd_ps(vp, vts, vts);
    const __m512 vem1 = _mm512_fmadd_ps(vp, vminus_two, vsm1);

    // Reconstruct tanh(-z) := expm1(-2z) / (2 + expm1(-2z))
    const __m512 vep1 = _mm512_sub_ps(vem1, vminus_two);
    const __m512 vabsy = _mm512_div_ps(vem1, vep1);

    // Reconstruct tanh[x] = copysign(tanh(abs(x)), x).
    const __m512 vy = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vabsy), _mm512_castps_si512(vx), vsign_mask, 0xD8));

    _mm512_store_ps(output, vy);
    output += 16;
  }
}
