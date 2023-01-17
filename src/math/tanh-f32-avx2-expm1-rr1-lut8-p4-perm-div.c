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


void xnn_math_f32_tanh__avx2_expm1_rr1_lut8_p4_perm_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m256) == 0);

  // Mask for the sign bit.
  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  // The largest z for which tanhf(z) is saturated at -1.0f.
  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.205968p+3f);
  // Large number such that ulp(magic bias) == exp2(-4).
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p19f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  // Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
  const __m256i vtable = _mm256_set_epi32(
    0x3F7AC0C7, 0x3F7744FD, 0x3F75672A, 0x3F7504F3, 0x3F75FED7, 0x3F7837F0, 0x3F7B95C2, 0x3F800000);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(-2t) - 1 ~ -2 * (t * (1 + t * (c2 + t * (c3 + t * c4))))
  // on [-log(2)/32, log(2)/32]
  const __m256 vc4 = _mm256_set1_ps(0x1.5558ECp-2f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555C20p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.000000p+0f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vtwo = _mm256_set1_ps(2.0f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_load_ps(input);
    input += 8;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f[x] :=
    //           \ -f[-x] if x >= 0
    //
    // First we compute f[z] := expm1(2z) / (2 + expm1(2z)) where z = -abs(x),
    // then replace result with -f[z] if x >= 0.
    __m256 vz = _mm256_or_ps(vx, vsign_mask);

    // Inverted mask for the sign of input: 0x00000000 for negative x, 0x80000000 for positive x.
    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);

    // The function f[z] saturates at -1 for large inputs: tanhf(x) == -1.0f for x <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. The order of operands in the [V]MAXPS instruction matters: it ensures that NaN
    // inputs are passed unchanged.
    vz = _mm256_max_ps(vsat_cutoff, vz);

    // Compute reduced argument n := round(z / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**18, i.e. |z| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x). We fixup the result for such
    // inputs at the very end of the algorithm.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. -17.328680 <= x <= 0.0. As
    // n has 4 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 3 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fetched value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 9.010913 we have -13 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -13.
    //
    // Shift bits 3:11 into 23:31 (position of floating-point exponent).
    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 20);

    // Use bits 0:3 bits of n, as integer, as an index for table lookup of l := 2**frac(2n).
    const __m256i vl = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn));

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));

    // Subtract the large number back to get final n := round(z / log(2), 4) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-4 polynomial approximation for exp(2t) - 1 on [-log(2)/32, log(2)/32].
    //   P(2t) = t * (1 + t * (c2 + t * (c3 + t * c4)))
    //          = t + t * (t * (c2 + t * (c3 + t * c4)))
    //          = 2 * (t + t * p)
    __m256 vp = _mm256_fmadd_ps(vc4, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + 2t * (1 + t * (c2 + t * (c3 + t * c4)))) - 1
    //              = (s - 1) + s * (2t) * (t + t * p)
    //              = (s - 1) + 2 * ((t * s) + (t * s) * p)
    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsm1 = _mm256_sub_ps(vs, vone);
    vp = _mm256_fmadd_ps(vp, vts, vts);
    const __m256 vem1 = _mm256_fmadd_ps(vp, vtwo, vsm1);

    // Reconstruct tanh(-z) := expm1(-2z) / (2 + expm1(-2z))
    const __m256 vep1 = _mm256_add_ps(vem1, vtwo);
    const __m256 vabsy = _mm256_div_ps(vem1, vep1);

    // Reconstruct tanh[x] = sign(x) * tanh[-abs(x)].
    // As tanh[-abs(x)] is negative, flips the sign bit if x is positive.
    const __m256 vy = _mm256_xor_ps(vabsy, vinvsignx);

    _mm256_store_ps(output, vy);
    output += 8;
  }
}
