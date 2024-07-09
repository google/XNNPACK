// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>

#include "xnnpack/math-stubs.h"


void xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.154246p+4f);
  // Large number such that ulp(magic bias) == exp2(-2)
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p21f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  // Table of exp2(k / 4) values decremented (as integer) by (k << 21), k = 0..3
  const __m256 vtable = _mm256_set_ps(
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f,
    0x1.EE89FAp-1f, 0x1.EA09E6p-1f, 0x1.F06FE0p-1f, 0x1.000000p+0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * (c3 + t * c4)))
  // on [-log(2)/8, log(2)/8]
  const __m256 vc4 = _mm256_set1_ps(0x1.554F9Ap-5f);
  const __m256 vc3 = _mm256_set1_ps(0x1.557082p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.000002p-1f);
  const __m256 vone = _mm256_set1_ps(1.0f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1f(sat_cutoff) == -1.0f. The order of operands in the VMAXPS instruction matters: it ensures that NaN
    // inputs are passed unchanged.
    vx = _mm256_max_ps(vsat_cutoff, vx);

    // Compute reduced argument n := round(x / log(2), 2).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 2 fractional bits, then
    // subtracing the large number back. The first addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**19,
    // i.e. |x| <= 0x1.62E43p+18 = 363408.75), but that is acceptable, because inputs x are restricted to
    // [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**n for valid inputs, i.e. -17.328680 <= x <= 0.0. As n
    // has 4 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s in two steps:
    // 1. Fetch 2**frac(n) from the table using the 4 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -17.328680 <= x <= 0.0 we have -25 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -25.
    //
    // Shift bits 2:10 into 23:31 (position of floating-point exponent).
    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 21);

    // Use bits 0:2 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const __m256i vl = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn)));

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));

    // Subtract the large number back to get final n := round(x / log(2), 2).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    // Compute degree-4 polynomial approximation for exp(t) - 1 on [-log(2)/16, log(2)/16].
    //   P(t) = t * (1 + t * (c2 + t * (c3 + t * c4))) = t + t * (t * (c2 + t * (c3 + t * c4))) = t + t * p
    __m256 vp = _mm256_fmadd_ps(vc4, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * (c3 + t * c4)))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt = _mm256_mul_ps(vt, vs);
    const __m256 vsm1 = _mm256_sub_ps(vs, vone);
    vp = _mm256_fmadd_ps(vp, vt, vt);
    const __m256 vf = _mm256_add_ps(vp, vsm1);

    _mm256_storeu_ps(output, vf);

    input += 8;
    output += 8;
  }
}
