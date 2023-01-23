// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>



void xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const __m512 vsat_cutoff = _mm512_set1_ps(-0x1.154246p+4f);
  // Large number such that ulp(magic bias) == exp2(-4)
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p19f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  // Table of exp2(k / 16) values decremented (as integer) by (k << 19), k = 0..15
  const __m512i vtable = _mm512_set_epi32(
    0x3F7D257D, 0x3F7AC0C7, 0x3F78CCDF, 0x3F7744FD, 0x3F76248C, 0x3F75672A, 0x3F7508A4, 0x3F7504F3,
    0x3F75583F, 0x3F75FED7, 0x3F76F532, 0x3F7837F0, 0x3F79C3D3, 0x3F7B95C2, 0x3F7DAAC3, 0x3F800000);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * c3))
  // on [-log(2)/32, log(2)/32]
  const __m512 vc3 = _mm512_set1_ps(0x1.55561Cp-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.0001ECp-1f);
  const __m512 vone = _mm512_set1_ps(1.0f);

  for (; n != 0; n -= 16 * sizeof(float)) {
    __m512 vx = _mm512_loadu_ps(input);

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1f(sat_cutoff) == -1.0f. The order of operands in the [V]MAXPS instruction matters: it ensures that NaN
    // inputs are passed unchanged.
    vx = _mm512_max_ps(vsat_cutoff, vx);

    // Compute reduced argument n := round(x / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits, then
    // subtracing the large number back. The first addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**18,
    // i.e. |x| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x are restricted to
    // [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m512 vn = _mm512_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**n for valid inputs, i.e. -17.328680 <= x <= 0.0. As n
    // has 4 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s in two steps:
    // 1. Fetch 2**frac(n) from the table using the 4 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -17.328680 <= x <= 0.0 we have -25 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -25.
    //
    // Shift bits 4:12 into 23:31 (position of floating-point exponent).
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 19);

    // Use bits 0:4 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));

    // Subtract the large number back to get final n := round(x / log(2), 4).
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vx);

    // Compute degree-3 polynomial approximation for exp(t) - 1 on [-log(2)/32, log(2)/32].
    //   P(t) = t * (1 + t * (c2 + t * c3)) = t + t * (t * (c2 + t * c3)) = t + t * p
    __m512 vp = _mm512_fmadd_ps(vc3, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * c3))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt = _mm512_mul_ps(vt, vs);
    const __m512 vsm1 = _mm512_sub_ps(vs, vone);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    const __m512 vf = _mm512_add_ps(vp, vsm1);

    _mm512_storeu_ps(output, vf);

    input += 16;
    output += 16;
  }
}
