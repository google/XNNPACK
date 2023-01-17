// Copyright 2022 Google LLC
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


void xnn_math_f32_tanh__avx512f_expm1_rr1_p6_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m512) == 0);

  // Mask for the sign bit.
  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);
  // The largest z for which tanhf(-z) is not saturated at -1.0f.
  const __m512 vsat_cutoff = _mm512_set1_ps(0x1.205966p+3f);
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.8000FEp+22f);
  const __m512 vminus_log2e = _mm512_set1_ps(-0x1.715476p+0f);
  const __m512 vln2 = _mm512_set1_ps(0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(-2t) - 1 ~ -2 * (t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))))
  // on [-log(2)/4, log(2)/4]
  const __m512 vc6 = _mm512_set1_ps(-0x1.6b7338p-5f);
  const __m512 vc5 = _mm512_set1_ps(0x1.12278Ep-3f);
  const __m512 vc4 = _mm512_set1_ps(-0x1.555716p-2f);
  const __m512 vc3 = _mm512_set1_ps(0x1.5554B0p-1f);
  const __m512 vc2 = _mm512_set1_ps(-0x1.FFFFFEp-1f);
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

    // Compute reduced argument n := round(-z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x). We fixup the result for such
    // inputs at the very end of the algorithm.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // -9.010913 <= z <= 0, and -13 <= n <= 0 accordingly.
    const __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));

    // Subtract the large number back to get final n := round(-z / log(2), 1) as a floating-point number.
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    // Compute degree-6 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(-2t) = t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //          = t + t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //          = -2 * (t + t * p)
    __m512 vp = _mm512_fmadd_ps(vc6, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 - 2t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))) - 1
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
