// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_tanh__sse2_expm1_rr1_p6_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m128) == 0);

  // Mask for the sign bit.
  const __m128 vsign_mask = _mm_set1_ps(-0.0f);
  // The largest z for which tanhf(z) is saturated at -1.0f.
  const __m128 vsat_cutoff = _mm_set1_ps(-0x1.205968p+3f);
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const __m128 vmagic_bias = _mm_set1_ps(0x1.8000FEp+22f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128 vminus_ln2 = _mm_set1_ps(-0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(-2t) - 1 ~ -2 * (t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))))
  // on [-log(2)/2, log(2)/2]
  const __m128 vc6 = _mm_set1_ps(0x1.6b7338p-5f);
  const __m128 vc5 = _mm_set1_ps(0x1.12278Ep-3f);
  const __m128 vc4 = _mm_set1_ps(0x1.555716p-2f);
  const __m128 vc3 = _mm_set1_ps(0x1.5554B0p-1f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFFFEp-1f);
  const __m128 vone = _mm_set1_ps(1.0f);
  const __m128 vtwo = _mm_set1_ps(2.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f[x] :=
    //           \ -f[-x] if x >= 0
    //
    // First we compute f[z] := expm1(2z) / (2 + expm1(2z)) where z = -abs(x),
    // then replace result with -f[z] if x >= 0.
    __m128 vz = _mm_or_ps(vx, vsign_mask);

    // Inverted mask for the sign of input: 0x00000000 for negative x, 0x80000000 for positive x.
    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    // The function f[z] saturates at -1 for large inputs: tanhf(x) == -1.0f for x <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. The order of operands in the [V]MAXPS instruction matters: it ensures that NaN
    // inputs are passed unchanged.
    vz = _mm_max_ps(vsat_cutoff, vz);

    // Compute reduced argument n := round(z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [-9.010913, 0]) saturate tanhf(x). We fixup the result for such
    // inputs at the very end of the algorithm.
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
    //   P(2t) = 2t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //          = 2t + 2t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //          = 2 * (t + t * p)
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc6, vt), vc5);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + 2t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))) - 1
    //              = (s - 1) + s * (2t) * (t + t * p)
    //              = (s - 1) + 2 * ((t * s) + (t * s) * p)
    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsm1 = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vts), vts);
    const __m128 vem1 = _mm_add_ps(_mm_mul_ps(vp, vtwo), vsm1);

    // Reconstruct tanh(-z) := expm1(-2z) / (2 + expm1(-2z))
    const __m128 vep1 = _mm_add_ps(vem1, vtwo);
    const __m128 vabsy = _mm_div_ps(vem1, vep1);

    // Reconstruct tanh[x] = sign(x) * tanh[-abs(x)].
    // As tanh[-abs(x)] is negative, flips the sign bit if x is positive.
    const __m128 vy = _mm_xor_ps(vabsy, vinvsignx);

    _mm_store_ps(output, vy);
    output += 4;
  }
}
