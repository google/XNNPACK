// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__avx512f_rr1_p5_nr1fma(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 vmagic_bias = _mm512_set1_ps(0x1.8000FEp23f);
  // The smallest x for which sigmoidf(x) is normalized.
  // This number is also the smallest x for which expf(x) is normalized.
  const __m512 vdenorm_cutoff = _mm512_set1_ps(-0x1.5D589Ep+6f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 vone = _mm512_set1_ps(1.0f);
  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);

  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[z] := exp(z) / (1 + exp(z)) where z = -abs(x),
    // then replace result with 1 - f[z] if x >= 0.
    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    // Compute reduced argument n := round(z / log(2)).
    // We do it by adding a large number (magic bias) to the product z * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but thats ok, because inputs x outside of [-87.336544, 17.328678] (i.e. z outsize
    // [0, 87.336544]) underflow or saturate sigmoidf(x) anyway. We fixup the result for such inputs at the very end of
    // the algorithm.
    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= z <= 0.0, and -126 <= n <= 0 accordingly.
    const __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));

    // Subtract the large number back to get final n := round(z / log(2)).
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);

    // Reconstruct the exp(z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm512_mul_ps(vt, vs);
    const __m512 ve = _mm512_fmadd_ps(vt, vp, vs);

    // Denominator of the sigmoid fraction: 1.0 + exp(z)
    const __m512 vd = _mm512_add_ps(ve, vone);

    // Use Newton-Raphson method (1 iteration) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    __m512 vr = _mm512_rcp14_ps(vd);
    vr = _mm512_fmadd_ps(_mm512_fnmadd_ps(vr, vd, vone), vr, vr);

    // Reconstruct sigmoid(-z) = exp(z) / (1.0 + exp(z))
    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is true, and outputs are left unchanged.
    __m512 vf = _mm512_maskz_mul_ps(_mm512_cmp_ps_mask(vz, vdenorm_cutoff, _CMP_NLT_US), ve, vr);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(z) : 1.0 - sigmoid(z)
    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_storeu_ps(output, vf);

    input += 16;
    output += 16;
  }
}
