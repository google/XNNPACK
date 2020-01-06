// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__avx2_rr1_p5_nr1fma(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  // The smallest x for which sigmoidf(x) is normalized.
  // This number is also the smallest x for which expf(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep+6f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);

  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[z] := exp(z) / (1 + exp(z)) where z = -abs(x),
    // then replace result with 1 - f[z] if x >= 0.
    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    // Compute reduced argument n := round(z / log(2)).
    // We do it by adding a large number (magic bias) to the product z * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but thats ok, because inputs x outside of [-87.336544, 17.328678] (i.e. z outsize
    // [0, 87.336544]) underflow or saturate sigmoidf(x) anyway. We fixup the result for such inputs at the very end of
    // the algorithm.
    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= z <= 0.0, and -126 <= n <= 0 accordingly.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final n := round(z / log(2)).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    // Reconstruct the exp(z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    // Denominator of the sigmoid fraction: 1.0 + exp(z)
    const __m256 vd = _mm256_add_ps(ve, vone);

    // Use Newton-Raphson method (1 iteration) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    __m256 vr = _mm256_rcp_ps(vd);
    vr = _mm256_fmadd_ps(_mm256_fnmadd_ps(vr, vd, vone), vr, vr);

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    __m256 vf = _mm256_mul_ps(ve, vr);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(z) : 1.0 - sigmoid(z)
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm256_storeu_ps(output, vf);

    input += 8;
    output += 8;
  }
}
