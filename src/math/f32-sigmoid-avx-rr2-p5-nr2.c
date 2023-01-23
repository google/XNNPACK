// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__avx_rr2_p5_nr2(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  // Floating-point mask with only the sign bit set
  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  // Last 7 bits are zeroes
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E400p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(-0x1.7F7D1Cp-20f);
  // Coefficient of polynomial approximation of
  // exp(t) ~ 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) on [-log(2)/2, log(2)/2]
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vtwo = _mm256_set1_ps(2.0f);
  // The smallest x for which sigmoidf(x) is normalized.
  // This number is also the smallest x for which expf(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep+6f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[z] := exp(z) / (1 + exp(z)) where z = -abs(x), then replace result with 1 - f[z] if x >= 0.
    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    // Compute reduced argument n := round(z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**22, i.e. |z| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs x outside
    // of [-87.336544, 17.328678] (i.e. z outsize [87.336544, 0]) underflow or saturate sigmoidf(x). We fixup the
    // result for such inputs at the very end of the algorithm.
    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= z <= 0.0, and -126 <= n <= 0 accordingly.
    const __m128 vs_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 23));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    // Subtract the large number back to get the final n := round(z / log(2)) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    //   P(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) = 1 + t * p
    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc5, vt), vc4);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc1);

    // Reconstruct the exp(z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_add_ps(_mm256_mul_ps(vt, vp), vs);

    // Denominator of the sigmoid fraction: 1.0 + exp(z)
    const __m256 vd = _mm256_add_ps(ve, vone);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    __m256 vr = _mm256_rcp_ps(vd);
    vr = _mm256_mul_ps(vr, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr, vd)));
    vr = _mm256_mul_ps(vr, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr, vd)));

    // Reconstruct sigmoid(z) = exp(z) / (1.0 + exp(z))
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
