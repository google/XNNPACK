// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_expminus__avx2_rr1_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation
  //   exp(t) ~ 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
  // on [-log(2)/2, log(2)/2]
  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);
  // The smallest x for which expf(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep6f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The first addition is combined with multiplication by
    // log2e into a single FMA instruction. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**22, i.e. |x| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs outside
    // of [-87.336540, 0.0] underflow expf(x) anyway. We fixup the result for such inputs at the very end of the
    // algorithm.
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final n := round(x / log(2)) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) = 1 + t * p
    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    // Reconstruct the exp(x) value:
    //   exp(x) = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //          = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //          = s + (t * s) * p
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);
    _mm256_storeu_ps(output, vf);

    input += 8;
    output += 8;
  }
}
