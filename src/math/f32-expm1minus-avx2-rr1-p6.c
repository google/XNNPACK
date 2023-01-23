// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_expm1minus__avx2_rr1_p6(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.154246p+4f);
  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/2, log(2)/2]
  const __m256 vc6 = _mm256_set1_ps(0x1.6b7338p-10f);
  const __m256 vc5 = _mm256_set1_ps(0x1.12278Ep-7f);
  const __m256 vc4 = _mm256_set1_ps(0x1.555716p-5f);
  const __m256 vc3 = _mm256_set1_ps(0x1.5554B0p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFFFEp-2f);
  const __m256 vone = _mm256_set1_ps(1.0f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1f(sat_cutoff) == -1.0f. The order of operands in the [V]MAXPS instruction matters: it ensures that NaN
    // inputs are passed unchanged.
    vx = _mm256_max_ps(vsat_cutoff, vx);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**22,
    // i.e. |x| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs x are restricted to
    // [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for valid inputs, i.e.
    // -17.328680 <= x <= 0.0, and -25 <= n <= 0 accordingly.
    // For NaN inputs, s would have zero mantissa and can have arbitrary sign and exponent, depending on the input
    // NaN payload. In these cases, n and t are NaNs with the same payload as input while s is non-NaN, and thus
    // input payload would be propagated in all computations.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    // Compute degree-6 polynomial approximation for exp(t) - 1 on [-log(2)/2, log(2)/2].
    //   P(t) = t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t + t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) = t + t * p
    __m256 vp = _mm256_fmadd_ps(vc6, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))) - 1
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
