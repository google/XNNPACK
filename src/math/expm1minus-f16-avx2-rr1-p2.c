// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_expm1minus__avx2_rr1_p2(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.0A4000p+3f);
  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * c2)
  // on [-log(2)/2, log(2)/2]
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFAEEp-2f);
  const __m256 vc1 = _mm256_set1_ps(0x1.028C1Cp0f);
  const __m256 vone = _mm256_set1_ps(1.0f);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    // The function saturates at -1 for large negative inputs: expm1h(x) == -1.0h for x <= sat_cutoff ~= -8.3203125.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1m(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vx = _mm256_max_ps(vx, vsat_cutoff);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The addition is combined with multiplication by log2e into a single FMA instruction. The
    // trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**9, i.e.
    // |x| <= 0x1.630p+8 = 355.0), but that is acceptable, because inputs x are restricted to [-8.3203125, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for valid inputs, i.e.
    // -8.3203125 <= x <= 0.0, and -12 <= n <= 0 accordingly.
    // For NaN inputs, s would have zero mantissa and can have arbitrary sign and exponent, depending on the input
    // NaN payload. In these cases, n and t are NaNs with the same payload as input while s is non-NaN, and thus
    // input payload would be propagated in all computations.
    __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    // Compute degree-2 polynomial approximation for exp(t) - 1 on [-log(2)/2, log(2)/2].
    //   P(t) = t * (c1 + t * (c2 + t * c3))
    //        = t * p
    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * p) - 1
    //              = (s - 1) + (s * t) * p
    //              = (t * s) * p + (s - 1)
    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_sub_ps(vs, vone);
    const __m256 vf = _mm256_fmadd_ps(vp, vt, vs);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
}
