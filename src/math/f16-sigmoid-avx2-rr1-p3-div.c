// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_sigmoid__avx2_rr1_p3_div(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // Floating-point mask with only the sign bit set
  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation of
  // exp(t) ~ 1 + t * (c1 + t * (c2 + t * c3)) on [-log(2)/2, log(2)/2]
  const __m256 vc3 = _mm256_set1_ps(0x1.5249A6p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.021D60p-1f);
  const __m256 vc1 = _mm256_set1_ps(0x1.000CD6p+0f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  // The smallest x for which sigmoidh(x) is normalized.
  // This number is also the smallest x for which exph(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.368000p+3f);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[z] := exp(z) / (1 + exp(z)) where z = -abs(x), then replace result with 1 - f[z] if x >= 0.
    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    // Compute reduced argument n := round(z / log(2)).
    // We do it by adding a large number (magic bias) to the product z * (1/log(2)), which cause rounding of the
    // result to an integer, then subtracing the large number back. The first addition is combined with multiplication
    // by log2e into a single FMA instruction. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**9, i.e. |z| <= 0x1.630p+8 = 355.0), but that is acceptable, because inputs x outside
    // of [-9.703125, 8.3125] (i.e. z outside [9.703125, 0]) underflow or saturate sigmoidh(x). We fixup the result for
    // such inputs at the very end of the algorithm.
    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -9.703125 <= z <= 0.0, and -14 <= n <= 0 accordingly.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get the final n := round(z / log(2)) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-3 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    //   P(t) = 1 + t * (c1 + t * (c2 + t * c3)) = 1 + t * p
    __m256 vp = _mm256_fmadd_ps(vc3, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    // Reconstruct the exp(z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * c3)))
    //     = s + (t * s) * (c1 + t * (c2 + t * c3))
    //     = s + (t * s) * p
    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    // Denominator of the sigmoid fraction: 1.0 + exp(z)
    const __m256 vd = _mm256_add_ps(ve, vone);

    // Reconstruct sigmoid(z) = exp(z) / (1.0 + exp(z))
    __m256 vf = _mm256_div_ps(ve, vd);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(z) : 1.0 - sigmoid(z)
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
}
