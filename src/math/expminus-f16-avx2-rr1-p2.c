// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_expminus__avx2_rr1_p2(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation of
  // exp(t) ~ 1 + t * (c1 + t * c2) on [-log(2)/2, log(2)/2]
  const __m256 vc2 = _mm256_set1_ps(0x1.FF3A32p-2f);
  const __m256 vc1 = _mm256_set1_ps(0x1.039E10p+0f);
  // The smallest x for which exph(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.368000p+3f);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the
    // result to an integer, then subtracing the large number back. The first addition is combined with multiplication
    // by log2e into a single FMA instruction. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**9, i.e. |x| <= 0x1.630p+8 = 355.0), but that is acceptable, because inputs x outside
    // of [-9.703125, 0] underflow expf(x). We fixup the result for such inputs at the very end of the algorithm.
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -9.703125 <= x <= 0.0, and -14 <= n <= 0 accordingly.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final n := round(x / log(2)) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (c1 + t * c2) = 1 + t * p
    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);

    // Reconstruct the exp(x) value:
    //   exp(x) = s * (1 + t * (c1 + t * c2))
    //          = s + (t * s) * (c1 + t * c2)
    //          = s + (t * s) * p
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);
    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
}
