// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx_rr2_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p+23f);
  // The smallest x for which expf(x) is non-zero.
  const __m256 vzero_cutoff = _mm256_set1_ps(-0x1.9FE368p+6f);
  // The largest x for which expf(x) is finite.
  const __m256 vinf_cutoff = _mm256_set1_ps(0x1.62E42Ep+6f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  // Last 8 bits are zeroes
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E400p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(-0x1.7F7D1Cp-20f);
  const __m256 vplus_inf = _mm256_set1_ps(INFINITY);

  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);

  const __m128i vmin_exponent = _mm_set1_epi32(0xC1000000);
  const __m128i vmax_exponent = _mm_set1_epi32(0x3F800000);
  const __m128i vdefault_exponent = vmax_exponent;

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but that's ok, because inputs outside of [-103.97207, 88.72283] underflow or
    // overflow expf(x) anyway. We fixup the result for such inputs at the very end of the algorithm.
    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vx, vlog2e), vmagic_bias);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -103.97207 <= x <= 88.72283, and -150 <= n <= 128 accordingly.
    // We need to use two numbers rather than one because a normalized single-precision exponent must be in [-127, 126]
    // range, which is insufficient to cover [-150, 128] range of n.
    // - When n is within [-127, 126], sn == 2**n and so == 1.0.
    // - When n < -127, sn == 2**(-127) and so == 2**(n + 127).
    // - When n > 126, sn == 2**126 and so == 2**(n - 126).
    __m128i veo_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23);
    __m128i veo_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 23);
    __m128i ven_lo = _mm_max_epi16(veo_lo, vmin_exponent);
    __m128i ven_hi = _mm_max_epi16(veo_hi, vmin_exponent);
    ven_lo = _mm_min_epi16(ven_lo, vmax_exponent);
    ven_hi = _mm_min_epi16(ven_hi, vmax_exponent);
    veo_lo = _mm_sub_epi32(veo_lo, ven_lo);
    veo_hi = _mm_sub_epi32(veo_hi, ven_hi);
    const __m128 vsn_lo = _mm_castsi128_ps(_mm_add_epi32(ven_lo, vdefault_exponent));
    const __m128 vsn_hi = _mm_castsi128_ps(_mm_add_epi32(ven_hi, vdefault_exponent));
    const __m128 vso_lo = _mm_castsi128_ps(_mm_add_epi32(veo_lo, vdefault_exponent));
    const __m128 vso_hi = _mm_castsi128_ps(_mm_add_epi32(veo_hi, vdefault_exponent));
    const __m256 vsn = _mm256_insertf128_ps(_mm256_castps128_ps256(vsn_lo), vsn_hi, 1);
    const __m256 vso = _mm256_insertf128_ps(_mm256_castps128_ps256(vso_lo), vso_hi, 1);

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc5, vt), vc4);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = so * sn * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = sn * (so + (t * so) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))))
    //     = sn * (so + (t * so) * p)
    vt = _mm256_mul_ps(vt, vso);
    __m256 vf = _mm256_mul_ps(vsn, _mm256_add_ps(_mm256_mul_ps(vt, vp), vso));

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vzero_cutoff, _CMP_LT_OS), vf);
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_blendv_ps(vf, vplus_inf, _mm256_cmp_ps(vx, vinf_cutoff, _CMP_GT_OS));
    _mm256_storeu_ps(output, vf);

    input += 8;
    output += 8;
  }
}
