// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx2_rr2_lut8_p3_perm(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p23f);
  // The smallest x for which expf(x) is non-zero.
  const __m256 vzero_cutoff = _mm256_set1_ps(-0x1.9FE368p6f);
  // The largest x for which expf(x) is finite.
  const __m256 vinf_cutoff = _mm256_set1_ps(0x1.62E42Ep6f);
  const __m256 vlog2e_x8  = _mm256_set1_ps(0x1.715476p3f);
  const __m256 vminus_ln2_o8_hi = _mm256_set1_ps(-0x1.62E43p-4f);
  const __m256 vminus_ln2_o8_lo = _mm256_set1_ps(0x1.05C61p-32f);
  const __m256 vplus_inf = _mm256_set1_ps(INFINITY);

  const __m256 vc2 = _mm256_set1_ps(0x1.00021Ep-1f);
  const __m256 vc3 = _mm256_set1_ps(0x1.55559Ap-3f);
  const __m256 vtable = _mm256_set_ps(
    0x1.D5818Ep+0f, 0x1.AE89FAp+0f, 0x1.8ACE54p+0f, 0x1.6A09E6p+0f,
    0x1.4BFDAEp+0f, 0x1.306FE0p+0f, 0x1.172B84p+0f, 0x1.000000p+0f);

  const __m256i vmin_exponent = _mm256_set1_epi32(0xC1000000);
  const __m256i vmax_exponent = _mm256_set1_epi32(0x3F800000);
  const __m256i vdefault_exponent = vmax_exponent;
  const __m256i vmantissa_mask = _mm256_set1_epi32(0x007FFFF8);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // Compute reduced argument n := round(x * 8 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but that's ok, because
    // inputs outside of [-103.97207, 88.72283] underflow or overflow expf(x) anyway. We fixup the result for such
    // inputs at the very end of the algorithm.
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e_x8, vmagic_bias);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -103.97207 <= x <= 88.72283, and -150 <= n <= 128 accordingly.
    // We need to use two numbers rather than one because a normalized single-precision exponent must be in [-127, 126]
    // range, which is insufficient to cover [-150, 128] range of n.
    // - When n is within [-127, 126], sn == 2**n and so == 1.0.
    // - When n < -127, sn == 2**(-127) and so == 2**(n + 127).
    // - When n > 126, sn == 2**126 and so == 2**(n - 126).
    __m256i veo = _mm256_slli_epi32(_mm256_and_si256(_mm256_castps_si256(vn), vmantissa_mask), 20);
    __m256i ven = _mm256_max_epi32(veo, vmin_exponent);
    ven = _mm256_min_epi32(ven, vmax_exponent);
    veo = _mm256_sub_epi32(veo, ven);
    const __m256 vsn = _mm256_castsi256_ps(_mm256_add_epi32(ven, vdefault_exponent));
    const __m256 vso = _mm256_castsi256_ps(_mm256_add_epi32(veo, vdefault_exponent));

    // Use the low 3 bits of n (as integer) for table lookup.
    __m256 vl = _mm256_permutevar8x32_ps(vtable, _mm256_castps_si256(vn));

    // Subtract the large number back to get final n := round(x * 8 / log(2)).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2) / 8.
    // Use Cody-Waite range reduction method (note two constants to represent log(2) / 8) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_o8_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_o8_lo, vt);

    // Compute degree-3 polynomial approximation for exp(t) on [-log(2)/16, log(2)/16].
    __m256 vp = _mm256_fmadd_ps(vt, vc3, vc2);

    // Reconstruct the final f value:
    //   f = so * sn * l * (1 + t * (1 + t * (c2 + t * c3)))
    //     = so * sn * (l + l * (t + t * (t * (c2 + t * c3))))
    //     = sn * ((l * so) + (l * so) * p)
    vl = _mm256_mul_ps(vl, vso);
    vp = _mm256_mul_ps(vp, vt);
    vp = _mm256_fmadd_ps(vt, vp, vt);
    __m256 vf = _mm256_fmadd_ps(vl, vp, vl);
    vf = _mm256_mul_ps(vf, vsn);

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
