// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx512f_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p+23f);
  // The smallest x for which expf(x) is non-zero.
  const __m512 vzero_cutoff = _mm512_set1_ps(-0x1.9FE368p+6f);
  // The largest x for which expf(x) is finite.
  const __m512 vinf_cutoff = _mm512_set1_ps(0x1.62E42Ep+6f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(0x1.05C61p-29f);
  const __m512 vplus_inf = _mm512_set1_ps(INFINITY);

  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);

  const __m512i vmin_exponent = _mm512_set1_epi32(0xC1000000);
  const __m512i vmax_exponent = _mm512_set1_epi32(0x3F800000);
  const __m512i vdefault_exponent = vmax_exponent;

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs outside of [-103.97207, 88.72283] underflow or overflow expf(x) anyway. We fixup the result for such
    // inputs at the very end of the algorithm.
    __m512 vn = _mm512_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -103.97207 <= x <= 88.72283, and -150 <= n <= 128 accordingly.
    // We need to use two numbers rather than one because a normalized single-precision exponent must be in [-127, 126]
    // range, which is insufficient to cover [-150, 128] range of n.
    // - When n is within [-127, 126], sn == 2**n and so == 1.0.
    // - When n < -127, sn == 2**(-127) and so == 2**(n + 127).
    // - When n > 126, sn == 2**126 and so == 2**(n - 126).
    __m512i veo = _mm512_slli_epi32(_mm512_castps_si512(vn), 23);
    __m512i ven = _mm512_max_epi32(veo, vmin_exponent);
    ven = _mm512_min_epi32(ven, vmax_exponent);
    veo = _mm512_sub_epi32(veo, ven);
    const __m512 vsn = _mm512_castsi512_ps(_mm512_add_epi32(ven, vdefault_exponent));
    const __m512 vso = _mm512_castsi512_ps(_mm512_add_epi32(veo, vdefault_exponent));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);

    // Reconstruct the final f value:
    //   f = so * sn * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = sn * (so + (t * so) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))))
    //     = sn * (so + (t * so) * p)
    vt = _mm512_mul_ps(vt, vso);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    __m512 vf = _mm512_maskz_fmadd_ps(_mm512_cmp_ps_mask(vx, vzero_cutoff, _CMP_NLT_US), vt, vp, vso);
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm512_mask_mul_ps(vplus_inf, _mm512_cmp_ps_mask(vx, vinf_cutoff, _CMP_NGT_US), vsn, vf);
    _mm512_storeu_ps(output, vf);

    input += 16;
    output += 16;
  }
}
