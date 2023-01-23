// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <emmintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__sse2_rr2_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p+23f);
  // The smallest x for which expf(x) is non-zero.
  const __m128 vzero_cutoff = _mm_set1_ps(-0x1.9FE368p+6f);
  // The largest x for which expf(x) is finite.
  const __m128 vinf_cutoff = _mm_set1_ps(0x1.62E42Ep+6f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  // Last 8 bits are zeroes
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vplus_inf = _mm_set1_ps(INFINITY);

  const __m128 vc1 = _mm_set1_ps(0x1.FFFFF6p-1f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFDC6p-2f);
  const __m128 vc3 = _mm_set1_ps(0x1.555A80p-3f);
  const __m128 vc4 = _mm_set1_ps(0x1.573A1Ap-5f);
  const __m128 vc5 = _mm_set1_ps(0x1.0F9F9Cp-7f);

  const __m128i vmin_exponent = _mm_set1_epi32(0xC1000000);
  const __m128i vmax_exponent = _mm_set1_epi32(0x3F800000);
  const __m128i vdefault_exponent = vmax_exponent;

  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but that's ok, because inputs outside of [-103.97207, 88.72283] underflow or
    // overflow expf(x) anyway. We fixup the result for such inputs at the very end of the algorithm.
    __m128 vn = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic_bias);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -103.97207 <= x <= 88.72283, and -150 <= n <= 128 accordingly.
    // We need to use two numbers rather than one because a normalized single-precision exponent must be in [-127, 126]
    // range, which is insufficient to cover [-150, 128] range of n.
    // - When n is within [-127, 126], sn == 2**n and so == 1.0.
    // - When n < -127, sn == 2**(-127) and so == 2**(n + 127).
    // - When n > 126, sn == 2**126 and so == 2**(n - 126).
    __m128i veo = _mm_slli_epi32(_mm_castps_si128(vn), 23);
    __m128i ven = _mm_max_epi16(veo, vmin_exponent);
    ven = _mm_min_epi16(ven, vmax_exponent);
    veo = _mm_sub_epi32(veo, ven);
    const __m128 vsn = _mm_castsi128_ps(_mm_add_epi32(ven, vdefault_exponent));
    const __m128 vso = _mm_castsi128_ps(_mm_add_epi32(veo, vdefault_exponent));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = so * sn * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = sn * (so + (t * so) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))))
    //     = sn * (so + (t * so) * p)
    vt = _mm_mul_ps(vt, vso);
    __m128 vf = _mm_mul_ps(vsn, _mm_add_ps(_mm_mul_ps(vt, vp), vso));

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vx, vzero_cutoff), vf);
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    const __m128 vm = _mm_cmpgt_ps(vx, vinf_cutoff);
    vf = _mm_or_ps(_mm_and_ps(vplus_inf, vm), _mm_andnot_ps(vm, vf));
    _mm_storeu_ps(output, vf);

    input += 4;
    output += 4;
  }
}
