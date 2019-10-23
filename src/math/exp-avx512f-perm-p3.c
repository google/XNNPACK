// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx512f_perm_p3(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p23f);
  // The smallest x for which expf(x) is non-zero.
  const __m512 vzero_cutoff = _mm512_set1_ps(-0x1.9FE368p6f);
  // The largest x for which expf(x) is finite.
  const __m512 vinf_cutoff = _mm512_set1_ps(0x1.62E42Ep6f);
  const __m512 vlog2e_x16  = _mm512_set1_ps(0x1.715476p4f);
  const __m512 vminus_ln2_o16_hi = _mm512_set1_ps(-0x1.62e43p-5f);
  const __m512 vminus_ln2_o16_lo = _mm512_set1_ps(0x1.05c61p-33f);
  const __m512 vplus_inf = _mm512_set1_ps(INFINITY);

  const __m512 vc2 = _mm512_set1_ps(0x1.00021Ep-1f);
  const __m512 vc3 = _mm512_set1_ps(0x1.55559Ap-3f);
  const __m512 vtable = _mm512_set_ps(
    0x1.EA4AFAp+0f, 0x1.D5818Ep+0f, 0x1.C199BEp+0f, 0x1.AE89FAp+0f,
    0x1.9C4918p+0f, 0x1.8ACE54p+0f, 0x1.7A1148p+0f, 0x1.6A09E6p+0f,
    0x1.5AB07Ep+0f, 0x1.4BFDAEp+0f, 0x1.3DEA64p+0f, 0x1.306FE0p+0f,
    0x1.2387A6p+0f, 0x1.172B84p+0f, 0x1.0B5586p+0f, 0x1.000000p+0f);

  const __m512i vmin_exponent = _mm512_set1_epi32(0xC1000000);
  const __m512i vmax_exponent = _mm512_set1_epi32(0x3F800000);
  const __m512i vdefault_exponent = vmax_exponent;
  const __m512i vmantissa_mask = _mm512_set1_epi32(0x007FFFF0);

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);

    // Compute reduced argument n := round(x * 16 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs outside of [-103.97207, 88.72283] underflow or overflow expf(x) anyway. We fixup the result for such
    // inputs at the very end of the algorithm.
    __m512 vn = _mm512_fmadd_ps(vx, vlog2e_x16, vmagic_bias);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -103.97207 <= x <= 88.72283, and -150 <= n <= 128 accordingly.
    // We need to use two numbers rather than one because a normalized single-precision exponent must be in [-127, 126]
    // range, which is insufficient to cover [-150, 128] range of n.
    // - When n is within [-127, 126], sn == 2**n and so == 1.0.
    // - When n < -127, sn == 2**(-127) and so == 2**(n + 127).
    // - When n > 126, sn == 2**126 and so == 2**(n - 126).
    __m512i veo = _mm512_slli_epi32(_mm512_and_si512(_mm512_castps_si512(vn), vmantissa_mask), 19);
    __m512i ven = _mm512_max_epi32(veo, vmin_exponent);
    ven = _mm512_min_epi32(ven, vmax_exponent);
    veo = _mm512_sub_epi32(veo, ven);
    const __m512 vsn = _mm512_castsi512_ps(_mm512_add_epi32(ven, vdefault_exponent));
    const __m512 vso = _mm512_castsi512_ps(_mm512_add_epi32(veo, vdefault_exponent));

    // Use the low 4 bits of n (as integer) for table lookup.
    const __m512 vl = _mm512_permutexvar_ps(_mm512_castps_si512(vn), vtable);

    // Subtract the large number back to get final n := round(x * 16 / log(2)).
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2) / 16.
    // Use Cody-Waite range reduction method (note two constants to represent log(2) / 16) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_o16_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_o16_lo, vt);

    // Compute degree-3 polynomial approxiatmion for exp(t) on [-log(2)/32, log(2)/32].
    __m512 vp = _mm512_fmadd_ps(vt, vc3, vc2);
    vp = _mm512_mul_ps(vp, vt);
    vp = _mm512_fmadd_ps(vt, vp, vt);

    // Reconstruct the final f value:
    //   f = so * sn * l * (1 + t * (1 + t * (c2 + t * c3)))
    //     = so * sn * (l + l * (t + t * (t * (c2 + t * c3))))
    //     = so * sn * (l + l * p)
    __m512 vf = _mm512_fmadd_ps(vl, vp, vl);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm512_maskz_mul_ps(_mm512_cmp_ps_mask(vx, vzero_cutoff, _CMP_NLT_US), vf, vsn);
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm512_mask_mul_ps(vplus_inf, _mm512_cmp_ps_mask(vx, vinf_cutoff, _CMP_NGT_US), vso, vf);
    _mm512_storeu_ps(output, vf);

    input += 16;
    output += 16;
  }
}
