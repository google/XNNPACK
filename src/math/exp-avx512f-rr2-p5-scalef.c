// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx512f_rr2_p5_scalef(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);

  // The smallest x for which expf(x) is non-zero.
  const __m512 vzero_cutoff = _mm512_set1_ps(-0x1.9FE368p+6f);
  // The largest x for which expf(x) is finite.
  const __m512 vinf_cutoff = _mm512_set1_ps(0x1.62E42Ep+6f);

  const __m512 vminus_ln2_hi = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(0x1.05C61p-29f);

  const __m512 vc0 = _mm512_set1_ps(1.0f);
  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);

    // Compute reduced argument n := round(x / log(2)).
    const __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vx, vlog2e), 0);

    // Detect underflow and overflow of expf(x) for further special handling.
    // For large positive or negative inputs the range reduction  may produce degenerate reduced arguments:
    // - Reduced argument t can fall outside of [-log(2)/2, log(2)/2] range, leading to polynomial approximation p
    //   being negative, and exp(n) * p being either -0.0f (in underflow case) or -inf (in overflow case) instead of
    //   +0.0f and +inf respectively.
    // - Reduced argument n can overflow and become +inf or -inf, and leading to NaN in reduced argument t.
    const __mmask16 vinvof = _mm512_cmp_ps_mask(vx, vinf_cutoff, _CMP_NGT_UQ);
    const __mmask16 vinvuf = _mm512_cmp_ps_mask(vx, vzero_cutoff, _CMP_NLT_UQ);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    // Use masking to explicitly zero the result for large positive inputs, to avoid propagating NaN in reduced
    // argument t into further computations. Zeroing the reduced argument t would instead result in polynomial
    // approximation being 1.0f, which correctly overflows to +inf when scaled by n = +inf.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_maskz_fmadd_ps(vinvof, vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vc0);

    // Reconstruct the final value as f = exp2(n) * p.
    // Use masking to explicitly zero (set to +0.0f) the result for large negative inputs, because for some of these
    // inputs the polynomial approximation p is negative and thus exp2(n) * p == -0.0f.
    const __m512 vf = _mm512_maskz_scalef_ps(vinvuf, vp, vn);
    _mm512_storeu_ps(output, vf);

    input += 16;
    output += 16;
  }
}
