// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__avx512f_rr2_p5_scalef_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  // Floating-point mask with only the sign bit set
  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p0f);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(0x1.05C61p-29f);
  // Coefficient of polynomial approximation of
  // exp(t) ~ 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) on [-log(2)/2, log(2)/2]
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vone = _mm512_set1_ps(1.0f);

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[z] := exp(z) / (1 + exp(z)) where z = -abs(x), then replace result with 1 - f[z] if x >= 0.
    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    // Compute reduced argument n := round(z / log(2)).
    const __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vz, vlog2e), 0);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vz);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    //   P(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) = p
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vone);

    // Reconstruct the exp(z) value: e = exp2(n) * p.
    const __m512 ve = _mm512_scalef_ps(vp, vn);

    // Denominator of the sigmoid fraction: 1.0 + exp(z)
    const __m512 vd = _mm512_add_ps(ve, vone);

    // Reconstruct sigmoid(z) = exp(z) / (1.0 + exp(z))
    __m512 vf = _mm512_div_ps(ve, vd);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(z) : 1.0 - sigmoid(z)
    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_storeu_ps(output, vf);

    input += 16;
    output += 16;
  }
}
