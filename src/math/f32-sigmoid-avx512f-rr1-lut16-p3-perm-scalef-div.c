// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__avx512f_rr1_lut16_p3_perm_scalef_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  // Floating-point mask with only the sign bit set
  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);
  // Large number such that ulp(magic bias) == exp2(-4)
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p19f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p0f);
  // Table of exp2(k / 16) values, k = 0..15
  const __m512 vtable = _mm512_set_ps(
    0x1.EA4AFAp+0f, 0x1.D5818Ep+0f, 0x1.C199BEp+0f, 0x1.AE89FAp+0f,
    0x1.9C4918p+0f, 0x1.8ACE54p+0f, 0x1.7A1148p+0f, 0x1.6A09E6p+0f,
    0x1.5AB07Ep+0f, 0x1.4BFDAEp+0f, 0x1.3DEA64p+0f, 0x1.306FE0p+0f,
    0x1.2387A6p+0f, 0x1.172B84p+0f, 0x1.0B5586p+0f, 0x1.000000p+0f);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation of
  // exp(t) ~ 1 + t * (1 + t * (c2 + t * c3)) on [-log(2)/32, log(2)/32]
  const __m512 vc3 = _mm512_set1_ps(0x1.55559Ap-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.00021Ep-1f);
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

    // Compute reduced argument n := round(z / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits, then
    // subtracing the large number back. The addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|z / log(2)| <= 2**18,
    // i.e. |z| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x outside of
    // [-87.336544, 17.328678] (i.e. z outsize [87.336544, 0]) underflow or saturate sigmoidf(x). We fixup the result
    // for such inputs at the very end of the algorithm.
    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Use the low 4 bits of n (as integer) for table lookup.
    const __m512 vl = _mm512_permutexvar_ps(_mm512_castps_si512(vn), vtable);

    // Subtract the large number back to get the final n := round(z / log(2), 4) as a floating-point number.
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-3 polynomial approximation for exp(t) on [-log(2)/32, log(2)/32].
    //   P(t) = 1 + t * (1 + t * (c2 + t * c3))
    //   p = l * P(t)
    //     = l + l * (t + t * (t * (c2 + t * c3)))
    __m512 vp = _mm512_fmadd_ps(vt, vc3, vc2);
    vp = _mm512_mul_ps(vp, vt);
    vp = _mm512_fmadd_ps(vt, vp, vt);
    vp = _mm512_fmadd_ps(vl, vp, vl);

    // Reconstruct the exp(z) value: e = exp2(floor(n)) * p.
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
