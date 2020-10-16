// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__avx512f_rr1_lut32_p2_perm2_scalef_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  // Floating-point mask with only the sign bit set
  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);
  // Large number such that ulp(magic bias) == exp2(-5)
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p18f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p0f);
  // Table of exp2(k / 32) values, k = 0..31
  const __m512 vtable_hi = _mm512_set_ps(
    0x1.F50766p+0f, 0x1.EA4AFAp+0f, 0x1.DFC974p+0f, 0x1.D5818Ep+0f,
    0x1.CB720Ep+0f, 0x1.C199BEp+0f, 0x1.B7F770p+0f, 0x1.AE89FAp+0f,
    0x1.A5503Cp+0f, 0x1.9C4918p+0f, 0x1.93737Cp+0f, 0x1.8ACE54p+0f,
    0x1.82589Ap+0f, 0x1.7A1148p+0f, 0x1.71F75Ep+0f, 0x1.6A09E6p+0f);
  const __m512 vtable_lo = _mm512_set_ps(
    0x1.6247ECp+0f, 0x1.5AB07Ep+0f, 0x1.5342B6p+0f, 0x1.4BFDAEp+0f,
    0x1.44E086p+0f, 0x1.3DEA64p+0f, 0x1.371A74p+0f, 0x1.306FE0p+0f,
    0x1.29E9E0p+0f, 0x1.2387A6p+0f, 0x1.1D4874p+0f, 0x1.172B84p+0f,
    0x1.11301Ep+0f, 0x1.0B5586p+0f, 0x1.059B0Ep+0f, 0x1.000000p+0f);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation of
  // exp(t) ~ 1 + t * (c1 + t * c2) on [-log(2)/64, log(2)/64]
  const __m512 vc2 = _mm512_set1_ps(0x1.000000p-1f);
  const __m512 vc1 = _mm512_set1_ps(0x1.0000F6p-0f);
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

    // Compute reduced argument n := round(z / log(2), 5).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 5 fractional bits, then
    // subtracing the large number back. The addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|z / log(2)| <= 2**17,
    // i.e. |z| <= 0x1.62E43p+16 = 90852.1875), but that is acceptable, because inputs x outside of
    // [-87.336544, 17.328678] (i.e. z outsize [87.336544, 0]) underflow or saturate sigmoidf(x). We fixup the result
    // for such inputs at the very end of the algorithm.
    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Use the low 5 bits of n (as integer) for table lookup.
    const __m512 vl = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn), vtable_hi);

    // Subtract the large number back to get the final n := round(z / log(2), 5) as a floating-point number.
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/64, log(2)/64].
    //   P(t) = 1 + t * (c1 + t * c2)
    //   p = l * P(t)
    //     = l + l * t * (c1 + t * c2)
    __m512 vp = _mm512_fmadd_ps(vt, vc2, vc1);
    vt = _mm512_mul_ps(vt, vl);
    vp = _mm512_fmadd_ps(vt, vp, vl);

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
