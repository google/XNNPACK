// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2_scalef(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p18f);
  const __m512 vlog2e  = _mm512_set1_ps(0x1.715476p0f);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(-0x1.62e43p-1f);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(0x1.05c61p-29f);

  const __m512 vc1 = _mm512_set1_ps(0x1.0000F6p-0f);
  const __m512 vc2 = _mm512_set1_ps(0x1.000000p-1f);
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

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);

    // Compute reduced argument n := round(x / log(2), 5).
    // We do it by adding a large number (magic bias), which cause rounding of result to 5 fractional bits, then
    // subtracing the large number back. The first addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|x| <= 2**17), but that's
    // ok, because inputs outside of [-103.97207, 88.72283] underflow or overflow expf(x) anyway. We fixup the result
    // for such inputs at the very end of the algorithm.
    __m512 vn = _mm512_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Use the low 5 bits of n (as integer) for table lookup.
    const __m512 vl = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn), vtable_hi);

    // Subtract the large number back to get final n := round(x / log(2), 5).
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/64, log(2)/64].
    //   p = l * (1 + t * (c1 + t * c2))
    //     = l + l * t * (c1 + t * c2)
    __m512 vp = _mm512_fmadd_ps(vt, vc2, vc1);
    vt = _mm512_mul_ps(vt, vl);
    vp = _mm512_fmadd_ps(vt, vp, vl);

    // Reconstruct the final value as f = exp2(floor(n)) * p.
    const __m512 vf = _mm512_scalef_ps(vp, vn);
    _mm512_storeu_ps(output, vf);

    input += 16;
    output += 16;
  }
}
