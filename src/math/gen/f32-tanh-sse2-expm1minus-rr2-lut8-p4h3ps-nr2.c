// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-sse-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr2(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m128) == 0);

  // Mask for the sign bit.
  const __m128 vsign_mask = _mm_set1_ps(-0.0f);
  // The largest z for which tanhf(z) is saturated at -1.0f.
  const __m128 vsat_cutoff = _mm_set1_ps(-0x1.205968p+3f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  // Large number such that ulp(magic bias) == exp2(-4)
  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p+19f);
  // Mask for the lowest 3 bits
  const __m128i vindex_mask = _mm_set1_epi32(0x7);
  // Last 7 bits are zeroes
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  // Coefficients of polynomial approximation
  //   exp(2t) - 1 ~ t * (2 + t * (c2 + t * (c3 + t * c4)))
  // on [-log(2)/32, log(2)/32]
  const __m128 vc4 = _mm_set1_ps(0x1.5558ECp-1f);
  const __m128 vc3 = _mm_set1_ps(0x1.555C20p+0f);
  const __m128 vc2 = _mm_set1_ps(0x1.000000p+1f);
  const __m128 vminus_two = _mm_set1_ps(-2.0f);
  const __m128 vminus_one = _mm_set1_ps(-1.0f);

  for (; n != 0; n -= sizeof(__m128)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f(x) :=
    //           \ -f(-x) if x >= 0
    //
    // First we compute f(z) := expm1(2z) / (2 + expm1(2z)) where z = -abs(x), then negate the result if x >= 0.
    __m128 vz = _mm_or_ps(vx, vsign_mask);

    // Inverted mask for the sign of input: 0x00000000 for negative x, 0x80000000 for positive x.
    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    // The function saturates at -1 for large negative inputs: tanhf(z) == -1.0f for z <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we compute the saturation mask here, and later use it to replace computed outputs
    // with the saturation value (-1). Note that for NaN inputs the saturation mask is inactive.
    const __m128 vm = _mm_cmple_ps(vz, vsat_cutoff);

    // Compute reduced argument n := round(z / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**18, i.e. |z| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [-9.010913, 0]) saturate tanhf(x).
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. -9.010913 <= z <= 0. As
    // n has 4 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 3 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fetched value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for -9.010913 <= z <= 0 we have -13 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -13.
    //
    // Shift bits 3:11 into 23:31 (position of floating-point exponent).
    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 20);

    // Use bits 0:3 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    #if XNN_ARCH_X86_64
      __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      vidx = _mm_unpackhi_epi64(vidx, vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(vidx);
      const __m128i vl0 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
      const __m128i vl1 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)]);
      const __m128i vl_lo = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl2 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi]);
      const __m128i vl3 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)]);
      const __m128i vl_hi = _mm_unpacklo_epi32(vl2, vl3);
    #else
      const __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx);
      const __m128i vl0 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx0]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx, 2);
      const __m128i vl1 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx1]);
      const __m128i vl_lo = _mm_unpacklo_epi32(vl0, vl1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx, 4);
      const __m128i vl2 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx2]);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx, 6);
      const __m128i vl3 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx3]);
      const __m128i vl_hi = _mm_unpacklo_epi32(vl2, vl3);
    #endif
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));

    // Subtract the large number back to get final n := round(z / log(2), 4) as a floating-point number.
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-4 polynomial approximation for exp(2t) - 1 on [-log(2)/32, log(2)/32].
    //   P(t) = t * (2 + t * (c2 + t * (c3 + t * c4)))
    //        = t * p
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc4, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    // Reconstruct the exp(2z) - 1 value:
    //   exp(2z) - 1 = s * (t * (2 + t * (c2 + t * (c3 + t * c4))) + 1) - 1
    //               = s * t * p + (s - 1)
    //               = (s - 1) + (p * s) * t
    const __m128 vps = _mm_mul_ps(vp, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vt, vps), vsmo);

    // Denominator of the tanh fraction: exp(2z) + 1 = expm1(2z) + 2
    const __m128 vepo = _mm_sub_ps(vminus_two, vemo);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of the denominator.
    // Note: 2 < exp(2z) + 1 <= 3, because z <= 0 and 0 < exp(2z) <= 1.
    // Thus the reciprocal of the denominator never overflows.
    __m128 vrepo = _mm_rcp_ps(vepo);
    vrepo = _mm_mul_ps(vrepo, _mm_add_ps(_mm_mul_ps(vrepo, vepo), vminus_two));
    vrepo = _mm_mul_ps(vrepo, _mm_sub_ps(_mm_mul_ps(vrepo, vepo), vminus_two));

    // Reconstruct tanh(z) := expm1(2z) / (2 + expm1(2z))
    __m128 vy = _mm_mul_ps(vemo, vrepo);

    // Saturate tanh(z) at -1 for large inputs.
    vy = _mm_or_ps(_mm_andnot_ps(vm, vy), _mm_and_ps(vminus_one, vm));

    // Reconstruct tanh(x):
    //
    //             / tanh(z) if x <= 0
    //   tanh(x) =
    //             \ -tanh(z) if x >= 0
    vy = _mm_xor_ps(vy, vinvsignx);

    _mm_store_ps(output, vy);
    output += 4;
  }
}
