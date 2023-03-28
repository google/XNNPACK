// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-avx-expm1minus.c.in
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

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3ps_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(__m256) == 0);

  // Mask for the sign bit.
  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  // The largest z for which tanhf(z) is saturated at -1.0f.
  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.205968p+3f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  // Large number such that ulp(magic bias) == exp2(-4)
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p+19f);
  // Mask for the lowest 3 bits
  const __m128i vindex_mask = _mm_set1_epi32(0x7);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E430p-1f);
  // Coefficients of polynomial approximation
  //   exp(2t) - 1 ~ t * (2 + t * (c2 + t * (c3 + t * c4)))
  // on [-log(2)/32, log(2)/32]
  const __m256 vc4 = _mm256_set1_ps(0x1.5558ECp-1f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555C20p+0f);
  const __m256 vc2 = _mm256_set1_ps(0x1.000000p+1f);
  const __m256 vtwo = _mm256_set1_ps(2.0f);
  const __m256 vminus_one = _mm256_set1_ps(-1.0f);

  for (; n != 0; n -= sizeof(__m256)) {
    const __m256 vx = _mm256_load_ps(input);
    input += 8;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f(x) :=
    //           \ -f(-x) if x >= 0
    //
    // First we compute f(z) := expm1(2z) / (2 + expm1(2z)) where z = -abs(x), then negate the result if x >= 0.
    __m256 vz = _mm256_or_ps(vx, vsign_mask);

    // Inverted mask for the sign of input: 0x00000000 for negative x, 0x80000000 for positive x.
    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);

    // The function saturates at -1 for large negative inputs: tanhf(z) == -1.0f for z <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vz = _mm256_max_ps(vsat_cutoff, vz);

    // Compute reduced argument n := round(z / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**18, i.e. |z| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [-9.010913, 0]) saturate tanhf(x).
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. -9.010913 <= z <= 0. As
    // n has 4 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 3 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fetched value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for -9.010913 <= z <= 0 we have -13 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -13.
    //
    // Shift bits 3:11 into 23:31 (position of floating-point exponent).
    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    const __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 20);
    const __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(vn_hi), 20);

    // Use bits 0:3 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const __m128i vidx_lo = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn)), vindex_mask);
    const __m128i vidx_hi = _mm_and_si128(_mm_castps_si128(vn_hi), vindex_mask);
    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
    #else
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx_lo);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx0]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx4]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi32(vidx_lo, 1);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi32(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx1], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx5], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi32(vidx_lo, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi32(vidx_hi, 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx2], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx6], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi32(vidx_lo, 3);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi32(vidx_hi, 3);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx3], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx7], 3);
    #endif

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ve_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ve_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    // Subtract the large number back to get final n := round(z / log(2), 4) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    const __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2), vz);

    // Compute degree-4 polynomial approximation for exp(2t) - 1 on [-log(2)/32, log(2)/32].
    //   P(t) = t * (2 + t * (c2 + t * (c3 + t * c4)))
    //        = t * p
    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc4, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vtwo);

    // Reconstruct the exp(2z) - 1 value:
    //   exp(2z) - 1 = s * (t * (2 + t * (c2 + t * (c3 + t * c4))) + 1) - 1
    //               = s * t * p + (s - 1)
    //               = (s - 1) + (p * s) * t
    const __m256 vps = _mm256_mul_ps(vp, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_add_ps(_mm256_mul_ps(vt, vps), vsmo);

    // Denominator of the tanh fraction: exp(2z) + 1 = expm1(2z) + 2
    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    // Reconstruct tanh(z) = expm1(2z) / (expm1(2z) + 2)
    __m256 vy = _mm256_div_ps(vemo, vepo);


    // Reconstruct tanh(x):
    //
    //             / tanh(z) if x <= 0
    //   tanh(x) =
    //             \ -tanh(z) if x >= 0
    vy = _mm256_xor_ps(vy, vinvsignx);

    _mm256_store_ps(output, vy);
    output += 8;
  }
}
