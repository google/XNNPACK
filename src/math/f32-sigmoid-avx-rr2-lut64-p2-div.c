// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 64) values decremented (as integer) by (k << 17), k = 0..63
extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_math_f32_sigmoid__avx_rr2_lut64_p2_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  // Floating-point mask with only the sign bit set
  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  // Large number such that ulp(magic bias) == exp2(-6)
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p17f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  // Mask for the lowest 6 bits
  const __m256 vindex_mask = _mm256_castsi256_ps(_mm256_set1_epi32(INT32_C(0x3F)));
  // Last 13 bits are zeroes
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.630000p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(0x1.BD0106p-13f);
  // Coefficient of polynomial approximation of exp(t) ~ 1 + t * (1 + t * c2) on [-log(2)/128, log(2)/128]
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFF0Ap-2f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  // The smallest x for which sigmoidf(x) is normalized.
  // This number is also the smallest x for which expf(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep+6f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[z] := exp(z) / (1 + exp(z)) where z = -abs(x), then replace result with 1 - f[z] if x >= 0.
    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    // Compute reduced argument n := round(z / log(2), 6).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 6 fractional bits, then
    // subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**16, i.e. |z| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs x
    // outside of [-87.336544, 17.328678] (i.e. z outsize [87.336544, 0]) underflow or saturate sigmoidf(x). We fixup
    // the result  for such inputs at the very end of the algorithm.
    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**n for such inputs that sigmoidf(z) is normalized,
    // i.e. -87.33642 <= z <= 0. As n has 6 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s
    // in two steps:
    // 1. Fetch 2**frac(n) from the table using the 6 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -87.33642 <= z <= 0 (inputs for which sigmoidf(z) is normalized) we have
    //    -126 <= int(n) <= 0, and thus the adjusted exponent is not lower than -126.
    //
    // Shift bits 6:14 into 23:31 (position of floating-point exponent).
    __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 17);
    __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 17);

    // Use bits 0:6 of n, as integer, as an index for table lookup of l := 2**frac(n).
    const __m256 vidx = _mm256_and_ps(vn, vindex_mask);
    const __m128i vidx_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx)), 2);
    const __m128i vidx_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx, 1)), 2);
#if XNN_ARCH_X86_64
    const uint64_t vidx_ll = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
    const uint64_t vidx_lh = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
    const uint64_t vidx_hl = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
    const uint64_t vidx_hh = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
    __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_ll)));
    __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_lh)));
    __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hl)));
    __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hh)));
    vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_ll >> 32))), 1);
    vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_lh >> 32))), 1);
    vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hl >> 32))), 1);
    vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hh >> 32))), 1);
#else
    __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_cvtsi128_si32(vidx_lo))));
    __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi32(vidx_lo, 2))));
    __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_cvtsi128_si32(vidx_hi))));
    __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi32(vidx_hi, 2))));
    vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi32(vidx_lo, 1))), 1);
    vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi32(vidx_lo, 3))), 1);
    vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi32(vidx_hi, 1))), 1);
    vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi32(vidx_hi, 3))), 1);
#endif
    const __m128i vl_lo = _mm_unpacklo_epi64(vl_ll, vl_lh);
    const __m128i vl_hi = _mm_unpacklo_epi64(vl_hl, vl_hh);
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ve_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ve_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    // Subtract the large number back to get the final n := round(z / log(2), 6) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/128, log(2)/128].
    //   P(t) = 1 + t * (1 + t * c2) = 1 + (t + t * (t * c2)) = 1 + p
    __m256 vp = _mm256_mul_ps(vt, vc2);
    vp = _mm256_add_ps(vt, _mm256_mul_ps(vp, vt));

    // Reconstruct the exp(z) value:
    //   e = s * (1 + t * (1 + t * c2))
    //     = s * (1 + p)
    //     = s + s * p
    const __m256 ve = _mm256_add_ps(vs, _mm256_mul_ps(vs, vp));

    // Denominator of the sigmoid fraction: 1.0 + exp(z)
    const __m256 vd = _mm256_add_ps(ve, vone);

    // Reconstruct sigmoid(z) = exp(z) / (1.0 + exp(z))
    __m256 vf = _mm256_div_ps(ve, vd);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(z) : 1.0 - sigmoid(z)
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm256_storeu_ps(output, vf);

    input += 8;
    output += 8;
  }
}
