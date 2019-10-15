// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx2_perm_p4(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m256 magic_bias = _mm256_set1_ps(0x1.800000p23f);
  // The smallest x for which expf(x) is non-zero.
  const __m256 zero_cutoff = _mm256_set1_ps(-0x1.9FE368p6f);
  // The largest x for which expf(x) is finite.
  const __m256 inf_cutoff = _mm256_set1_ps(0x1.62E42Ep6f);
  const __m256 log2e_x8  = _mm256_set1_ps(0x1.715476p3f);
  const __m256 minus_ln2_o8_hi = _mm256_set1_ps(-0x1.62E43p-4f);
  const __m256 minus_ln2_o8_lo = _mm256_set1_ps(0x1.05C61p-32f);
  const __m256 plus_inf = _mm256_set1_ps(INFINITY);

  const __m256 c2 = _mm256_set1_ps(0x1.000000p-1f);
  const __m256 c3 = _mm256_set1_ps(0x1.555C82p-3f);
  const __m256 c4 = _mm256_set1_ps(0x1.5558A8p-5f);

  const __m256 table = _mm256_set_ps(
    0x1.D5818Ep+0f, 0x1.AE89FAp+0f, 0x1.8ACE54p+0f, 0x1.6A09E6p+0f,
    0x1.4BFDAEp+0f, 0x1.306FE0p+0f, 0x1.172B84p+0f, 0x1.000000p+0f);

  const __m256i min_exponent = _mm256_set1_epi32(0xC1000000);
  const __m256i max_exponent = _mm256_set1_epi32(0x3F800000);
  const __m256i default_exponent = max_exponent;
  const __m256i mantissa_mask = _mm256_set1_epi32(0x007FFFF8);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 x = _mm256_loadu_ps(input);
    __m256 t = _mm256_fmadd_ps(x, log2e_x8, magic_bias);
    __m256i eo = _mm256_slli_epi32(_mm256_and_si256(_mm256_castps_si256(t), mantissa_mask), 20);
    __m256i en = _mm256_max_epi32(eo, min_exponent);
    en = _mm256_min_epi32(en, max_exponent);
    eo = _mm256_sub_epi32(eo, en);
    const __m256 sn = _mm256_castsi256_ps(_mm256_add_epi32(en, default_exponent));
    const __m256 so = _mm256_castsi256_ps(_mm256_add_epi32(eo, default_exponent));
    __m256 tf = _mm256_permutevar8x32_ps(table, _mm256_castps_si256(t));
    t = _mm256_sub_ps(t, magic_bias);
    __m256 rx = _mm256_fmadd_ps(t, minus_ln2_o8_hi, x);
    rx = _mm256_fmadd_ps(t, minus_ln2_o8_lo, rx);
    // f = so * sn * t * (1 + x * (1 + x * (c2 + x * c3)))
    //   = sn * ((so * t) + (so * t) * x * (1 + x * (c2 + x * c3)))
    //   = sn * ((so * t) + (so * t) * x + (so * t) * x * x * (c2 + x * c3)))
    __m256 rf = _mm256_fmadd_ps(rx, c4, c3);
    rf = _mm256_fmadd_ps(rf, rx, c2);
    tf = _mm256_mul_ps(tf, so);
    rf = _mm256_fmadd_ps(_mm256_mul_ps(rx, rx), rf, rx);
    __m256 f = _mm256_fmadd_ps(tf, rf, tf);
    f = _mm256_mul_ps(f, sn);
    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    f = _mm256_andnot_ps(_mm256_cmp_ps(x, zero_cutoff, _CMP_LT_OS), f);
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    f = _mm256_blendv_ps(f, plus_inf, _mm256_cmp_ps(x, inf_cutoff, _CMP_GT_OS));
    _mm256_storeu_ps(output, f);

    input += 8;
    output += 8;
  }
}
