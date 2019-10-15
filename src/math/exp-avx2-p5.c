// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx2_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const __m256 magic_bias = _mm256_set1_ps(0x1.800000p+23f);
  // The smallest x for which expf(x) is non-zero.
  const __m256 zero_cutoff = _mm256_set1_ps(-0x1.9FE368p+6f);
  // The largest x for which expf(x) is finite.
  const __m256 inf_cutoff = _mm256_set1_ps(0x1.62E42Ep+6f);
  const __m256 log2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 minus_ln2_hi = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 minus_ln2_lo = _mm256_set1_ps(0x1.05C61p-29f);
  const __m256 plus_inf = _mm256_set1_ps(INFINITY);

  const __m256 c1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 c2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 c3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 c4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 c5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);

  const __m256i min_exponent = _mm256_set1_epi32(0xC1000000);
  const __m256i max_exponent = _mm256_set1_epi32(0x3F800000);
  const __m256i default_exponent = max_exponent;

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 x = _mm256_loadu_ps(input);
    __m256 t = _mm256_fmadd_ps(x, log2e, magic_bias);
    __m256i eo = _mm256_slli_epi32(_mm256_castps_si256(t), 23);
    __m256i en = _mm256_max_epi32(eo, min_exponent);
    en = _mm256_min_epi32(en, max_exponent);
    eo = _mm256_sub_epi32(eo, en);
    const __m256 sn = _mm256_castsi256_ps(_mm256_add_epi32(en, default_exponent));
    const __m256 so = _mm256_castsi256_ps(_mm256_add_epi32(eo, default_exponent));
    t = _mm256_sub_ps(t, magic_bias);
    __m256 rx = _mm256_fmadd_ps(t, minus_ln2_hi, x);
    rx = _mm256_fmadd_ps(t, minus_ln2_lo, rx);
    // f = so * sn * (1 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))))
    //   = sn * (so + (x * so) * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5))))))
    __m256 rf = _mm256_fmadd_ps(c5, rx, c4);
    rf = _mm256_fmadd_ps(rf, rx, c3);
    rf = _mm256_fmadd_ps(rf, rx, c2);
    rf = _mm256_fmadd_ps(rf, rx, c1);
    rx = _mm256_mul_ps(rx, so);
    __m256 f = _mm256_mul_ps(sn, _mm256_fmadd_ps(rx, rf, so));
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
