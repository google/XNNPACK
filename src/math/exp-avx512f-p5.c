// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx512f_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 magic_bias = _mm512_set1_ps(0x1.800000p+23f);
  // The smallest x for which expf(x) is non-zero.
  const __m512 zero_cutoff = _mm512_set1_ps(-0x1.9FE368p+6f);
  // The largest x for which expf(x) is finite.
  const __m512 inf_cutoff = _mm512_set1_ps(0x1.62E42Ep+6f);
  const __m512 log2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 minus_ln2_hi = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 minus_ln2_lo = _mm512_set1_ps(0x1.05C61p-29f);
  const __m512 plus_inf = _mm512_set1_ps(INFINITY);

  const __m512 c1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 c2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 c3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 c4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 c5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);

  const __m512i min_exponent = _mm512_set1_epi32(0xC1000000);
  const __m512i max_exponent = _mm512_set1_epi32(0x3F800000);
  const __m512i default_exponent = max_exponent;

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 x = _mm512_loadu_ps(input);
    __m512 t = _mm512_fmadd_ps(x, log2e, magic_bias);
    __m512i eo = _mm512_slli_epi32(_mm512_castps_si512(t), 23);
    __m512i en = _mm512_max_epi32(eo, min_exponent);
    en = _mm512_min_epi32(en, max_exponent);
    eo = _mm512_sub_epi32(eo, en);
    const __m512 sn = _mm512_castsi512_ps(_mm512_add_epi32(en, default_exponent));
    const __m512 so = _mm512_castsi512_ps(_mm512_add_epi32(eo, default_exponent));
    t = _mm512_sub_ps(t, magic_bias);
    __m512 rx = _mm512_fmadd_ps(t, minus_ln2_hi, x);
    rx = _mm512_fmadd_ps(t, minus_ln2_lo, rx);
    __m512 rf = _mm512_fmadd_ps(c5, rx, c4);
    rf = _mm512_fmadd_ps(rf, rx, c3);
    rf = _mm512_fmadd_ps(rf, rx, c2);
    rf = _mm512_fmadd_ps(rf, rx, c1);
    rx = _mm512_mul_ps(rx, so);
    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    __m512 f = _mm512_maskz_fmadd_ps(_mm512_cmpnlt_ps_mask(x, zero_cutoff), rf, rx, so);
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    f = _mm512_mask_mul_ps(plus_inf, _mm512_cmp_ps_mask(x, inf_cutoff, _CMP_NGT_US), sn, f);
    _mm512_storeu_ps(output, f);

    input += 16;
    output += 16;
  }
}
