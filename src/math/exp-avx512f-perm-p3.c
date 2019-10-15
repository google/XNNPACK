// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx512f_perm_p3(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 magic_bias = _mm512_set1_ps(0x1.800000p23f);
  // The smallest x for which expf(x) is non-zero.
  const __m512 zero_cutoff = _mm512_set1_ps(-0x1.9FE368p6f);
  // The largest x for which expf(x) is finite.
  const __m512 inf_cutoff = _mm512_set1_ps(0x1.62E42Ep6f);
  const __m512 log2e_x16  = _mm512_set1_ps(0x1.715476p4f);
  const __m512 minus_ln2_o16_hi = _mm512_set1_ps(-0x1.62e43p-5f);
  const __m512 minus_ln2_o16_lo = _mm512_set1_ps(0x1.05c61p-33f);
  const __m512 plus_inf = _mm512_set1_ps(INFINITY);

  const __m512 c2 = _mm512_set1_ps(0x1.00021Ep-1f);
  const __m512 c3 = _mm512_set1_ps(0x1.55559Ap-3f);
  const __m512 table = _mm512_set_ps(
    0x1.EA4AFAp+0f, 0x1.D5818Ep+0f, 0x1.C199BEp+0f, 0x1.AE89FAp+0f,
    0x1.9C4918p+0f, 0x1.8ACE54p+0f, 0x1.7A1148p+0f, 0x1.6A09E6p+0f,
    0x1.5AB07Ep+0f, 0x1.4BFDAEp+0f, 0x1.3DEA64p+0f, 0x1.306FE0p+0f,
    0x1.2387A6p+0f, 0x1.172B84p+0f, 0x1.0B5586p+0f, 0x1.000000p+0f);

  const __m512i min_exponent = _mm512_set1_epi32(0xC1000000);
  const __m512i max_exponent = _mm512_set1_epi32(0x3F800000);
  const __m512i default_exponent = max_exponent;
  const __m512i mantissa_mask = _mm512_set1_epi32(0x007FFFF0);

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 x = _mm512_loadu_ps(input);
    __m512 t = _mm512_fmadd_ps(x, log2e_x16, magic_bias);
    __m512i eo = _mm512_slli_epi32(_mm512_and_si512(_mm512_castps_si512(t), mantissa_mask), 19);
    __m512i en = _mm512_max_epi32(eo, min_exponent);
    en = _mm512_min_epi32(en, max_exponent);
    eo = _mm512_sub_epi32(eo, en);
    const __m512 sn = _mm512_castsi512_ps(_mm512_add_epi32(en, default_exponent));
    const __m512 so = _mm512_castsi512_ps(_mm512_add_epi32(eo, default_exponent));
    const __m512 tf = _mm512_permutexvar_epi32(_mm512_castps_si512(t), table);
    t = _mm512_sub_ps(t, magic_bias);
    __m512 rx = _mm512_fmadd_ps(t, minus_ln2_o16_hi, x);
    rx = _mm512_fmadd_ps(t, minus_ln2_o16_lo, rx);
    __m512 rf = _mm512_fmadd_ps(rx, c3, c2);
    rf = _mm512_mul_ps(rf, rx);
    rf = _mm512_fmadd_ps(rx, rf, rx);
    __m512 f = _mm512_fmadd_ps(tf, rf, tf);
    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    f = _mm512_maskz_mul_ps(_mm512_cmpnlt_ps_mask(x, zero_cutoff), f, sn);
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    f = _mm512_mask_mul_ps(plus_inf, _mm512_cmp_ps_mask(x, inf_cutoff, _CMP_NGT_US), so, f);
    _mm512_storeu_ps(output, f);

    input += 16;
    output += 16;
  }
}
