// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__avx512f_p5_scalef(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 log2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 minus_ln2_hi = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 minus_ln2_lo = _mm512_set1_ps(0x1.05C61p-29f);

  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 c2 = _mm512_set1_ps(0x1.FFFDFCp-2f);
  const __m512 c3 = _mm512_set1_ps(0x1.5557ACp-3f);
  const __m512 c4 = _mm512_set1_ps(0x1.572A12p-5f);
  const __m512 c5 = _mm512_set1_ps(0x1.1063E2p-7f);

  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 x = _mm512_loadu_ps(input);
    const __m512 s = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), 0);
    __m512 rx = _mm512_fmadd_ps(s, minus_ln2_hi, x);
    rx = _mm512_fmadd_ps(s, minus_ln2_lo, rx);
    __m512 rf = _mm512_fmadd_ps(c5, rx, c4);
    rf = _mm512_fmadd_ps(rf, rx, c3);
    rf = _mm512_fmadd_ps(rf, rx, c2);
    rf = _mm512_fmadd_ps(rf, rx, one);
    rf = _mm512_fmadd_ps(rf, rx, one);
    __m512 f = _mm512_scalef_ps(rf, s);
    _mm512_storeu_ps(output, f);

    input += 16;
    output += 16;
  }
}
