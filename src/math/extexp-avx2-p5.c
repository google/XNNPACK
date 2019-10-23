// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_extexp__avx2_p5(
    size_t n,
    const float* input,
    float* output_mantissa,
    float* output_exponent)
{
  assert(n % (8 * sizeof(float)) == 0);

  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(0x1.05C61p-29f);

  const __m256 vc0 = _mm256_set1_ps(1.0f);
  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // Compute reduced argument n := round(x / log(2)).
    const __m256 vn = _mm256_round_ps(_mm256_mul_ps(vx, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);
    vp = _mm256_fmadd_ps(vp, vt, vc0);

    // Skip reconstruction step and separately store "mantissa" and "exponent" of the output.
    _mm256_storeu_ps(output_mantissa, vp);
    _mm256_storeu_ps(output_exponent, vn);

    input += 8;
    output_mantissa += 8;
    output_exponent += 8;
  }
}
