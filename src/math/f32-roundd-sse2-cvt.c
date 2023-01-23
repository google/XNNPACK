// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <emmintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundd__sse2_cvt(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // This magic number serves two purposes:
  // 1. Set the bit corresponding to the sign of a floating-point number in a bitmask.
  // 2. Check if the input to CVTTPS2DQ (_mm_cvttps_epi32) is out-of-range, which results in 0x80000000 output.
  const __m128i vmagic = _mm_set1_epi32(0x80000000);
  // Unit constant to decrement results rounded "wrong way" (i.e. up) in the round-towards-zero operation.
  const __m128 vone = _mm_set1_ps(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // Convert floating-point value x to integer, with rounding towards zero.
    // If x is beyond [-2**31, 2**31-1] range or x is NaN, the result is -2**31 (0x80000000).
    const __m128i vintx = _mm_cvttps_epi32(vx);

    // Compute bitmask for the bits we want to copy from the rounded x. Other bits will be copied from x.
    // If x is out-of-range for CVTTPS2DQ, we want all bits from x.
    // If x is in-range for CVTTPS2DQ, we want all but the sign bit from the rounded x and the sign bit from x.
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));

    // Convert integer back to floating-point.
    // We binary OR the result with the sign of x to restore the sign of negative zero.
    const __m128 vprerndx = _mm_cvtepi32_ps(vintx);

    // Combine x rounded via conversion to integer and the initial x value.
    // For -2**31 < x < 2**31, the result is x rounded via conversion to integer.
    // Otherwise (including NaN inputs), the result is x itself.
    const __m128 vrndx = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vprerndx));

    // Adjust x rounded towards zero to get x rounded down.
    // Note: subtraction implicitly converts SNaN inputs to QNaNs.
    const __m128 vy = _mm_sub_ps(vrndx, _mm_and_ps(_mm_cmpgt_ps(vrndx, vx), vone));

    _mm_store_ps(output, vy);
    output += 4;
  }
}
