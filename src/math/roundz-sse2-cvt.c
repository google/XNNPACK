// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <emmintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundz__sse2_cvt(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // This magic number with a bit representation 0x80000000 serves two purposes:
  // 1. Extract the sign of a floating-point number.
  // 2. Check if the input to CVTTPS2DQ (_mm_cvttps_epi32) is out-of-range, which results in 0x80000000 output.
  const __m128 vmagic = _mm_set1_ps(-0.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // Extract the sign of the input.
    // We need the sign to preserve negative zero value, which would otherwise get lost in FP->INT->FP conversion.
    const __m128 vsignx = _mm_and_ps(vx, vmagic);
    // Convert floating-point value x to integer, with rounding towards zero.
    // If x is beyond [-2**31, 2**31-1] range or x is NaN, the result is -2**31 (0x80000000).
    const __m128i vintx = _mm_cvttps_epi32(vx);

    // Compute bitmask for out-of-range conversion input.
    // The bitmask is set to all ones when x is out-of-range for CVTTPS2DQ, and also when x == -2**31. The latter case
    // is ok, because this x is already an integer, and can be passed to output as is.
    const __m128 vrndmask = _mm_castsi128_ps(_mm_cmpeq_epi32(vintx, _mm_castps_si128(vmagic)));

    // Convert integer back to floating-point.
    // We binary OR the result with the sign of x to restore the sign of negative zero.
    const __m128 vrndx = _mm_or_ps(_mm_cvtepi32_ps(vintx), vsignx);

    // Combine x rounded via conversion to integer and the initial x value.
    // For -2**31 < x < 2**31, the result is x rounded via conversion to integer.
    // Otherwise (including NaN inputs), the result is x itself.
    const __m128 vy = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vrndx));

    _mm_store_ps(output, vy);
    output += 4;
  }
}
