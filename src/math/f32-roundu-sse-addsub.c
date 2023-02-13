// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xmmintrin.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundu__sse_addsub(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Mask for all bits of a floating-point number except the sign bit.
  const __m128 vnonsign_mask = _mm_set1_ps(math_nonsign_mask_f32());
  // Addition of this number to a floating-point number x cause rounding of the result to an integer. Then this magic
  // number is subtracted back from the result to get original x rounded to integer. This trick works only for
  // 0 <= x < 2**24, but all numbers in 2**23 <= x < 2**24 range are integers, so we can further restrict it to
  // 0 <= x < 2**23. Then the upper bound of the validity interval is conveniently the same as the magic number.
  const __m128 vmagic_number = _mm_set1_ps(0x1.000000p+23f);
  // Unit constant to increment results rounded "wrong way" (i.e. down) in the round-to-nearest-even operation.
  const __m128 vone = _mm_set1_ps(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-to-nearest-even because it is an odd function.
    const __m128 vabsx = _mm_and_ps(vx, vnonsign_mask);

    // Compute bitmask for the bits we want to copy from the rounded abs(x). Other bits will be copied from x.
    // If abs(x) >= 2**23, we want all bits from x.
    // If abs(x) < 2**23 or x is NaN, we want all but the sign bit from the rounded abs(x) and the sign bit from x.
    const __m128 vrndmask = _mm_andnot_ps(_mm_cmpge_ps(vabsx, vmagic_number), vnonsign_mask);
    // Addition-subtraction trick with the magic number to cause rounding to integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    // Note: addition-subtraction implicitly converts SNaN inputs to QNaNs.
    const __m128 vrndabsx = _mm_sub_ps(_mm_add_ps(vabsx, vmagic_number), vmagic_number);

    // Combine abs(x) rounded via addition-subtraction trick and the input x value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick with the sign of x.
    // For NaN inputs, the result is x converted to QNaN as a side-effect of addition-subtraction.
    // For abs(x) >= 2**23, the result is x itself.
    const __m128 vrndx = _mm_or_ps(_mm_and_ps(vrndabsx, vrndmask), _mm_andnot_ps(vrndmask, vx));

    // Compute bitmask for the bits to copy from the adjusted rounded x. Other bits will be copied from rounded x.
    // If rounded x < x, we want all but the sign bit from the adjusted rounded x and the sign bit from rounded x (same
    // as the sign bit of x).
    // If rounded x >= x or rounded x is NaN (implies x is NaN), we want all bits from rounded x.
    const __m128 vadjmask = _mm_and_ps(_mm_cmplt_ps(vrndx, vx), vnonsign_mask);
    // Compute adjusted rounded x value.
    // The adjusted value is a unit above the rounded-to-nearest-even x value, but is used only if the rounded value is
    // below x. In this cases, the adjusted value is x rounded up.
    const __m128 vadjrndx = _mm_add_ps(vrndx, vone);

    // Combine the adjusted rounded x and the original rounded to nearest-even x.
    // For rounded x < x, the result is the absolute value of adjusted rounded-to-nearest-even x with the sign of
    // rounded-to-nearest-even x (same as sign of x). Propagating the sign of x is important to produce negative zero
    // for -1.0 < x < -0.5 inputs, where otherwise we would get -1.0 (rounded x) + 1.0 (adjustment) = +0.0.
    // For rounded x >= x, the result is the rounded-to-nearest-even x.
    // For NaN inputs, the result is rounded x (same as x converted to QNaN as a side-effect of addition-subtraction).
    const __m128 vy = _mm_or_ps(_mm_and_ps(vadjrndx, vadjmask), _mm_andnot_ps(vadjmask, vrndx));

    _mm_store_ps(output, vy);
    output += 4;
  }
}
