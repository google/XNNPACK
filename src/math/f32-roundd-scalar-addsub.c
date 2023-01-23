// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundd__scalar_addsub(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // Addition of this number to a floating-point number x cause rounding of the result to an integer. Then this magic
  // number is subtracted back from the result to get original x rounded to integer. This trick works only for
  // 0 <= x < 2**24, but all numbers in 2**23 <= x < 2**24 range are integers, so we can further restrict it to
  // 0 <= x < 2**23. Then the upper bound of the validity interval is conveniently the same as the magic number.
  const float vmagic_number = 0x1.000000p+23f;
  // Unit constant to decrement results rounded "wrong way" (i.e. up) in the round-to-nearest-even operation.
  const float vone = 1.0f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-to-nearest-even because it is an odd function.
    const float vabsx = fabsf(vx);
    // Addition-subtraction trick with the magic number to cause rounding to integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    // Note: addition-subtraction implicitly converts SNaN inputs to QNaNs.
    const float vprerndabsx = (vabsx + vmagic_number) - vmagic_number;

    // Select between the abs(x) rounded using addition-subtraction trick and the abs(x) value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick.
    // For abs(x) >= 2**23, the result is abs(x) itself (already an integer).
    // For NaN inputs, the result is abs(x) converted to QNaN as a side-effect of addition-subtraction.
    const float vrndabsx = XNN_UNPREDICTABLE(vabsx >= vmagic_number) ? vabsx : vprerndabsx;
    // Restore the sign of the rounded value.
    const float vrndx = copysignf(vrndabsx, vx);

    // Adjust x rounded to nearest-even to get x rounded down.
    const float vy = XNN_UNPREDICTABLE(vrndx > vx) ? vrndx - vone : vrndx;

    *output++ = vy;
  }
}
