// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundz__scalar_addsub(
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
  // Unit constant to decrement absolute values rounded "wrong way" (i.e. away from zero) in the round-to-nearest-even
  // operation.
  const float vone = 1.0f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-towards-zero because it is an odd function.
    const float vabsx = fabsf(vx);
    // Addition-subtraction trick with the magic number to cause rounding to the nearest-even integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    // Note: addition-subtraction implicitly converts SNaN inputs to QNaNs.
    const float vrndabsx = (vabsx + vmagic_number) - vmagic_number;

    // Adjust abs(x) rounded to nearest-even via the addition-subtraction trick to get abs(x) rounded down.
    // Note: subtraction implicitly converts SNaN inputs to QNaNs.
    const float vflrabsx = XNN_UNPREDICTABLE(vrndabsx <= vabsx) ? vrndabsx : vrndabsx - vone;

    // Select between the abs(x) rounded down using addition-subtraction trick with adjustment and the abs(x) value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick.
    // For abs(x) >= 2**23, the result is abs(x) itself (already an integer).
    // For NaN inputs, the result is abs(x) converted to QNaN as a side-effect of addition-subtraction.
    const float vabsy = XNN_UNPREDICTABLE(vabsx >= vmagic_number) ? vabsx : vflrabsx;
    // Restore the sign of the rounded value.
    const float vy = copysignf(vabsy, vx);

    *output++ = vy;
  }
}
