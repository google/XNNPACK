// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundu__scalar_cvt(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // Threshold of non-integral values in single-precision floating-point representation.
  // All inputs above this threshold (by absolute value) are integer numbers.
  const float vintegral_threshold = 0x1.000000p+23f;
  // Unit constant to increment results rounded "wrong way" (i.e. down) in the round-towards-zero operation.
  const float vone = 1.0f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // Convert floating-point value x to integer, with rounding towards zero, and then back to floating-point.
    // Note: the result is valid only for abs(x) < 2**31, but we further restrict its use to 2**23.
    const float vprerndx = (float) (int32_t) vx;
    // Compute abs(x) to check if the FP->INT->FP conversion result is valid.
    const float vabsx = fabsf(vx);

    // Select between the x rounded via FP->INT->FP conversion and the original x value.
    const float vrndx = XNN_UNPREDICTABLE(vabsx < vintegral_threshold) ? vprerndx : vx;

    // Adjust x rounded towards zero to get x rounded up.
    // Note: addition implicitly converts SNaN inputs to QNaNs.
    const float vprey = XNN_UNPREDICTABLE(vrndx >= vx) ? vrndx : vrndx + vone;
    // Restore the sign of -0.0f lost in the FP->INT->FP conversion and adjustment.
    const float vy = copysignf(vprey, vx);

    *output++ = vy;
  }
}
