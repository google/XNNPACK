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


void xnn_math_f32_roundz__scalar_cvt(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // Threshold of non-integral values in single-precision floating-point representation.
  // All inputs above this threshold (by absolute value) are integer numbers.
  const float vintegral_threshold = 0x1.000000p+23f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // Convert floating-point value x to integer, with rounding towards zero, and then back to floating-point.
    // Note: the result is valid only for abs(x) < 2**31, but we further restrict its use to 2**23.
    const float vrndx = (float) (int32_t) vx;
    // Compute abs(x) to check if the FP->INT->FP conversion result is valid.
    const float vabsx = fabsf(vx);

    // Select between the x rounded via FP->INT->FP conversion and the original x value.
    const float vprey = XNN_UNPREDICTABLE(vabsx < vintegral_threshold) ? vrndx : vx;
    // Restore the sign of -0.0f lost in the FP->INT->FP conversion.
    const float vy = copysignf(vprey, vx);

    *output++ = vy;
  }
}
