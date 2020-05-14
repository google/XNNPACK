// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundu__neon_cvt(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Threshold of non-integral values in single-precision floating-point representation.
  // All inputs above this threshold (by absolute value) are integer numbers.
  const float32x4_t vintegral_threshold = vmovq_n_f32(0x1.000000p+23f);
  // Mask for the sign of a single-precision floating-point number.
  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));
  // Unit constant to increment results rounded "wrong way" (i.e. down) in the round-towards-zero operation.
  const float32x4_t vone = vmovq_n_f32(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // Convert floating-point value x to integer, with rounding towards zero, and then back to floating-point.
    // Note: the result is valid only for abs(x) < 2**31, but we further restrict its use to 2**23.
    const float32x4_t vprerndx = vcvtq_f32_s32(vcvtq_s32_f32(vx));

    // Compute bitmask for the bits we want to copy from the rounded x. Other bits will be copied from x.
    // If abs(x) is below the integral threshold, use all but the sign bit from the rounded x and the sign bit from x.
    // If x is guaranteed integral or NaN, use all bits from x.
    const uint32x4_t vrndmask = vbicq_u32(vcaltq_f32(vx, vintegral_threshold), vsign_mask);

    // Combine x rounded towardz zero via FP->INT->FP conversion and the input x value.
    // For 0.0 <= x < 2**23, the result is x rounded via FP->INT->FP conversion.
    // For -2**23 < x <= -0.0, the result is abs(x) rounded via FP->INT->FP conversion with the sign of x.
    // For abs(x) >= 2**23 or NaN inputs, the result is x itself.
    const float32x4_t vrndx = vbslq_f32(vrndmask, vprerndx, vx);

    // Compute bitmask for the bits to copy from the rounded x. Other bits will be copied from the adjusted rounded x.
    // If rounded x >= x, we want all bits from rounded x.
    // If rounded x < x or rounded x is NaN (implies x is NaN), we want all but the sign bit from the adjusted rounded
    // x and the sign bit from rounded x (same as the sign bit of x).
    const uint32x4_t vadjmask = vorrq_u32(vcgeq_f32(vrndx, vx), vsign_mask);
    // Adjust the rounded x value.
    // The adjusted value is a unit above the rounded-towards-zero x value, but is used only if the rounded value is
    // below x. In these cases, the adjusted value is x rounded up.
    // Note: addition implicitly converts SNaN inputs to QNaNs.
    const float32x4_t vadjrndx = vaddq_f32(vrndx, vone);

    // Combine the adjusted rounded x and the original rounded towards zero x.
    // For rounded x < x, the result is the absolute value of adjusted rounded-towards-zero x with the sign of
    // rounded-towards x (same as sign of x). Propagating the sign of x is important to produce negative zero
    // for -1.0 < x < -0.5 inputs, where otherwise we would get -1.0 (rounded x) + 1.0 (adjustment) = +0.0.
    // For rounded x >= x, the result is the rounded-towards-zero x.
    // For NaN inputs, the result is rounded x (same as x converted to QNaN as a side-effect of adjustment).
    const float32x4_t vy = vbslq_f32(vadjmask, vrndx, vadjrndx);

    vst1q_f32(output, vy); output += 4;
  }
}
