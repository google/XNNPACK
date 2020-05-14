// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundd__neon_cvt(
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
  // Unit constant to decrement results rounded "wrong way" (i.e. up) in the round-to-nearest-even operation.
  const uint32x4_t vone = vmovq_n_u32(UINT32_C(0x3F800000));

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

    // Adjust x rounded towards nearest-even to get x rounded down.
    // Note: subtraction implicitly converts SNaN inputs to QNaNs.
    const float32x4_t vy = vsubq_f32(vrndx, vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(vrndx, vx), vone)));

    vst1q_f32(output, vy); output += 4;
  }
}
