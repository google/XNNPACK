// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundz__neon_cvt(
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

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // Convert floating-point value x to integer, with rounding towards zero, and then back to floating-point.
    // Note: the result is valid only for abs(x) < 2**31, but we further restrict its use to 2**23.
    const float32x4_t vrndx = vcvtq_f32_s32(vcvtq_s32_f32(vx));
    // Extract the sign of the input.
    // We need the sign to preserve negative zero value, which would otherwise get lost in FP->INT->FP conversion.
    const uint32x4_t vsignx = vandq_u32(vreinterpretq_u32_f32(vrndx), vsign_mask);

    // Compute bitmask for non-integral input.
    // The bitmask is set to all ones when x is potentially non-integral, and we round it using FP->INT->FP conversion.
    const uint32x4_t vrndmask = vcaltq_f32(vx, vintegral_threshold);

    // Combine x rounded towardz zero via FP->INT->FP conversion and the input x value.
    // For 0.0 <= x < 2**23, the result is x rounded via FP->INT->FP conversion.
    // For -2**23 < x <= -0.0, the result is abs(x) rounded via FP->INT->FP conversion with the sign of x.
    // For abs(x) >= 2**23 or NaN inputs, the result is x itself.
    const float32x4_t vy = vbslq_f32(vbicq_u32(vrndmask, vsignx), vrndx, vx);

    vst1q_f32(output, vy); output += 4;
  }
}
