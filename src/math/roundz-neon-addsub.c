// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundz__neon_addsub(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Addition of this number to a floating-point number x cause rounding of the result to an integer. Then this magic
  // number is subtracted back from the result to get original x rounded to integer. This trick works only for
  // 0 <= x < 2**24, but all numbers in 2**23 <= x < 2**24 range are integers, so we can further restrict it to
  // 0 <= x < 2**23. Then the upper bound of the validity interval is conveniently the same as the magic number.
  const float32x4_t vmagic_number = vmovq_n_f32(0x1.000000p+23f);
  // Mask for the sign bit of a floating-point number.
  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));
  // Unit constant to decrement absolute values rounded "wrong way" (i.e. away from zero) in the round-to-nearest-even
  // operation.
  const uint32x4_t vone = vmovq_n_u32(UINT32_C(0x3F800000));

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-towards-zero because it is an odd function.
    const float32x4_t vabsx = vabsq_f32(vx);
    // Compute bitmask for the bits we want to copy from the rounded abs(x). Other bits will be copied from x.
    // If abs(x) >= 2**23, we want all bits from x.
    // If abs(x) < 2**23 or x is NaN, we want all but the sign bit from the rounded abs(x) and the sign bit from x.
    // Note: we do vcaltq_f32(vmagic_number, vx) instead of vcltq_f32(vmagic_number, vabsx) to reduce dependency chain.
    const uint32x4_t vrndmask = vorrq_u32(vcaltq_f32(vmagic_number, vx), vsign_mask);

    // Addition-subtraction trick with the magic number to cause rounding to the nearest-even integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    // Note: addition-subtraction implicitly converts SNaN inputs to QNaNs.
    const float32x4_t vrndabsx = vsubq_f32(vaddq_f32(vabsx, vmagic_number), vmagic_number);

    // Compute adjustment to be subtracted from the rounded-to-nearest-even abs(x) value.
    // Adjustment is one if the rounded value is greater than the abs(x) value and zero otherwise (including NaN input).
    const float32x4_t vadjustment = vreinterpretq_f32_u32(vandq_u32(vone, vcgtq_f32(vrndabsx, vabsx)));
    // Adjust abs(x) rounded to nearest-even via the addition-subtraction trick to get abs(x) rounded down.
    // Note: subtraction implicitly converts SNaN inputs to QNaNs.
    const float32x4_t vflrabsx = vsubq_f32(vrndabsx, vadjustment);

    // Combine abs(x) rounded via addition-subtraction trick with adjustment and the input x value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick with the sign of x.
    // For NaN inputs, the result is x converted to QNaN as a side-effect of addition-subtraction and adjustment.
    // For abs(x) >= 2**23, the result is x itself.
    const float32x4_t vy = vbslq_f32(vrndmask, vx, vflrabsx);

    vst1q_f32(output, vy); output += 4;
  }
}
