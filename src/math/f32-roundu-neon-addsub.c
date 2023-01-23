// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundu__neon_addsub(
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
  // Unit constant to increment results rounded "wrong way" (i.e. down) in the round-to-nearest-even operation.
  const float32x4_t vone = vmovq_n_f32(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-to-nearest-even because it is an odd function.
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

    // Combine abs(x) rounded via addition-subtraction trick and the input x value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick with the sign of x.
    // For NaN inputs, the result is x converted to QNaN as a side-effect of addition-subtraction.
    // For abs(x) >= 2**23, the result is x itself.
    const float32x4_t vrndx = vbslq_f32(vrndmask, vx, vrndabsx);

    // Compute bitmask for the bits to copy from the adjusted rounded x. Other bits will be copied from rounded x.
    // If rounded x < x, we want all but the sign bit from the adjusted rounded x and the sign bit from rounded x (same
    // as the sign bit of x).
    // If rounded x >= x or rounded x is NaN (implies x is NaN), we want all bits from rounded x.
    const uint32x4_t vadjmask = vbicq_u32(vcltq_f32(vrndx, vx), vsign_mask);
    // Adjust the rounded x value.
    // The adjusted value is a unit above the rounded-to-nearest-even x value, but is used only if the rounded value is
    // below x. In these cases, the adjusted value is x rounded up.
    const float32x4_t vadjrndx = vaddq_f32(vrndx, vone);

    // Combine the adjusted rounded x and the original rounded to nearest-even x.
    // For rounded x < x, the result is the absolute value of adjusted rounded-to-nearest-even x with the sign of
    // rounded-to-nearest-even x (same as sign of x). Propagating the sign of x is important to produce negative zero
    // for -1.0 < x < -0.5 inputs, where otherwise we would get -1.0 (rounded x) + 1.0 (adjustment) = +0.0.
    // For rounded x >= x, the result is the rounded-to-nearest-even x.
    // For NaN inputs, the result is rounded x (same as x converted to QNaN as a side-effect of addition-subtraction).
    const float32x4_t vy = vbslq_f32(vadjmask, vadjrndx, vrndx);

    vst1q_f32(output, vy); output += 4;
  }
}
