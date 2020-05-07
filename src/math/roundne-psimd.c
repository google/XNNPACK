// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <psimd.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundne__psimd(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Mask for all bits of a floating-point number except the sign bit.
  const psimd_s32 vnonsign_mask = psimd_splat_s32(INT32_C(0x7FFFFFFF));
  // Addition of this number to a floating-point number x cause rounding of the result to an integer. Then this magic
  // number is subtracted back from the result to get original x rounded to integer. This trick works only for
  // 0 <= x < 2**24, but all numbers in 2**23 <= x < 2**24 range are integers, so we can further restrict it to
  // 0 <= x < 2**23. Then the upper bound of the validity interval is conveniently the same as the magic number.
  const psimd_f32 vmagic_number = psimd_splat_f32(0x1.000000p+23f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const psimd_f32 vx = psimd_load_f32(input);
    input += 4;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-to-nearest-even because it is an odd function.
    const psimd_f32 vabsx = psimd_andmask_f32(vnonsign_mask, vx);

    // Compute bitmask for the bits we want to copy from the rounded abs(x). Other bits will be copied from x.
    // If abs(x) < 2**23, we want all but the sign bit from the rounded abs(x) and the sign bit from x.
    // Otherwise (abs(x) >= 2**23 or x is NaN), we want all bits from x.
    const psimd_s32 vrndmask = vnonsign_mask & (vabsx < vmagic_number);
    // Addition-subtraction trick with the magic number to cause rounding to integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    const psimd_f32 vrndabsx = psimd_sub_f32(psimd_add_f32(vabsx, vmagic_number), vmagic_number);

    // Combine abs(x) rounded via addition-subtraction trick and the input x value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick with the sign of x.
    // For abs(x) >= 2**23 and NaN inputs, the result is x itself.
    const psimd_f32 vy = psimd_blend_f32(vrndmask, vrndabsx, vx);

    psimd_store_f32(output, vy);
    output += 4;
  }
}
