// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundu__wasmsimd_addsub(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Mask for the sign bit of a floating-point number.
  const v128_t vsign_mask = wasm_i32x4_const_splat(INT32_C(0x80000000));
  // Addition of this number to a floating-point number x cause rounding of the result to an integer. Then this magic
  // number is subtracted back from the result to get original x rounded to integer. This trick works only for
  // 0 <= x < 2**24, but all numbers in 2**23 <= x < 2**24 range are integers, so we can further restrict it to
  // 0 <= x < 2**23. Then the upper bound of the validity interval is conveniently the same as the magic number.
  const v128_t vmagic_number = wasm_f32x4_const_splat(0x1.000000p+23f);
  // Unit constant to increment results rounded "wrong way" (i.e. down) in the round-to-nearest-even operation.
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-to-nearest-even because it is an odd function.
    const v128_t vabsx = wasm_v128_andnot(vx, vsign_mask);

    // Compute bitmask for the bits we want to copy from x. Other bits will be copied from the rounded abs(x).
    // If abs(x) < 2**23 or x is NaN, we want the sign bit from x and the rest from the rounded abs(x).
    // Otherwise (abs(x) >= 2**23), we want all bits from x.
    const v128_t vrndmask = wasm_v128_or(vsign_mask, wasm_f32x4_ge(vabsx, vmagic_number));
    // Addition-subtraction trick with the magic number to cause rounding to integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    // Note: addition-subtraction implicitly converts SNaN inputs to QNaNs.
    const v128_t vrndabsx = wasm_f32x4_sub(wasm_f32x4_add(vabsx, vmagic_number), vmagic_number);

    // Combine abs(x) rounded via addition-subtraction trick and the input x value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick with the sign of x.
    // For NaN inputs, the result is x converted to QNaN as a side-effect of addition-subtraction.
    // For abs(x) >= 2**23, the result is x itself.
    const v128_t vrndx = wasm_v128_bitselect(vx, vrndabsx, vrndmask);

    // Compute bitmask for the bits to copy from the rounded x. Other bits will be copied from the adjusted rounded x.
    // If rounded x >= x, we want all bits from rounded x.
    // If rounded x < x or rounded x is NaN (implies x is NaN), we want all but the sign bit from the adjusted rounded
    // x and the sign bit from rounded x (same as the sign bit of x).
    const v128_t vadjmask = wasm_v128_or(wasm_f32x4_ge(vrndx, vx), vsign_mask);
    // Adjust the rounded x value.
    // The adjusted value is a unit above the rounded-to-nearest-even x value, but is used only if the rounded value is
    // below x. In these cases, the adjusted value is x rounded up.
    // Note: addition implicitly converts SNaN inputs to QNaNs.
    const v128_t vadjrndx = wasm_f32x4_add(vrndx, vone);

    // Combine the adjusted rounded x and the original rounded towards zero x.
    // For rounded x < x, the result is the absolute value of adjusted rounded-towards-zero x with the sign of
    // rounded-towards x (same as sign of x). Propagating the sign of x is important to produce negative zero
    // for -1.0 < x < -0.5 inputs, where otherwise we would get -1.0 (rounded x) + 1.0 (adjustment) = +0.0.
    // For rounded x >= x, the result is the rounded-towards-zero x.
    // For NaN inputs, the result is rounded x (same as x converted to QNaN as a side-effect of adjustment).
    const v128_t vy = wasm_v128_bitselect(vrndx, vadjrndx, vadjmask);

    wasm_v128_store(output, vy);
    output += 4;
  }
}
