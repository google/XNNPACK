// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundz__wasmsimd_addsub(
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
  // Unit constant to decrement absolute values rounded "wrong way" (i.e. away from zero) in the round-to-nearest-even
  // operation.
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-toward-zero because it is an odd function.
    const v128_t vabsx = wasm_v128_andnot(vx, vsign_mask);

    // Compute bitmask for the bits we want to copy from x. Other bits will be copied from the rounded abs(x).
    // If abs(x) < 2**23 or x is NaN, we want the sign bit from x and the rest from the rounded abs(x).
    // Otherwise (abs(x) >= 2**23), we want all bits from x.
    const v128_t vrndmask = wasm_v128_or(vsign_mask, wasm_f32x4_ge(vabsx, vmagic_number));
    // Addition-subtraction trick with the magic number to cause rounding to integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    // Note: addition-subtraction implicitly converts SNaN inputs to QNaNs.
    const v128_t vrndabsx = wasm_f32x4_sub(wasm_f32x4_add(vabsx, vmagic_number), vmagic_number);

    // Compute adjustment to be subtracted from the rounded-to-nearest-even abs(x) value.
    // Adjustment is one if the rounded value is greater than the abs(x) value and zero otherwise (including NaN input).
    const v128_t vadjustment = wasm_v128_and(wasm_f32x4_gt(vrndabsx, vabsx), vone);
    // Adjust abs(x) rounded to nearest-even via the addition-subtraction trick to get abs(x) rounded down.
    // Note: subtraction implicitly converts SNaN inputs to QNaNs.
    const v128_t vflrabsx = wasm_f32x4_sub(vrndabsx, vadjustment);

    // Combine abs(x) rounded down via addition-subtraction trick with adjustment and the input x value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick with the sign of x.
    // For NaN inputs, the result is x converted to QNaN as a side-effect of addition-subtraction and adjustment.
    // For abs(x) >= 2**23, the result is x itself.
    const v128_t vy = wasm_v128_bitselect(vx, vflrabsx, vrndmask);

    wasm_v128_store(output, vy);
    output += 4;
  }
}
