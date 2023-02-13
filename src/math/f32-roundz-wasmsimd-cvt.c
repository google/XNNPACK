// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundz__wasmsimd_cvt(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Threshold of non-integral values in single-precision floating-point representation.
  // All inputs above this threshold (by absolute value) are integer numbers.
  const v128_t vintegral_threshold = wasm_f32x4_const_splat(0x1.000000p+23f);
  // Mask for the sign of a single-precision floating-point number.
  const v128_t vsign_mask = wasm_f32x4_const_splat(-0.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    // Convert floating-point value x to integer, with rounding towards zero, and then back to floating-point.
    // Note: the result is valid only for abs(x) < 2**31, but we further restrict its use to 2**23.
    const v128_t vrndx = wasm_f32x4_convert_i32x4(wasm_i32x4_trunc_sat_f32x4(vx));

    // Compute bitmask for the bits we want to copy from the rounded x. Other bits will be copied from x.
    // If abs(x) is below the integral threshold, use all but the sign bit from the rounded x and the sign bit from x.
    // If x is guaranteed integral or NaN, use all bits from x.
    const v128_t vrndmask = wasm_v128_andnot(wasm_f32x4_lt(wasm_f32x4_abs(vx), vintegral_threshold), vsign_mask);

    // Combine x rounded towardz zero via FP->INT->FP conversion and the input x value.
    // For 0.0 <= x < 2**23, the result is x rounded via FP->INT->FP conversion.
    // For -2**23 < x <= -0.0, the result is abs(x) rounded via FP->INT->FP conversion with the sign of x.
    // For abs(x) >= 2**23 or NaN inputs, the result is x itself.
    const v128_t vy = wasm_v128_bitselect(vrndx, vx, vrndmask);

    wasm_v128_store(output, vy);
    output += 4;
  }
}
