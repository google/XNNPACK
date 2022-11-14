// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 16) values decremented (as integer) by (k << 19), k = 0..15
extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Large number such that ulp(magic bias) == exp2(-4)
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.800000p19f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  // Mask for the lowest 4 bits
  const v128_t vindex_mask = wasm_i32x4_const_splat(0xF);
  // The largest x for which expm1f(x) is saturated at -1.0f.
  const v128_t vsat_cutoff = wasm_f32x4_const_splat(-0x1.154246p+4f);
  // Last 9 bits are zeroes
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E400p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(-0x1.7F7D1Cp-20f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * c3))
  // on [-log(2)/32, log(2)/32]
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.55561Cp-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.0001ECp-1f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(input);

    // Compute reduced argument n := round(x / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits, then
    // subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**18, i.e. |x| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x are
    // restricted to [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**n for valid inputs, i.e. -17.328680 <= x <= 0.0. As n
    // has 4 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s in two steps:
    // 1. Fetch 2**frac(n) from the table using the 4 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -17.328680 <= x <= 0.0 we have -25 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -25.
    //
    // Shift bits 4:12 into 23:31 (position of floating-point exponent).
    const v128_t ven = wasm_i32x4_shl(vn, 19);

    // Use bits 0:4 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const uint32_t vidx0 = wasm_u32x4_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0));
    const uint32_t vidx1 = wasm_u32x4_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1), vl, 1);
    const uint32_t vidx2 = wasm_u32x4_extract_lane(vidx, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2), vl, 2);
    const uint32_t vidx3 = wasm_u32x4_extract_lane(vidx, 3);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3), vl, 3);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    v128_t vs = wasm_i32x4_add(vl, ven);

    // Subtract the large number back to get final n := round(x / log(2), 4).
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vx);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we zero out s (scale) and t (reduced argument) for x <= sat_cutoff.
    const v128_t vm = wasm_f32x4_le(vx, vsat_cutoff);
    vs = wasm_v128_andnot(vs, vm);
    vt = wasm_v128_andnot(vt, vm);

    // Compute degree-3 polynomial approximation for exp(t) - 1 on [-log(2)/32, log(2)/32].
    //   P(t) = t * (1 + t * (c2 + t * c3)) = t + t * (t * (c2 + t * c3)) = t + t * p
    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt), vc2);
    vp = wasm_f32x4_mul(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * c3))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt = wasm_f32x4_mul(vt, vs);
    const v128_t vsm1 = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vt);
    const v128_t vf = wasm_f32x4_add(vp, vsm1);

    wasm_v128_store(output, vf);

    input += 4;
    output += 4;
  }
}
