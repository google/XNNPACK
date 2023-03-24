// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_pmax(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(v128_t) == 0);

  // Mask for the sign bit.
  const v128_t vsign_mask = wasm_f32x4_const_splat(-0.0f);
  // The largest z for which tanhf(z) is saturated at -1.0f.
  const v128_t vsat_cutoff = wasm_f32x4_const_splat(-0x1.205968p+3f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  // Large number such that ulp(magic bias) == exp2(-4)
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.800000p+19f);
  // Mask for the lowest 3 bits
  const v128_t vindex_mask = wasm_u32x4_const_splat(UINT32_C(0x7));
  const v128_t vminus_ln2 = wasm_f32x4_const_splat(-0x1.62E430p-1f);
  // Coefficients of polynomial approximation
  //   exp(2t) - 1 ~ t * (2 + t * (c2 + t * (c3 + t * c4)))
  // on [-log(2)/32, log(2)/32]
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.5558ECp-1f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.555C20p+0f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.000000p+1f);
  const v128_t vtwo = wasm_f32x4_const_splat(2.0f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f(x) :=
    //           \ -f(-x) if x >= 0
    //
    // First we compute f(z) := expm1(2z) / (2 + expm1(2z)) where z = -abs(x), then negate the result if x >= 0.
    v128_t vz = wasm_v128_or(vx, vsign_mask);

    // Inverted mask for the sign of input: 0x00000000 for negative x, 0x80000000 for positive x.
    const v128_t vinvsignx = wasm_v128_xor(vx, vz);

    // The function saturates at -1 for large negative inputs: tanhf(z) == -1.0f for z <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vz = wasm_f32x4_pmax(vz, vsat_cutoff);

    // Compute reduced argument n := round(z / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**18, i.e. |z| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [-9.010913, 0]) saturate tanhf(x).
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. -9.010913 <= z <= 0. As
    // n has 4 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 3 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fetched value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for -9.010913 <= z <= 0 we have -13 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -13.
    //
    // Shift bits 3:11 into 23:31 (position of floating-point exponent).
    const v128_t ve = wasm_i32x4_shl(vn, 20);

    // Use bits 0:3 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const uint64_t vidx_lo = wasm_u64x2_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx_lo));
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx_lo >> 32)), vl, 1);
    const uint64_t vidx_hi = wasm_u64x2_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx_hi), vl, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx_hi >> 32)), vl, 3);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const v128_t vs = wasm_i32x4_add(vl, ve);

    // Subtract the large number back to get final n := round(z / log(2), 4) as a floating-point number.
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    const v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2), vz);

    // Compute degree-4 polynomial approximation for exp(2t) - 1 on [-log(2)/32, log(2)/32].
    //   P(t) = t * (2 + t * (c2 + t * (c3 + t * c4)))
    //        = t * p
    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc4, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vtwo);

    // Reconstruct the exp(2z) - 1 value:
    //   exp(2z) - 1 = s * (t * (2 + t * (c2 + t * (c3 + t * c4))) + 1) - 1
    //               = s * t * p + (s - 1)
    //               = (s - 1) + (p * s) * t
    const v128_t vps = wasm_f32x4_mul(vp, vs);
    const v128_t vsmo = wasm_f32x4_sub(vs, vone);
    const v128_t vemo = wasm_f32x4_add(wasm_f32x4_mul(vt, vps), vsmo);

    // Reconstruct tanh(z) = expm1(2z) / (expm1(2z) + 2)
    const v128_t vepo = wasm_f32x4_add(vemo, vtwo);
    v128_t vy = wasm_f32x4_div(vemo, vepo);

    // Reconstruct tanh(x):
    //
    //             / tanh(z) if x <= 0
    //   tanh(x) =
    //             \ -tanh(z) if x >= 0
    vy = wasm_v128_xor(vy, vinvsignx);

    wasm_v128_store(output, vy);
    output += 4;
  }
}
