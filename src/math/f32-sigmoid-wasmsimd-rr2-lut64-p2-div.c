// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 64) values decremented (as integer) by (k << 17), k = 0..63
extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_math_f32_sigmoid__wasmsimd_rr2_lut64_p2_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Large number such that ulp(magic bias) == exp2(-6)
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.800000p17f);
  const v128_t vminus_log2e = wasm_f32x4_const_splat(-0x1.715476p0f);
  // Mask for the lowest 6 bits
  const v128_t vindex_mask = wasm_i32x4_const_splat(INT32_C(0x3F));
  // Last 13 bits are zeroes
  const v128_t vln2_hi = wasm_f32x4_const_splat(0x1.630000p-1f);
  const v128_t vln2_lo = wasm_f32x4_const_splat(-0x1.BD0106p-13f);
  // Coefficient of polynomial approximation of exp(-t) ~ 1 + t * (1 + t * c2) on [-log(2)/128, log(2)/128]
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFF0Ap-2f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const v128_t vdenorm_cutoff = wasm_f32x4_const_splat(0x1.5D589Ep+6f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const v128_t vz = wasm_f32x4_abs(vx);

    // Compute reduced argument n := round(-z / log(2), 6).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**16, i.e. |z| <= 0x1.62E43p+15 = 5814540.0), but that is acceptable, because inputs x
    // outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup
    // the result for such inputs at the very end of the algorithm.
    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));

    // Create a floating-point number s (scale) such that s := 2**n for such inputs that sigmoidf(-z) is normalized,
    // i.e. 0 <= z <= 87.33642. As n has 6 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s
    // in two steps:
    // 1. Fetch 2**frac(n) from the table using the 6 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 87.33642 (inputs for which sigmoidf(z) is normalized) we have
    //    -126 <= int(n) <= 0, and thus the adjusted exponent is not lower than -126.
    //
    // Shift bits 6:14 into 23:31 (position of floating-point exponent).
    const v128_t ve = wasm_i32x4_shl(vn, 17);

    // Use bits 0:6 of n, as integer, as an index for table lookup of l := 2**frac(n).
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const uint32_t vidx0 = wasm_u32x4_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx0));
    const uint32_t vidx1 = wasm_u32x4_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx1), vl, 1);
    const uint32_t vidx2 = wasm_u32x4_extract_lane(vidx, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx2), vl, 2);
    const uint32_t vidx3 = wasm_u32x4_extract_lane(vidx, 3);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx3), vl, 3);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const v128_t vs = wasm_i32x4_add(vl, ve);

    // Subtract the large number back to get the final n := round(-z / log(2), 6) as a floating-point number.
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := (z + n * log(2)). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    // Compute degree-2 polynomial approximation for exp(-t) on [-log(2)/128, log(2)/128].
    //   P(t) = 1 + t * (-1 + t * c2) = 1 - (t - t * (t * c2)) = 1 - p
    v128_t vp = wasm_f32x4_mul(vt, vc2);
    vp = wasm_f32x4_sub(vt, wasm_f32x4_mul(vp, vt));

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (-1 + t * c2))
    //     = s * (1 - p)
    //     = s - s * p
    const v128_t vy = wasm_f32x4_sub(vs, wasm_f32x4_mul(vs, vp));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    v128_t vf = wasm_f32x4_div(vy, wasm_f32x4_add(vy, vone));

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = wasm_v128_bitselect(vf, wasm_f32x4_sub(vone, vf), wasm_i32x4_shr(vx, 31));

    wasm_v128_store(output, vf);
    output += 4;
  }
}
