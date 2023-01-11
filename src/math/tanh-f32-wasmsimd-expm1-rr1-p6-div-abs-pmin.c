// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6_div_abs_pmin(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(v128_t) == 0);

  // The smallest z for which tanhf(-z) is saturated at -1.0f.
  const v128_t vsat_cutoff = wasm_f32x4_const_splat(0x1.205968p+3f);
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp+22f);
  const v128_t vminus_log2e = wasm_f32x4_const_splat(-0x1.715476p+0f);
  const v128_t vln2 = wasm_f32x4_const_splat(0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(-2t) - 1 ~ -2 * (t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))))
  // on [-log(2)/2, log(2)/2]
  const v128_t vc6 = wasm_f32x4_const_splat(-0x1.6b7338p-5f);
  const v128_t vc5 = wasm_f32x4_const_splat(0x1.12278Ep-3f);
  const v128_t vc4 = wasm_f32x4_const_splat(-0x1.555716p-2f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.5554B0p-1f);
  const v128_t vc2 = wasm_f32x4_const_splat(-0x1.FFFFFEp-1f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);
  const v128_t vminus_two = wasm_f32x4_const_splat(-2.0f);
  // Mask for the sign bit.
  const v128_t vsign_mask = wasm_f32x4_const_splat(-0.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f[x] :=
    //           \ -f[-x] if x >= 0
    //
    // First we compute f[-z] := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
    // then replace result with -f[-z] if x >= 0.
    v128_t vz = wasm_f32x4_abs(vx);

    // The function f[-z] saturates at -1 for large inputs: tanhf(x) == -1.0f for x <= sat_cutoff ~= -9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. The order of operands in the f32x4.pmin instruction matters: it ensures that NaN
    // inputs are passed unchanged.
    vz = wasm_f32x4_pmin(vz, vsat_cutoff);

    // Compute reduced argument n := round(-z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x). We fixup the result for such
    // inputs at the very end of the algorithm.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vminus_log2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // 0 <= z <= 9.010913, and -13 <= n <= 0 accordingly.
    const v128_t vs = wasm_i32x4_shl(vn, 23);

    // Subtract the large number back to get final n := round(-z / log(2), 1) as a floating-point number.
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vln2), vz);

    // Compute degree-6 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(-2t) = t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //          = t + t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //          = -2 * (t + t * p)
    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt), vc5);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
    vp = wasm_f32x4_mul(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 - 2t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))) - 1
    //              = (s - 1) + s * (-2t) * (t + t * p)
    //              = (s - 1) - 2 * ((t * s) + (t * s) * p)
    vt = wasm_f32x4_mul(vt, vs);
    const v128_t vsm1 = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vt);
    const v128_t vem1 = wasm_f32x4_add(wasm_f32x4_mul(vp, vminus_two), vsm1);

    // Reconstruct tanh(-z) := expm1(-2z) / (2 + expm1(-2z))
    const v128_t vep1 = wasm_f32x4_sub(vem1, vminus_two);
    v128_t vabsy = wasm_f32x4_div(vem1, vep1);

    // Reconstruct tanh[x] = sign(x) * tanh[-abs(x)]
    const v128_t vy = wasm_v128_bitselect(vx, vabsy, vsign_mask);

    wasm_v128_store(output, vy);
    output += 4;
  }
}
