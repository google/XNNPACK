// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__wasmsimd_rr2_p5_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vminus_log2e = wasm_f32x4_const_splat(-0x1.715476p+0f);
  // Last 7 bits are zeroes
  const v128_t vln2_hi = wasm_f32x4_const_splat(0x1.62E400p-1f);
  const v128_t vln2_lo = wasm_f32x4_const_splat(0x1.7F7D1Cp-20f);
  // Coefficient of polynomial approximation of
  // exp(-t) ~ 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) on [-log(2)/2, log(2)/2]
  const v128_t vc5 = wasm_f32x4_const_splat(-0x1.0F9F9Cp-7f);
  const v128_t vc4 = wasm_f32x4_const_splat( 0x1.573A1Ap-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(-0x1.555A80p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat( 0x1.FFFDC6p-2f);
  const v128_t vc1 = wasm_f32x4_const_splat(-0x1.FFFFF6p-1f);
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

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**22, i.e. |z| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs x
    // outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup
    // the result for such inputs at the very end of the algorithm.
    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const v128_t vs = wasm_i32x4_shl(vn, 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) = 1 + t * p
    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt, vc5));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt, vp));

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s * (1 + t * p)
    //     = s + (t * s) * p
    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    v128_t vf = wasm_f32x4_div(ve, wasm_f32x4_add(ve, vone));

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = wasm_v128_bitselect(vf, wasm_f32x4_sub(vone, vf), wasm_i32x4_shr(vx, 31));

    wasm_v128_store(output, vf);
    output += 4;
  }
}
