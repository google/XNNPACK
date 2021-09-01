// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <wasm_simd128.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const v128_t vsat_cutoff = wasm_f32x4_const_splat(-0x1.154246p+4f);
  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  // Last 5 bits are zeroes
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E440p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(0x1.0105C6p-21f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/2, log(2)/2]
  const v128_t vc6 = wasm_f32x4_const_splat(0x1.6b7338p-10f);
  const v128_t vc5 = wasm_f32x4_const_splat(0x1.12278Ep-7f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.555716p-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.5554B0p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFFFEp-2f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(input);

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1f(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vx = wasm_f32x4_max(vx, vsat_cutoff);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**22, i.e. |x| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs x are
    // restricted to [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for valid inputs, i.e.
    // -17.328680 <= x <= 0.0, and -25 <= n <= 0 accordingly.
    // For NaN inputs, s would have zero mantissa and can have arbitrary sign and exponent, depending on the input
    // NaN payload. In these cases, n and t are NaNs with the same payload as input while s is non-NaN, and thus
    // input payload would be propagated in all computations.
    const v128_t vs = wasm_i32x4_shl(vn, 23);

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vx);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);

    // Compute degree-6 polynomial approximation for exp(t) - 1 on [-log(2)/2, log(2)/2].
    //   P(t) = t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t + t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) = t + t * p
    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt), vc5);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
    vp = wasm_f32x4_mul(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))) - 1
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
