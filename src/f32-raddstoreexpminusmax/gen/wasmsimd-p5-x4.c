// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/wasmsimd-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/raddstoreexpminusmax.h>


void xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_p5_x4(
    size_t elements,
    const float* input,
    float* output,
    float* sum,
    float max) XNN_DISABLE_TSAN
{
  assert(elements % sizeof(float) == 0);

  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  // The smallest x for which expf(x) is normalized.
  const v128_t vdenorm_cutoff = wasm_f32x4_const_splat(-0x1.5D589Ep6f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  // Last 7 bits are zeroes
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E400p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(-0x1.7F7D1Cp-20f);

  const v128_t vc1 = wasm_f32x4_const_splat(0x1.FFFFF6p-1f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFDC6p-2f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.555A80p-3f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.573A1Ap-5f);
  const v128_t vc5 = wasm_f32x4_const_splat(0x1.0F9F9Cp-7f);

  const v128_t vi_max = wasm_f32x4_splat(max);

  v128_t vacc0 = wasm_f32x4_const_splat(0.0f);
  for (; elements >= 4 * sizeof(float); elements -= 4 * sizeof(float)) {
    // Load 4 (1x4) inputs at a time.
    const v128_t vi0123 = wasm_v128_load(input);
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const v128_t vx0123 = wasm_f32x4_sub(vi0123, vi_max);

    // Compute reduced argument elements := round(x / log(2)).
    v128_t vn0123 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vx0123, vlog2e));

    // Create a floating-point number s (scale) such that s == 2**elements for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= elements <= 0 accordingly.
    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);

    // Subtract the large number back to get final elements := round(x / log(2)).
    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt0123 = wasm_f32x4_add(vx0123, wasm_f32x4_mul(vn0123, vminus_ln2_hi));

    vt0123 = wasm_f32x4_add(vt0123, wasm_f32x4_mul(vn0123, vminus_ln2_lo));

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    v128_t vp0123 = wasm_f32x4_add(vc4, wasm_f32x4_mul(vc5, vt0123));

    vp0123 = wasm_f32x4_add(vc3, wasm_f32x4_mul(vp0123, vt0123));

    vp0123 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vp0123, vt0123));

    vp0123 = wasm_f32x4_add(vc1, wasm_f32x4_mul(vp0123, vt0123));

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0123 = wasm_f32x4_mul(vt0123, vs0123);

    v128_t vf0123 = wasm_f32x4_add(vs0123, wasm_f32x4_mul(vt0123, vp0123));

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_lt(vx0123, vdenorm_cutoff));

    // Store 4 (1x4) outputs at a time.
    wasm_v128_store(output, vf0123);
    output += 4;

    // Accumulate computed exponents.
    vacc0 = wasm_f32x4_add(vacc0, vf0123);
  }

  v128_t vacc = vacc0;
  for (; elements >= 4 * sizeof(float); elements -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const v128_t vi = wasm_v128_load(input);
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    // Compute reduced argument elements := round(x / log(2)).
    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vx, vlog2e));

    // Create a floating-point number s (scale) such that s == 2**elements for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= elements <= 0 accordingly.
    const v128_t vs = wasm_i32x4_shl(vn, 23);

    // Subtract the large number back to get final elements := round(x / log(2)).
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(vx, wasm_f32x4_mul(vn, vminus_ln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vminus_ln2_lo));

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vc5, vt));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vp, vt));

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    // Store 4 outputs at a time.
    wasm_v128_store(output, vf);
    output += 4;

    // Accumulate computed exponents.
    vacc = wasm_f32x4_add(vacc, vf);
  }
  vacc = wasm_f32x4_add(vacc, wasm_v32x4_shuffle(vacc, vacc, 2, 3, 2, 3));
  float vsum = wasm_f32x4_extract_lane(vacc, 0) + wasm_f32x4_extract_lane(vacc, 1);
  if (elements != 0) {
    assert(elements >= 1 * sizeof(float));
    assert(elements <= 3 * sizeof(float));
    // Load 4 inputs at a time.
    const v128_t vi = wasm_v128_load(input);

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    // Compute reduced argument elements := round(x / log(2)).
    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vx, vlog2e));

    // Create a floating-point number s (scale) such that s == 2**elements for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= elements <= 0 accordingly.
    const v128_t vs = wasm_i32x4_shl(vn, 23);

    // Subtract the large number back to get final elements := round(x / log(2)).
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(vx, wasm_f32x4_mul(vn, vminus_ln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vminus_ln2_lo));

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vc5, vt));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vp, vt));

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    if (elements & (2 * sizeof(float))) {
      // Store and accumulate 2 outputs at a time.
      const float vf0 = wasm_f32x4_extract_lane(vf, 0);
      output[0] = vf0;
      vsum += vf0;

      const float vf1 = wasm_f32x4_extract_lane(vf, 1);
      output[1] = vf1;
      vsum += vf1;

      vf = wasm_v32x4_shuffle(vf, vf, 2, 3, 2, 3);
      output += 2;
    }
    if (elements & (1 * sizeof(float))) {
      // Store 1 output at a time.
      const float vf0 = wasm_f32x4_extract_lane(vf, 0);
      *output = vf0;
      vsum += vf0;
    }
  }
  // Reduce 4 elements in the SIMD register
  *sum = vsum;
}
