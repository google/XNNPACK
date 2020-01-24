// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/psimd-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/raddstoreexpminusmax.h>


void xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x4(
    size_t elements,
    const float* input,
    float* output,
    float* sum,
    float max)
{
  assert(elements % sizeof(float) == 0);

  const psimd_f32 vmagic_bias = psimd_splat_f32(0x1.8000FEp23f);
  // The smallest x for which expf(x) is normalized.
  const psimd_f32 vdenorm_cutoff = psimd_splat_f32(-0x1.5D589Ep6f);
  const psimd_f32 vlog2e = psimd_splat_f32(0x1.715476p+0f);
  // Last 7 bits are zeroes
  const psimd_f32 vminus_ln2_hi = psimd_splat_f32(-0x1.62E400p-1f);
  const psimd_f32 vminus_ln2_lo = psimd_splat_f32(-0x1.7F7D1Cp-20f);

  const psimd_f32 vc1 = psimd_splat_f32(0x1.FFFFF6p-1f);
  const psimd_f32 vc2 = psimd_splat_f32(0x1.FFFDC6p-2f);
  const psimd_f32 vc3 = psimd_splat_f32(0x1.555A80p-3f);
  const psimd_f32 vc4 = psimd_splat_f32(0x1.573A1Ap-5f);
  const psimd_f32 vc5 = psimd_splat_f32(0x1.0F9F9Cp-7f);

  const psimd_f32 vi_max = psimd_splat_f32(max);

  psimd_f32 vacc0 = psimd_zero_f32();
  for (; elements >= 4 * sizeof(float); elements -= 4 * sizeof(float)) {
    // Load 4 (1x4) inputs at a time.
    const psimd_f32 vi0123 = psimd_load_f32(input);
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const psimd_f32 vx0123 = psimd_sub_f32(vi0123, vi_max);

    // Compute reduced argument elements := round(x / log(2)).
    psimd_f32 vn0123 = psimd_qfma_f32(vmagic_bias, vx0123, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**elements for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= elements <= 0 accordingly.
    const psimd_f32 vs0123 = (psimd_f32) ((psimd_u32) vn0123 << 23);

    // Subtract the large number back to get final elements := round(x / log(2)).
    vn0123 = psimd_sub_f32(vn0123, vmagic_bias);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    psimd_f32 vt0123 = psimd_qfma_f32(vx0123, vn0123, vminus_ln2_hi);

    vt0123 = psimd_qfma_f32(vt0123, vn0123, vminus_ln2_lo);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    psimd_f32 vp0123 = psimd_qfma_f32(vc4, vc5, vt0123);

    vp0123 = psimd_qfma_f32(vc3, vp0123, vt0123);

    vp0123 = psimd_qfma_f32(vc2, vp0123, vt0123);

    vp0123 = psimd_qfma_f32(vc1, vp0123, vt0123);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0123 = psimd_mul_f32(vt0123, vs0123);

    psimd_f32 vf0123 = psimd_qfma_f32(vs0123, vt0123, vp0123);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = psimd_andnotmask_f32(vx0123 < vdenorm_cutoff, vf0123);

    // Store 4 (1x4) outputs at a time.
    psimd_store_f32(output, vf0123);
    output += 4;

    // Accumulate computed exponents.
    vacc0 = psimd_add_f32(vacc0, vf0123);
  }

  psimd_f32 vacc = vacc0;
  for (; elements >= 4 * sizeof(float); elements -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const psimd_f32 vi = psimd_load_f32(input);
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const psimd_f32 vx = psimd_sub_f32(vi, vi_max);

    // Compute reduced argument elements := round(x / log(2)).
    psimd_f32 vn = psimd_qfma_f32(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**elements for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= elements <= 0 accordingly.
    const psimd_f32 vs = (psimd_f32) ((psimd_u32) vn << 23);

    // Subtract the large number back to get final elements := round(x / log(2)).
    vn = psimd_sub_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    psimd_f32 vt = psimd_qfma_f32(vx, vn, vminus_ln2_hi);
    vt = psimd_qfma_f32(vt, vn, vminus_ln2_lo);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    psimd_f32 vp = psimd_qfma_f32(vc4, vc5, vt);
    vp = psimd_qfma_f32(vc3, vp, vt);
    vp = psimd_qfma_f32(vc2, vp, vt);
    vp = psimd_qfma_f32(vc1, vp, vt);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = psimd_mul_f32(vt, vs);
    psimd_f32 vf = psimd_qfma_f32(vs, vt, vp);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = psimd_andnotmask_f32(vx < vdenorm_cutoff, vf);

    // Store 4 outputs at a time.
    psimd_store_f32(output, vf);
    output += 4;

    // Accumulate computed exponents.
    vacc = psimd_add_f32(vacc, vf);
  }
  if (elements != 0) {
    assert(elements >= 1 * sizeof(float));
    assert(elements <= 3 * sizeof(float));
    // Load 4 inputs at a time.
    const psimd_f32 vi = psimd_load_f32(input);

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const psimd_f32 vx = psimd_sub_f32(vi, vi_max);

    // Compute reduced argument elements := round(x / log(2)).
    psimd_f32 vn = psimd_qfma_f32(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**elements for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= elements <= 0 accordingly.
    const psimd_f32 vs = (psimd_f32) ((psimd_u32) vn << 23);

    // Subtract the large number back to get final elements := round(x / log(2)).
    vn = psimd_sub_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    psimd_f32 vt = psimd_qfma_f32(vx, vn, vminus_ln2_hi);
    vt = psimd_qfma_f32(vt, vn, vminus_ln2_lo);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    psimd_f32 vp = psimd_qfma_f32(vc4, vc5, vt);
    vp = psimd_qfma_f32(vc3, vp, vt);
    vp = psimd_qfma_f32(vc2, vp, vt);
    vp = psimd_qfma_f32(vc1, vp, vt);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = psimd_mul_f32(vt, vs);
    psimd_f32 vf = psimd_qfma_f32(vs, vt, vp);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = psimd_andnotmask_f32(vx < vdenorm_cutoff, vf);

    if (elements & (2 * sizeof(float))) {
      // Store 2 outputs at a time.
      psimd_store2_f32(output, vf);
      output += 2;

      // Accumulate 2 computed exponents.
      vacc = psimd_add_f32(vacc, psimd_concat_lo_f32(vf, psimd_zero_f32()));

      vf = psimd_concat_hi_f32(vf, vf);
    }
    if (elements & (1 * sizeof(float))) {
      // Store 1 output at a time.
      psimd_store1_f32(output, vf);

      // Accumulate 1 computed exponent.
      const psimd_f32 vzero = psimd_zero_f32();
      vf = psimd_concat_lo_f32(vf, vzero);
      vf = psimd_concat_even_f32(vf, vzero);
      vacc = psimd_add_f32(vacc, vf);
    }
  }
  // Reduce 4 elements in the SIMD register
  *sum = psimd_reduce_sum_f32(vacc);
}
