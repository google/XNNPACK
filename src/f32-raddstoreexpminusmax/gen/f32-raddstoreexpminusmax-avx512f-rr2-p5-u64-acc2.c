// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/avx512f-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const struct xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.8000FEp23f);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(-0x1.62E400p-1f);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(-0x1.7F7D1Cp-20f);
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vdenorm_cutoff = _mm512_set1_ps(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const __m512 vi_max = _mm512_set1_ps(*max);
  const __m512 vzero = _mm512_setzero_ps();

  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    // Load 64 (4x16) inputs at a time.
    const __m512 vi0 = _mm512_loadu_ps(input);
    const __m512 vi1 = _mm512_loadu_ps(input + 16);
    const __m512 vi2 = _mm512_loadu_ps(input + 32);
    const __m512 vi3 = _mm512_loadu_ps(input + 48);
    input += 64;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m512 vx0 = _mm512_sub_ps(vi0, vi_max);
    const __m512 vx1 = _mm512_sub_ps(vi1, vi_max);
    const __m512 vx2 = _mm512_sub_ps(vi2, vi_max);
    const __m512 vx3 = _mm512_sub_ps(vi3, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m512 vn0 = _mm512_fmadd_ps(vx0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vx1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vx2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vx3, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m512 vs0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn0), 23));
    const __m512 vs1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn1), 23));
    const __m512 vs2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn2), 23));
    const __m512 vs3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn3), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_hi, vx0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2_hi, vx1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2_hi, vx2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2_hi, vx3);

    vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_lo, vt0);
    vt1 = _mm512_fmadd_ps(vn1, vminus_ln2_lo, vt1);
    vt2 = _mm512_fmadd_ps(vn2, vminus_ln2_lo, vt2);
    vt3 = _mm512_fmadd_ps(vn3, vminus_ln2_lo, vt3);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp0 = _mm512_fmadd_ps(vc5, vt0, vc4);
    __m512 vp1 = _mm512_fmadd_ps(vc5, vt1, vc4);
    __m512 vp2 = _mm512_fmadd_ps(vc5, vt2, vc4);
    __m512 vp3 = _mm512_fmadd_ps(vc5, vt3, vc4);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc3);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc2);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc1);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc1);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc1);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 = _mm512_mul_ps(vt0, vs0);
    vt1 = _mm512_mul_ps(vt1, vs1);
    vt2 = _mm512_mul_ps(vt2, vs2);
    vt3 = _mm512_mul_ps(vt3, vs3);

    __m512 vf0 = _mm512_fmadd_ps(vt0, vp0, vs0);
    __m512 vf1 = _mm512_fmadd_ps(vt1, vp1, vs1);
    __m512 vf2 = _mm512_fmadd_ps(vt2, vp2, vs2);
    __m512 vf3 = _mm512_fmadd_ps(vt3, vp3, vs3);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx0, vdenorm_cutoff, _CMP_LT_OS), vf0, vzero);
    vf1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx1, vdenorm_cutoff, _CMP_LT_OS), vf1, vzero);
    vf2 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx2, vdenorm_cutoff, _CMP_LT_OS), vf2, vzero);
    vf3 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx3, vdenorm_cutoff, _CMP_LT_OS), vf3, vzero);

    // Store 64 (4x16) outputs at a time.
    _mm512_storeu_ps(output, vf0);
    _mm512_storeu_ps(output + 16, vf1);
    _mm512_storeu_ps(output + 32, vf2);
    _mm512_storeu_ps(output + 48, vf3);

    output += 64;

    // Accumulate computed exponents.
    vacc0 = _mm512_add_ps(vacc0, vf0);
    vacc1 = _mm512_add_ps(vacc1, vf1);
    vacc0 = _mm512_add_ps(vacc0, vf2);
    vacc1 = _mm512_add_ps(vacc1, vf3);
  }
  // Add up all accumulators to vacc0
  vacc0 = _mm512_add_ps(vacc0, vacc1);

  __m512 vacc = vacc0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    // Load 16 inputs at a time.
    const __m512 vi = _mm512_loadu_ps(input);
    input += 16;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m512 vx = _mm512_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m512 vn = _mm512_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm512_mul_ps(vt, vs);
    __m512 vf = _mm512_fmadd_ps(vt, vp, vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx, vdenorm_cutoff, _CMP_LT_OS), vf, vzero);

    // Store 16 outputs at a time.
    _mm512_storeu_ps(output, vf);
    output += 16;

    // Accumulate computed exponents.
    vacc = _mm512_add_ps(vacc, vf);
  }
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit batch (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    // Load 16 inputs at a time.
    const __m512 vi = _mm512_maskz_loadu_ps(vmask, input);

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m512 vx = _mm512_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m512 vn = _mm512_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm512_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm512_mul_ps(vt, vs);
    __m512 vf = _mm512_fmadd_ps(vt, vp, vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx, vdenorm_cutoff, _CMP_LT_OS), vf, vzero);

    _mm512_mask_storeu_ps(output, vmask, vf);

    vacc = _mm512_mask_add_ps(vacc, vmask, vacc, vf);
  }

  *sum = _mm512_reduce_add_ps(vacc);
}


