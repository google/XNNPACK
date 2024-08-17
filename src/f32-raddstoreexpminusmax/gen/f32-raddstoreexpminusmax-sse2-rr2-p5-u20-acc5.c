// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc5(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.8000FEp23f);
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vc5 = _mm_set1_ps(0x1.0F9F9Cp-7f);
  const __m128 vc4 = _mm_set1_ps(0x1.573A1Ap-5f);
  const __m128 vc3 = _mm_set1_ps(0x1.555A80p-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFDC6p-2f);
  const __m128 vc1 = _mm_set1_ps(0x1.FFFFF6p-1f);
  const __m128 vdenorm_cutoff = _mm_set1_ps(-0x1.5D589Ep6f);

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

  const __m128 vi_max = _mm_load1_ps(max);

  __m128 vacc0 = _mm_setzero_ps();
  __m128 vacc1 = _mm_setzero_ps();
  __m128 vacc2 = _mm_setzero_ps();
  __m128 vacc3 = _mm_setzero_ps();
  __m128 vacc4 = _mm_setzero_ps();
  for (; batch >= 20 * sizeof(float); batch -= 20 * sizeof(float)) {
    // Load 20 (5x4) inputs at a time.
    const __m128 vi0123 = _mm_loadu_ps(input);
    const __m128 vi4567 = _mm_loadu_ps(input + 4);
    const __m128 vi89AB = _mm_loadu_ps(input + 8);
    const __m128 viCDEF = _mm_loadu_ps(input + 12);
    const __m128 viGHIJ = _mm_loadu_ps(input + 16);
    input += 20;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx0123 = _mm_sub_ps(vi0123, vi_max);
    const __m128 vx4567 = _mm_sub_ps(vi4567, vi_max);
    const __m128 vx89AB = _mm_sub_ps(vi89AB, vi_max);
    const __m128 vxCDEF = _mm_sub_ps(viCDEF, vi_max);
    const __m128 vxGHIJ = _mm_sub_ps(viGHIJ, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vx0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vx4567, vlog2e), vmagic_bias);
    __m128 vn89AB = _mm_add_ps(_mm_mul_ps(vx89AB, vlog2e), vmagic_bias);
    __m128 vnCDEF = _mm_add_ps(_mm_mul_ps(vxCDEF, vlog2e), vmagic_bias);
    __m128 vnGHIJ = _mm_add_ps(_mm_mul_ps(vxGHIJ, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0123), 23));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn4567), 23));
    const __m128 vs89AB = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn89AB), 23));
    const __m128 vsCDEF = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnCDEF), 23));
    const __m128 vsGHIJ = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnGHIJ), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);
    vn89AB = _mm_sub_ps(vn89AB, vmagic_bias);
    vnCDEF = _mm_sub_ps(vnCDEF, vmagic_bias);
    vnGHIJ = _mm_sub_ps(vnGHIJ, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_hi), vx0123);
    __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_hi), vx4567);
    __m128 vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_hi), vx89AB);
    __m128 vtCDEF = _mm_add_ps(_mm_mul_ps(vnCDEF, vminus_ln2_hi), vxCDEF);
    __m128 vtGHIJ = _mm_add_ps(_mm_mul_ps(vnGHIJ, vminus_ln2_hi), vxGHIJ);

    vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_lo), vt4567);
    vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_lo), vt89AB);
    vtCDEF = _mm_add_ps(_mm_mul_ps(vnCDEF, vminus_ln2_lo), vtCDEF);
    vtGHIJ = _mm_add_ps(_mm_mul_ps(vnGHIJ, vminus_ln2_lo), vtGHIJ);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc5, vt0123), vc4);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc5, vt4567), vc4);
    __m128 vp89AB = _mm_add_ps(_mm_mul_ps(vc5, vt89AB), vc4);
    __m128 vpCDEF = _mm_add_ps(_mm_mul_ps(vc5, vtCDEF), vc4);
    __m128 vpGHIJ = _mm_add_ps(_mm_mul_ps(vc5, vtGHIJ), vc4);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc3);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc3);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc3);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc3);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc3);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc2);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc2);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc2);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc2);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc2);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc1);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc1);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc1);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc1);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0123 = _mm_mul_ps(vt0123, vs0123);
    vt4567 = _mm_mul_ps(vt4567, vs4567);
    vt89AB = _mm_mul_ps(vt89AB, vs89AB);
    vtCDEF = _mm_mul_ps(vtCDEF, vsCDEF);
    vtGHIJ = _mm_mul_ps(vtGHIJ, vsGHIJ);

    __m128 vf0123 = _mm_add_ps(_mm_mul_ps(vt0123, vp0123), vs0123);
    __m128 vf4567 = _mm_add_ps(_mm_mul_ps(vt4567, vp4567), vs4567);
    __m128 vf89AB = _mm_add_ps(_mm_mul_ps(vt89AB, vp89AB), vs89AB);
    __m128 vfCDEF = _mm_add_ps(_mm_mul_ps(vtCDEF, vpCDEF), vsCDEF);
    __m128 vfGHIJ = _mm_add_ps(_mm_mul_ps(vtGHIJ, vpGHIJ), vsGHIJ);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = _mm_andnot_ps(_mm_cmplt_ps(vx0123, vdenorm_cutoff), vf0123);
    vf4567 = _mm_andnot_ps(_mm_cmplt_ps(vx4567, vdenorm_cutoff), vf4567);
    vf89AB = _mm_andnot_ps(_mm_cmplt_ps(vx89AB, vdenorm_cutoff), vf89AB);
    vfCDEF = _mm_andnot_ps(_mm_cmplt_ps(vxCDEF, vdenorm_cutoff), vfCDEF);
    vfGHIJ = _mm_andnot_ps(_mm_cmplt_ps(vxGHIJ, vdenorm_cutoff), vfGHIJ);

    // Store 20 (5x4) outputs at a time.
    _mm_storeu_ps(output, vf0123);
    _mm_storeu_ps(output + 4, vf4567);
    _mm_storeu_ps(output + 8, vf89AB);
    _mm_storeu_ps(output + 12, vfCDEF);
    _mm_storeu_ps(output + 16, vfGHIJ);
    output += 20;

    // Accumulate computed exponents.
    vacc0 = _mm_add_ps(vacc0, vf0123);
    vacc4 = _mm_add_ps(vacc4, vf4567);
    vacc3 = _mm_add_ps(vacc3, vf89AB);
    vacc2 = _mm_add_ps(vacc2, vfCDEF);
    vacc1 = _mm_add_ps(vacc1, vfGHIJ);
  }
  // Add up all accumulators to vacc0
  vacc0 = _mm_add_ps(vacc0, vacc1);
  vacc2 = _mm_add_ps(vacc2, vacc3);
  vacc0 = _mm_add_ps(vacc0, vacc2);
  vacc0 = _mm_add_ps(vacc0, vacc4);

  __m128 vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const __m128 vi = _mm_loadu_ps(input);
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx = _mm_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm_mul_ps(vt, vs);
    __m128 vf = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vx, vdenorm_cutoff), vf);

    // Store 4 outputs at a time.
    _mm_storeu_ps(output, vf);
    output += 4;

    // Accumulate computed exponents.
    vacc = _mm_add_ps(vacc, vf);
  }
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    // Load 4 inputs at a time.
    const __m128 vi = _mm_loadu_ps(input);

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx = _mm_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm_mul_ps(vt, vs);
    __m128 vf = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vx, vdenorm_cutoff), vf);

    if (batch & (2 * sizeof(float))) {
      // Store 2 outputs at a time.
      _mm_storel_pi((__m64*) output, vf);
      output += 2;

      // Accumulate 2 computed exponents.
      vacc = _mm_add_ps(vacc, _mm_movelh_ps(vf, _mm_setzero_ps()));

      vf = _mm_movehl_ps(vf, vf);
    }
    if (batch & (1 * sizeof(float))) {
      // Store 1 output at a time.
      _mm_store_ss(output, vf);

      // Accumulate 1 computed exponent.
      vacc = _mm_add_ss(vacc, vf);
    }
  }
  // Reduce 4 batch in the SIMD register
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_shuffle_ps(vacc, vacc, _MM_SHUFFLE(2, 3, 0, 1)));
  _mm_store_ss(sum, vacc);
}
