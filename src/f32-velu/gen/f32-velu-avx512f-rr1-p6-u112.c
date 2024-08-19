// Auto-generated file. Do not edit!
//   Template: src/f32-velu/avx512f-rr1-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


void xnn_f32_velu_ukernel__avx512f_rr1_p6_u112(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vsat_cutoff = _mm512_set1_ps(-0x1.154246p+4f);
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.8000FEp23f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E430p-1f);
  const __m512 vc6 = _mm512_set1_ps(0x1.6B7338p-10f);
  const __m512 vc5 = _mm512_set1_ps(0x1.12278Ep-7f);
  const __m512 vc4 = _mm512_set1_ps(0x1.555716p-5f);
  const __m512 vc3 = _mm512_set1_ps(0x1.5554B0p-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFFFEp-2f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc6);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  
  const __m512 vprescale = _mm512_set1_ps(params->scalar.prescale);
  const __m512 valpha = _mm512_set1_ps(params->scalar.alpha);
  const __m512 vbeta = _mm512_set1_ps(params->scalar.beta);

  for (; batch >= 112 * sizeof(float); batch -= 112 * sizeof(float)) {
    __m512 vx0 = _mm512_loadu_ps(input);
    __m512 vx1 = _mm512_loadu_ps(input + 16);
    __m512 vx2 = _mm512_loadu_ps(input + 32);
    __m512 vx3 = _mm512_loadu_ps(input + 48);
    __m512 vx4 = _mm512_loadu_ps(input + 64);
    __m512 vx5 = _mm512_loadu_ps(input + 80);
    __m512 vx6 = _mm512_loadu_ps(input + 96);
    input += 112;

    const __m512 vz0 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx0, vprescale));
    const __m512 vz1 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx1, vprescale));
    const __m512 vz2 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx2, vprescale));
    const __m512 vz3 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx3, vprescale));
    const __m512 vz4 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx4, vprescale));
    const __m512 vz5 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx5, vprescale));
    const __m512 vz6 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx6, vprescale));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vlog2e, vmagic_bias);
    __m512 vn4 = _mm512_fmadd_ps(vz4, vlog2e, vmagic_bias);
    __m512 vn5 = _mm512_fmadd_ps(vz5, vlog2e, vmagic_bias);
    __m512 vn6 = _mm512_fmadd_ps(vz6, vlog2e, vmagic_bias);

    __m512 vs0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn0), 23));
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    __m512 vs1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn1), 23));
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    __m512 vs2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn2), 23));
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    __m512 vs3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn3), 23));
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);
    __m512 vs4 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn4), 23));
    vn4 = _mm512_sub_ps(vn4, vmagic_bias);
    __m512 vs5 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn5), 23));
    vn5 = _mm512_sub_ps(vn5, vmagic_bias);
    __m512 vs6 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn6), 23));
    vn6 = _mm512_sub_ps(vn6, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vz2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2, vz3);
    __m512 vt4 = _mm512_fmadd_ps(vn4, vminus_ln2, vz4);
    __m512 vt5 = _mm512_fmadd_ps(vn5, vminus_ln2, vz5);
    __m512 vt6 = _mm512_fmadd_ps(vn6, vminus_ln2, vz6);

    __m512 vp0 = _mm512_fmadd_ps(vc6, vt0, vc5);
    __m512 vp1 = _mm512_fmadd_ps(vc6, vt1, vc5);
    __m512 vp2 = _mm512_fmadd_ps(vc6, vt2, vc5);
    __m512 vp3 = _mm512_fmadd_ps(vc6, vt3, vc5);
    __m512 vp4 = _mm512_fmadd_ps(vc6, vt4, vc5);
    __m512 vp5 = _mm512_fmadd_ps(vc6, vt5, vc5);
    __m512 vp6 = _mm512_fmadd_ps(vc6, vt6, vc5);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc4);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc4);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc4);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc4);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc4);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc4);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc4);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc3);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc2);

    vp0 = _mm512_mul_ps(vp0, vt0);
    vt0 = _mm512_mul_ps(vt0, vs0);
    vp1 = _mm512_mul_ps(vp1, vt1);
    vt1 = _mm512_mul_ps(vt1, vs1);
    vp2 = _mm512_mul_ps(vp2, vt2);
    vt2 = _mm512_mul_ps(vt2, vs2);
    vp3 = _mm512_mul_ps(vp3, vt3);
    vt3 = _mm512_mul_ps(vt3, vs3);
    vp4 = _mm512_mul_ps(vp4, vt4);
    vt4 = _mm512_mul_ps(vt4, vs4);
    vp5 = _mm512_mul_ps(vp5, vt5);
    vt5 = _mm512_mul_ps(vt5, vs5);
    vp6 = _mm512_mul_ps(vp6, vt6);
    vt6 = _mm512_mul_ps(vt6, vs6);

    vs0 = _mm512_fmsub_ps(vs0, valpha, valpha);
    vs1 = _mm512_fmsub_ps(vs1, valpha, valpha);
    vs2 = _mm512_fmsub_ps(vs2, valpha, valpha);
    vs3 = _mm512_fmsub_ps(vs3, valpha, valpha);
    vs4 = _mm512_fmsub_ps(vs4, valpha, valpha);
    vs5 = _mm512_fmsub_ps(vs5, valpha, valpha);
    vs6 = _mm512_fmsub_ps(vs6, valpha, valpha);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vt0);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vt1);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vt2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vt3);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vt4);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vt5);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vt6);

    const __m512 vzero = _mm512_setzero_ps();
    __m512 vy0 = _mm512_fmadd_ps(vp0, valpha, vs0);
    const __mmask16 vsign0 = _mm512_cmp_ps_mask(vx0, vzero, _CMP_NLT_US);
    __m512 vy1 = _mm512_fmadd_ps(vp1, valpha, vs1);
    const __mmask16 vsign1 = _mm512_cmp_ps_mask(vx1, vzero, _CMP_NLT_US);
    __m512 vy2 = _mm512_fmadd_ps(vp2, valpha, vs2);
    const __mmask16 vsign2 = _mm512_cmp_ps_mask(vx2, vzero, _CMP_NLT_US);
    __m512 vy3 = _mm512_fmadd_ps(vp3, valpha, vs3);
    const __mmask16 vsign3 = _mm512_cmp_ps_mask(vx3, vzero, _CMP_NLT_US);
    __m512 vy4 = _mm512_fmadd_ps(vp4, valpha, vs4);
    const __mmask16 vsign4 = _mm512_cmp_ps_mask(vx4, vzero, _CMP_NLT_US);
    __m512 vy5 = _mm512_fmadd_ps(vp5, valpha, vs5);
    const __mmask16 vsign5 = _mm512_cmp_ps_mask(vx5, vzero, _CMP_NLT_US);
    __m512 vy6 = _mm512_fmadd_ps(vp6, valpha, vs6);
    const __mmask16 vsign6 = _mm512_cmp_ps_mask(vx6, vzero, _CMP_NLT_US);

    vy0 = _mm512_mask_mul_ps(vy0, vsign0, vx0, vbeta);
    vy1 = _mm512_mask_mul_ps(vy1, vsign1, vx1, vbeta);
    vy2 = _mm512_mask_mul_ps(vy2, vsign2, vx2, vbeta);
    vy3 = _mm512_mask_mul_ps(vy3, vsign3, vx3, vbeta);
    vy4 = _mm512_mask_mul_ps(vy4, vsign4, vx4, vbeta);
    vy5 = _mm512_mask_mul_ps(vy5, vsign5, vx5, vbeta);
    vy6 = _mm512_mask_mul_ps(vy6, vsign6, vx6, vbeta);

    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    _mm512_storeu_ps(output + 48, vy3);
    _mm512_storeu_ps(output + 64, vy4);
    _mm512_storeu_ps(output + 80, vy5);
    _mm512_storeu_ps(output + 96, vy6);
    output += 112;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc6, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc6, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
