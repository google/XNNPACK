// Auto-generated file. Do not edit!
//   Template: src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in
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


void xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u48(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_ALIGN(64) static const uint32_t table[16] = {
    UINT32_C(0x3F800000), UINT32_C(0x3F7DAAC3), UINT32_C(0x3F7B95C2), UINT32_C(0x3F79C3D3),
    UINT32_C(0x3F7837F0), UINT32_C(0x3F76F532), UINT32_C(0x3F75FED7), UINT32_C(0x3F75583F),
    UINT32_C(0x3F7504F3), UINT32_C(0x3F7508A4), UINT32_C(0x3F75672A), UINT32_C(0x3F76248C),
    UINT32_C(0x3F7744FD), UINT32_C(0x3F78CCDF), UINT32_C(0x3F7AC0C7), UINT32_C(0x3F7D257D),
  };
  const __m512i vtable = _mm512_load_si512(table);

  const __m512 vsat_cutoff = _mm512_set1_ps(-0x1.154246p+4f);
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p19f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E430p-1f);
  const __m512 vc3 = _mm512_set1_ps(0x1.55561Cp-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.0001ECp-1f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);

  const __m512 vprescale = _mm512_set1_ps(params->scalar.prescale);
  const __m512 valpha = _mm512_set1_ps(params->scalar.alpha);
  const __m512 vbeta = _mm512_set1_ps(params->scalar.beta);

  for (; batch >= 48 * sizeof(float); batch -= 48 * sizeof(float)) {
    __m512 vx0 = _mm512_loadu_ps(input);
    __m512 vx1 = _mm512_loadu_ps(input + 16);
    __m512 vx2 = _mm512_loadu_ps(input + 32);
    input += 48;

    const __m512 vz0 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx0, vprescale));
    const __m512 vz1 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx1, vprescale));
    const __m512 vz2 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx2, vprescale));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);

    const __m512i ven0 = _mm512_slli_epi32(_mm512_castps_si512(vn0), 19);
    const __m512i vl0 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn0), vtable);
    const __m512i ven1 = _mm512_slli_epi32(_mm512_castps_si512(vn1), 19);
    const __m512i vl1 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn1), vtable);
    const __m512i ven2 = _mm512_slli_epi32(_mm512_castps_si512(vn2), 19);
    const __m512i vl2 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn2), vtable);

    __m512 vs0 = _mm512_castsi512_ps(_mm512_add_epi32(vl0, ven0));
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    __m512 vs1 = _mm512_castsi512_ps(_mm512_add_epi32(vl1, ven1));
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    __m512 vs2 = _mm512_castsi512_ps(_mm512_add_epi32(vl2, ven2));
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vz2);

    __m512 vp0 = _mm512_fmadd_ps(vc3, vt0, vc2);
    __m512 vp1 = _mm512_fmadd_ps(vc3, vt1, vc2);
    __m512 vp2 = _mm512_fmadd_ps(vc3, vt2, vc2);

    vp0 = _mm512_mul_ps(vp0, vt0);
    vt0 = _mm512_mul_ps(vt0, vs0);
    vp1 = _mm512_mul_ps(vp1, vt1);
    vt1 = _mm512_mul_ps(vt1, vs1);
    vp2 = _mm512_mul_ps(vp2, vt2);
    vt2 = _mm512_mul_ps(vt2, vs2);

    vs0 = _mm512_fmsub_ps(vs0, valpha, valpha);
    vs1 = _mm512_fmsub_ps(vs1, valpha, valpha);
    vs2 = _mm512_fmsub_ps(vs2, valpha, valpha);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vt0);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vt1);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vt2);

    const __m512 vzero = _mm512_setzero_ps();
    __m512 vy0 = _mm512_fmadd_ps(vp0, valpha, vs0);
    const __mmask16 vsign0 = _mm512_cmp_ps_mask(vx0, vzero, _CMP_NLT_US);
    __m512 vy1 = _mm512_fmadd_ps(vp1, valpha, vs1);
    const __mmask16 vsign1 = _mm512_cmp_ps_mask(vx1, vzero, _CMP_NLT_US);
    __m512 vy2 = _mm512_fmadd_ps(vp2, valpha, vs2);
    const __mmask16 vsign2 = _mm512_cmp_ps_mask(vx2, vzero, _CMP_NLT_US);

    vy0 = _mm512_mask_mul_ps(vy0, vsign0, vx0, vbeta);
    vy1 = _mm512_mask_mul_ps(vy1, vsign1, vx1, vbeta);
    vy2 = _mm512_mask_mul_ps(vy2, vsign2, vx2, vbeta);

    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    output += 48;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 19);
    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);
    __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc3, vt, vc2);
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
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 19);
    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);
    __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc3, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
