// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/avx512skx-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


void xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vsat_cutoff = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.sat_cutoff);
  const __m512 vminus_log2e = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.minus_log2e);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.magic_bias);
  const __m512i vtable = _mm512_load_si512(params->avx512_expm1minus_rr1_lut8_p4h3_perm.table);
  const __m512 vln2 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.ln2);
  const __m512 vc4 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.c4);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.c2);
  const __m512 vminus_two = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.minus_two);
  const __m512 vone = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut8_p4h3_perm.one);
  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512_expm1minus_rr1_lut8_p4h3_perm.sign_mask);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_range_ps(vsat_cutoff, vx, 0xA);
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    const __m512i ve = _mm512_slli_epi32(_mm512_castps_si512(vn), 20);

    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);

    const __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ve));

    vn = _mm512_sub_ps(vn, vmagic_bias);

    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    __m512 vp = vc4;
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vminus_two);

    const __m512 vts = _mm512_mul_ps(vt, vs);
    const __m512 vsmo = _mm512_sub_ps(vs, vone);
    const __m512 vemo = _mm512_fmadd_ps(vp, vts, vsmo);
    const __m512 vepo = _mm512_sub_ps(vemo, vminus_two);

    __m512 vy = _mm512_div_ps(vemo, vepo);
    vy = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy), _mm512_castps_si512(vx), vsign_mask, 0xD8));

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    const __m512 vz = _mm512_range_ps(vsat_cutoff, vx, 0xA);
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    const __m512i ve = _mm512_slli_epi32(_mm512_castps_si512(vn), 20);

    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);

    const __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ve));

    vn = _mm512_sub_ps(vn, vmagic_bias);

    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    __m512 vp = vc4;
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vminus_two);

    const __m512 vts = _mm512_mul_ps(vt, vs);
    const __m512 vsmo = _mm512_sub_ps(vs, vone);
    const __m512 vemo = _mm512_fmadd_ps(vp, vts, vsmo);
    const __m512 vepo = _mm512_sub_ps(vemo, vminus_two);

    __m512 vy = _mm512_div_ps(vemo, vepo);
    vy = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy), _mm512_castps_si512(vx), vsign_mask, 0xD8));

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
