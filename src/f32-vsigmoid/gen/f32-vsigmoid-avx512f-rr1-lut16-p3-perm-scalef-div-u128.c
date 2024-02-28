// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512_rr1_lut16_p3.sign_mask);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_rr1_lut16_p3.magic_bias);
  const __m512 vlog2e = _mm512_set1_ps(params->avx512_rr1_lut16_p3.log2e);
  const __m512 vtable = _mm512_load_ps(params->avx512_rr1_lut16_p3.table);
  const __m512 vminus_ln2 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.minus_ln2);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.c2);
  const __m512 vone = _mm512_set1_ps(params->avx512_rr1_lut16_p3.one);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    const __m512 vx2 = _mm512_loadu_ps(input + 32);
    const __m512 vx3 = _mm512_loadu_ps(input + 48);
    const __m512 vx4 = _mm512_loadu_ps(input + 64);
    const __m512 vx5 = _mm512_loadu_ps(input + 80);
    const __m512 vx6 = _mm512_loadu_ps(input + 96);
    const __m512 vx7 = _mm512_loadu_ps(input + 112);
    input += 128;

    const __m512 vz0 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx0), vsign_mask));
    const __m512 vz1 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx1), vsign_mask));
    const __m512 vz2 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx2), vsign_mask));
    const __m512 vz3 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx3), vsign_mask));
    const __m512 vz4 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx4), vsign_mask));
    const __m512 vz5 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx5), vsign_mask));
    const __m512 vz6 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx6), vsign_mask));
    const __m512 vz7 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx7), vsign_mask));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vlog2e, vmagic_bias);
    __m512 vn4 = _mm512_fmadd_ps(vz4, vlog2e, vmagic_bias);
    __m512 vn5 = _mm512_fmadd_ps(vz5, vlog2e, vmagic_bias);
    __m512 vn6 = _mm512_fmadd_ps(vz6, vlog2e, vmagic_bias);
    __m512 vn7 = _mm512_fmadd_ps(vz7, vlog2e, vmagic_bias);

    const __m512 vl0 = _mm512_permutexvar_ps(_mm512_castps_si512(vn0), vtable);
    const __m512 vl1 = _mm512_permutexvar_ps(_mm512_castps_si512(vn1), vtable);
    const __m512 vl2 = _mm512_permutexvar_ps(_mm512_castps_si512(vn2), vtable);
    const __m512 vl3 = _mm512_permutexvar_ps(_mm512_castps_si512(vn3), vtable);
    const __m512 vl4 = _mm512_permutexvar_ps(_mm512_castps_si512(vn4), vtable);
    const __m512 vl5 = _mm512_permutexvar_ps(_mm512_castps_si512(vn5), vtable);
    const __m512 vl6 = _mm512_permutexvar_ps(_mm512_castps_si512(vn6), vtable);
    const __m512 vl7 = _mm512_permutexvar_ps(_mm512_castps_si512(vn7), vtable);

    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);
    vn4 = _mm512_sub_ps(vn4, vmagic_bias);
    vn5 = _mm512_sub_ps(vn5, vmagic_bias);
    vn6 = _mm512_sub_ps(vn6, vmagic_bias);
    vn7 = _mm512_sub_ps(vn7, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vz2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2, vz3);
    __m512 vt4 = _mm512_fmadd_ps(vn4, vminus_ln2, vz4);
    __m512 vt5 = _mm512_fmadd_ps(vn5, vminus_ln2, vz5);
    __m512 vt6 = _mm512_fmadd_ps(vn6, vminus_ln2, vz6);
    __m512 vt7 = _mm512_fmadd_ps(vn7, vminus_ln2, vz7);

    __m512 vp0 = _mm512_fmadd_ps(vt0, vc3, vc2);
    __m512 vp1 = _mm512_fmadd_ps(vt1, vc3, vc2);
    __m512 vp2 = _mm512_fmadd_ps(vt2, vc3, vc2);
    __m512 vp3 = _mm512_fmadd_ps(vt3, vc3, vc2);
    __m512 vp4 = _mm512_fmadd_ps(vt4, vc3, vc2);
    __m512 vp5 = _mm512_fmadd_ps(vt5, vc3, vc2);
    __m512 vp6 = _mm512_fmadd_ps(vt6, vc3, vc2);
    __m512 vp7 = _mm512_fmadd_ps(vt7, vc3, vc2);

    vp0 = _mm512_mul_ps(vp0, vt0);
    vp1 = _mm512_mul_ps(vp1, vt1);
    vp2 = _mm512_mul_ps(vp2, vt2);
    vp3 = _mm512_mul_ps(vp3, vt3);
    vp4 = _mm512_mul_ps(vp4, vt4);
    vp5 = _mm512_mul_ps(vp5, vt5);
    vp6 = _mm512_mul_ps(vp6, vt6);
    vp7 = _mm512_mul_ps(vp7, vt7);

    vp0 = _mm512_fmadd_ps(vt0, vp0, vt0);
    vp1 = _mm512_fmadd_ps(vt1, vp1, vt1);
    vp2 = _mm512_fmadd_ps(vt2, vp2, vt2);
    vp3 = _mm512_fmadd_ps(vt3, vp3, vt3);
    vp4 = _mm512_fmadd_ps(vt4, vp4, vt4);
    vp5 = _mm512_fmadd_ps(vt5, vp5, vt5);
    vp6 = _mm512_fmadd_ps(vt6, vp6, vt6);
    vp7 = _mm512_fmadd_ps(vt7, vp7, vt7);

    vp0 = _mm512_fmadd_ps(vl0, vp0, vl0);
    vp1 = _mm512_fmadd_ps(vl1, vp1, vl1);
    vp2 = _mm512_fmadd_ps(vl2, vp2, vl2);
    vp3 = _mm512_fmadd_ps(vl3, vp3, vl3);
    vp4 = _mm512_fmadd_ps(vl4, vp4, vl4);
    vp5 = _mm512_fmadd_ps(vl5, vp5, vl5);
    vp6 = _mm512_fmadd_ps(vl6, vp6, vl6);
    vp7 = _mm512_fmadd_ps(vl7, vp7, vl7);

    const __m512 ve0 = _mm512_scalef_ps(vp0, vn0);
    const __m512 ve1 = _mm512_scalef_ps(vp1, vn1);
    const __m512 ve2 = _mm512_scalef_ps(vp2, vn2);
    const __m512 ve3 = _mm512_scalef_ps(vp3, vn3);
    const __m512 ve4 = _mm512_scalef_ps(vp4, vn4);
    const __m512 ve5 = _mm512_scalef_ps(vp5, vn5);
    const __m512 ve6 = _mm512_scalef_ps(vp6, vn6);
    const __m512 ve7 = _mm512_scalef_ps(vp7, vn7);

    const __m512 vd0 = _mm512_add_ps(ve0, vone);
    const __m512 vd1 = _mm512_add_ps(ve1, vone);
    const __m512 vd2 = _mm512_add_ps(ve2, vone);
    const __m512 vd3 = _mm512_add_ps(ve3, vone);
    const __m512 vd4 = _mm512_add_ps(ve4, vone);
    const __m512 vd5 = _mm512_add_ps(ve5, vone);
    const __m512 vd6 = _mm512_add_ps(ve6, vone);
    const __m512 vd7 = _mm512_add_ps(ve7, vone);

    __m512 vf0 = _mm512_div_ps(ve0, vd0);
    __m512 vf1 = _mm512_div_ps(ve1, vd1);
    __m512 vf2 = _mm512_div_ps(ve2, vd2);
    __m512 vf3 = _mm512_div_ps(ve3, vd3);
    __m512 vf4 = _mm512_div_ps(ve4, vd4);
    __m512 vf5 = _mm512_div_ps(ve5, vd5);
    __m512 vf6 = _mm512_div_ps(ve6, vd6);
    __m512 vf7 = _mm512_div_ps(ve7, vd7);

    vf0 = _mm512_mask_sub_ps(vf0, _mm512_testn_epi32_mask(_mm512_castps_si512(vx0), vsign_mask), vone, vf0);
    vf1 = _mm512_mask_sub_ps(vf1, _mm512_testn_epi32_mask(_mm512_castps_si512(vx1), vsign_mask), vone, vf1);
    vf2 = _mm512_mask_sub_ps(vf2, _mm512_testn_epi32_mask(_mm512_castps_si512(vx2), vsign_mask), vone, vf2);
    vf3 = _mm512_mask_sub_ps(vf3, _mm512_testn_epi32_mask(_mm512_castps_si512(vx3), vsign_mask), vone, vf3);
    vf4 = _mm512_mask_sub_ps(vf4, _mm512_testn_epi32_mask(_mm512_castps_si512(vx4), vsign_mask), vone, vf4);
    vf5 = _mm512_mask_sub_ps(vf5, _mm512_testn_epi32_mask(_mm512_castps_si512(vx5), vsign_mask), vone, vf5);
    vf6 = _mm512_mask_sub_ps(vf6, _mm512_testn_epi32_mask(_mm512_castps_si512(vx6), vsign_mask), vone, vf6);
    vf7 = _mm512_mask_sub_ps(vf7, _mm512_testn_epi32_mask(_mm512_castps_si512(vx7), vsign_mask), vone, vf7);

    _mm512_storeu_ps(output, vf0);
    _mm512_storeu_ps(output + 16, vf1);
    _mm512_storeu_ps(output + 32, vf2);
    _mm512_storeu_ps(output + 48, vf3);
    _mm512_storeu_ps(output + 64, vf4);
    _mm512_storeu_ps(output + 80, vf5);
    _mm512_storeu_ps(output + 96, vf6);
    _mm512_storeu_ps(output + 112, vf7);
    output += 128;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutexvar_ps(_mm512_castps_si512(vn), vtable);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vt, vc3, vc2);
    vp = _mm512_mul_ps(vp, vt);
    vp = _mm512_fmadd_ps(vt, vp, vt);
    vp = _mm512_fmadd_ps(vl, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_storeu_ps(output, vf);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutexvar_ps(_mm512_castps_si512(vn), vtable);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vt, vc3, vc2);
    vp = _mm512_mul_ps(vp, vt);
    vp = _mm512_fmadd_ps(vt, vp, vt);
    vp = _mm512_fmadd_ps(vl, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_mask_storeu_ps(output, vmask, vf);
  }
}
