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


void xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u160(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vsat_cutoff = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.sat_cutoff);
  const __m512 vminus_log2e = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.minus_log2e);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.magic_bias);
  const __m512 vln2 = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.ln2);
  const __m512 vc6 = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.c6);
  const __m512 vc5 = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.c5);
  const __m512 vc4 = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.c4);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.c2);
  const __m512 vminus_two = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.minus_two);
  const __m512 vone = _mm512_set1_ps(params->avx512_expm1minus_rr1_p6h5.one);
  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512_expm1minus_rr1_p6h5.sign_mask);

  for (; batch >= 160 * sizeof(float); batch -= 160 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    const __m512 vx2 = _mm512_loadu_ps(input + 32);
    const __m512 vx3 = _mm512_loadu_ps(input + 48);
    const __m512 vx4 = _mm512_loadu_ps(input + 64);
    const __m512 vx5 = _mm512_loadu_ps(input + 80);
    const __m512 vx6 = _mm512_loadu_ps(input + 96);
    const __m512 vx7 = _mm512_loadu_ps(input + 112);
    const __m512 vx8 = _mm512_loadu_ps(input + 128);
    const __m512 vx9 = _mm512_loadu_ps(input + 144);
    input += 160;

    const __m512 vz0 = _mm512_range_ps(vsat_cutoff, vx0, 0xA);
    const __m512 vz1 = _mm512_range_ps(vsat_cutoff, vx1, 0xA);
    const __m512 vz2 = _mm512_range_ps(vsat_cutoff, vx2, 0xA);
    const __m512 vz3 = _mm512_range_ps(vsat_cutoff, vx3, 0xA);
    const __m512 vz4 = _mm512_range_ps(vsat_cutoff, vx4, 0xA);
    const __m512 vz5 = _mm512_range_ps(vsat_cutoff, vx5, 0xA);
    const __m512 vz6 = _mm512_range_ps(vsat_cutoff, vx6, 0xA);
    const __m512 vz7 = _mm512_range_ps(vsat_cutoff, vx7, 0xA);
    const __m512 vz8 = _mm512_range_ps(vsat_cutoff, vx8, 0xA);
    const __m512 vz9 = _mm512_range_ps(vsat_cutoff, vx9, 0xA);
    __m512 vn0 = _mm512_fmadd_ps(vz0, vminus_log2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vminus_log2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vminus_log2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vminus_log2e, vmagic_bias);
    __m512 vn4 = _mm512_fmadd_ps(vz4, vminus_log2e, vmagic_bias);
    __m512 vn5 = _mm512_fmadd_ps(vz5, vminus_log2e, vmagic_bias);
    __m512 vn6 = _mm512_fmadd_ps(vz6, vminus_log2e, vmagic_bias);
    __m512 vn7 = _mm512_fmadd_ps(vz7, vminus_log2e, vmagic_bias);
    __m512 vn8 = _mm512_fmadd_ps(vz8, vminus_log2e, vmagic_bias);
    __m512 vn9 = _mm512_fmadd_ps(vz9, vminus_log2e, vmagic_bias);

    const __m512 vs0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn0), 23));
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    const __m512 vs1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn1), 23));
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    const __m512 vs2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn2), 23));
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    const __m512 vs3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn3), 23));
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);
    const __m512 vs4 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn4), 23));
    vn4 = _mm512_sub_ps(vn4, vmagic_bias);
    const __m512 vs5 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn5), 23));
    vn5 = _mm512_sub_ps(vn5, vmagic_bias);
    const __m512 vs6 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn6), 23));
    vn6 = _mm512_sub_ps(vn6, vmagic_bias);
    const __m512 vs7 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn7), 23));
    vn7 = _mm512_sub_ps(vn7, vmagic_bias);
    const __m512 vs8 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn8), 23));
    vn8 = _mm512_sub_ps(vn8, vmagic_bias);
    const __m512 vs9 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn9), 23));
    vn9 = _mm512_sub_ps(vn9, vmagic_bias);

    const __m512 vt0 = _mm512_fmadd_ps(vn0, vln2, vz0);
    const __m512 vt1 = _mm512_fmadd_ps(vn1, vln2, vz1);
    const __m512 vt2 = _mm512_fmadd_ps(vn2, vln2, vz2);
    const __m512 vt3 = _mm512_fmadd_ps(vn3, vln2, vz3);
    const __m512 vt4 = _mm512_fmadd_ps(vn4, vln2, vz4);
    const __m512 vt5 = _mm512_fmadd_ps(vn5, vln2, vz5);
    const __m512 vt6 = _mm512_fmadd_ps(vn6, vln2, vz6);
    const __m512 vt7 = _mm512_fmadd_ps(vn7, vln2, vz7);
    const __m512 vt8 = _mm512_fmadd_ps(vn8, vln2, vz8);
    const __m512 vt9 = _mm512_fmadd_ps(vn9, vln2, vz9);

    __m512 vp0 = vc6;
    __m512 vp1 = vc6;
    __m512 vp2 = vc6;
    __m512 vp3 = vc6;
    __m512 vp4 = vc6;
    __m512 vp5 = vc6;
    __m512 vp6 = vc6;
    __m512 vp7 = vc6;
    __m512 vp8 = vc6;
    __m512 vp9 = vc6;
    vp0 = _mm512_fmadd_ps(vp0, vt0, vc5);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc5);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc5);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc5);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc5);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc5);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc5);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc5);
    vp8 = _mm512_fmadd_ps(vp8, vt8, vc5);
    vp9 = _mm512_fmadd_ps(vp9, vt9, vc5);
    vp0 = _mm512_fmadd_ps(vp0, vt0, vc4);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc4);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc4);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc4);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc4);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc4);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc4);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc4);
    vp8 = _mm512_fmadd_ps(vp8, vt8, vc4);
    vp9 = _mm512_fmadd_ps(vp9, vt9, vc4);
    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc3);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc3);
    vp8 = _mm512_fmadd_ps(vp8, vt8, vc3);
    vp9 = _mm512_fmadd_ps(vp9, vt9, vc3);
    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc2);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc2);
    vp8 = _mm512_fmadd_ps(vp8, vt8, vc2);
    vp9 = _mm512_fmadd_ps(vp9, vt9, vc2);
    vp0 = _mm512_fmadd_ps(vp0, vt0, vminus_two);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vminus_two);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vminus_two);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vminus_two);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vminus_two);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vminus_two);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vminus_two);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vminus_two);
    vp8 = _mm512_fmadd_ps(vp8, vt8, vminus_two);
    vp9 = _mm512_fmadd_ps(vp9, vt9, vminus_two);

    const __m512 vts0 = _mm512_mul_ps(vt0, vs0);
    const __m512 vsmo0 = _mm512_sub_ps(vs0, vone);
    const __m512 vts1 = _mm512_mul_ps(vt1, vs1);
    const __m512 vsmo1 = _mm512_sub_ps(vs1, vone);
    const __m512 vts2 = _mm512_mul_ps(vt2, vs2);
    const __m512 vsmo2 = _mm512_sub_ps(vs2, vone);
    const __m512 vts3 = _mm512_mul_ps(vt3, vs3);
    const __m512 vsmo3 = _mm512_sub_ps(vs3, vone);
    const __m512 vts4 = _mm512_mul_ps(vt4, vs4);
    const __m512 vsmo4 = _mm512_sub_ps(vs4, vone);
    const __m512 vts5 = _mm512_mul_ps(vt5, vs5);
    const __m512 vsmo5 = _mm512_sub_ps(vs5, vone);
    const __m512 vts6 = _mm512_mul_ps(vt6, vs6);
    const __m512 vsmo6 = _mm512_sub_ps(vs6, vone);
    const __m512 vts7 = _mm512_mul_ps(vt7, vs7);
    const __m512 vsmo7 = _mm512_sub_ps(vs7, vone);
    const __m512 vts8 = _mm512_mul_ps(vt8, vs8);
    const __m512 vsmo8 = _mm512_sub_ps(vs8, vone);
    const __m512 vts9 = _mm512_mul_ps(vt9, vs9);
    const __m512 vsmo9 = _mm512_sub_ps(vs9, vone);
    const __m512 vemo0 = _mm512_fmadd_ps(vp0, vts0, vsmo0);
    const __m512 vemo1 = _mm512_fmadd_ps(vp1, vts1, vsmo1);
    const __m512 vemo2 = _mm512_fmadd_ps(vp2, vts2, vsmo2);
    const __m512 vemo3 = _mm512_fmadd_ps(vp3, vts3, vsmo3);
    const __m512 vemo4 = _mm512_fmadd_ps(vp4, vts4, vsmo4);
    const __m512 vemo5 = _mm512_fmadd_ps(vp5, vts5, vsmo5);
    const __m512 vemo6 = _mm512_fmadd_ps(vp6, vts6, vsmo6);
    const __m512 vemo7 = _mm512_fmadd_ps(vp7, vts7, vsmo7);
    const __m512 vemo8 = _mm512_fmadd_ps(vp8, vts8, vsmo8);
    const __m512 vemo9 = _mm512_fmadd_ps(vp9, vts9, vsmo9);
    const __m512 vepo0 = _mm512_sub_ps(vemo0, vminus_two);
    const __m512 vepo1 = _mm512_sub_ps(vemo1, vminus_two);
    const __m512 vepo2 = _mm512_sub_ps(vemo2, vminus_two);
    const __m512 vepo3 = _mm512_sub_ps(vemo3, vminus_two);
    const __m512 vepo4 = _mm512_sub_ps(vemo4, vminus_two);
    const __m512 vepo5 = _mm512_sub_ps(vemo5, vminus_two);
    const __m512 vepo6 = _mm512_sub_ps(vemo6, vminus_two);
    const __m512 vepo7 = _mm512_sub_ps(vemo7, vminus_two);
    const __m512 vepo8 = _mm512_sub_ps(vemo8, vminus_two);
    const __m512 vepo9 = _mm512_sub_ps(vemo9, vminus_two);

    __m512 vy0 = _mm512_div_ps(vemo0, vepo0);
    __m512 vy1 = _mm512_div_ps(vemo1, vepo1);
    __m512 vy2 = _mm512_div_ps(vemo2, vepo2);
    __m512 vy3 = _mm512_div_ps(vemo3, vepo3);
    __m512 vy4 = _mm512_div_ps(vemo4, vepo4);
    __m512 vy5 = _mm512_div_ps(vemo5, vepo5);
    __m512 vy6 = _mm512_div_ps(vemo6, vepo6);
    __m512 vy7 = _mm512_div_ps(vemo7, vepo7);
    __m512 vy8 = _mm512_div_ps(vemo8, vepo8);
    __m512 vy9 = _mm512_div_ps(vemo9, vepo9);
    vy0 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy0), _mm512_castps_si512(vx0), vsign_mask, 0xD8));
    vy1 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy1), _mm512_castps_si512(vx1), vsign_mask, 0xD8));
    vy2 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy2), _mm512_castps_si512(vx2), vsign_mask, 0xD8));
    vy3 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy3), _mm512_castps_si512(vx3), vsign_mask, 0xD8));
    vy4 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy4), _mm512_castps_si512(vx4), vsign_mask, 0xD8));
    vy5 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy5), _mm512_castps_si512(vx5), vsign_mask, 0xD8));
    vy6 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy6), _mm512_castps_si512(vx6), vsign_mask, 0xD8));
    vy7 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy7), _mm512_castps_si512(vx7), vsign_mask, 0xD8));
    vy8 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy8), _mm512_castps_si512(vx8), vsign_mask, 0xD8));
    vy9 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy9), _mm512_castps_si512(vx9), vsign_mask, 0xD8));

    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    _mm512_storeu_ps(output + 48, vy3);
    _mm512_storeu_ps(output + 64, vy4);
    _mm512_storeu_ps(output + 80, vy5);
    _mm512_storeu_ps(output + 96, vy6);
    _mm512_storeu_ps(output + 112, vy7);
    _mm512_storeu_ps(output + 128, vy8);
    _mm512_storeu_ps(output + 144, vy9);
    output += 160;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_range_ps(vsat_cutoff, vx, 0xA);
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    const __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));

    vn = _mm512_sub_ps(vn, vmagic_bias);

    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    __m512 vp = vc6;
    vp = _mm512_fmadd_ps(vp, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
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

    const __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));

    vn = _mm512_sub_ps(vn, vmagic_bias);

    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    __m512 vp = vc6;
    vp = _mm512_fmadd_ps(vp, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
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
