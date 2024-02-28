// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/raddstoreexpminusmax.h>


void xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc4(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const __m512 vi_max = _mm512_set1_ps(*max);
  const __m512 vlog2e = _mm512_set1_ps(params->avx512_rr1_p5.log2e);
  const __m512 vminus_ln2 = _mm512_set1_ps(params->avx512_rr1_p5.minus_ln2);
  const __m512 vc5 = _mm512_set1_ps(params->avx512_rr1_p5.c5);
  const __m512 vc4 = _mm512_set1_ps(params->avx512_rr1_p5.c4);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_rr1_p5.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_rr1_p5.c2);
  const __m512 vc1 = _mm512_set1_ps(params->avx512_rr1_p5.c1);
  const __m512 vc0 = _mm512_set1_ps(params->avx512_rr1_p5.c0);

  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  __m512 vacc2 = _mm512_setzero_ps();
  __m512 vacc3 = _mm512_setzero_ps();
  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    const __m512 vi0 = _mm512_loadu_ps(input);
    const __m512 vi1 = _mm512_loadu_ps(input + 16);
    const __m512 vi2 = _mm512_loadu_ps(input + 32);
    const __m512 vi3 = _mm512_loadu_ps(input + 48);
    const __m512 vi4 = _mm512_loadu_ps(input + 64);
    const __m512 vi5 = _mm512_loadu_ps(input + 80);
    const __m512 vi6 = _mm512_loadu_ps(input + 96);
    const __m512 vi7 = _mm512_loadu_ps(input + 112);
    input += 128;

    const __m512 vx0 = _mm512_sub_ps(vi0, vi_max);
    const __m512 vx1 = _mm512_sub_ps(vi1, vi_max);
    const __m512 vx2 = _mm512_sub_ps(vi2, vi_max);
    const __m512 vx3 = _mm512_sub_ps(vi3, vi_max);
    const __m512 vx4 = _mm512_sub_ps(vi4, vi_max);
    const __m512 vx5 = _mm512_sub_ps(vi5, vi_max);
    const __m512 vx6 = _mm512_sub_ps(vi6, vi_max);
    const __m512 vx7 = _mm512_sub_ps(vi7, vi_max);

    const __m512 vn0 = _mm512_roundscale_ps(_mm512_mul_ps(vx0, vlog2e), 0);
    const __m512 vn1 = _mm512_roundscale_ps(_mm512_mul_ps(vx1, vlog2e), 0);
    const __m512 vn2 = _mm512_roundscale_ps(_mm512_mul_ps(vx2, vlog2e), 0);
    const __m512 vn3 = _mm512_roundscale_ps(_mm512_mul_ps(vx3, vlog2e), 0);
    const __m512 vn4 = _mm512_roundscale_ps(_mm512_mul_ps(vx4, vlog2e), 0);
    const __m512 vn5 = _mm512_roundscale_ps(_mm512_mul_ps(vx5, vlog2e), 0);
    const __m512 vn6 = _mm512_roundscale_ps(_mm512_mul_ps(vx6, vlog2e), 0);
    const __m512 vn7 = _mm512_roundscale_ps(_mm512_mul_ps(vx7, vlog2e), 0);

    const __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vx0);
    const __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vx1);
    const __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vx2);
    const __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2, vx3);
    const __m512 vt4 = _mm512_fmadd_ps(vn4, vminus_ln2, vx4);
    const __m512 vt5 = _mm512_fmadd_ps(vn5, vminus_ln2, vx5);
    const __m512 vt6 = _mm512_fmadd_ps(vn6, vminus_ln2, vx6);
    const __m512 vt7 = _mm512_fmadd_ps(vn7, vminus_ln2, vx7);

    __m512 vp0 = _mm512_fmadd_ps(vc5, vt0, vc4);
    __m512 vp1 = _mm512_fmadd_ps(vc5, vt1, vc4);
    __m512 vp2 = _mm512_fmadd_ps(vc5, vt2, vc4);
    __m512 vp3 = _mm512_fmadd_ps(vc5, vt3, vc4);
    __m512 vp4 = _mm512_fmadd_ps(vc5, vt4, vc4);
    __m512 vp5 = _mm512_fmadd_ps(vc5, vt5, vc4);
    __m512 vp6 = _mm512_fmadd_ps(vc5, vt6, vc4);
    __m512 vp7 = _mm512_fmadd_ps(vc5, vt7, vc4);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc3);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc3);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc2);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc2);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc1);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc1);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc1);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc1);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc1);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc1);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc1);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc1);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc0);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc0);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc0);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc0);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc0);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc0);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc0);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc0);

    const __m512 vf0 = _mm512_scalef_ps(vp0, vn0);
    const __m512 vf1 = _mm512_scalef_ps(vp1, vn1);
    const __m512 vf2 = _mm512_scalef_ps(vp2, vn2);
    const __m512 vf3 = _mm512_scalef_ps(vp3, vn3);
    const __m512 vf4 = _mm512_scalef_ps(vp4, vn4);
    const __m512 vf5 = _mm512_scalef_ps(vp5, vn5);
    const __m512 vf6 = _mm512_scalef_ps(vp6, vn6);
    const __m512 vf7 = _mm512_scalef_ps(vp7, vn7);

    _mm512_storeu_ps(output, vf0);
    _mm512_storeu_ps(output + 16, vf1);
    _mm512_storeu_ps(output + 32, vf2);
    _mm512_storeu_ps(output + 48, vf3);
    _mm512_storeu_ps(output + 64, vf4);
    _mm512_storeu_ps(output + 80, vf5);
    _mm512_storeu_ps(output + 96, vf6);
    _mm512_storeu_ps(output + 112, vf7);
    output += 128;

    vacc0 = _mm512_add_ps(vacc0, vf0);
    vacc1 = _mm512_add_ps(vacc1, vf1);
    vacc2 = _mm512_add_ps(vacc2, vf2);
    vacc3 = _mm512_add_ps(vacc3, vf3);
    vacc0 = _mm512_add_ps(vacc0, vf4);
    vacc1 = _mm512_add_ps(vacc1, vf5);
    vacc2 = _mm512_add_ps(vacc2, vf6);
    vacc3 = _mm512_add_ps(vacc3, vf7);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  vacc2 = _mm512_add_ps(vacc2, vacc3);
  vacc0 = _mm512_add_ps(vacc0, vacc2);

  __m512 vacc = vacc0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vi = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vx = _mm512_sub_ps(vi, vi_max);

    const __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vx, vlog2e), 0);

    const __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vx);

    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vc0);

    const __m512 vf = _mm512_scalef_ps(vp, vn);

    _mm512_storeu_ps(output, vf);
    output += 16;

    vacc = _mm512_add_ps(vacc, vf);
  }
  if (batch != 0) {
    // Prepare mask for valid 32-bit batch (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vi = _mm512_maskz_loadu_ps(vmask, input);

    const __m512 vx = _mm512_sub_ps(vi, vi_max);

    const __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vx, vlog2e), 0);

    const __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vx);

    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vc0);

    const __m512 vf = _mm512_scalef_ps(vp, vn);

    _mm512_mask_storeu_ps(output, vmask, vf);

    vacc = _mm512_mask_add_ps(vacc, vmask, vacc, vf);
  }
  *sum = _mm512_reduce_add_ps(vacc);
}
