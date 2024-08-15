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

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc4(
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

  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E430p-1f);
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vc0 = _mm512_set1_ps(1.0f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vc0);

  const __m512 vi_max = _mm512_set1_ps(*max);

  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  __m512 vacc2 = _mm512_setzero_ps();
  __m512 vacc3 = _mm512_setzero_ps();
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vi0 = _mm512_loadu_ps(input);
    const __m512 vi1 = _mm512_loadu_ps(input + 16);
    const __m512 vi2 = _mm512_loadu_ps(input + 32);
    const __m512 vi3 = _mm512_loadu_ps(input + 48);
    input += 64;

    const __m512 vx0 = _mm512_sub_ps(vi0, vi_max);
    const __m512 vx1 = _mm512_sub_ps(vi1, vi_max);
    const __m512 vx2 = _mm512_sub_ps(vi2, vi_max);
    const __m512 vx3 = _mm512_sub_ps(vi3, vi_max);

    const __m512 vn0 = _mm512_roundscale_ps(_mm512_mul_ps(vx0, vlog2e), 0);
    const __m512 vn1 = _mm512_roundscale_ps(_mm512_mul_ps(vx1, vlog2e), 0);
    const __m512 vn2 = _mm512_roundscale_ps(_mm512_mul_ps(vx2, vlog2e), 0);
    const __m512 vn3 = _mm512_roundscale_ps(_mm512_mul_ps(vx3, vlog2e), 0);

    const __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vx0);
    const __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vx1);
    const __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vx2);
    const __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2, vx3);

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

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc0);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc0);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc0);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc0);

    const __m512 vf0 = _mm512_scalef_ps(vp0, vn0);
    const __m512 vf1 = _mm512_scalef_ps(vp1, vn1);
    const __m512 vf2 = _mm512_scalef_ps(vp2, vn2);
    const __m512 vf3 = _mm512_scalef_ps(vp3, vn3);

    _mm512_storeu_ps(output, vf0);
    _mm512_storeu_ps(output + 16, vf1);
    _mm512_storeu_ps(output + 32, vf2);
    _mm512_storeu_ps(output + 48, vf3);
    output += 64;

    vacc0 = _mm512_add_ps(vacc0, vf0);
    vacc1 = _mm512_add_ps(vacc1, vf1);
    vacc2 = _mm512_add_ps(vacc2, vf2);
    vacc3 = _mm512_add_ps(vacc3, vf3);
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
