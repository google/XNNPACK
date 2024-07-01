// Auto-generated file. Do not edit!
//   Template: src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vscaleexpminusmax.h"


void xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u16(
    size_t batch,
    const float* input,
    float* output,
    float scale,
    float max)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(0x1.05C61p-29f);

  const __m512 vc0 = _mm512_set1_ps(1.0f);
  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);

  const __m512 vscale = _mm512_set1_ps(scale);
  const __m512 vi_max = _mm512_set1_ps(max);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    // Load 16 (1x16) inputs at a time.
    const __m512 vi0 = _mm512_loadu_ps(input);
    input += 16;

    // Subtract maximum input x := i - i_max.
    const __m512 vx0 = _mm512_sub_ps(vi0, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m512 vn0 = _mm512_roundscale_ps(_mm512_mul_ps(vx0, vlog2e), 0);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_hi, vx0);

    vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_lo, vt0);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp0 = _mm512_fmadd_ps(vc5, vt0, vc4);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc1);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc0);

    // Reconstruct the final f value:
    //   f = 2**batch * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = 2**batch * p
    __m512 vf0 = _mm512_scalef_ps(vp0, vn0);

    // Multiply by scale.
    vf0 = _mm512_mul_ps(vf0, vscale);

    // Store 16 (1x16) outputs at a time.
    _mm512_storeu_ps(output, vf0);
    _mm512_storeu_ps(output + 0, vf0);
    output += 16;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    // Load 16 inputs at a time.
    const __m512 vi = _mm512_loadu_ps(input);
    input += 16;

    // Subtract maximum input x := i - i_max.
    const __m512 vx = _mm512_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vx, vlog2e), 0);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vc0);

    // Reconstruct the final f value:
    //   f = 2**batch * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = 2**batch * p
    __m512 vf = _mm512_scalef_ps(vp, vn);

    // Multiply by scale.
    vf = _mm512_mul_ps(vf, vscale);

    // Store 16 outputs at a time.
    _mm512_storeu_ps(output, vf);
    output += 16;
  }
  if (batch != 0) {
    // Prepare mask for valid 32-bit batch (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    // Load up to 15 inputs at a time.
    const __m512 vi = _mm512_mask_loadu_ps(_mm512_undefined_ps(), vmask, input);

    // Subtract maximum input x := i - i_max.
    const __m512 vx = _mm512_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vx, vlog2e), 0);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vc0);

    // Reconstruct the final f value:
    //   f = 2**batch * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = 2**batch * p
    __m512 vf = _mm512_scalef_ps(vp, vn);

    // Multiply by scale.
    vf = _mm512_mul_ps(vf, vscale);

    // Store up to 15 outputs at a time.
    _mm512_mask_storeu_ps(output, vmask, vf);
  }
}
