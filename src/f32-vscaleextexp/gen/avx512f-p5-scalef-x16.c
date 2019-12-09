// Auto-generated file. Do not edit!
//   Template: src/f32-vscaleextexp/avx512f-p5-scalef.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vscaleextexp.h>


void xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x16(
    size_t elements,
    const float* x,
    float* y,
    float scale_value,
    float scale_exp)
{
  assert(elements % sizeof(float) == 0);

  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p+0f);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(0x1.05C61p-29f);

  const __m512 vc0 = _mm512_set1_ps(1.0f);
  const __m512 vc1 = _mm512_set1_ps(0x1.FFFFF6p-1f);
  const __m512 vc2 = _mm512_set1_ps(0x1.FFFDC6p-2f);
  const __m512 vc3 = _mm512_set1_ps(0x1.555A80p-3f);
  const __m512 vc4 = _mm512_set1_ps(0x1.573A1Ap-5f);
  const __m512 vc5 = _mm512_set1_ps(0x1.0F9F9Cp-7f);

  const __m512 vscalev = _mm512_set1_ps(scale_value);
  const __m512 vscalee = _mm512_set1_ps(scale_exp);

  for (; elements >= 16 * sizeof(float); elements -= 16 * sizeof(float)) {
    // Load 16 (1x16) inputs at a time.
    const __m512 vx0 = _mm512_loadu_ps(x);
    x += 16;

    // Compute reduced argument elements := round(x / log(2)).
    const __m512 vn0 = _mm512_roundscale_ps(_mm512_mul_ps(vx0, vlog2e), 0);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_hi, vx0);

    vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_lo, vt0);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp0 = _mm512_fmadd_ps(vc5, vt0, vc4);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc1);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc0);

    // Multiply "extended" floating-point numbers in ("mantissa", "exponent") representation where
    //  - vnX is "exponent"
    //  - vpX is "mantissa"
    //   
    // exp2(ae) * av * exp2(be) * bv = 
    //   = exp2(ae + be) * (av * bv)
    __m512 vf0 = _mm512_mul_ps(vp0, vscalev);

    const __m512 ve0 = _mm512_add_ps(vn0, vscalee);

    // Multiply "mantissa" by the exp2("exponent").
    vf0 = _mm512_scalef_ps(vf0, ve0);

    // Store 128 (8x16) results at a time.
    _mm512_storeu_ps(y, vf0);
    _mm512_storeu_ps(y + 0, vf0);
    y += 16;
  }

  for (; elements >= 16 * sizeof(float); elements -= 16 * sizeof(float)) {
    // Load 16 inputs at a time.
    const __m512 vx = _mm512_loadu_ps(x);
    x += 16;

    // Compute reduced argument elements := round(x / log(2)).
    const __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vx, vlog2e), 0);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vc0);

    // Multiply "extended" floating-point numbers in ("mantissa", "exponent") representation.
    __m512 vf = _mm512_mul_ps(vp, vscalev);
    const __m512 ve = _mm512_add_ps(vn, vscalee);

    // Multiply "mantissa" by the exp2("exponent").
    vf = _mm512_scalef_ps(vf, ve);

    // Store 16 results at a time.
    _mm512_storeu_ps(y, vf);
    y += 16;
  }
  if XNN_UNLIKELY(elements != 0) {
    // Prepare mask for valid 32-bit elements (depends on elements).
    elements >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << elements) - UINT32_C(1)));

    // Load up to 15 inputs at a time.
    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);

    // Compute reduced argument elements := round(x / log(2)).
    const __m512 vn = _mm512_roundscale_ps(_mm512_mul_ps(vx, vlog2e), 0);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m512 vp = _mm512_fmadd_ps(vc5, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vc1);
    vp = _mm512_fmadd_ps(vp, vt, vc0);

    // Multiply "extended" floating-point numbers in ("mantissa", "exponent") representation.
    __m512 vf = _mm512_mul_ps(vp, vscalev);
    const __m512 ve = _mm512_add_ps(vn, vscalee);

    // Multiply "mantissa" by the exp2("exponent").
    vf = _mm512_scalef_ps(vf, ve);

    // Store up to 15 results at a time.
    _mm512_mask_storeu_ps(y, vmask, vf);
  }
  _mm256_zeroupper();
}
