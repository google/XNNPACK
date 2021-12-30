// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/avx512f-nr1fma1adj.c.in
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


void xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x32(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 vhalf = _mm512_set1_ps(params->avx512.half);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(x);
    const __m512 vx1 = _mm512_loadu_ps(x + 16);
    x += 32;

    const __m512 vrsqrtx0 = _mm512_rsqrt14_ps(vx0);
    const __m512 vrsqrtx1 = _mm512_rsqrt14_ps(vx1);

    __m512 vsqrtx0 = _mm512_mul_ps(vrsqrtx0, vx0);
    __m512 vhalfrsqrtx0 = _mm512_mul_ps(vrsqrtx0, vhalf);
    __m512 vsqrtx1 = _mm512_mul_ps(vrsqrtx1, vx1);
    __m512 vhalfrsqrtx1 = _mm512_mul_ps(vrsqrtx1, vhalf);

    const __m512 vresidual0 = _mm512_fnmadd_ps(vsqrtx0, vhalfrsqrtx0, vhalf);
    const __m512 vresidual1 = _mm512_fnmadd_ps(vsqrtx1, vhalfrsqrtx1, vhalf);

    vhalfrsqrtx0 = _mm512_fmadd_ps(vhalfrsqrtx0, vresidual0, vhalfrsqrtx0);
    vsqrtx0 = _mm512_fmadd_ps(vsqrtx0, vresidual0, vsqrtx0);
    vhalfrsqrtx1 = _mm512_fmadd_ps(vhalfrsqrtx1, vresidual1, vhalfrsqrtx1);
    vsqrtx1 = _mm512_fmadd_ps(vsqrtx1, vresidual1, vsqrtx1);

    const __m512 vadjustment0 = _mm512_fnmadd_ps(vsqrtx0, vsqrtx0, vx0);
    const __m512 vadjustment1 = _mm512_fnmadd_ps(vsqrtx1, vsqrtx1, vx1);

    const __m512 vy0 = _mm512_fmadd_ps(vhalfrsqrtx0, vadjustment0, vsqrtx0);
    const __m512 vy1 = _mm512_fmadd_ps(vhalfrsqrtx1, vadjustment1, vsqrtx1);

    _mm512_storeu_ps(y, vy0);
    _mm512_storeu_ps(y + 16, vy1);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vrsqrtx = _mm512_rsqrt14_ps(vx);
    __m512 vsqrtx = _mm512_mul_ps(vrsqrtx, vx);
    __m512 vhalfrsqrtx = _mm512_mul_ps(vrsqrtx, vhalf);
    const __m512 vresidual = _mm512_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm512_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm512_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m512 vadjustment = _mm512_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m512 vy = _mm512_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vrsqrtx = _mm512_maskz_rsqrt14_ps(vmask, vx);
    __m512 vsqrtx = _mm512_mul_ps(vrsqrtx, vx);
    __m512 vhalfrsqrtx = _mm512_mul_ps(vrsqrtx, vhalf);
    const __m512 vresidual = _mm512_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm512_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm512_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m512 vadjustment = _mm512_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m512 vy = _mm512_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}
