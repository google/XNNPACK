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


void xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x96(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vhalf = _mm512_set1_ps(params->avx512.half);
  for (; batch >= 96 * sizeof(float); batch -= 96 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    const __m512 vx2 = _mm512_loadu_ps(input + 32);
    const __m512 vx3 = _mm512_loadu_ps(input + 48);
    const __m512 vx4 = _mm512_loadu_ps(input + 64);
    const __m512 vx5 = _mm512_loadu_ps(input + 80);
    input += 96;

    const __m512 vrsqrtx0 = _mm512_rsqrt14_ps(vx0);
    const __m512 vrsqrtx1 = _mm512_rsqrt14_ps(vx1);
    const __m512 vrsqrtx2 = _mm512_rsqrt14_ps(vx2);
    const __m512 vrsqrtx3 = _mm512_rsqrt14_ps(vx3);
    const __m512 vrsqrtx4 = _mm512_rsqrt14_ps(vx4);
    const __m512 vrsqrtx5 = _mm512_rsqrt14_ps(vx5);

    __m512 vsqrtx0 = _mm512_mul_ps(vrsqrtx0, vx0);
    __m512 vhalfrsqrtx0 = _mm512_mul_ps(vrsqrtx0, vhalf);
    __m512 vsqrtx1 = _mm512_mul_ps(vrsqrtx1, vx1);
    __m512 vhalfrsqrtx1 = _mm512_mul_ps(vrsqrtx1, vhalf);
    __m512 vsqrtx2 = _mm512_mul_ps(vrsqrtx2, vx2);
    __m512 vhalfrsqrtx2 = _mm512_mul_ps(vrsqrtx2, vhalf);
    __m512 vsqrtx3 = _mm512_mul_ps(vrsqrtx3, vx3);
    __m512 vhalfrsqrtx3 = _mm512_mul_ps(vrsqrtx3, vhalf);
    __m512 vsqrtx4 = _mm512_mul_ps(vrsqrtx4, vx4);
    __m512 vhalfrsqrtx4 = _mm512_mul_ps(vrsqrtx4, vhalf);
    __m512 vsqrtx5 = _mm512_mul_ps(vrsqrtx5, vx5);
    __m512 vhalfrsqrtx5 = _mm512_mul_ps(vrsqrtx5, vhalf);

    const __m512 vresidual0 = _mm512_fnmadd_ps(vsqrtx0, vhalfrsqrtx0, vhalf);
    const __m512 vresidual1 = _mm512_fnmadd_ps(vsqrtx1, vhalfrsqrtx1, vhalf);
    const __m512 vresidual2 = _mm512_fnmadd_ps(vsqrtx2, vhalfrsqrtx2, vhalf);
    const __m512 vresidual3 = _mm512_fnmadd_ps(vsqrtx3, vhalfrsqrtx3, vhalf);
    const __m512 vresidual4 = _mm512_fnmadd_ps(vsqrtx4, vhalfrsqrtx4, vhalf);
    const __m512 vresidual5 = _mm512_fnmadd_ps(vsqrtx5, vhalfrsqrtx5, vhalf);

    vhalfrsqrtx0 = _mm512_fmadd_ps(vhalfrsqrtx0, vresidual0, vhalfrsqrtx0);
    vsqrtx0 = _mm512_fmadd_ps(vsqrtx0, vresidual0, vsqrtx0);
    vhalfrsqrtx1 = _mm512_fmadd_ps(vhalfrsqrtx1, vresidual1, vhalfrsqrtx1);
    vsqrtx1 = _mm512_fmadd_ps(vsqrtx1, vresidual1, vsqrtx1);
    vhalfrsqrtx2 = _mm512_fmadd_ps(vhalfrsqrtx2, vresidual2, vhalfrsqrtx2);
    vsqrtx2 = _mm512_fmadd_ps(vsqrtx2, vresidual2, vsqrtx2);
    vhalfrsqrtx3 = _mm512_fmadd_ps(vhalfrsqrtx3, vresidual3, vhalfrsqrtx3);
    vsqrtx3 = _mm512_fmadd_ps(vsqrtx3, vresidual3, vsqrtx3);
    vhalfrsqrtx4 = _mm512_fmadd_ps(vhalfrsqrtx4, vresidual4, vhalfrsqrtx4);
    vsqrtx4 = _mm512_fmadd_ps(vsqrtx4, vresidual4, vsqrtx4);
    vhalfrsqrtx5 = _mm512_fmadd_ps(vhalfrsqrtx5, vresidual5, vhalfrsqrtx5);
    vsqrtx5 = _mm512_fmadd_ps(vsqrtx5, vresidual5, vsqrtx5);

    const __m512 vadjustment0 = _mm512_fnmadd_ps(vsqrtx0, vsqrtx0, vx0);
    const __m512 vadjustment1 = _mm512_fnmadd_ps(vsqrtx1, vsqrtx1, vx1);
    const __m512 vadjustment2 = _mm512_fnmadd_ps(vsqrtx2, vsqrtx2, vx2);
    const __m512 vadjustment3 = _mm512_fnmadd_ps(vsqrtx3, vsqrtx3, vx3);
    const __m512 vadjustment4 = _mm512_fnmadd_ps(vsqrtx4, vsqrtx4, vx4);
    const __m512 vadjustment5 = _mm512_fnmadd_ps(vsqrtx5, vsqrtx5, vx5);

    const __m512 vy0 = _mm512_fmadd_ps(vhalfrsqrtx0, vadjustment0, vsqrtx0);
    const __m512 vy1 = _mm512_fmadd_ps(vhalfrsqrtx1, vadjustment1, vsqrtx1);
    const __m512 vy2 = _mm512_fmadd_ps(vhalfrsqrtx2, vadjustment2, vsqrtx2);
    const __m512 vy3 = _mm512_fmadd_ps(vhalfrsqrtx3, vadjustment3, vsqrtx3);
    const __m512 vy4 = _mm512_fmadd_ps(vhalfrsqrtx4, vadjustment4, vsqrtx4);
    const __m512 vy5 = _mm512_fmadd_ps(vhalfrsqrtx5, vadjustment5, vsqrtx5);

    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    _mm512_storeu_ps(output + 48, vy3);
    _mm512_storeu_ps(output + 64, vy4);
    _mm512_storeu_ps(output + 80, vy5);
    output += 96;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vrsqrtx = _mm512_rsqrt14_ps(vx);
    __m512 vsqrtx = _mm512_mul_ps(vrsqrtx, vx);
    __m512 vhalfrsqrtx = _mm512_mul_ps(vrsqrtx, vhalf);
    const __m512 vresidual = _mm512_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm512_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm512_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m512 vadjustment = _mm512_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m512 vy = _mm512_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vrsqrtx = _mm512_maskz_rsqrt14_ps(vmask, vx);
    __m512 vsqrtx = _mm512_mul_ps(vrsqrtx, vx);
    __m512 vhalfrsqrtx = _mm512_mul_ps(vrsqrtx, vhalf);
    const __m512 vresidual = _mm512_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm512_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm512_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m512 vadjustment = _mm512_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m512 vy = _mm512_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
