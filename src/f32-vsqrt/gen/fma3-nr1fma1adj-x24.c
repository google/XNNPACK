// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/fma3-nr1fma1adj.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x24(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vhalf = _mm256_load_ps(params->fma.half);
  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    const __m256 vx2 = _mm256_loadu_ps(input + 16);
    input += 24;

    const __m256 vrsqrtx0 = _mm256_rsqrt_ps(vx0);
    const __m256 vrsqrtx1 = _mm256_rsqrt_ps(vx1);
    const __m256 vrsqrtx2 = _mm256_rsqrt_ps(vx2);

    __m256 vsqrtx0 = _mm256_mul_ps(vrsqrtx0, vx0);
    __m256 vhalfrsqrtx0 = _mm256_mul_ps(vrsqrtx0, vhalf);
    __m256 vsqrtx1 = _mm256_mul_ps(vrsqrtx1, vx1);
    __m256 vhalfrsqrtx1 = _mm256_mul_ps(vrsqrtx1, vhalf);
    __m256 vsqrtx2 = _mm256_mul_ps(vrsqrtx2, vx2);
    __m256 vhalfrsqrtx2 = _mm256_mul_ps(vrsqrtx2, vhalf);

    const __m256 vresidual0 = _mm256_fnmadd_ps(vsqrtx0, vhalfrsqrtx0, vhalf);
    const __m256 vresidual1 = _mm256_fnmadd_ps(vsqrtx1, vhalfrsqrtx1, vhalf);
    const __m256 vresidual2 = _mm256_fnmadd_ps(vsqrtx2, vhalfrsqrtx2, vhalf);

    vhalfrsqrtx0 = _mm256_fmadd_ps(vhalfrsqrtx0, vresidual0, vhalfrsqrtx0);
    vsqrtx0 = _mm256_fmadd_ps(vsqrtx0, vresidual0, vsqrtx0);
    vhalfrsqrtx1 = _mm256_fmadd_ps(vhalfrsqrtx1, vresidual1, vhalfrsqrtx1);
    vsqrtx1 = _mm256_fmadd_ps(vsqrtx1, vresidual1, vsqrtx1);
    vhalfrsqrtx2 = _mm256_fmadd_ps(vhalfrsqrtx2, vresidual2, vhalfrsqrtx2);
    vsqrtx2 = _mm256_fmadd_ps(vsqrtx2, vresidual2, vsqrtx2);

    const __m256 vadjustment0 = _mm256_fnmadd_ps(vsqrtx0, vsqrtx0, vx0);
    const __m256 vadjustment1 = _mm256_fnmadd_ps(vsqrtx1, vsqrtx1, vx1);
    const __m256 vadjustment2 = _mm256_fnmadd_ps(vsqrtx2, vsqrtx2, vx2);

    const __m256 vy0 = _mm256_fmadd_ps(vhalfrsqrtx0, vadjustment0, vsqrtx0);
    const __m256 vy1 = _mm256_fmadd_ps(vhalfrsqrtx1, vadjustment1, vsqrtx1);
    const __m256 vy2 = _mm256_fmadd_ps(vhalfrsqrtx2, vadjustment2, vsqrtx2);

    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    _mm256_storeu_ps(output + 16, vy2);
    output += 24;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vrsqrtx = _mm256_rsqrt_ps(vx);
    __m256 vsqrtx = _mm256_mul_ps(vrsqrtx, vx);
    __m256 vhalfrsqrtx = _mm256_mul_ps(vrsqrtx, vhalf);
    const __m256 vresidual = _mm256_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm256_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm256_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m256 vadjustment = _mm256_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m256 vy = _mm256_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->fma.mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    const __m256 vrsqrtx = _mm256_rsqrt_ps(vx);
    __m256 vsqrtx = _mm256_mul_ps(vrsqrtx, vx);
    __m256 vhalfrsqrtx = _mm256_mul_ps(vrsqrtx, vhalf);
    const __m256 vresidual = _mm256_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm256_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm256_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m256 vadjustment = _mm256_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m256 vy = _mm256_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}
