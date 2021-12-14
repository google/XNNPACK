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


static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x32(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m256 vhalf = _mm256_broadcast_ss(&params->fma.half);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(x);
    const __m256 vx1 = _mm256_loadu_ps(x + 8);
    const __m256 vx2 = _mm256_loadu_ps(x + 16);
    const __m256 vx3 = _mm256_loadu_ps(x + 24);
    x += 32;

    const __m256 vrsqrtx0 = _mm256_rsqrt_ps(vx0);
    const __m256 vrsqrtx1 = _mm256_rsqrt_ps(vx1);
    const __m256 vrsqrtx2 = _mm256_rsqrt_ps(vx2);
    const __m256 vrsqrtx3 = _mm256_rsqrt_ps(vx3);

    __m256 vsqrtx0 = _mm256_mul_ps(vrsqrtx0, vx0);
    __m256 vhalfrsqrtx0 = _mm256_mul_ps(vrsqrtx0, vhalf);
    __m256 vsqrtx1 = _mm256_mul_ps(vrsqrtx1, vx1);
    __m256 vhalfrsqrtx1 = _mm256_mul_ps(vrsqrtx1, vhalf);
    __m256 vsqrtx2 = _mm256_mul_ps(vrsqrtx2, vx2);
    __m256 vhalfrsqrtx2 = _mm256_mul_ps(vrsqrtx2, vhalf);
    __m256 vsqrtx3 = _mm256_mul_ps(vrsqrtx3, vx3);
    __m256 vhalfrsqrtx3 = _mm256_mul_ps(vrsqrtx3, vhalf);

    const __m256 vresidual0 = _mm256_fnmadd_ps(vsqrtx0, vhalfrsqrtx0, vhalf);
    const __m256 vresidual1 = _mm256_fnmadd_ps(vsqrtx1, vhalfrsqrtx1, vhalf);
    const __m256 vresidual2 = _mm256_fnmadd_ps(vsqrtx2, vhalfrsqrtx2, vhalf);
    const __m256 vresidual3 = _mm256_fnmadd_ps(vsqrtx3, vhalfrsqrtx3, vhalf);

    vhalfrsqrtx0 = _mm256_fmadd_ps(vhalfrsqrtx0, vresidual0, vhalfrsqrtx0);
    vsqrtx0 = _mm256_fmadd_ps(vsqrtx0, vresidual0, vsqrtx0);
    vhalfrsqrtx1 = _mm256_fmadd_ps(vhalfrsqrtx1, vresidual1, vhalfrsqrtx1);
    vsqrtx1 = _mm256_fmadd_ps(vsqrtx1, vresidual1, vsqrtx1);
    vhalfrsqrtx2 = _mm256_fmadd_ps(vhalfrsqrtx2, vresidual2, vhalfrsqrtx2);
    vsqrtx2 = _mm256_fmadd_ps(vsqrtx2, vresidual2, vsqrtx2);
    vhalfrsqrtx3 = _mm256_fmadd_ps(vhalfrsqrtx3, vresidual3, vhalfrsqrtx3);
    vsqrtx3 = _mm256_fmadd_ps(vsqrtx3, vresidual3, vsqrtx3);

    const __m256 vadjustment0 = _mm256_fnmadd_ps(vsqrtx0, vsqrtx0, vx0);
    const __m256 vadjustment1 = _mm256_fnmadd_ps(vsqrtx1, vsqrtx1, vx1);
    const __m256 vadjustment2 = _mm256_fnmadd_ps(vsqrtx2, vsqrtx2, vx2);
    const __m256 vadjustment3 = _mm256_fnmadd_ps(vsqrtx3, vsqrtx3, vx3);

    const __m256 vy0 = _mm256_fmadd_ps(vhalfrsqrtx0, vadjustment0, vsqrtx0);
    const __m256 vy1 = _mm256_fmadd_ps(vhalfrsqrtx1, vadjustment1, vsqrtx1);
    const __m256 vy2 = _mm256_fmadd_ps(vhalfrsqrtx2, vadjustment2, vsqrtx2);
    const __m256 vy3 = _mm256_fmadd_ps(vhalfrsqrtx3, vadjustment3, vsqrtx3);

    _mm256_storeu_ps(y, vy0);
    _mm256_storeu_ps(y + 8, vy1);
    _mm256_storeu_ps(y + 16, vy2);
    _mm256_storeu_ps(y + 24, vy3);
    y += 32;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vrsqrtx = _mm256_rsqrt_ps(vx);
    __m256 vsqrtx = _mm256_mul_ps(vrsqrtx, vx);
    __m256 vhalfrsqrtx = _mm256_mul_ps(vrsqrtx, vhalf);
    const __m256 vresidual = _mm256_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm256_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm256_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m256 vadjustment = _mm256_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m256 vy = _mm256_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);

    const __m256 vrsqrtx = _mm256_rsqrt_ps(vx);
    __m256 vsqrtx = _mm256_mul_ps(vrsqrtx, vx);
    __m256 vhalfrsqrtx = _mm256_mul_ps(vrsqrtx, vhalf);
    const __m256 vresidual = _mm256_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm256_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm256_fmadd_ps(vsqrtx, vresidual, vsqrtx);
    const __m256 vadjustment = _mm256_fnmadd_ps(vsqrtx, vsqrtx, vx);
    const __m256 vy = _mm256_fmadd_ps(vhalfrsqrtx, vadjustment, vsqrtx);

    #if XNN_COMPILER_HAS_FEATURE(memory_sanitizer)
      __m128 vy_lo = _mm256_castps256_ps128(vy);
      if (n & (4 * sizeof(float))) {
        _mm_storeu_ps(y, vy_lo);
        vy_lo = _mm256_extractf128_ps(vy, 1);
        y += 4;
      }
      if (n & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) y, vy_lo);
        vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
        y += 2;
      }
      if (n & (1 * sizeof(float))) {
        _mm_store_ss(y, vy_lo);
      }
    #else
      // Triggers spurious MSan failures in the calling code.
      _mm256_maskstore_ps(y, vmask, vy);
    #endif
  }
}
