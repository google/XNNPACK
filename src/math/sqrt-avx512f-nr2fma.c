// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sqrt__avx512f_nr2fma(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (16 * sizeof(float)) == 0);

  const __m512 vhalf = _mm512_set1_ps(0.5f);
  for (; n != 0; n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_load_ps(input);
    input += 16;

    // Initial approximation
    const __m512 vrsqrtx = _mm512_rsqrt14_ps(vx);
    __m512 vsqrtx = _mm512_mul_ps(vrsqrtx, vx);
    __m512 vhalfrsqrtx = _mm512_mul_ps(vrsqrtx, vhalf);

    // Netwon-Raphson iteration:
    //   residual   <- 0.5 - sqrtx * halfrsqrtx
    //   halfrsqrtx <- halfrsqrtx + halfrsqrtx * residual
    //   sqrtx      <- sqrtx + sqrtx * residual
    __m512 vresidual = _mm512_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vhalfrsqrtx = _mm512_fmadd_ps(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = _mm512_fmadd_ps(vsqrtx, vresidual, vsqrtx);

    vresidual = _mm512_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vsqrtx = _mm512_fmadd_ps(vsqrtx, vresidual, vsqrtx);

    const __m512 vy = vsqrtx;

    _mm512_store_ps(output, vy);
    output += 16;
  }
}
