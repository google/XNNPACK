// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sqrt__fma3_nr1fma(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const __m256 vhalf = _mm256_set1_ps(0.5f);
  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_load_ps(input);
    input += 8;

    // Initial approximation
    const __m256 vrsqrtx = _mm256_rsqrt_ps(vx);
    __m256 vsqrtx = _mm256_mul_ps(vrsqrtx, vx);
    const __m256 vhalfrsqrtx = _mm256_mul_ps(vrsqrtx, vhalf);

    // Netwon-Raphson iteration:
    //   residual   <- 0.5 - sqrtx * halfrsqrtx
    //   sqrtx      <- sqrtx + sqrtx * residual
    const __m256 vresidual = _mm256_fnmadd_ps(vsqrtx, vhalfrsqrtx, vhalf);
    vsqrtx = _mm256_fmadd_ps(vsqrtx, vresidual, vsqrtx);

    const __m256 vy = vsqrtx;

    _mm256_store_ps(output, vy);
    output += 8;
  }
}
