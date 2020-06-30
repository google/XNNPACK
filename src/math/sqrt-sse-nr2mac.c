// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xmmintrin.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sqrt__sse_nr2mac(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const __m128 vthree_halfs = _mm_set1_ps(1.5f);
  const __m128 vhalf = _mm_set1_ps(0.5f);
  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // Initial approximation
    __m128 vrsqrtx = _mm_rsqrt_ps(vx);
    const __m128 vhalfx = _mm_mul_ps(vx, vhalf);

    // Netwon-Raphson iteration: rsqrt_x <- rsqrt_x * (3/2 - x/2 * rsqrt_x * rsqrt_x)
    // Note: half_x * (rsqrt_x * rsqrt_x) is less accurate than (half_x * rsqrt_x) * rsqrt_x
    vrsqrtx = _mm_mul_ps(vrsqrtx, _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(vhalfx, vrsqrtx), vrsqrtx), vthree_halfs));
    vrsqrtx = _mm_mul_ps(vrsqrtx, _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(vhalfx, vrsqrtx), vrsqrtx), vthree_halfs));

    // Reconstruct sqrt(x) = rsqrt(x) * x
    const __m128 vy = _mm_mul_ps(vrsqrtx, vx);

    _mm_store_ps(output, vy);
    output += 4;
  }
}
