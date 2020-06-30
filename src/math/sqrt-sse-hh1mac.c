// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xmmintrin.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sqrt__sse_hh1mac(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const __m128 vc1875 = _mm_set1_ps(1.875f);
  const __m128 vc0375 = _mm_set1_ps(0.375f);
  const __m128 vc1250 = _mm_set1_ps(1.250f);
  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    // Initial approximation
    __m128 vrsqrtx = _mm_rsqrt_ps(vx);

    // Householder (order 2) iteration:
    //   rsqrt_x <- rsqrt_x * (1.875 + t * (0.375 * t - 1.25)) where t = x * rsqrt_x * rsqrt_x
    // Note: half_x * (rsqrt_x * rsqrt_x) is less accurate than (half_x * rsqrt_x) * rsqrt_x
    const __m128 vt = _mm_mul_ps(_mm_mul_ps(vx, vrsqrtx), vrsqrtx);
    vrsqrtx = _mm_mul_ps(vrsqrtx, _mm_add_ps(_mm_mul_ps(vt, _mm_sub_ps(_mm_mul_ps(vt, vc0375), vc1250)), vc1875));

    // Reconstruct sqrt(x) = rsqrt(x) * x
    const __m128 vy = _mm_mul_ps(vrsqrtx, vx);

    _mm_store_ps(output, vy);
    output += 4;
  }
}
