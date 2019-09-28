/*
 * Copyright 2019 Google LLC
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/hswish.h>


void xnn_f32_hswish_ukernel__sse(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m128 vsixth = _mm_load_ps(params->sse.sixth);
  const __m128 vhalf = _mm_load_ps(params->sse.half);
  const __m128 vone = _mm_load_ps(params->sse.one);
  const __m128 vzero = _mm_setzero_ps();

  for (; n >= 16; n -= 16) {
    const __m128 vx = _mm_loadu_ps(x);
    x += 4;

    const __m128 vt = _mm_min_ps(_mm_max_ps(_mm_add_ps(_mm_mul_ps(vx, vsixth), vhalf), vzero), vone);
    const __m128 vy = _mm_mul_ps(vt, vx);

    _mm_storeu_ps(y, vy);
    y += 4;
  }
  if (n != 0) {
    const __m128 vx = _mm_loadu_ps(x);
    x += 4;

    const __m128 vt = _mm_min_ps(_mm_max_ps(_mm_add_ps(_mm_mul_ps(vx, vsixth), vhalf), vzero), vone);
    __m128 vy = _mm_mul_ps(vt, vx);

    if (n & 8) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & 4) {
      _mm_store_ss(y, vy);
    }
  }
}
