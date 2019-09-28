/*
 * Copyright 2019 Google LLC
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vmul.h>


void xnn_f32_vmul_ukernel__sse(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m128 vy_min = _mm_load_ps(params->sse.min);
  const __m128 vy_max = _mm_load_ps(params->sse.max);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(a);
    const __m128 va1 = _mm_loadu_ps(a + 4);
    a += 8;

    const __m128 vb0 = _mm_loadu_ps(b);
    const __m128 vb1 = _mm_loadu_ps(b + 4);
    b += 8;

    const __m128 vacc0 = _mm_mul_ps(va0, vb0);
    const __m128 vacc1 = _mm_mul_ps(va1, vb1);
    const __m128 vy0 = _mm_min_ps(_mm_max_ps(vacc0, vy_min), vy_max);
    const __m128 vy1 = _mm_min_ps(_mm_max_ps(vacc1, vy_min), vy_max);

    _mm_storeu_ps(y, vy0);
    _mm_storeu_ps(y + 4, vy1);
    y += 8;
  }
  if (n >= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(a);
    a += 4;
    const __m128 vb = _mm_loadu_ps(b);
    b += 4;
    const __m128 vacc = _mm_mul_ps(va, vb);
    const __m128 vy = _mm_min_ps(_mm_max_ps(vacc, vy_min), vy_max);
    _mm_storeu_ps(y, vy);
    y += 4;
    n -= 4 * sizeof(float);
  }
  if (n != 0) {
    const __m128 va = _mm_loadu_ps(a);
    const __m128 vb = _mm_loadu_ps(b);
    const __m128 vacc = _mm_mul_ps(va, vb);
    __m128 vy = _mm_min_ps(_mm_max_ps(vacc, vy_min), vy_max);
    if (n & 2 * sizeof(float)) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & 1 * sizeof(float)) {
      _mm_store_ss(y, vy);
    }
  }
}
