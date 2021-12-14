// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vlrelu_ukernel__sse_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m128 vslope = _mm_load_ps(params->sse.slope);
  const __m128 vzero = _mm_setzero_ps();
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(x);
    __m128 vx4567 = _mm_loadu_ps(x + 4);
    x += 8;

    __m128 vacc0123 = _mm_max_ps(_mm_setzero_ps(), vx0123);
    vx0123 = _mm_min_ps(vx0123, vzero);
    __m128 vacc4567 = _mm_max_ps(_mm_setzero_ps(), vx4567);
    vx4567 = _mm_min_ps(vx4567, vzero);

    vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vx0123, vslope));
    vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vx4567, vslope));

    _mm_storeu_ps(y, vacc0123);
    _mm_storeu_ps(y + 4, vacc4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    __m128 vx = _mm_loadu_ps(x);
    x += 4;

    __m128 vacc = _mm_max_ps(_mm_setzero_ps(), vx);
    vx = _mm_min_ps(vx, vzero);
    vacc = _mm_add_ps(vacc, _mm_mul_ps(vx, vslope));

    _mm_storeu_ps(y, vacc);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    __m128 vx = _mm_loadu_ps(x);

    __m128 vacc = _mm_max_ps(_mm_setzero_ps(), vx);
    vx = _mm_min_ps(vx, vzero);
    vacc = _mm_add_ps(vacc, _mm_mul_ps(vx, vslope));

    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vacc);
    }
  }
}
