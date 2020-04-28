// Auto-generated file. Do not edit!
//   Template: src/f32-hswish/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vbinary.h>


void xnn_f32_hswish_ukernel__sse_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m128 vsixth = _mm_load_ps(params->sse.sixth);
  const __m128 vhalf = _mm_load_ps(params->sse.half);
  const __m128 vone = _mm_load_ps(params->sse.one);
  const __m128 vzero = _mm_setzero_ps();

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(x);
    x += 4;

    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);

    vacc0123 = _mm_add_ps(vacc0123, vhalf);

    vacc0123 = _mm_max_ps(vacc0123, vzero);

    vacc0123 = _mm_min_ps(vacc0123, vone);

    vacc0123 = _mm_mul_ps(vacc0123, vx0123);

    _mm_storeu_ps(y, vacc0123);
    y += 4;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(x);
    x += 4;
    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);
    vacc0123 = _mm_add_ps(vacc0123, vhalf);
    vacc0123 = _mm_max_ps(vacc0123, vzero);
    vacc0123 = _mm_min_ps(vacc0123, vone);
    vacc0123 = _mm_mul_ps(vacc0123, vx0123);
    _mm_storeu_ps(y, vacc0123);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m128 vx0123 = _mm_loadu_ps_notsan(x);
    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);
    vacc0123 = _mm_add_ps(vacc0123, vhalf);
    vacc0123 = _mm_max_ps(vacc0123, vzero);
    vacc0123 = _mm_min_ps(vacc0123, vone);
    vacc0123 = _mm_mul_ps(vacc0123, vx0123);

    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vacc0123);
      vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vacc0123);
    }
  }
}
