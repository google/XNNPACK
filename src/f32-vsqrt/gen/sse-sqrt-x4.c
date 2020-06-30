// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/sse-sqrt.c.in
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


void xnn_f32_vsqrt_ukernel__sse_sqrt_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(x);
    x += 4;
    const __m128 vy = _mm_sqrt_ps(vx);
    _mm_storeu_ps(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128 vx = _mm_load_ss(x++);
      const __m128 vy = _mm_sqrt_ss(vx);
      _mm_store_ss(y++, vy);
      n -= sizeof(float);
    } while (n != 0);
  }
}
