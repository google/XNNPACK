// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/clamp.h>


void xnn_f32_clamp_ukernel__sse(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m128 voutput_max = _mm_load_ps(params->sse.max);
  const __m128 voutput_min = _mm_load_ps(params->sse.min);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(x);
    x += 4;

    const __m128 vy = _mm_min_ps(_mm_max_ps(vx, voutput_min), voutput_max);

    _mm_storeu_ps(y, vy);
    y += 4;
  }
  if (n != 0) {
    const __m128 vx = _mm_loadu_ps(x);

    __m128 vy = _mm_min_ps(_mm_max_ps(vx, voutput_min), voutput_max);

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
