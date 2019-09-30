// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__sse(
    size_t n,
    const float* x,
    float* y)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  __m128 vmax0 = _mm_load_ss(x);
  vmax0 = _mm_shuffle_ps(vmax0, vmax0, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 vmax1 = vmax0;
  __m128 vmax2 = vmax0;
  __m128 vmax3 = vmax0;
  for (; n >= 64; n -= 64) {
    const __m128 vx0 = _mm_loadu_ps(x);
    const __m128 vx1 = _mm_loadu_ps(x + 4);
    const __m128 vx2 = _mm_loadu_ps(x + 8);
    const __m128 vx3 = _mm_loadu_ps(x + 12);
    x += 16;

    vmax0 = _mm_max_ps(vmax0, vx0);
    vmax1 = _mm_max_ps(vmax1, vx1);
    vmax2 = _mm_max_ps(vmax2, vx2);
    vmax3 = _mm_max_ps(vmax3, vx3);
  }
  __m128 vmax = _mm_max_ps(_mm_max_ps(vmax0, vmax1), _mm_max_ps(vmax2, vmax3));
  for (; n >= 16; n -= 16) {
    const __m128 vx = _mm_loadu_ps(x);
    vmax = _mm_max_ps(vmax, vx);
    x += 4;
  }
  __m128 vmax_lo = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vmax_lo = _mm_max_ss(vmax_lo, _mm_shuffle_ps(vmax_lo, vmax_lo, _MM_SHUFFLE(3, 3, 1, 1)));
  if XNN_UNLIKELY(n != 0) {
    do {
      vmax_lo = _mm_max_ss(vmax_lo, _mm_load_ss(x));
      x += 1;
      n -= 4;
    } while (n != 0);
  }
  _mm_store_ss(y, vmax_lo);
}
