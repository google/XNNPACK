// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__sse(
    size_t batch,
    const float* input,
    float* output)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m128 vmax0 = _mm_load_ss(input);
  vmax0 = _mm_shuffle_ps(vmax0, vmax0, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 vmax1 = vmax0;
  __m128 vmax2 = vmax0;
  __m128 vmax3 = vmax0;
  for (; batch >= 64; batch -= 64) {
    const __m128 vx0 = _mm_loadu_ps(input);
    const __m128 vx1 = _mm_loadu_ps(input + 4);
    const __m128 vx2 = _mm_loadu_ps(input + 8);
    const __m128 vx3 = _mm_loadu_ps(input + 12);
    input += 16;

    vmax0 = _mm_max_ps(vmax0, vx0);
    vmax1 = _mm_max_ps(vmax1, vx1);
    vmax2 = _mm_max_ps(vmax2, vx2);
    vmax3 = _mm_max_ps(vmax3, vx3);
  }
  __m128 vmax = _mm_max_ps(_mm_max_ps(vmax0, vmax1), _mm_max_ps(vmax2, vmax3));
  for (; batch >= 16; batch -= 16) {
    const __m128 vx = _mm_loadu_ps(input);
    vmax = _mm_max_ps(vmax, vx);
    input += 4;
  }
  __m128 vmax_lo = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vmax_lo = _mm_max_ss(vmax_lo, _mm_shuffle_ps(vmax_lo, vmax_lo, _MM_SHUFFLE(3, 3, 1, 1)));
  if XNN_UNLIKELY(batch != 0) {
    do {
      vmax_lo = _mm_max_ss(vmax_lo, _mm_load_ss(input));
      input += 1;
      batch -= 4;
    } while (batch != 0);
  }
  _mm_store_ss(output, vmax_lo);
}
