// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/packx.h>


void xnn_x32_packx_ukernel_4x__sse(
    size_t m,
    size_t k,
    const uint32_t* restrict x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(k != 0);

  const float* x0 = (const float*) x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (m < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (m <= 2) {
    x2 = x1;
  }
  const float* x3 = (const float*) ((uintptr_t) x2 + x_stride);
  if (m != 4) {
    x3 = x2;
  }

  float* restrict y_f32 = (float*) y;

  for (; k >= 4; k -= 4) {
    const __m128 vx0 = _mm_loadu_ps(x0);
    x0 += 4;
    const __m128 vx1 = _mm_loadu_ps(x1);
    x1 += 4;
    const __m128 vx2 = _mm_loadu_ps(x2);
    x2 += 4;
    const __m128 vx3 = _mm_loadu_ps(x3);
    x3 += 4;

    const __m128 vt0 = _mm_unpacklo_ps(vx0, vx1);
    const __m128 vt1 = _mm_unpackhi_ps(vx0, vx1);
    const __m128 vt2 = _mm_unpacklo_ps(vx2, vx3);
    const __m128 vt3 = _mm_unpackhi_ps(vx2, vx3);

    const __m128 vy0 = _mm_movelh_ps(vt0, vt2);
    _mm_store_ps(y_f32, vy0);

    const __m128 vy1 = _mm_movehl_ps(vt2, vt0);
    _mm_store_ps(y_f32 + 4, vy1);

    const __m128 vy2 = _mm_movelh_ps(vt1, vt3);
    _mm_store_ps(y_f32 + 8, vy2);

    const __m128 vy3 = _mm_movehl_ps(vt3, vt1);
    _mm_store_ps(y_f32 + 12, vy3);

    y_f32 += 16;
  }
  if XNN_UNLIKELY(k != 0) {
    do {
      const __m128 vx0 = _mm_load_ss(x0);
      x0 += 1;
      const __m128 vx1 = _mm_load_ss(x1);
      x1 += 1;
      const __m128 vx2 = _mm_load_ss(x2);
      x2 += 1;
      const __m128 vx3 = _mm_load_ss(x3);
      x3 += 1;

      const __m128 vx01 = _mm_unpacklo_ps(vx0, vx1);
      const __m128 vx23 = _mm_unpacklo_ps(vx2, vx3);
      const __m128 vy = _mm_movelh_ps(vx01, vx23);

      _mm_store_ps(y_f32, vy);
      y_f32 += 4;
    } while (--k != 0);
  }
}
