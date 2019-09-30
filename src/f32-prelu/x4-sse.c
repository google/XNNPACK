// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/prelu.h>


void xnn_f32_prelu_ukernel_x4__sse(
    size_t mr,
    size_t n,
    const float* x,
    size_t x_stride,
    const float* w,
    float* y,
    size_t y_stride,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float* x0 = x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (mr < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (mr <= 2) {
    x2 = x1;
  }
  const float* x3 = (const float*) ((uintptr_t) x2 + x_stride);
  if (mr != 4) {
    x3 = x2;
  }

  float* y0 = y;
  float* y1 = (float*) ((uintptr_t) y0 + y_stride);
  if (mr < 2) {
    y1 = y0;
  }
  float* y2 = (float*) ((uintptr_t) y1 + y_stride);
  if (mr <= 2) {
    y2 = y1;
  }
  float* y3 = (float*) ((uintptr_t) y2 + y_stride);
  if (mr != 4) {
    y3 = y2;
  }

  const __m128 vy_min = _mm_load_ps(params->sse.min);
  const __m128 vy_max = _mm_load_ps(params->sse.max);
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vw = _mm_loadu_ps(w);
    w += 4;
    const __m128 vx0 = _mm_loadu_ps(x0);
    x0 += 4;
    const __m128 vx1 = _mm_loadu_ps(x1);
    x1 += 4;
    const __m128 vx2 = _mm_loadu_ps(x2);
    x2 += 4;
    const __m128 vx3 = _mm_loadu_ps(x3);
    x3 += 4;

    const __m128 vwx0 = _mm_mul_ps(vx0, vw);
    const __m128 vwx1 = _mm_mul_ps(vx1, vw);
    const __m128 vwx2 = _mm_mul_ps(vx2, vw);
    const __m128 vwx3 = _mm_mul_ps(vx3, vw);

    const __m128i vmask0 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0));
    const __m128i vmask1 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx1));
    const __m128i vmask2 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx2));
    const __m128i vmask3 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx3));

    const __m128i vacc0 = _mm_or_si128(_mm_andnot_si128(vmask0, _mm_castps_si128(vx0)), _mm_and_si128(vmask0, _mm_castps_si128(vwx0)));
    const __m128i vacc1 = _mm_or_si128(_mm_andnot_si128(vmask1, _mm_castps_si128(vx1)), _mm_and_si128(vmask1, _mm_castps_si128(vwx1)));
    const __m128i vacc2 = _mm_or_si128(_mm_andnot_si128(vmask2, _mm_castps_si128(vx2)), _mm_and_si128(vmask2, _mm_castps_si128(vwx2)));
    const __m128i vacc3 = _mm_or_si128(_mm_andnot_si128(vmask3, _mm_castps_si128(vx3)), _mm_and_si128(vmask3, _mm_castps_si128(vwx3)));

    const __m128 vy0 = _mm_min_ps(_mm_max_ps(_mm_castsi128_ps(vacc0), vy_min), vy_max);
    const __m128 vy1 = _mm_min_ps(_mm_max_ps(_mm_castsi128_ps(vacc1), vy_min), vy_max);
    const __m128 vy2 = _mm_min_ps(_mm_max_ps(_mm_castsi128_ps(vacc2), vy_min), vy_max);
    const __m128 vy3 = _mm_min_ps(_mm_max_ps(_mm_castsi128_ps(vacc3), vy_min), vy_max);

    _mm_storeu_ps(y0, vy0);
    y0 += 4;
    _mm_storeu_ps(y1, vy1);
    y1 += 4;
    _mm_storeu_ps(y2, vy2);
    y2 += 4;
    _mm_storeu_ps(y3, vy3);
    y3 += 4;
  }
  if (n != 0) {
    const __m128 vw = _mm_loadu_ps(w);
    const __m128 vx0 = _mm_loadu_ps(x0);
    const __m128 vx1 = _mm_loadu_ps(x1);
    const __m128 vx2 = _mm_loadu_ps(x2);
    const __m128 vx3 = _mm_loadu_ps(x3);

    const __m128 vwx0 = _mm_mul_ps(vx0, vw);
    const __m128 vwx1 = _mm_mul_ps(vx1, vw);
    const __m128 vwx2 = _mm_mul_ps(vx2, vw);
    const __m128 vwx3 = _mm_mul_ps(vx3, vw);

    const __m128i vmask0 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0));
    const __m128i vmask1 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx1));
    const __m128i vmask2 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx2));
    const __m128i vmask3 = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx3));

    const __m128i vacc0 = _mm_or_si128(_mm_andnot_si128(vmask0, _mm_castps_si128(vx0)), _mm_and_si128(vmask0, _mm_castps_si128(vwx0)));
    const __m128i vacc1 = _mm_or_si128(_mm_andnot_si128(vmask1, _mm_castps_si128(vx1)), _mm_and_si128(vmask1, _mm_castps_si128(vwx1)));
    const __m128i vacc2 = _mm_or_si128(_mm_andnot_si128(vmask2, _mm_castps_si128(vx2)), _mm_and_si128(vmask2, _mm_castps_si128(vwx2)));
    const __m128i vacc3 = _mm_or_si128(_mm_andnot_si128(vmask3, _mm_castps_si128(vx3)), _mm_and_si128(vmask3, _mm_castps_si128(vwx3)));

    __m128 vy0 = _mm_min_ps(_mm_max_ps(vacc0, vy_min), vy_max);
    __m128 vy1 = _mm_min_ps(_mm_max_ps(vacc1, vy_min), vy_max);
    __m128 vy2 = _mm_min_ps(_mm_max_ps(vacc2, vy_min), vy_max);
    __m128 vy3 = _mm_min_ps(_mm_max_ps(vacc3, vy_min), vy_max);

    if (n & 2 * sizeof(float)) {
      _mm_storel_pi((__m64*) y0, vy0);
      _mm_storel_pi((__m64*) y1, vy1);
      _mm_storel_pi((__m64*) y2, vy2);
      _mm_storel_pi((__m64*) y3, vy3);

      vy0 = _mm_movehl_ps(vy0, vy0);
      vy1 = _mm_movehl_ps(vy1, vy1);
      vy2 = _mm_movehl_ps(vy2, vy2);
      vy3 = _mm_movehl_ps(vy3, vy3);

      y0 += 2;
      y1 += 2;
      y2 += 2;
      y3 += 2;
    }
    if (n & 1 * sizeof(float)) {
      _mm_store_ss(y0, vy0);
      _mm_store_ss(y1, vy1);
      _mm_store_ss(y2, vy2);
      _mm_store_ss(y3, vy3);
    }
  }
}
