// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/sse4.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>


void xnn_qs8_f32_vcvt_ukernel__sse41_x8(
    size_t n,
    const int8_t* x,
    float* y,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(int8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vminus_zero_point = _mm_load_si128((const __m128i*) params->sse4.minus_zero_point);
  const __m128 vscale = _mm_load_ps(params->sse4.scale);
  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    __m128i vx0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(x));
    __m128i vx4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 4));
    x += 8;

    vx0123 = _mm_add_epi32(vx0123, vminus_zero_point);
    vx4567 = _mm_add_epi32(vx4567, vminus_zero_point);

    __m128 vy0123 = _mm_cvtepi32_ps(vx0123);
    __m128 vy4567 = _mm_cvtepi32_ps(vx4567);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);

    _mm_storeu_ps(y, vy0123);
    _mm_storeu_ps(y + 4, vy4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    __m128i vx = _mm_cvtepi8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);
    x += 4;

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    _mm_storeu_ps(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(int8_t));
    assert(n <= 3 * sizeof(int8_t));

    __m128i vx = _mm_cvtepi8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    if (n & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & (1 * sizeof(int8_t))) {
      _mm_store_ss(y, vy);
    }
  }
}
