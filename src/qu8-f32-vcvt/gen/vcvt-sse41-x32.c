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


void xnn_qu8_f32_vcvt_ukernel__sse41_x32(
    size_t n,
    const uint8_t* x,
    float* y,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vminus_zero_point = _mm_load_si128((const __m128i*) params->sse4.minus_zero_point);
  const __m128 vscale = _mm_load_ps(params->sse4.scale);
  for (; n >= 32 * sizeof(uint8_t); n -= 32 * sizeof(uint8_t)) {
    __m128i vx0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(x));
    __m128i vx4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 4));
    __m128i vx89AB = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 8));
    __m128i vxCDEF = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 12));
    __m128i vxGHIJ = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 16));
    __m128i vxKLMN = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 20));
    __m128i vxOPQR = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 24));
    __m128i vxSTUV = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 28));
    x += 32;

    vx0123 = _mm_add_epi32(vx0123, vminus_zero_point);
    vx4567 = _mm_add_epi32(vx4567, vminus_zero_point);
    vx89AB = _mm_add_epi32(vx89AB, vminus_zero_point);
    vxCDEF = _mm_add_epi32(vxCDEF, vminus_zero_point);
    vxGHIJ = _mm_add_epi32(vxGHIJ, vminus_zero_point);
    vxKLMN = _mm_add_epi32(vxKLMN, vminus_zero_point);
    vxOPQR = _mm_add_epi32(vxOPQR, vminus_zero_point);
    vxSTUV = _mm_add_epi32(vxSTUV, vminus_zero_point);

    __m128 vy0123 = _mm_cvtepi32_ps(vx0123);
    __m128 vy4567 = _mm_cvtepi32_ps(vx4567);
    __m128 vy89AB = _mm_cvtepi32_ps(vx89AB);
    __m128 vyCDEF = _mm_cvtepi32_ps(vxCDEF);
    __m128 vyGHIJ = _mm_cvtepi32_ps(vxGHIJ);
    __m128 vyKLMN = _mm_cvtepi32_ps(vxKLMN);
    __m128 vyOPQR = _mm_cvtepi32_ps(vxOPQR);
    __m128 vySTUV = _mm_cvtepi32_ps(vxSTUV);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);
    vy89AB = _mm_mul_ps(vy89AB, vscale);
    vyCDEF = _mm_mul_ps(vyCDEF, vscale);
    vyGHIJ = _mm_mul_ps(vyGHIJ, vscale);
    vyKLMN = _mm_mul_ps(vyKLMN, vscale);
    vyOPQR = _mm_mul_ps(vyOPQR, vscale);
    vySTUV = _mm_mul_ps(vySTUV, vscale);

    _mm_storeu_ps(y, vy0123);
    _mm_storeu_ps(y + 4, vy4567);
    _mm_storeu_ps(y + 8, vy89AB);
    _mm_storeu_ps(y + 12, vyCDEF);
    _mm_storeu_ps(y + 16, vyGHIJ);
    _mm_storeu_ps(y + 20, vyKLMN);
    _mm_storeu_ps(y + 24, vyOPQR);
    _mm_storeu_ps(y + 28, vySTUV);
    y += 32;
  }
  for (; n >= 4 * sizeof(uint8_t); n -= 4 * sizeof(uint8_t)) {
    __m128i vx = _mm_cvtepu8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);
    x += 4;

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    _mm_storeu_ps(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 3 * sizeof(uint8_t));

    __m128i vx = _mm_cvtepu8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    if (n & (2 * sizeof(uint8_t))) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      _mm_store_ss(y, vy);
    }
  }
}
