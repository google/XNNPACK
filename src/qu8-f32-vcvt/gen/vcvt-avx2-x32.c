// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/avx2.c.in
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


void xnn_qu8_f32_vcvt_ukernel__avx2_x32(
    size_t n,
    const uint8_t* x,
    float* y,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256i vminus_zero_point = _mm256_load_si256((const __m256i*) params->avx.minus_zero_point);
  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  for (; n >= 32 * sizeof(uint8_t); n -= 32 * sizeof(uint8_t)) {
    __m256i vx01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) x));
    __m256i vx89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (x + 8)));
    __m256i vxGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (x + 16)));
    __m256i vxOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (x + 24)));
    x += 32;

    vx01234567 = _mm256_add_epi32(vx01234567, vminus_zero_point);
    vx89ABCDEF = _mm256_add_epi32(vx89ABCDEF, vminus_zero_point);
    vxGHIJKLMN = _mm256_add_epi32(vxGHIJKLMN, vminus_zero_point);
    vxOPQRSTUV = _mm256_add_epi32(vxOPQRSTUV, vminus_zero_point);

    __m256 vy01234567 = _mm256_cvtepi32_ps(vx01234567);
    __m256 vy89ABCDEF = _mm256_cvtepi32_ps(vx89ABCDEF);
    __m256 vyGHIJKLMN = _mm256_cvtepi32_ps(vxGHIJKLMN);
    __m256 vyOPQRSTUV = _mm256_cvtepi32_ps(vxOPQRSTUV);

    vy01234567 = _mm256_mul_ps(vy01234567, vscale);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vscale);
    vyGHIJKLMN = _mm256_mul_ps(vyGHIJKLMN, vscale);
    vyOPQRSTUV = _mm256_mul_ps(vyOPQRSTUV, vscale);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    _mm256_storeu_ps(y + 16, vyGHIJKLMN);
    _mm256_storeu_ps(y + 24, vyOPQRSTUV);
    y += 32;
  }
  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    __m256i vx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) x));
    vx = _mm256_add_epi32(vx, vminus_zero_point);
    x += 8;

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 7 * sizeof(uint8_t));

    __m256i vx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) x));
    vx = _mm256_add_epi32(vx, vminus_zero_point);

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(uint8_t))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(uint8_t))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}
