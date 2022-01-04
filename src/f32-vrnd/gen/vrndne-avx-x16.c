// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrndne_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_round_ps(vx01234567, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vy89ABCDEF = _mm256_round_ps(vx89ABCDEF, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}
