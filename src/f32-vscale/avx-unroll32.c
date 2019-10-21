// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vscale.h>


void xnn_f32_vscale_ukernel__avx_unroll32(
    size_t n,
    const float* x,
    float* y,
    float c)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  __m256 vc = _mm256_set1_ps(c);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(x);
    const __m256 vx1 = _mm256_loadu_ps(x + 8);
    const __m256 vx2 = _mm256_loadu_ps(x + 16);
    const __m256 vx3 = _mm256_loadu_ps(x + 24);
    x += 32;

    const __m256 vy0 = _mm256_mul_ps(vx0, vc);
    const __m256 vy1 = _mm256_mul_ps(vx1, vc);
    const __m256 vy2 = _mm256_mul_ps(vx2, vc);
    const __m256 vy3 = _mm256_mul_ps(vx3, vc);

    _mm256_storeu_ps(y, vy0);
    _mm256_storeu_ps(y + 8, vy1);
    _mm256_storeu_ps(y + 16, vy2);
    _mm256_storeu_ps(y + 24, vy3);
    y += 32;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vy = _mm256_mul_ps(vx, vc);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128 vx = _mm_load_ss(x);
      x += 1;

      const __m128 vy = _mm_mul_ss(vx, _mm256_castps256_ps128(vc));

      _mm_store_ss(y, vy);
      y += 1;

      n -= sizeof(float);
    } while (n != 0);
  }
}
