// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__avx512f(
    size_t n,
    const float* x,
    float* y)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  __m512 vmax0 = _mm512_broadcastss_ps(_mm_load_ss(x));
  __m512 vmax1 = vmax0;
  __m512 vmax2 = vmax0;
  __m512 vmax3 = vmax0;
  for (; n >= 256; n -= 256) {
    const __m512 vx0 = _mm512_loadu_ps(x);
    const __m512 vx1 = _mm512_loadu_ps(x + 16);
    const __m512 vx2 = _mm512_loadu_ps(x + 32);
    const __m512 vx3 = _mm512_loadu_ps(x + 48);
    x += 64;

    vmax0 = _mm512_max_ps(vmax0, vx0);
    vmax1 = _mm512_max_ps(vmax1, vx1);
    vmax2 = _mm512_max_ps(vmax2, vx2);
    vmax3 = _mm512_max_ps(vmax3, vx3);
  }
  __m512 vmax = _mm512_max_ps(_mm512_max_ps(vmax0, vmax1), _mm512_max_ps(vmax2, vmax3));
  for (; n >= 64; n -= 64) {
    const __m512 vx = _mm512_loadu_ps(x);
    vmax = _mm512_max_ps(vmax, vx);
    x += 16;
  }
  __m256 vmax_lo = _mm256_max_ps(_mm512_castps512_ps256(vmax), _mm512_castps512_ps256(_mm512_shuffle_f32x4(vmax, vmax, _MM_SHUFFLE(3, 2, 3, 2))));
  __m128 vmax_ll = _mm_max_ps(_mm256_castps256_ps128(vmax_lo), _mm256_extractf128_ps(vmax_lo, 1));
  for (; n >= 16; n -= 16) {
    const __m128 vx = _mm_loadu_ps(x);
    vmax_ll = _mm_max_ps(vmax_ll, vx);
    x += 4;
  }
  vmax_ll = _mm_max_ps(vmax_ll, _mm_movehl_ps(vmax_ll, vmax_ll));
  vmax_ll = _mm_max_ss(vmax_ll, _mm_shuffle_ps(vmax_ll, vmax_ll, _MM_SHUFFLE(3, 3, 1, 1)));
  if XNN_UNLIKELY(n != 0) {
    do {
      vmax_ll = _mm_max_ss(vmax_ll, _mm_load_ss(x));
      x += 1;
      n -= 4;
    } while (n != 0);
  }
  _mm_store_ss(y, vmax_ll);
}
