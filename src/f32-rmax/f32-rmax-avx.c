// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__avx(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m256 vmax0 = _mm256_broadcast_ss(input);
  __m256 vmax1 = vmax0;
  __m256 vmax2 = vmax0;
  __m256 vmax3 = vmax0;
  for (; batch >= 128; batch -= 128) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    const __m256 vx2 = _mm256_loadu_ps(input + 16);
    const __m256 vx3 = _mm256_loadu_ps(input + 24);
    input += 32;

    vmax0 = _mm256_max_ps(vmax0, vx0);
    vmax1 = _mm256_max_ps(vmax1, vx1);
    vmax2 = _mm256_max_ps(vmax2, vx2);
    vmax3 = _mm256_max_ps(vmax3, vx3);
  }
  __m256 vmax = _mm256_max_ps(_mm256_max_ps(vmax0, vmax1), _mm256_max_ps(vmax2, vmax3));
  for (; batch >= 32; batch -= 32) {
    const __m256 vx = _mm256_loadu_ps(input);
    vmax = _mm256_max_ps(vmax, vx);
    input += 8;
  }
  __m128 vmax_lo = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
  vmax_lo = _mm_max_ps(vmax_lo, _mm_movehl_ps(vmax_lo, vmax_lo));
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
