// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vlrelu_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m256 vslope = _mm256_load_ps(params->avx.slope);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    __m256 vacc01234567 = _mm256_mul_ps(vx01234567, vslope);
    __m256 vacc89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vslope);

    vacc01234567 = _mm256_blendv_ps(vx01234567, vacc01234567, vx01234567);
    vacc89ABCDEF = _mm256_blendv_ps(vx89ABCDEF, vacc89ABCDEF, vx89ABCDEF);

    _mm256_storeu_ps(y, vacc01234567);
    _mm256_storeu_ps(y + 8, vacc89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);
    _mm256_storeu_ps(y, vacc);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vacc_lo);
    }
  }
}
