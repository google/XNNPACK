// Auto-generated file. Do not edit!
//   Template: src/f32-hswish/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_hswish_ukernel__avx_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m256 vsixth = _mm256_broadcast_ps((const __m128*) params->sse.sixth);
  const __m256 vhalf = _mm256_broadcast_ps((const __m128*) params->sse.half);
  const __m256 vone = _mm256_broadcast_ps((const __m128*) params->sse.one);
  const __m256 vzero = _mm256_setzero_ps();

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    x += 8;

    __m256 vacc01234567 = _mm256_mul_ps(vx01234567, vsixth);

    vacc01234567 = _mm256_add_ps(vacc01234567, vhalf);

    vacc01234567 = _mm256_max_ps(vacc01234567, vzero);

    vacc01234567 = _mm256_min_ps(vacc01234567, vone);

    vacc01234567 = _mm256_mul_ps(vacc01234567, vx01234567);

    _mm256_storeu_ps(y, vacc01234567);
    y += 8;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    __m256 vacc = _mm256_mul_ps(vx, vsixth);
    vacc = _mm256_add_ps(vacc, vhalf);
    vacc = _mm256_max_ps(vacc, vzero);
    vacc = _mm256_min_ps(vacc, vone);
    vacc = _mm256_mul_ps(vacc, vx);
    _mm256_storeu_ps(y, vacc);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    __m256 vacc = _mm256_mul_ps(vx, vsixth);
    vacc = _mm256_add_ps(vacc, vhalf);
    vacc = _mm256_max_ps(vacc, vzero);
    vacc = _mm256_min_ps(vacc, vone);
    vacc = _mm256_mul_ps(vacc, vx);

    // _mm256_maskstore_ps(y, vmask, vacc) could be used here, but triggers msan failures (probably an msan bug).
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
