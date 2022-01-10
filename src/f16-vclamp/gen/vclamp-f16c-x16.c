// Auto-generated file. Do not edit!
//   Template: src/f16-vclamp/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f16_vclamp_ukernel__f16c_x16(
    size_t n,
    const void* restrict x_ptr,
    void* restrict y_ptr,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(x_ptr != NULL);
  assert(y_ptr != NULL);

  const uint16_t* x = (const uint16_t*) x_ptr;
  uint16_t* y = (uint16_t*) y_ptr;

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    __m256 vacc01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) x));
    __m256 vacc89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (x + 8)));
    x += 16;

    vacc01234567 = _mm256_max_ps(vacc01234567, vy_min);
    vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEF, vy_min);

    vacc01234567 = _mm256_min_ps(vacc01234567, vy_max);
    vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (y + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_NO_EXC));
    y += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) x));
    x += 8;
    vacc = _mm256_max_ps(vacc, vy_min);
    vacc = _mm256_min_ps(vacc, vy_max);
    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vacc, _MM_FROUND_NO_EXC));
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) x));
    vacc = _mm256_max_ps(vacc, vy_min);
    vacc = _mm256_min_ps(vacc, vy_max);

    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_NO_EXC);
    if (n & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) y, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      y += 4;
    }
    if (n & (2 * sizeof(uint16_t))) {
      *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vh);
      vh = _mm_srli_epi64(vh, 32);
      y += 2;
    }
    if (n & (1 * sizeof(uint16_t))) {
      *y = _mm_extract_epi16(vh, 0);
    }
  }
}
