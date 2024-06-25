// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f16_rmax_ukernel__f16c_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != 0);
  assert(output != 0);

  const uint16_t* i = (const uint16_t*) input;
  __m128i vmax_init = _mm_shufflelo_epi16(_mm_loadl_epi64((const __m128i*) i), _MM_SHUFFLE(0, 0, 0, 0));
  vmax_init = _mm_unpacklo_epi64(vmax_init, vmax_init);
  __m256 vmax0 = _mm256_cvtph_ps(vmax_init);
  __m256 vmax1 = vmax0;
  __m256 vmax2 = vmax0;
  __m256 vmax3 = vmax0;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vx2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vx3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    vmax0 = _mm256_max_ps(vmax0, vx0);
    vmax1 = _mm256_max_ps(vmax1, vx1);
    vmax2 = _mm256_max_ps(vmax2, vx2);
    vmax3 = _mm256_max_ps(vmax3, vx3);
  }
  __m256 vmax = _mm256_max_ps(_mm256_max_ps(vmax0, vmax1), _mm256_max_ps(vmax2, vmax3));
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    vmax = _mm256_max_ps(vmax, vx);
  }
  __m128 vmax_lo = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
  if XNN_UNLIKELY(batch != 0) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m128 vx_lo = _mm256_castps256_ps128(vx);
    if (batch & (4 * sizeof(uint16_t))) {
      vmax_lo = _mm_max_ps(vmax_lo, vx_lo);
      vx_lo = _mm256_extractf128_ps(vx, 1);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmax_lo = _mm_blend_ps(_mm_max_ps(vmax_lo, vx_lo), vmax_lo, 0xC);
      vx_lo = _mm_movehl_ps(vx_lo, vx_lo);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmax_lo = _mm_max_ss(vmax_lo, vx_lo);
    }
  }
  vmax_lo = _mm_max_ps(vmax_lo, _mm_movehl_ps(vmax_lo, vmax_lo));
  vmax_lo = _mm_max_ss(vmax_lo, _mm_movehdup_ps(vmax_lo));
  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(vmax_lo, _MM_FROUND_TO_NEAREST_INT), 0);
}
