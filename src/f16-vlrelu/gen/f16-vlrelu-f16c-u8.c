// Auto-generated file. Do not edit!
//   Template: src/f16-vlrelu/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


void xnn_f16_vlrelu_ukernel__f16c_u8(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vslope = _mm256_set1_ps(params->avx.slope);
  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);

    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = _mm_extract_epi16(vh, 0);
    }
  }
}
