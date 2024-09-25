// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vop-f16c.c.in
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
#include "xnnpack/vbinary.h"


void xnn_f16_vprelu_ukernel__f16c_u8(
    size_t batch,
    const xnn_float16* restrict input_a,
    const xnn_float16* restrict input_b,
    xnn_float16* restrict output,
    const struct xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_blendv_ps(va, vy, va), _MM_FROUND_TO_NEAREST_INT));

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_blendv_ps(va, vy, va), _MM_FROUND_TO_NEAREST_INT));

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
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
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
