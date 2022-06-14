// Auto-generated file. Do not edit!
//   Template: src/f16-vrnd/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f16_vrndu_ukernel__f16c_x16(
    size_t n,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vacc0 = _mm256_round_ps(vacc0, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    vacc1 = _mm256_round_ps(vacc1, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_NO_EXC));
    o += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_NO_EXC));
    o += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint16_t));
    assert(n <= 7 * sizeof(uint16_t));
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_NO_EXC);
    if (n & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (n & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (n & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
