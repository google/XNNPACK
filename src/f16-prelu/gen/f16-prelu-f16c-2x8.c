// Auto-generated file. Do not edit!
//   Template: src/f16-prelu/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_f16_prelu_ukernel__f16c_2x8(
    size_t rows,
    size_t channels,
    const void* restrict input,
    size_t input_stride,
    const void* restrict weights,
    void* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  const uint16_t* i0 = (const uint16_t*) input;
  uint16_t* o0 = (uint16_t*) output;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const __m256 vw01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));
      w += 8;

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      __m256 vacc0x01234567 = _mm256_mul_ps(vi0x01234567, vw01234567);
      __m256 vacc1x01234567 = _mm256_mul_ps(vi1x01234567, vw01234567);

      vacc0x01234567 = _mm256_blendv_ps(vi0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = _mm256_blendv_ps(vi1x01234567, vacc1x01234567, vi1x01234567);

      _mm_storeu_si128((__m128i*) o0, _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT));
      o0 += 8;
      _mm_storeu_si128((__m128i*) o1, _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT));
      o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m256 vw01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 = (const uint16_t*) ((uintptr_t) i0 + c);
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 = (const uint16_t*) ((uintptr_t) i1 + c);

      __m256 vacc0x01234567 = _mm256_mul_ps(vi0x01234567, vw01234567);
      __m256 vacc1x01234567 = _mm256_mul_ps(vi1x01234567, vw01234567);

      vacc0x01234567 = _mm256_blendv_ps(vi0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = _mm256_blendv_ps(vi1x01234567, vacc1x01234567, vi1x01234567);

      __m128i vh0x01234567 = _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1x01234567 = _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & (4 * sizeof(uint16_t))) {
        _mm_storel_epi64((__m128i*) o0, vh0x01234567);
        _mm_storel_epi64((__m128i*) o1, vh1x01234567);

        vh0x01234567 = _mm_unpackhi_epi64(vh0x01234567, vh0x01234567);
        vh1x01234567 = _mm_unpackhi_epi64(vh1x01234567, vh1x01234567);

        o0 += 4;
        o1 += 4;
      }
      if (c & (2 * sizeof(uint16_t))) {
        _mm_storeu_si32(o0, vh0x01234567);
        _mm_storeu_si32(o1, vh1x01234567);

        vh0x01234567 = _mm_srli_epi64(vh0x01234567, 32);
        vh1x01234567 = _mm_srli_epi64(vh1x01234567, 32);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(uint16_t))) {
        *o0 = (uint16_t) _mm_extract_epi16(vh0x01234567, 0);
        *o1 = (uint16_t) _mm_extract_epi16(vh1x01234567, 0);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
