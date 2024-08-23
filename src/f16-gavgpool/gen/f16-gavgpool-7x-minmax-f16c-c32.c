// Auto-generated file. Do not edit!
//   Template: src/f16-gavgpool/unipass-f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const uint16_t*) zero;
  }
  uint16_t* o = (uint16_t*) output;

  const __m256 vscale = _mm256_set1_ps(params->avx.scale);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  for (; channels >= 32; channels -= 32) {
    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
    const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
    const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 16)));
    const __m256 vi0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 24)));
    i0 += 32;
    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
    const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
    const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 16)));
    const __m256 vi1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 24)));
    i1 += 32;

    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
    __m128i vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(vi0x89ABCDEF, vi1x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 16)));
    __m128i vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(vi0xGHIJKLMN, vi1xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 24)));
    __m128i vaccOPQRSTUV = _mm256_cvtps_ph(_mm256_add_ps(vi0xOPQRSTUV, vi1xOPQRSTUV), _MM_FROUND_TO_NEAREST_INT);
    i2 += 32;

    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi2x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 16)));
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi2xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 24)));
    i3 += 32;
    vaccOPQRSTUV = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccOPQRSTUV), vi2xOPQRSTUV), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi3x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 16)));
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi3xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 24)));
    i4 += 32;
    vaccOPQRSTUV = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccOPQRSTUV), vi3xOPQRSTUV), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi4x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 16)));
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi4xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 24)));
    i5 += 32;
    vaccOPQRSTUV = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccOPQRSTUV), vi4xOPQRSTUV), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi5x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 16)));
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi5xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 24)));
    i6 += 32;
    vaccOPQRSTUV = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccOPQRSTUV), vi5xOPQRSTUV), _MM_FROUND_TO_NEAREST_INT);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi6x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi6xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    vaccOPQRSTUV = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccOPQRSTUV), vi6xOPQRSTUV), _MM_FROUND_TO_NEAREST_INT);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc89ABCDEF), vscale), _MM_FROUND_TO_NEAREST_INT);
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vscale), _MM_FROUND_TO_NEAREST_INT);
    vaccOPQRSTUV = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vaccOPQRSTUV), vscale), _MM_FROUND_TO_NEAREST_INT);

    __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
    __m256 vout89ABCDEF = _mm256_max_ps(_mm256_cvtph_ps(vacc89ABCDEF), vmin);
    __m256 voutGHIJKLMN = _mm256_max_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vmin);
    __m256 voutOPQRSTUV = _mm256_max_ps(_mm256_cvtph_ps(vaccOPQRSTUV), vmin);

    vout01234567 = _mm256_min_ps(vout01234567, vmax);
    vout89ABCDEF = _mm256_min_ps(vout89ABCDEF, vmax);
    voutGHIJKLMN = _mm256_min_ps(voutGHIJKLMN, vmax);
    voutOPQRSTUV = _mm256_min_ps(voutOPQRSTUV, vmax);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vout89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(voutGHIJKLMN, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(voutOPQRSTUV, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
      i2 += 8;

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

      vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);
      __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
      vout01234567 = _mm256_min_ps(vout01234567, vmax);

      if XNN_LIKELY(channels >= 8) {
        _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT));
        o += 8;
        channels -= 8;
      } else {
        __m128i vh01234567 = _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT);
        if (channels & 4) {
          _mm_storel_epi64((__m128i*) o, vh01234567);
          o += 4;
          vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        }
        if (channels & 2) {
          _mm_storeu_si32(o, vh01234567);
          o += 2;
          vh01234567 = _mm_srli_epi64(vh01234567, 32);
        }
        if (channels & 1) {
          *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
