// Auto-generated file. Do not edit!
//   Template: src/f16-gavgpool/multipass-f16c.c.in
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
#include "xnnpack/math.h"


void xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* buffer,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(uint16_t);

  uint16_t* b = buffer;
  size_t c = channels;
  for (; c >= 24; c -= 24) {
    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
    const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
    const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
    const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
    const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;

    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    __m128i vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(vi0x89ABCDEF, vi1x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    __m128i vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(vi0xGHIJKLMN, vi1xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);

    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi2x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi2xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi3x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi3xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi4x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi4xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi5x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi5xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi6x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi6xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);

    _mm_store_si128((__m128i*) b, vacc01234567); b += 8;
    _mm_store_si128((__m128i*) b, vacc89ABCDEF); b += 8;
    _mm_store_si128((__m128i*) b, vaccGHIJKLMN); b += 8;
  }
  if XNN_UNLIKELY(c != 0) {
    do {
      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

      _mm_store_si128((__m128i*) b, vacc01234567); b += 8;

      c = doz(c, 8);
    } while (c != 0);
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);

    uint16_t* b = buffer;
    size_t c = channels;
    for (; c >= 24; c -= 24) {
      __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) b);
      __m128i vacc89ABCDEF = _mm_loadu_si128((const __m128i*) (b + 8));
      __m128i vaccGHIJKLMN = _mm_loadu_si128((const __m128i*) (b + 16));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
      const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi0x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi0xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi1x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi1xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi2x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi2xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi3x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi3xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi4x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi4xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi5x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi5xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);
      vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi6x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
      vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi6xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);

      _mm_store_si128((__m128i*) b, vacc01234567); b += 8;
      _mm_store_si128((__m128i*) b, vacc89ABCDEF); b += 8;
      _mm_store_si128((__m128i*) b, vaccGHIJKLMN); b += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) b);
        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
        vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_TO_NEAREST_INT);
        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
        vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
        vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
        vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
        vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
        vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
        vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

        _mm_store_si128((__m128i*) b, vacc01234567);
        b += 8;

        c = doz(c, 8);
      } while (c != 0);
    }
  }

  i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
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
  for (; channels >= 24; channels -= 24) {
    __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;
    __m128i vacc89ABCDEF = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;
    __m128i vaccGHIJKLMN = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;

    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
    const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
    const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;

    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi0x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi0xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi1x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi1xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi2x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi2xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi3x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi3xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi4x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi4xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi5x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi5xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc89ABCDEF), vi6x89ABCDEF), _MM_FROUND_TO_NEAREST_INT);
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vi6xGHIJKLMN), _MM_FROUND_TO_NEAREST_INT);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc89ABCDEF), vscale), _MM_FROUND_TO_NEAREST_INT);
    vaccGHIJKLMN = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vscale), _MM_FROUND_TO_NEAREST_INT);

    __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
    __m256 vout89ABCDEF = _mm256_max_ps(_mm256_cvtph_ps(vacc89ABCDEF), vmin);
    __m256 voutGHIJKLMN = _mm256_max_ps(_mm256_cvtph_ps(vaccGHIJKLMN), vmin);

    vout01234567 = _mm256_min_ps(vout01234567, vmax);
    vout89ABCDEF = _mm256_min_ps(vout89ABCDEF, vmax);
    voutGHIJKLMN = _mm256_min_ps(voutGHIJKLMN, vmax);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) ((uint16_t*) o + 8), _mm256_cvtps_ph(vout89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) ((uint16_t*) o + 16), _mm256_cvtps_ph(voutGHIJKLMN, _MM_FROUND_TO_NEAREST_INT));
    o += 24;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
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
