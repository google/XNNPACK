// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vunary.h>


void xnn_f16_f32_vcvt_ukernel__f16c_x16(
    size_t n,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    const __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    _mm256_storeu_ps(output, vacc0);
    _mm256_storeu_ps(output + 8, vacc1);
    output += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    _mm256_storeu_ps(output, vacc);
    output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint16_t));
    assert(n <= 7 * sizeof(uint16_t));
    const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (n & (4 * sizeof(uint16_t))) {
      _mm_storeu_ps(output, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      output += 4;
    }
    if (n & (2 * sizeof(uint16_t))) {
      _mm_storel_pi((__m64*) output, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      output += 2;
    }
    if (n & (1 * sizeof(uint16_t))) {
      _mm_store_ss(output, vacc_lo);
    }
  }
}

void xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8(
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
  for (; c != 0; c = doz(c, 8)) {
    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;

    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_NO_EXC);

    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_NO_EXC);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_NO_EXC);

    _mm_store_si128((__m128i*) b, vacc01234567); b += 8;
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
    for (; c != 0; c = doz(c, 8)) {
      __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) b);

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_NO_EXC);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_NO_EXC);

      _mm_store_si128((__m128i*) b, vacc01234567); b += 8;
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

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
  const __m256 vmax = _mm256_load_ps(params->avx.max);
  for (; channels >= 8; channels -= 8) {
    __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;

    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;

    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_NO_EXC);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_NO_EXC);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_NO_EXC);

    __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);

    vout01234567 = _mm256_min_ps(vout01234567, vmax);

    _mm_storeu_si128((__m128i*) output, _mm256_cvtps_ph(vout01234567, _MM_FROUND_NO_EXC));
    output = (uint16_t*) output + 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_NO_EXC);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_NO_EXC);

      vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_NO_EXC);
      __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
      vout01234567 = _mm256_min_ps(vout01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vout01234567, _MM_FROUND_NO_EXC);
      if (channels & 4) {
        _mm_storel_epi64((__m128i*) output, vh01234567);
        output = (uint16_t*) output + 4;
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
      }
      if (channels & 2) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vh01234567);
        output = (uint16_t*) output + 2;
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
      }
      if (channels & 1) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
      }
    }
  }
}

void xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8(
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

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
  const __m256 vmax = _mm256_load_ps(params->avx.max);
  for (; channels >= 8; channels -= 8) {
    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
    i0 += 8;
    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
    i1 += 8;

    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_NO_EXC);
    i2 += 8;

    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_NO_EXC);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
    i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_NO_EXC);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_NO_EXC);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_NO_EXC);

    __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);

    vout01234567 = _mm256_min_ps(vout01234567, vmax);

    _mm_storeu_si128((__m128i*) output, _mm256_cvtps_ph(vout01234567, _MM_FROUND_NO_EXC));
    output = (uint16_t*) output + 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_NO_EXC);

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_NO_EXC);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_NO_EXC);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_NO_EXC);

      vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_NO_EXC);
      __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
      vout01234567 = _mm256_min_ps(vout01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vout01234567, _MM_FROUND_NO_EXC);
      if (channels & 4) {
        _mm_storel_epi64((__m128i*) output, vh01234567);
        output = (uint16_t*) output + 4;
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
      }
      if (channels & 2) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vh01234567);
        output = (uint16_t*) output + 2;
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
      }
      if (channels & 1) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
      }
    }
  }
}

void xnn_f16_vadd_minmax_ukernel__f16c_x16(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const uint16_t* a = (const uint16_t*) a_ptr;
  const uint16_t* b = (const uint16_t*) b_ptr;
  uint16_t* y = (uint16_t*) y_ptr;

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va01234567, vb01234567), _MM_FROUND_NO_EXC));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va456789AB, vb456789AB), _MM_FROUND_NO_EXC));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy01234567, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (y + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_NO_EXC));
    y += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC));
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC);
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
      *y = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vaddc_minmax_ukernel__f16c_x16(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const uint16_t* a = (const uint16_t*) a_ptr;
  const uint16_t* b = (const uint16_t*) b_ptr;
  uint16_t* y = (uint16_t*) y_ptr;

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va01234567, vb), _MM_FROUND_NO_EXC));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va456789AB, vb), _MM_FROUND_NO_EXC));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy01234567, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (y + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_NO_EXC));
    y += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC));
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC);
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
      *y = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vmul_minmax_ukernel__f16c_x16(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const uint16_t* a = (const uint16_t*) a_ptr;
  const uint16_t* b = (const uint16_t*) b_ptr;
  uint16_t* y = (uint16_t*) y_ptr;

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va01234567, vb01234567), _MM_FROUND_NO_EXC));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va456789AB, vb456789AB), _MM_FROUND_NO_EXC));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy01234567, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (y + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_NO_EXC));
    y += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC));
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC);
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
      *y = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vmulc_minmax_ukernel__f16c_x16(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const uint16_t* a = (const uint16_t*) a_ptr;
  const uint16_t* b = (const uint16_t*) b_ptr;
  uint16_t* y = (uint16_t*) y_ptr;

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va01234567, vb), _MM_FROUND_NO_EXC));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va456789AB, vb), _MM_FROUND_NO_EXC));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy01234567, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (y + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_NO_EXC));
    y += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC));
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_NO_EXC));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC);
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
      *y = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vhswish_ukernel__f16c_x16(
    size_t n,
    const void* restrict x_ptr,
    void* restrict y_ptr,
    const union xnn_f16_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);

  const uint16_t* x = (const uint16_t*) x_ptr;
  uint16_t* y = (uint16_t*) y_ptr;

  const __m256 vsixth = _mm256_load_ps(params->avx.sixth);
  const __m256 vthree = _mm256_load_ps(params->avx.three);
  const __m128i vsix = _mm_load_si128((const __m128i*) params->avx.six);
  const __m128i vzero = _mm_setzero_si128();

  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    __m256 vx01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) x));
    __m256 vx89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (x + 8)));
    x += 16;

    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vx01234567, vthree), _MM_FROUND_NO_EXC);
    vx01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx01234567, vsixth), _MM_FROUND_NO_EXC));
    __m128i vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(vx89ABCDEF, vthree), _MM_FROUND_NO_EXC);
    vx89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx89ABCDEF, vsixth), _MM_FROUND_NO_EXC));

    vacc01234567 = _mm_max_epi16(vacc01234567, vzero);
    vacc89ABCDEF = _mm_max_epi16(vacc89ABCDEF, vzero);

    vacc01234567 = _mm_min_epi16(vacc01234567, vsix);
    vacc89ABCDEF = _mm_min_epi16(vacc89ABCDEF, vsix);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vx01234567), _MM_FROUND_NO_EXC);
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc89ABCDEF), vx89ABCDEF), _MM_FROUND_NO_EXC);

    _mm_storeu_si128((__m128i*) y, vacc01234567);
    _mm_storeu_si128((__m128i*) (y + 8), vacc89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) x));
    x += 8;
    __m128i vacc = _mm256_cvtps_ph(_mm256_add_ps(vx, vthree), _MM_FROUND_NO_EXC);
    vx = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx, vsixth), _MM_FROUND_NO_EXC));
    vacc = _mm_max_epi16(vacc, vzero);
    vacc = _mm_min_epi16(vacc, vsix);
    vacc = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc), vx), _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i*) y, vacc);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) x));
    __m128i vacc = _mm256_cvtps_ph(_mm256_add_ps(vx, vthree), _MM_FROUND_NO_EXC);
    vx = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx, vsixth), _MM_FROUND_NO_EXC));
    vacc = _mm_max_epi16(vacc, vzero);
    vacc = _mm_min_epi16(vacc, vsix);
    vacc = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc), vx), _MM_FROUND_NO_EXC);

    if (n & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) y, vacc);
      vacc = _mm_unpackhi_epi64(vacc, vacc);
      y += 4;
    }
    if (n & (2 * sizeof(uint16_t))) {
      *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vacc);
      vacc = _mm_srli_epi64(vacc, 32);
      y += 2;
    }
    if (n & (1 * sizeof(uint16_t))) {
      *y = (uint16_t) _mm_extract_epi16(vacc, 0);
    }
  }
}

void xnn_f32_f16_vcvt_ukernel__f16c_x16(
    size_t n,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vf0 = _mm256_loadu_ps(input);
    const __m256 vf1 = _mm256_loadu_ps(input + 8);
    input += 16;

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf0, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vf1, _MM_FROUND_NO_EXC));
    o += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vf = _mm256_loadu_ps(input);
    input += 8;

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_NO_EXC));
    o += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->f16c.mask_table[7] - n));

    const __m256 vf = _mm256_maskload_ps(input, vmask);

    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (n & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, _mm_cvtps_ph(vf_lo, _MM_FROUND_NO_EXC));
      vf_lo = _mm256_extractf128_ps(vf, 1);
      o += 4;
    }
    __m128i vh = _mm_cvtps_ph(vf_lo, _MM_FROUND_NO_EXC);
    if (n & (2 * sizeof(float))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (n & (1 * sizeof(float))) {
      *((uint16_t*) o) = _mm_extract_epi16(vh, 0);
    }
  }
}
