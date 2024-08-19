// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-fma3.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_dwconv_minmax_ukernel_25p32c__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint16_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 32; c -= 32) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
      __m256 vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      __m256 vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m256 vi0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 24)));
      i0 += 32;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      const __m256 vk0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xGHIJKLMN, vk0xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xOPQRSTUV, vk0xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m256 vi1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 24)));
      i1 += 32;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      const __m256 vk1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 80)));
      const __m256 vk1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xGHIJKLMN, vk1xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xOPQRSTUV, vk1xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m256 vi2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 24)));
      i2 += 32;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 96)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 104)));
      const __m256 vk2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 112)));
      const __m256 vk2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xGHIJKLMN, vk2xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xOPQRSTUV, vk2xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
      const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m256 vi3xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 24)));
      i3 += 32;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 128)));
      const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 136)));
      const __m256 vk3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 144)));
      const __m256 vk3xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xGHIJKLMN, vk3xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xOPQRSTUV, vk3xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
      const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m256 vi4xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 24)));
      i4 += 32;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 160)));
      const __m256 vk4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 168)));
      const __m256 vk4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 176)));
      const __m256 vk4xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 184)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xGHIJKLMN, vk4xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xOPQRSTUV, vk4xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
      const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m256 vi5xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 24)));
      i5 += 32;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 192)));
      const __m256 vk5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 200)));
      const __m256 vk5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 208)));
      const __m256 vk5xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 216)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xGHIJKLMN, vk5xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xOPQRSTUV, vk5xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
      const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m256 vi6xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 24)));
      i6 += 32;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 224)));
      const __m256 vk6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 232)));
      const __m256 vk6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 240)));
      const __m256 vk6xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 248)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xGHIJKLMN, vk6xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xOPQRSTUV, vk6xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      const __m256 vi7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 8)));
      const __m256 vi7xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m256 vi7xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 24)));
      i7 += 32;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 256)));
      const __m256 vk7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 264)));
      const __m256 vk7xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 272)));
      const __m256 vk7xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 280)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xGHIJKLMN, vk7xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xOPQRSTUV, vk7xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      const __m256 vi8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 8)));
      const __m256 vi8xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m256 vi8xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 24)));
      i8 += 32;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 288)));
      const __m256 vk8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 296)));
      const __m256 vk8xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 304)));
      const __m256 vk8xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 312)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8xGHIJKLMN, vk8xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8xOPQRSTUV, vk8xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));
      const __m256 vi9x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i9 + 8)));
      const __m256 vi9xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m256 vi9xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i9 + 24)));
      i9 += 32;

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 320)));
      const __m256 vk9x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 328)));
      const __m256 vk9xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 336)));
      const __m256 vk9xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 344)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x89ABCDEF, vk9x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9xGHIJKLMN, vk9xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9xOPQRSTUV, vk9xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));
      const __m256 vi10x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i10 + 8)));
      const __m256 vi10xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m256 vi10xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i10 + 24)));
      i10 += 32;

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 352)));
      const __m256 vk10x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 360)));
      const __m256 vk10xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 368)));
      const __m256 vk10xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 376)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x89ABCDEF, vk10x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10xGHIJKLMN, vk10xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10xOPQRSTUV, vk10xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));
      const __m256 vi11x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i11 + 8)));
      const __m256 vi11xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m256 vi11xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i11 + 24)));
      i11 += 32;

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 384)));
      const __m256 vk11x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 392)));
      const __m256 vk11xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 400)));
      const __m256 vk11xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 408)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x89ABCDEF, vk11x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11xGHIJKLMN, vk11xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11xOPQRSTUV, vk11xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));
      const __m256 vi12x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i12 + 8)));
      const __m256 vi12xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m256 vi12xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i12 + 24)));
      i12 += 32;

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 416)));
      const __m256 vk12x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 424)));
      const __m256 vk12xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 432)));
      const __m256 vk12xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 440)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x89ABCDEF, vk12x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12xGHIJKLMN, vk12xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12xOPQRSTUV, vk12xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));
      const __m256 vi13x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i13 + 8)));
      const __m256 vi13xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m256 vi13xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i13 + 24)));
      i13 += 32;

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 448)));
      const __m256 vk13x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 456)));
      const __m256 vk13xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 464)));
      const __m256 vk13xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 472)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x89ABCDEF, vk13x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13xGHIJKLMN, vk13xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13xOPQRSTUV, vk13xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));
      const __m256 vi14x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i14 + 8)));
      const __m256 vi14xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m256 vi14xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i14 + 24)));
      i14 += 32;

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 480)));
      const __m256 vk14x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 488)));
      const __m256 vk14xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 496)));
      const __m256 vk14xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 504)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x89ABCDEF, vk14x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14xGHIJKLMN, vk14xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14xOPQRSTUV, vk14xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));
      const __m256 vi15x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i15 + 8)));
      const __m256 vi15xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m256 vi15xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i15 + 24)));
      i15 += 32;

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 512)));
      const __m256 vk15x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 520)));
      const __m256 vk15xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 528)));
      const __m256 vk15xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 536)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x89ABCDEF, vk15x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15xGHIJKLMN, vk15xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15xOPQRSTUV, vk15xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));
      const __m256 vi16x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i16 + 8)));
      const __m256 vi16xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m256 vi16xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i16 + 24)));
      i16 += 32;

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 544)));
      const __m256 vk16x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 552)));
      const __m256 vk16xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 560)));
      const __m256 vk16xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 568)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x89ABCDEF, vk16x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16xGHIJKLMN, vk16xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16xOPQRSTUV, vk16xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));
      const __m256 vi17x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i17 + 8)));
      const __m256 vi17xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m256 vi17xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i17 + 24)));
      i17 += 32;

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 576)));
      const __m256 vk17x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 584)));
      const __m256 vk17xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 592)));
      const __m256 vk17xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 600)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x89ABCDEF, vk17x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17xGHIJKLMN, vk17xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17xOPQRSTUV, vk17xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));
      const __m256 vi18x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i18 + 8)));
      const __m256 vi18xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m256 vi18xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i18 + 24)));
      i18 += 32;

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 608)));
      const __m256 vk18x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 616)));
      const __m256 vk18xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 624)));
      const __m256 vk18xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 632)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x89ABCDEF, vk18x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18xGHIJKLMN, vk18xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18xOPQRSTUV, vk18xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));
      const __m256 vi19x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i19 + 8)));
      const __m256 vi19xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m256 vi19xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i19 + 24)));
      i19 += 32;

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 640)));
      const __m256 vk19x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 648)));
      const __m256 vk19xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 656)));
      const __m256 vk19xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 664)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x89ABCDEF, vk19x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19xGHIJKLMN, vk19xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19xOPQRSTUV, vk19xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));
      const __m256 vi20x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i20 + 8)));
      const __m256 vi20xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m256 vi20xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i20 + 24)));
      i20 += 32;

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 672)));
      const __m256 vk20x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 680)));
      const __m256 vk20xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 688)));
      const __m256 vk20xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 696)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x89ABCDEF, vk20x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20xGHIJKLMN, vk20xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20xOPQRSTUV, vk20xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));
      const __m256 vi21x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i21 + 8)));
      const __m256 vi21xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m256 vi21xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i21 + 24)));
      i21 += 32;

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 704)));
      const __m256 vk21x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 712)));
      const __m256 vk21xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 720)));
      const __m256 vk21xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 728)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x89ABCDEF, vk21x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21xGHIJKLMN, vk21xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21xOPQRSTUV, vk21xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));
      const __m256 vi22x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i22 + 8)));
      const __m256 vi22xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m256 vi22xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i22 + 24)));
      i22 += 32;

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 736)));
      const __m256 vk22x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 744)));
      const __m256 vk22xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 752)));
      const __m256 vk22xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 760)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x89ABCDEF, vk22x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22xGHIJKLMN, vk22xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22xOPQRSTUV, vk22xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));
      const __m256 vi23x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i23 + 8)));
      const __m256 vi23xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m256 vi23xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i23 + 24)));
      i23 += 32;

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 768)));
      const __m256 vk23x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 776)));
      const __m256 vk23xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 784)));
      const __m256 vk23xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 792)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x89ABCDEF, vk23x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23xGHIJKLMN, vk23xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23xOPQRSTUV, vk23xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));
      const __m256 vi24x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i24 + 8)));
      const __m256 vi24xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m256 vi24xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i24 + 24)));
      i24 += 32;

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 800)));
      const __m256 vk24x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 808)));
      const __m256 vk24xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 816)));
      const __m256 vk24xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 824)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x89ABCDEF, vk24x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24xGHIJKLMN, vk24xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24xOPQRSTUV, vk24xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      w += 832;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      __m256 vaccGHIJKLMN = _mm256_max_ps(vaccGHIJKLMNp0, vmin);
      __m256 vaccOPQRSTUV = _mm256_max_ps(vaccOPQRSTUVp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);
      vaccGHIJKLMN = _mm256_min_ps(vaccGHIJKLMN, vmax);
      vaccOPQRSTUV = _mm256_min_ps(vaccOPQRSTUV, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vaccGHIJKLMN, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vaccOPQRSTUV, _MM_FROUND_TO_NEAREST_INT));
      o += 32;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 224)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 256)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 288)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 320)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 352)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 384)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 416)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 448)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 480)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 512)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 544)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 576)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 608)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 640)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 672)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 704)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 736)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 768)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 800)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 224)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 256)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 288)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 320)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 352)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 384)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 416)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 448)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 480)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 512)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 544)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 576)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 608)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 640)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 672)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 704)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 736)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 768)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 800)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}
