// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/multipass-fma3.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    size_t kernel_size,
    void* buffer,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 8);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const uint16_t* w = weights;

    // First pass to process 8 inputs.
    {
      uint16_t* b = buffer;
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
      input += 8;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 32; c -= 32) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
        __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        __m256 vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        __m256 vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));

        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
        const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const __m256 vi0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 24)));
        i0 += 32;

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        const __m256 vk0xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        const __m256 vk0xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xGHIJKLMN, vk0xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xOPQRSTUV, vk0xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
        const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const __m256 vi1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 24)));
        i1 += 32;

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 72)));
        const __m256 vk1xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
        const __m256 vk1xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 88)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xGHIJKLMN, vk1xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xOPQRSTUV, vk1xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
        const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const __m256 vi2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 24)));
        i2 += 32;

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
        const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 104)));
        const __m256 vk2xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
        const __m256 vk2xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 120)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xGHIJKLMN, vk2xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xOPQRSTUV, vk2xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
        const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const __m256 vi3xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 24)));
        i3 += 32;

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
        const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 136)));
        const __m256 vk3xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
        const __m256 vk3xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 152)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xGHIJKLMN, vk3xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xOPQRSTUV, vk3xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
        const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const __m256 vi4xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 24)));
        i4 += 32;

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
        const __m256 vk4x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 168)));
        const __m256 vk4xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 176)));
        const __m256 vk4xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 184)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xGHIJKLMN, vk4xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xOPQRSTUV, vk4xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
        const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 16)));
        const __m256 vi5xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 24)));
        i5 += 32;

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
        const __m256 vk5x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 200)));
        const __m256 vk5xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 208)));
        const __m256 vk5xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 216)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xGHIJKLMN, vk5xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xOPQRSTUV, vk5xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
        const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 16)));
        const __m256 vi6xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 24)));
        i6 += 32;

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 224)));
        const __m256 vk6x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 232)));
        const __m256 vk6xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 240)));
        const __m256 vk6xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 248)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xGHIJKLMN, vk6xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xOPQRSTUV, vk6xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        const __m256 vi7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 8)));
        const __m256 vi7xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 16)));
        const __m256 vi7xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 24)));
        i7 += 32;

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 256)));
        const __m256 vk7x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 264)));
        const __m256 vk7xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 272)));
        const __m256 vk7xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 280)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xGHIJKLMN, vk7xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xOPQRSTUV, vk7xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        w += 288;


        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
        _mm_store_si128((__m128i*) (b + 8), _mm256_cvtps_ph(vacc89ABCDEFp0, _MM_FROUND_TO_NEAREST_INT));
        _mm_store_si128((__m128i*) (b + 16), _mm256_cvtps_ph(vaccGHIJKLMNp0, _MM_FROUND_TO_NEAREST_INT));
        _mm_store_si128((__m128i*) (b + 24), _mm256_cvtps_ph(vaccOPQRSTUVp0, _MM_FROUND_TO_NEAREST_INT));
        b += 32;
      }

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        i5 += 8;

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        i6 += 8;

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        i7 += 8;

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        w += 72;


        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        w += 72;


        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
      }
    }

    // Middle pass to process 8 inputs in each iteration.
    for (size_t ks = kernel_size - 8; ks > 9; ks -= 8) {
      uint16_t* b = buffer;
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
      input += 8;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 32; c -= 32) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));
        __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b + 8)));
        __m256 vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b + 16)));
        __m256 vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b + 24)));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
        const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const __m256 vi0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 24)));
        i0 += 32;

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        const __m256 vk0xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        const __m256 vk0xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xGHIJKLMN, vk0xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xOPQRSTUV, vk0xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
        const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const __m256 vi1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 24)));
        i1 += 32;

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        const __m256 vk1xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        const __m256 vk1xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xGHIJKLMN, vk1xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xOPQRSTUV, vk1xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
        const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const __m256 vi2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 24)));
        i2 += 32;

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 72)));
        const __m256 vk2xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
        const __m256 vk2xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 88)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xGHIJKLMN, vk2xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xOPQRSTUV, vk2xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
        const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const __m256 vi3xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 24)));
        i3 += 32;

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
        const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 104)));
        const __m256 vk3xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
        const __m256 vk3xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 120)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xGHIJKLMN, vk3xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xOPQRSTUV, vk3xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
        const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const __m256 vi4xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 24)));
        i4 += 32;

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
        const __m256 vk4x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 136)));
        const __m256 vk4xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
        const __m256 vk4xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 152)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xGHIJKLMN, vk4xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xOPQRSTUV, vk4xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
        const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 16)));
        const __m256 vi5xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 24)));
        i5 += 32;

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
        const __m256 vk5x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 168)));
        const __m256 vk5xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 176)));
        const __m256 vk5xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 184)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xGHIJKLMN, vk5xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xOPQRSTUV, vk5xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
        const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 16)));
        const __m256 vi6xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 24)));
        i6 += 32;

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
        const __m256 vk6x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 200)));
        const __m256 vk6xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 208)));
        const __m256 vk6xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 216)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xGHIJKLMN, vk6xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xOPQRSTUV, vk6xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        const __m256 vi7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 8)));
        const __m256 vi7xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 16)));
        const __m256 vi7xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 24)));
        i7 += 32;

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 224)));
        const __m256 vk7x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 232)));
        const __m256 vk7xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 240)));
        const __m256 vk7xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 248)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xGHIJKLMN, vk7xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xOPQRSTUV, vk7xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        w += 256;


        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
        _mm_store_si128((__m128i*) (b + 8), _mm256_cvtps_ph(vacc89ABCDEFp0, _MM_FROUND_TO_NEAREST_INT));
        _mm_store_si128((__m128i*) (b + 16), _mm256_cvtps_ph(vaccGHIJKLMNp0, _MM_FROUND_TO_NEAREST_INT));
        _mm_store_si128((__m128i*) (b + 24), _mm256_cvtps_ph(vaccOPQRSTUVp0, _MM_FROUND_TO_NEAREST_INT));
        b += 32;
      }

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        i5 += 8;

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        i6 += 8;

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        i7 += 8;

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        w += 64;


        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        w += 64;


        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
      }
    }

    // Last pass to process up to 9 inputs.
    {
      uint16_t* b = buffer;
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

      size_t c = channels;
      for (; c >= 32; c -= 32) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));
        __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b + 8)));
        __m256 vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b + 16)));
        __m256 vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b + 24)));
        b += 32;


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
        const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const __m256 vi0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 24)));
        i0 += 32;

        __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        __m256 vk0xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        __m256 vk0xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xGHIJKLMN, vk0xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xOPQRSTUV, vk0xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
        const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const __m256 vi1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 24)));
        i1 += 32;

        __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        __m256 vk1xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        __m256 vk1xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xGHIJKLMN, vk1xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xOPQRSTUV, vk1xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
        const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const __m256 vi2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 24)));
        i2 += 32;

        __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 72)));
        __m256 vk2xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
        __m256 vk2xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 88)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xGHIJKLMN, vk2xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xOPQRSTUV, vk2xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
        const __m256 vi3xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const __m256 vi3xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 24)));
        i3 += 32;

        __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
        __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 104)));
        __m256 vk3xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
        __m256 vk3xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 120)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xGHIJKLMN, vk3xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3xOPQRSTUV, vk3xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
        const __m256 vi4xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const __m256 vi4xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 24)));
        i4 += 32;

        __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
        __m256 vk4x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 136)));
        __m256 vk4xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
        __m256 vk4xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 152)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xGHIJKLMN, vk4xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4xOPQRSTUV, vk4xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
        const __m256 vi5xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 16)));
        const __m256 vi5xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 24)));
        i5 += 32;

        __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
        __m256 vk5x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 168)));
        __m256 vk5xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 176)));
        __m256 vk5xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 184)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xGHIJKLMN, vk5xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5xOPQRSTUV, vk5xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
        const __m256 vi6xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 16)));
        const __m256 vi6xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 24)));
        i6 += 32;

        __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
        __m256 vk6x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 200)));
        __m256 vk6xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 208)));
        __m256 vk6xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 216)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xGHIJKLMN, vk6xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6xOPQRSTUV, vk6xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        const __m256 vi7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 8)));
        const __m256 vi7xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 16)));
        const __m256 vi7xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 24)));
        i7 += 32;

        __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 224)));
        __m256 vk7x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 232)));
        __m256 vk7xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 240)));
        __m256 vk7xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 248)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xGHIJKLMN, vk7xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7xOPQRSTUV, vk7xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8)));
        const __m256 vi8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 8)));
        const __m256 vi8xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 16)));
        const __m256 vi8xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 24)));
        i8 += 32;

        __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 256)));
        __m256 vk8x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 264)));
        __m256 vk8xGHIJKLMN = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 272)));
        __m256 vk8xOPQRSTUV = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 280)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
        vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
        vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8xGHIJKLMN, vk8xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
        vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8xOPQRSTUV, vk8xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

        w += 288;


        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
        __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
        __m256 vaccGHIJKLMN = _mm256_max_ps(vaccGHIJKLMNp0, vmin);
        __m256 vaccOPQRSTUV = _mm256_max_ps(vaccOPQRSTUVp0, vmin);

        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
        vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);
        vaccGHIJKLMN = _mm256_min_ps(vaccGHIJKLMN, vmax);
        vaccOPQRSTUV = _mm256_min_ps(vaccOPQRSTUV, vmax);

        _mm_storeu_si128((__m128i*) output, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128((__m128i*) ((uint16_t*) output + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128((__m128i*) ((uint16_t*) output + 16), _mm256_cvtps_ph(vaccGHIJKLMN, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128((__m128i*) ((uint16_t*) output + 24), _mm256_cvtps_ph(vaccOPQRSTUV, _MM_FROUND_TO_NEAREST_INT));
        output = (uint16_t*) output + 32;
      }


      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));
        b += 8;


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        i0 += 8;

        __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        i1 += 8;

        __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        i2 += 8;

        __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        i3 += 8;

        __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        i4 += 8;

        __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        i5 += 8;

        __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        i6 += 8;

        __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        i7 += 8;

        __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8)));
        i8 += 8;

        __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        w += 72;



        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);

        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

        _mm_storeu_si128((__m128i*) output, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
        output = (uint16_t*) output + 8;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));

        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
        __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));


        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

        __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
        if (c & 4) {
          _mm_storel_epi64((__m128i*) output, vh01234567);
          vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
          output = (uint16_t*) output + 4;
        }
        if (c & 2) {
          _mm_storeu_si32(output, vh01234567);
          vh01234567 = _mm_srli_epi64(vh01234567, 32);
          output = (uint16_t*) output + 2;
        }
        if (c & 1) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
          output = (uint16_t*) output + 1;
        }
      }

    }
    input = (const void**) (const uint16_t**) ((uintptr_t) input + input_stride);
    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
