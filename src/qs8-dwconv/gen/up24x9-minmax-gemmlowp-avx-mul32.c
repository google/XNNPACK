// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-sse-mul32.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up24x9__avx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 24; c -= 24) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));
      __m128i vaccGHIJ = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 16));
      __m128i vaccKLMN = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 20));


      const __m128i vi0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i0));
      const __m128i vk0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m128i vi0x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i0 + 4));
      const __m128i vk0x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 4 * sizeof(int8_t))));
      const __m128i vi0x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i0 + 8));
      const __m128i vk0x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      const __m128i vi0xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i0 + 12));
      const __m128i vk0xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 12 * sizeof(int8_t))));
      const __m128i vi0xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i0 + 16));
      const __m128i vk0xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m128i vi0xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i0 + 20));
      const __m128i vk0xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 20 * sizeof(int8_t))));
      i0 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi0x0123, vk0x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi0x4567, vk0x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi0x89AB, vk0x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi0xCDEF, vk0xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi0xGHIJ, vk0xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi0xKLMN, vk0xKLMN));

      const __m128i vi1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i1));
      const __m128i vk1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      const __m128i vi1x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i1 + 4));
      const __m128i vk1x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 28 * sizeof(int8_t))));
      const __m128i vi1x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i1 + 8));
      const __m128i vk1x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m128i vi1xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i1 + 12));
      const __m128i vk1xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 36 * sizeof(int8_t))));
      const __m128i vi1xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i1 + 16));
      const __m128i vk1xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      const __m128i vi1xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i1 + 20));
      const __m128i vk1xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 44 * sizeof(int8_t))));
      i1 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi1x0123, vk1x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi1x4567, vk1x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi1x89AB, vk1x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi1xCDEF, vk1xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi1xGHIJ, vk1xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi1xKLMN, vk1xKLMN));

      const __m128i vi2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i2));
      const __m128i vk2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      const __m128i vi2x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i2 + 4));
      const __m128i vk2x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 52 * sizeof(int8_t))));
      const __m128i vi2x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i2 + 8));
      const __m128i vk2x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      const __m128i vi2xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i2 + 12));
      const __m128i vk2xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 60 * sizeof(int8_t))));
      const __m128i vi2xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i2 + 16));
      const __m128i vk2xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m128i vi2xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i2 + 20));
      const __m128i vk2xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 68 * sizeof(int8_t))));
      i2 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi2x0123, vk2x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi2x4567, vk2x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi2x89AB, vk2x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi2xCDEF, vk2xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi2xGHIJ, vk2xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi2xKLMN, vk2xKLMN));

      const __m128i vi3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i3));
      const __m128i vk3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 72 * sizeof(int8_t))));
      const __m128i vi3x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i3 + 4));
      const __m128i vk3x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 76 * sizeof(int8_t))));
      const __m128i vi3x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i3 + 8));
      const __m128i vk3x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      const __m128i vi3xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i3 + 12));
      const __m128i vk3xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 84 * sizeof(int8_t))));
      const __m128i vi3xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i3 + 16));
      const __m128i vk3xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 88 * sizeof(int8_t))));
      const __m128i vi3xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i3 + 20));
      const __m128i vk3xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 92 * sizeof(int8_t))));
      i3 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi3x0123, vk3x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi3x4567, vk3x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi3x89AB, vk3x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi3xCDEF, vk3xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi3xGHIJ, vk3xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi3xKLMN, vk3xKLMN));

      const __m128i vi4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i4));
      const __m128i vk4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m128i vi4x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i4 + 4));
      const __m128i vk4x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 100 * sizeof(int8_t))));
      const __m128i vi4x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i4 + 8));
      const __m128i vk4x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 104 * sizeof(int8_t))));
      const __m128i vi4xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i4 + 12));
      const __m128i vk4xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 108 * sizeof(int8_t))));
      const __m128i vi4xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i4 + 16));
      const __m128i vk4xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      const __m128i vi4xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i4 + 20));
      const __m128i vk4xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 116 * sizeof(int8_t))));
      i4 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi4x0123, vk4x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi4x4567, vk4x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi4x89AB, vk4x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi4xCDEF, vk4xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi4xGHIJ, vk4xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi4xKLMN, vk4xKLMN));

      const __m128i vi5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i5));
      const __m128i vk5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 120 * sizeof(int8_t))));
      const __m128i vi5x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i5 + 4));
      const __m128i vk5x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 124 * sizeof(int8_t))));
      const __m128i vi5x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i5 + 8));
      const __m128i vk5x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m128i vi5xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i5 + 12));
      const __m128i vk5xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 132 * sizeof(int8_t))));
      const __m128i vi5xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i5 + 16));
      const __m128i vk5xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 136 * sizeof(int8_t))));
      const __m128i vi5xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i5 + 20));
      const __m128i vk5xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 140 * sizeof(int8_t))));
      i5 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi5x0123, vk5x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi5x4567, vk5x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi5x89AB, vk5x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi5xCDEF, vk5xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi5xGHIJ, vk5xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi5xKLMN, vk5xKLMN));

      const __m128i vi6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i6));
      const __m128i vk6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      const __m128i vi6x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i6 + 4));
      const __m128i vk6x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 148 * sizeof(int8_t))));
      const __m128i vi6x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i6 + 8));
      const __m128i vk6x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 152 * sizeof(int8_t))));
      const __m128i vi6xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i6 + 12));
      const __m128i vk6xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 156 * sizeof(int8_t))));
      const __m128i vi6xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i6 + 16));
      const __m128i vk6xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m128i vi6xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i6 + 20));
      const __m128i vk6xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 164 * sizeof(int8_t))));
      i6 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi6x0123, vk6x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi6x4567, vk6x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi6x89AB, vk6x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi6xCDEF, vk6xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi6xGHIJ, vk6xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi6xKLMN, vk6xKLMN));

      const __m128i vi7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i7));
      const __m128i vk7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 168 * sizeof(int8_t))));
      const __m128i vi7x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i7 + 4));
      const __m128i vk7x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 172 * sizeof(int8_t))));
      const __m128i vi7x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i7 + 8));
      const __m128i vk7x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      const __m128i vi7xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i7 + 12));
      const __m128i vk7xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 180 * sizeof(int8_t))));
      const __m128i vi7xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i7 + 16));
      const __m128i vk7xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 184 * sizeof(int8_t))));
      const __m128i vi7xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i7 + 20));
      const __m128i vk7xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 188 * sizeof(int8_t))));
      i7 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi7x0123, vk7x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi7x4567, vk7x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi7x89AB, vk7x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi7xCDEF, vk7xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi7xGHIJ, vk7xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi7xKLMN, vk7xKLMN));

      const __m128i vi8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i8));
      const __m128i vk8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m128i vi8x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i8 + 4));
      const __m128i vk8x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 196 * sizeof(int8_t))));
      const __m128i vi8x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(i8 + 8));
      const __m128i vk8x89AB = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 200 * sizeof(int8_t))));
      const __m128i vi8xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(i8 + 12));
      const __m128i vk8xCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 204 * sizeof(int8_t))));
      const __m128i vi8xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(i8 + 16));
      const __m128i vk8xGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      const __m128i vi8xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(i8 + 20));
      const __m128i vk8xKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 212 * sizeof(int8_t))));
      i8 += 24;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi8x0123, vk8x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi8x4567, vk8x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi8x89AB, vk8x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi8xCDEF, vk8xCDEF));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_mullo_epi32(vi8xGHIJ, vk8xGHIJ));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_mullo_epi32(vi8xKLMN, vk8xKLMN));

      w = (const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 216 * sizeof(int8_t));

      const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.multiplier);
      const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.rounding);

      const __m128i vacc13 = _mm_shuffle_epi32(vacc0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprod02 = _mm_add_epi64(_mm_mul_epi32(vacc0123, vmultiplier), vrounding);
      const __m128i vprod13 = _mm_add_epi64(_mm_mul_epi32(vacc13, vmultiplier), vrounding);
      const __m128i vacc57 = _mm_shuffle_epi32(vacc4567, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprod46 = _mm_add_epi64(_mm_mul_epi32(vacc4567, vmultiplier), vrounding);
      const __m128i vprod57 = _mm_add_epi64(_mm_mul_epi32(vacc57, vmultiplier), vrounding);
      const __m128i vacc9B = _mm_shuffle_epi32(vacc89AB, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprod8A = _mm_add_epi64(_mm_mul_epi32(vacc89AB, vmultiplier), vrounding);
      const __m128i vprod9B = _mm_add_epi64(_mm_mul_epi32(vacc9B, vmultiplier), vrounding);
      const __m128i vaccDF = _mm_shuffle_epi32(vaccCDEF, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprodCE = _mm_add_epi64(_mm_mul_epi32(vaccCDEF, vmultiplier), vrounding);
      const __m128i vprodDF = _mm_add_epi64(_mm_mul_epi32(vaccDF, vmultiplier), vrounding);
      const __m128i vaccHJ = _mm_shuffle_epi32(vaccGHIJ, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprodGI = _mm_add_epi64(_mm_mul_epi32(vaccGHIJ, vmultiplier), vrounding);
      const __m128i vprodHJ = _mm_add_epi64(_mm_mul_epi32(vaccHJ, vmultiplier), vrounding);
      const __m128i vaccLN = _mm_shuffle_epi32(vaccKLMN, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprodKM = _mm_add_epi64(_mm_mul_epi32(vaccKLMN, vmultiplier), vrounding);
      const __m128i vprodLN = _mm_add_epi64(_mm_mul_epi32(vaccLN, vmultiplier), vrounding);

      const __m128i vq31prod02 = _mm_srli_epi64(vprod02, 31);
      const __m128i vq31prod13 = _mm_add_epi64(vprod13, vprod13);
      const __m128i vq31prod46 = _mm_srli_epi64(vprod46, 31);
      const __m128i vq31prod57 = _mm_add_epi64(vprod57, vprod57);
      const __m128i vq31prod8A = _mm_srli_epi64(vprod8A, 31);
      const __m128i vq31prod9B = _mm_add_epi64(vprod9B, vprod9B);
      const __m128i vq31prodCE = _mm_srli_epi64(vprodCE, 31);
      const __m128i vq31prodDF = _mm_add_epi64(vprodDF, vprodDF);
      const __m128i vq31prodGI = _mm_srli_epi64(vprodGI, 31);
      const __m128i vq31prodHJ = _mm_add_epi64(vprodHJ, vprodHJ);
      const __m128i vq31prodKM = _mm_srli_epi64(vprodKM, 31);
      const __m128i vq31prodLN = _mm_add_epi64(vprodLN, vprodLN);

      const __m128i vq31prod0123 = _mm_blend_epi16(vq31prod02, vq31prod13, 0xCC);
      const __m128i vq31prod4567 = _mm_blend_epi16(vq31prod46, vq31prod57, 0xCC);
      const __m128i vq31prod89AB = _mm_blend_epi16(vq31prod8A, vq31prod9B, 0xCC);
      const __m128i vq31prodCDEF = _mm_blend_epi16(vq31prodCE, vq31prodDF, 0xCC);
      const __m128i vq31prodGHIJ = _mm_blend_epi16(vq31prodGI, vq31prodHJ, 0xCC);
      const __m128i vq31prodKLMN = _mm_blend_epi16(vq31prodKM, vq31prodLN, 0xCC);

      const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_mask);
      const __m128i vrem0123 =
        _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));
      const __m128i vrem4567 =
        _mm_add_epi32(_mm_and_si128(vq31prod4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod4567));
      const __m128i vrem89AB =
        _mm_add_epi32(_mm_and_si128(vq31prod89AB, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod89AB));
      const __m128i vremCDEF =
        _mm_add_epi32(_mm_and_si128(vq31prodCDEF, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prodCDEF));
      const __m128i vremGHIJ =
        _mm_add_epi32(_mm_and_si128(vq31prodGHIJ, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prodGHIJ));
      const __m128i vremKLMN =
        _mm_add_epi32(_mm_and_si128(vq31prodKLMN, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prodKLMN));

      const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_threshold);
      const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse4.shift);
      vacc0123 =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
      vacc4567 =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));
      vacc89AB =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod89AB, vshift), _mm_cmpgt_epi32(vrem89AB, vremainder_threshold));
      vaccCDEF =
        _mm_sub_epi32(_mm_sra_epi32(vq31prodCDEF, vshift), _mm_cmpgt_epi32(vremCDEF, vremainder_threshold));
      vaccGHIJ =
        _mm_sub_epi32(_mm_sra_epi32(vq31prodGHIJ, vshift), _mm_cmpgt_epi32(vremGHIJ, vremainder_threshold));
      vaccKLMN =
        _mm_sub_epi32(_mm_sra_epi32(vq31prodKLMN, vshift), _mm_cmpgt_epi32(vremKLMN, vremainder_threshold));

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);
      __m128i voutGHIJKLMN = _mm_adds_epi16(_mm_packs_epi32(vaccGHIJ, vaccKLMN), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_min);
      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_max);
      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);
      __m128i voutGHIJKLMNGHIJKLMN = _mm_packs_epi16(voutGHIJKLMN, voutGHIJKLMN);
      voutGHIJKLMNGHIJKLMN = _mm_max_epi8(voutGHIJKLMNGHIJKLMN, voutput_min);
      voutGHIJKLMNGHIJKLMN = _mm_min_epi8(voutGHIJKLMNGHIJKLMN, voutput_max);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      _mm_storel_epi64((__m128i*) (output + 16), voutGHIJKLMNGHIJKLMN);
      output += 24;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 24);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);

        const __m128i vi0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i0));
        const __m128i vk0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(k));
        i0 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi0x0123, vk0x0123));
        const __m128i vi1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i1));
        const __m128i vk1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 24)));
        i1 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi1x0123, vk1x0123));
        const __m128i vi2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i2));
        const __m128i vk2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 48)));
        i2 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi2x0123, vk2x0123));
        const __m128i vi3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i3));
        const __m128i vk3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 72)));
        i3 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi3x0123, vk3x0123));
        const __m128i vi4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i4));
        const __m128i vk4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 96)));
        i4 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi4x0123, vk4x0123));
        const __m128i vi5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i5));
        const __m128i vk5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 120)));
        i5 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi5x0123, vk5x0123));
        const __m128i vi6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i6));
        const __m128i vk6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 144)));
        i6 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi6x0123, vk6x0123));
        const __m128i vi7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i7));
        const __m128i vk7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 168)));
        i7 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi7x0123, vk7x0123));
        const __m128i vi8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i8));
        const __m128i vk8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 192)));
        i8 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi8x0123, vk8x0123));

        k += 4;

        const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.multiplier);
        const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.rounding);

        const __m128i vacc13 = _mm_shuffle_epi32(vacc0123, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vprod02 = _mm_add_epi64(_mm_mul_epi32(vacc0123, vmultiplier), vrounding);
        const __m128i vprod13 = _mm_add_epi64(_mm_mul_epi32(vacc13, vmultiplier), vrounding);

        const __m128i vq31prod02 = _mm_srli_epi64(vprod02, 31);
        const __m128i vq31prod13 = _mm_add_epi64(vprod13, vprod13);

        const __m128i vq31prod0123 = _mm_blend_epi16(vq31prod02, vq31prod13, 0xCC);

        const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_mask);
        const __m128i vrem0123 =
          _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));

        const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_threshold);
        const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse4.shift);
        vacc0123 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));

        w = (const void*) ((const int32_t*) w + 4);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_zero_point);
        __m128i vout0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc0123), voutput_zero_point);

        vout0123 = _mm_packs_epi16(vout0123, vout0123);
        vout0123 = _mm_max_epi8(vout0123, _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_min));
        vout0123 = _mm_min_epi8(vout0123, _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_max));

        if XNN_LIKELY(c >= 4) {
          _mm_storeu_si32(output, vout0123);
          output += 4;
          c -= 4;
        } else {
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123, 0);
            vout0123 = _mm_srli_epi32(vout0123, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
