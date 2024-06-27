// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-avx2-mul32.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));
      __m256i vaccGHIJKLMN = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 16));
      __m256i vaccOPQRSTUV = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 24));


      const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      const __m256i vi0xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 16)));
      const __m256i vk0xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m256i vi0xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 24)));
      const __m256i vk0xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      i0 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi0xGHIJKLMN, vk0xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi0xOPQRSTUV, vk0xOPQRSTUV));

      const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      const __m256i vi1xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 16)));
      const __m256i vk1xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      const __m256i vi1xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 24)));
      const __m256i vk1xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      i1 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi1xGHIJKLMN, vk1xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi1xOPQRSTUV, vk1xOPQRSTUV));

      const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m256i vi2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 72 * sizeof(int8_t))));
      const __m256i vi2xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 16)));
      const __m256i vk2xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      const __m256i vi2xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 24)));
      const __m256i vk2xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 88 * sizeof(int8_t))));
      i2 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi2xGHIJKLMN, vk2xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi2xOPQRSTUV, vk2xOPQRSTUV));

      const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m256i vi3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 104 * sizeof(int8_t))));
      const __m256i vi3xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 16)));
      const __m256i vk3xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      const __m256i vi3xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 24)));
      const __m256i vk3xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 120 * sizeof(int8_t))));
      i3 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi3xGHIJKLMN, vk3xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi3xOPQRSTUV, vk3xOPQRSTUV));

      const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m256i vi4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 136 * sizeof(int8_t))));
      const __m256i vi4xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 16)));
      const __m256i vk4xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      const __m256i vi4xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 24)));
      const __m256i vk4xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 152 * sizeof(int8_t))));
      i4 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi4xGHIJKLMN, vk4xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi4xOPQRSTUV, vk4xOPQRSTUV));

      const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m256i vi5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 168 * sizeof(int8_t))));
      const __m256i vi5xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 16)));
      const __m256i vk5xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      const __m256i vi5xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 24)));
      const __m256i vk5xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 184 * sizeof(int8_t))));
      i5 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi5xGHIJKLMN, vk5xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi5xOPQRSTUV, vk5xOPQRSTUV));

      const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m256i vi6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 200 * sizeof(int8_t))));
      const __m256i vi6xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 16)));
      const __m256i vk6xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      const __m256i vi6xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 24)));
      const __m256i vk6xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 216 * sizeof(int8_t))));
      i6 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi6xGHIJKLMN, vk6xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi6xOPQRSTUV, vk6xOPQRSTUV));

      const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m256i vi7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 232 * sizeof(int8_t))));
      const __m256i vi7xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 16)));
      const __m256i vk7xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      const __m256i vi7xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 24)));
      const __m256i vk7xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 248 * sizeof(int8_t))));
      i7 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi7xGHIJKLMN, vk7xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi7xOPQRSTUV, vk7xOPQRSTUV));

      const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m256i vi8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 264 * sizeof(int8_t))));
      const __m256i vi8xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 16)));
      const __m256i vk8xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      const __m256i vi8xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 24)));
      const __m256i vk8xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 280 * sizeof(int8_t))));
      i8 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi8xGHIJKLMN, vk8xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi8xOPQRSTUV, vk8xOPQRSTUV));

      const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
      const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m256i vi9x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i9 + 8)));
      const __m256i vk9x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 296 * sizeof(int8_t))));
      const __m256i vi9xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i9 + 16)));
      const __m256i vk9xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      const __m256i vi9xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i9 + 24)));
      const __m256i vk9xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 312 * sizeof(int8_t))));
      i9 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi9x89ABCDEF, vk9x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi9xGHIJKLMN, vk9xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi9xOPQRSTUV, vk9xOPQRSTUV));

      const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
      const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m256i vi10x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i10 + 8)));
      const __m256i vk10x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 328 * sizeof(int8_t))));
      const __m256i vi10xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i10 + 16)));
      const __m256i vk10xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      const __m256i vi10xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i10 + 24)));
      const __m256i vk10xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 344 * sizeof(int8_t))));
      i10 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi10x89ABCDEF, vk10x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi10xGHIJKLMN, vk10xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi10xOPQRSTUV, vk10xOPQRSTUV));

      const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
      const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m256i vi11x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i11 + 8)));
      const __m256i vk11x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 360 * sizeof(int8_t))));
      const __m256i vi11xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i11 + 16)));
      const __m256i vk11xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      const __m256i vi11xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i11 + 24)));
      const __m256i vk11xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 376 * sizeof(int8_t))));
      i11 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi11x89ABCDEF, vk11x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi11xGHIJKLMN, vk11xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi11xOPQRSTUV, vk11xOPQRSTUV));

      const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
      const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m256i vi12x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i12 + 8)));
      const __m256i vk12x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 392 * sizeof(int8_t))));
      const __m256i vi12xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i12 + 16)));
      const __m256i vk12xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(int8_t))));
      const __m256i vi12xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i12 + 24)));
      const __m256i vk12xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 408 * sizeof(int8_t))));
      i12 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi12x89ABCDEF, vk12x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi12xGHIJKLMN, vk12xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi12xOPQRSTUV, vk12xOPQRSTUV));

      const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
      const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t))));
      const __m256i vi13x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i13 + 8)));
      const __m256i vk13x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 424 * sizeof(int8_t))));
      const __m256i vi13xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i13 + 16)));
      const __m256i vk13xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(int8_t))));
      const __m256i vi13xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i13 + 24)));
      const __m256i vk13xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 440 * sizeof(int8_t))));
      i13 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi13x89ABCDEF, vk13x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi13xGHIJKLMN, vk13xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi13xOPQRSTUV, vk13xOPQRSTUV));

      const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
      const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t))));
      const __m256i vi14x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i14 + 8)));
      const __m256i vk14x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 456 * sizeof(int8_t))));
      const __m256i vi14xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i14 + 16)));
      const __m256i vk14xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(int8_t))));
      const __m256i vi14xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i14 + 24)));
      const __m256i vk14xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 472 * sizeof(int8_t))));
      i14 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi14x89ABCDEF, vk14x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi14xGHIJKLMN, vk14xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi14xOPQRSTUV, vk14xOPQRSTUV));

      const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
      const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t))));
      const __m256i vi15x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i15 + 8)));
      const __m256i vk15x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 488 * sizeof(int8_t))));
      const __m256i vi15xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i15 + 16)));
      const __m256i vk15xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(int8_t))));
      const __m256i vi15xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i15 + 24)));
      const __m256i vk15xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 504 * sizeof(int8_t))));
      i15 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi15x89ABCDEF, vk15x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi15xGHIJKLMN, vk15xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi15xOPQRSTUV, vk15xOPQRSTUV));

      const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
      const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t))));
      const __m256i vi16x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i16 + 8)));
      const __m256i vk16x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 520 * sizeof(int8_t))));
      const __m256i vi16xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i16 + 16)));
      const __m256i vk16xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(int8_t))));
      const __m256i vi16xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i16 + 24)));
      const __m256i vk16xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 536 * sizeof(int8_t))));
      i16 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi16x89ABCDEF, vk16x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi16xGHIJKLMN, vk16xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi16xOPQRSTUV, vk16xOPQRSTUV));

      const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
      const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t))));
      const __m256i vi17x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i17 + 8)));
      const __m256i vk17x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 552 * sizeof(int8_t))));
      const __m256i vi17xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i17 + 16)));
      const __m256i vk17xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(int8_t))));
      const __m256i vi17xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i17 + 24)));
      const __m256i vk17xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 568 * sizeof(int8_t))));
      i17 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi17x89ABCDEF, vk17x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi17xGHIJKLMN, vk17xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi17xOPQRSTUV, vk17xOPQRSTUV));

      const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
      const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t))));
      const __m256i vi18x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i18 + 8)));
      const __m256i vk18x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 584 * sizeof(int8_t))));
      const __m256i vi18xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i18 + 16)));
      const __m256i vk18xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(int8_t))));
      const __m256i vi18xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i18 + 24)));
      const __m256i vk18xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 600 * sizeof(int8_t))));
      i18 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi18x89ABCDEF, vk18x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi18xGHIJKLMN, vk18xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi18xOPQRSTUV, vk18xOPQRSTUV));

      const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
      const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t))));
      const __m256i vi19x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i19 + 8)));
      const __m256i vk19x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 616 * sizeof(int8_t))));
      const __m256i vi19xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i19 + 16)));
      const __m256i vk19xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(int8_t))));
      const __m256i vi19xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i19 + 24)));
      const __m256i vk19xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 632 * sizeof(int8_t))));
      i19 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi19x89ABCDEF, vk19x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi19xGHIJKLMN, vk19xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi19xOPQRSTUV, vk19xOPQRSTUV));

      const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
      const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t))));
      const __m256i vi20x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i20 + 8)));
      const __m256i vk20x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 648 * sizeof(int8_t))));
      const __m256i vi20xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i20 + 16)));
      const __m256i vk20xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(int8_t))));
      const __m256i vi20xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i20 + 24)));
      const __m256i vk20xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 664 * sizeof(int8_t))));
      i20 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi20x89ABCDEF, vk20x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi20xGHIJKLMN, vk20xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi20xOPQRSTUV, vk20xOPQRSTUV));

      const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
      const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t))));
      const __m256i vi21x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i21 + 8)));
      const __m256i vk21x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 680 * sizeof(int8_t))));
      const __m256i vi21xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i21 + 16)));
      const __m256i vk21xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(int8_t))));
      const __m256i vi21xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i21 + 24)));
      const __m256i vk21xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 696 * sizeof(int8_t))));
      i21 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi21x89ABCDEF, vk21x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi21xGHIJKLMN, vk21xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi21xOPQRSTUV, vk21xOPQRSTUV));

      const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
      const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t))));
      const __m256i vi22x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i22 + 8)));
      const __m256i vk22x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 712 * sizeof(int8_t))));
      const __m256i vi22xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i22 + 16)));
      const __m256i vk22xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(int8_t))));
      const __m256i vi22xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i22 + 24)));
      const __m256i vk22xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 728 * sizeof(int8_t))));
      i22 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi22x89ABCDEF, vk22x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi22xGHIJKLMN, vk22xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi22xOPQRSTUV, vk22xOPQRSTUV));

      const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
      const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t))));
      const __m256i vi23x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i23 + 8)));
      const __m256i vk23x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 744 * sizeof(int8_t))));
      const __m256i vi23xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i23 + 16)));
      const __m256i vk23xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(int8_t))));
      const __m256i vi23xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i23 + 24)));
      const __m256i vk23xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 760 * sizeof(int8_t))));
      i23 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi23x89ABCDEF, vk23x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi23xGHIJKLMN, vk23xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi23xOPQRSTUV, vk23xOPQRSTUV));

      const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
      const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t))));
      const __m256i vi24x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i24 + 8)));
      const __m256i vk24x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 776 * sizeof(int8_t))));
      const __m256i vi24xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i24 + 16)));
      const __m256i vk24xGHIJKLMN = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(int8_t))));
      const __m256i vi24xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i24 + 24)));
      const __m256i vk24xOPQRSTUV = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 792 * sizeof(int8_t))));
      i24 += 32;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi24x89ABCDEF, vk24x89ABCDEF));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi24xGHIJKLMN, vk24xGHIJKLMN));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi24xOPQRSTUV, vk24xOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);
      __m256 vscaledGHIJKLMN = _mm256_cvtepi32_ps(vaccGHIJKLMN);
      __m256 vscaledOPQRSTUV = _mm256_cvtepi32_ps(vaccOPQRSTUV);

      const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w);
      const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
      const __m256 vscaleGHIJKLMN = _mm256_loadu_ps((const float*) w + 16);
      const __m256 vscaleOPQRSTUV = _mm256_loadu_ps((const float*) w + 24);
      w = (const void*) ((const float*) w + 32);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale89ABCDEF);
      vscaledGHIJKLMN = _mm256_mul_ps(vscaledGHIJKLMN, vscaleGHIJKLMN);
      vscaledOPQRSTUV = _mm256_mul_ps(vscaledOPQRSTUV, vscaleOPQRSTUV);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMN = _mm256_min_ps(vscaledGHIJKLMN, voutput_max_less_zero_point);
      vscaledOPQRSTUV = _mm256_min_ps(vscaledOPQRSTUV, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);
      vaccGHIJKLMN = _mm256_cvtps_epi32(vscaledGHIJKLMN);
      vaccOPQRSTUV = _mm256_cvtps_epi32(vscaledOPQRSTUV);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(vaccGHIJKLMN, vaccOPQRSTUV), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV), _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 32);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) k));
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32)));
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64)));
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96)));
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128)));
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 160)));
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 192)));
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 224)));
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 256)));
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
        const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 288)));
        i9 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));

        const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
        const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 320)));
        i10 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));

        const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
        const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 352)));
        i11 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));

        const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
        const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 384)));
        i12 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));

        const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
        const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 416)));
        i13 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));

        const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
        const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 448)));
        i14 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));

        const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
        const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 480)));
        i15 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));

        const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
        const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 512)));
        i16 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));

        const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
        const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 544)));
        i17 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));

        const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
        const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 576)));
        i18 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));

        const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
        const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 608)));
        i19 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));

        const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
        const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 640)));
        i20 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));

        const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
        const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 672)));
        i21 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));

        const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
        const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 704)));
        i22 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));

        const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
        const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 736)));
        i23 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));

        const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
        const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 768)));
        i24 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t)));
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
