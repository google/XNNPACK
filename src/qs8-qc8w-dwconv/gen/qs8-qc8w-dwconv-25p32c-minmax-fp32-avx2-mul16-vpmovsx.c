// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in
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


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul16_vpmovsx(
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
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((uintptr_t) w + 8 * sizeof(int32_t)));
      __m256i vaccGHIJKLMN = _mm256_loadu_si256((const __m256i*) ((uintptr_t) w + 16 * sizeof(int32_t)));
      __m256i vaccOPQRSTUV = _mm256_loadu_si256((const __m256i*) ((uintptr_t) w + 24 * sizeof(int32_t)));


      const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
      const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m256i vk0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      const __m256i vprod0x0123456789ABCDEF =  _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
      const __m128i vprod0x89ABCDEF = _mm256_extracti128_si256(vprod0x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod0x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod0x89ABCDEF));
      const __m256i vprod0xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV);
      const __m128i vprod0xOPQRSTUV = _mm256_extracti128_si256(vprod0xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod0xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod0xOPQRSTUV));

      const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
      const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m256i vk1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      const __m256i vprod1x0123456789ABCDEF =  _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
      const __m128i vprod1x89ABCDEF = _mm256_extracti128_si256(vprod1x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod1x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod1x89ABCDEF));
      const __m256i vprod1xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV);
      const __m128i vprod1xOPQRSTUV = _mm256_extracti128_si256(vprod1xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod1xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod1xOPQRSTUV));

      const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
      const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m256i vi2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m256i vk2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      const __m256i vprod2x0123456789ABCDEF =  _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
      const __m128i vprod2x89ABCDEF = _mm256_extracti128_si256(vprod2x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod2x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod2x89ABCDEF));
      const __m256i vprod2xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV);
      const __m128i vprod2xOPQRSTUV = _mm256_extracti128_si256(vprod2xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod2xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod2xOPQRSTUV));

      const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
      const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m256i vi3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m256i vk3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      const __m256i vprod3x0123456789ABCDEF =  _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
      const __m128i vprod3x89ABCDEF = _mm256_extracti128_si256(vprod3x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod3x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod3x89ABCDEF));
      const __m256i vprod3xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV);
      const __m128i vprod3xOPQRSTUV = _mm256_extracti128_si256(vprod3xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod3xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod3xOPQRSTUV));

      const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
      const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m256i vi4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m256i vk4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      const __m256i vprod4x0123456789ABCDEF =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
      const __m128i vprod4x89ABCDEF = _mm256_extracti128_si256(vprod4x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod4x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod4x89ABCDEF));
      const __m256i vprod4xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV);
      const __m128i vprod4xOPQRSTUV = _mm256_extracti128_si256(vprod4xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod4xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod4xOPQRSTUV));

      const __m256i vi5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
      const __m256i vk5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m256i vi5xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m256i vk5xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      const __m256i vprod5x0123456789ABCDEF =  _mm256_mullo_epi16(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF);
      const __m128i vprod5x89ABCDEF = _mm256_extracti128_si256(vprod5x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod5x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod5x89ABCDEF));
      const __m256i vprod5xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV);
      const __m128i vprod5xOPQRSTUV = _mm256_extracti128_si256(vprod5xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod5xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod5xOPQRSTUV));

      const __m256i vi6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
      const __m256i vk6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m256i vi6xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m256i vk6xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      const __m256i vprod6x0123456789ABCDEF =  _mm256_mullo_epi16(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF);
      const __m128i vprod6x89ABCDEF = _mm256_extracti128_si256(vprod6x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod6x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod6x89ABCDEF));
      const __m256i vprod6xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV);
      const __m128i vprod6xOPQRSTUV = _mm256_extracti128_si256(vprod6xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod6xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod6xOPQRSTUV));

      const __m256i vi7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
      const __m256i vk7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m256i vi7xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m256i vk7xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      const __m256i vprod7x0123456789ABCDEF =  _mm256_mullo_epi16(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF);
      const __m128i vprod7x89ABCDEF = _mm256_extracti128_si256(vprod7x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod7x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod7x89ABCDEF));
      const __m256i vprod7xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV);
      const __m128i vprod7xOPQRSTUV = _mm256_extracti128_si256(vprod7xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod7xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod7xOPQRSTUV));

      const __m256i vi8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));
      const __m256i vk8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m256i vi8xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m256i vk8xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      const __m256i vprod8x0123456789ABCDEF =  _mm256_mullo_epi16(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF);
      const __m128i vprod8x89ABCDEF = _mm256_extracti128_si256(vprod8x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod8x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod8x89ABCDEF));
      const __m256i vprod8xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV);
      const __m128i vprod8xOPQRSTUV = _mm256_extracti128_si256(vprod8xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod8xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod8xOPQRSTUV));

      const __m256i vi9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i9));
      const __m256i vk9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m256i vi9xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m256i vk9xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      i9 += 32;

      const __m256i vprod9x0123456789ABCDEF =  _mm256_mullo_epi16(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF);
      const __m128i vprod9x89ABCDEF = _mm256_extracti128_si256(vprod9x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod9x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod9x89ABCDEF));
      const __m256i vprod9xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV);
      const __m128i vprod9xOPQRSTUV = _mm256_extracti128_si256(vprod9xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod9xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod9xOPQRSTUV));

      const __m256i vi10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i10));
      const __m256i vk10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m256i vi10xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m256i vk10xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      i10 += 32;

      const __m256i vprod10x0123456789ABCDEF =  _mm256_mullo_epi16(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF);
      const __m128i vprod10x89ABCDEF = _mm256_extracti128_si256(vprod10x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod10x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod10x89ABCDEF));
      const __m256i vprod10xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV);
      const __m128i vprod10xOPQRSTUV = _mm256_extracti128_si256(vprod10xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod10xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod10xOPQRSTUV));

      const __m256i vi11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i11));
      const __m256i vk11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m256i vi11xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m256i vk11xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      i11 += 32;

      const __m256i vprod11x0123456789ABCDEF =  _mm256_mullo_epi16(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF);
      const __m128i vprod11x89ABCDEF = _mm256_extracti128_si256(vprod11x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod11x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod11x89ABCDEF));
      const __m256i vprod11xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV);
      const __m128i vprod11xOPQRSTUV = _mm256_extracti128_si256(vprod11xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod11xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod11xOPQRSTUV));

      const __m256i vi12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i12));
      const __m256i vk12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m256i vi12xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m256i vk12xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(int8_t))));
      i12 += 32;

      const __m256i vprod12x0123456789ABCDEF =  _mm256_mullo_epi16(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF);
      const __m128i vprod12x89ABCDEF = _mm256_extracti128_si256(vprod12x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod12x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod12x89ABCDEF));
      const __m256i vprod12xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV);
      const __m128i vprod12xOPQRSTUV = _mm256_extracti128_si256(vprod12xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod12xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod12xOPQRSTUV));

      const __m256i vi13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i13));
      const __m256i vk13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t))));
      const __m256i vi13xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m256i vk13xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(int8_t))));
      i13 += 32;

      const __m256i vprod13x0123456789ABCDEF =  _mm256_mullo_epi16(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF);
      const __m128i vprod13x89ABCDEF = _mm256_extracti128_si256(vprod13x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod13x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod13x89ABCDEF));
      const __m256i vprod13xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV);
      const __m128i vprod13xOPQRSTUV = _mm256_extracti128_si256(vprod13xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod13xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod13xOPQRSTUV));

      const __m256i vi14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i14));
      const __m256i vk14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t))));
      const __m256i vi14xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m256i vk14xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(int8_t))));
      i14 += 32;

      const __m256i vprod14x0123456789ABCDEF =  _mm256_mullo_epi16(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF);
      const __m128i vprod14x89ABCDEF = _mm256_extracti128_si256(vprod14x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod14x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod14x89ABCDEF));
      const __m256i vprod14xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV);
      const __m128i vprod14xOPQRSTUV = _mm256_extracti128_si256(vprod14xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod14xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod14xOPQRSTUV));

      const __m256i vi15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i15));
      const __m256i vk15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t))));
      const __m256i vi15xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m256i vk15xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(int8_t))));
      i15 += 32;

      const __m256i vprod15x0123456789ABCDEF =  _mm256_mullo_epi16(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF);
      const __m128i vprod15x89ABCDEF = _mm256_extracti128_si256(vprod15x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod15x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod15x89ABCDEF));
      const __m256i vprod15xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV);
      const __m128i vprod15xOPQRSTUV = _mm256_extracti128_si256(vprod15xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod15xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod15xOPQRSTUV));

      const __m256i vi16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i16));
      const __m256i vk16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t))));
      const __m256i vi16xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m256i vk16xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(int8_t))));
      i16 += 32;

      const __m256i vprod16x0123456789ABCDEF =  _mm256_mullo_epi16(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF);
      const __m128i vprod16x89ABCDEF = _mm256_extracti128_si256(vprod16x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod16x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod16x89ABCDEF));
      const __m256i vprod16xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV);
      const __m128i vprod16xOPQRSTUV = _mm256_extracti128_si256(vprod16xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod16xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod16xOPQRSTUV));

      const __m256i vi17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i17));
      const __m256i vk17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t))));
      const __m256i vi17xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m256i vk17xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(int8_t))));
      i17 += 32;

      const __m256i vprod17x0123456789ABCDEF =  _mm256_mullo_epi16(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF);
      const __m128i vprod17x89ABCDEF = _mm256_extracti128_si256(vprod17x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod17x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod17x89ABCDEF));
      const __m256i vprod17xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV);
      const __m128i vprod17xOPQRSTUV = _mm256_extracti128_si256(vprod17xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod17xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod17xOPQRSTUV));

      const __m256i vi18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i18));
      const __m256i vk18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t))));
      const __m256i vi18xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m256i vk18xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(int8_t))));
      i18 += 32;

      const __m256i vprod18x0123456789ABCDEF =  _mm256_mullo_epi16(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF);
      const __m128i vprod18x89ABCDEF = _mm256_extracti128_si256(vprod18x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod18x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod18x89ABCDEF));
      const __m256i vprod18xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV);
      const __m128i vprod18xOPQRSTUV = _mm256_extracti128_si256(vprod18xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod18xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod18xOPQRSTUV));

      const __m256i vi19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i19));
      const __m256i vk19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t))));
      const __m256i vi19xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m256i vk19xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(int8_t))));
      i19 += 32;

      const __m256i vprod19x0123456789ABCDEF =  _mm256_mullo_epi16(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF);
      const __m128i vprod19x89ABCDEF = _mm256_extracti128_si256(vprod19x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod19x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod19x89ABCDEF));
      const __m256i vprod19xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV);
      const __m128i vprod19xOPQRSTUV = _mm256_extracti128_si256(vprod19xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod19xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod19xOPQRSTUV));

      const __m256i vi20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i20));
      const __m256i vk20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t))));
      const __m256i vi20xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m256i vk20xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(int8_t))));
      i20 += 32;

      const __m256i vprod20x0123456789ABCDEF =  _mm256_mullo_epi16(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF);
      const __m128i vprod20x89ABCDEF = _mm256_extracti128_si256(vprod20x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod20x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod20x89ABCDEF));
      const __m256i vprod20xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV);
      const __m128i vprod20xOPQRSTUV = _mm256_extracti128_si256(vprod20xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod20xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod20xOPQRSTUV));

      const __m256i vi21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i21));
      const __m256i vk21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t))));
      const __m256i vi21xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m256i vk21xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(int8_t))));
      i21 += 32;

      const __m256i vprod21x0123456789ABCDEF =  _mm256_mullo_epi16(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF);
      const __m128i vprod21x89ABCDEF = _mm256_extracti128_si256(vprod21x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod21x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod21x89ABCDEF));
      const __m256i vprod21xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV);
      const __m128i vprod21xOPQRSTUV = _mm256_extracti128_si256(vprod21xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod21xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod21xOPQRSTUV));

      const __m256i vi22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i22));
      const __m256i vk22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t))));
      const __m256i vi22xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m256i vk22xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(int8_t))));
      i22 += 32;

      const __m256i vprod22x0123456789ABCDEF =  _mm256_mullo_epi16(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF);
      const __m128i vprod22x89ABCDEF = _mm256_extracti128_si256(vprod22x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod22x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod22x89ABCDEF));
      const __m256i vprod22xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV);
      const __m128i vprod22xOPQRSTUV = _mm256_extracti128_si256(vprod22xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod22xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod22xOPQRSTUV));

      const __m256i vi23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i23));
      const __m256i vk23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t))));
      const __m256i vi23xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m256i vk23xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(int8_t))));
      i23 += 32;

      const __m256i vprod23x0123456789ABCDEF =  _mm256_mullo_epi16(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF);
      const __m128i vprod23x89ABCDEF = _mm256_extracti128_si256(vprod23x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod23x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod23x89ABCDEF));
      const __m256i vprod23xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV);
      const __m128i vprod23xOPQRSTUV = _mm256_extracti128_si256(vprod23xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod23xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod23xOPQRSTUV));

      const __m256i vi24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i24));
      const __m256i vk24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t))));
      const __m256i vi24xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m256i vk24xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(int8_t))));
      i24 += 32;

      const __m256i vprod24x0123456789ABCDEF =  _mm256_mullo_epi16(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF);
      const __m128i vprod24x89ABCDEF = _mm256_extracti128_si256(vprod24x0123456789ABCDEF, 1);
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod24x0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod24x89ABCDEF));
      const __m256i vprod24xGHIJKLMNOPQRSTUV =  _mm256_mullo_epi16(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV);
      const __m128i vprod24xOPQRSTUV = _mm256_extracti128_si256(vprod24xGHIJKLMNOPQRSTUV, 1);
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod24xGHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(vprod24xOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

      __m256 vfpacc01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vfpacc89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);
      __m256 vfpaccGHIJKLMN = _mm256_cvtepi32_ps(vaccGHIJKLMN);
      __m256 vfpaccOPQRSTUV = _mm256_cvtepi32_ps(vaccOPQRSTUV);

      const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w);
      const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(float)));
      const __m256 vscaleGHIJKLMN = _mm256_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(float)));
      const __m256 vscaleOPQRSTUV = _mm256_loadu_ps((const float*) ((uintptr_t) w + 24 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 32 * sizeof(float));
      vfpacc01234567 = _mm256_mul_ps(vfpacc01234567, vscale01234567);
      vfpacc89ABCDEF = _mm256_mul_ps(vfpacc89ABCDEF, vscale89ABCDEF);
      vfpaccGHIJKLMN = _mm256_mul_ps(vfpaccGHIJKLMN, vscaleGHIJKLMN);
      vfpaccOPQRSTUV = _mm256_mul_ps(vfpaccOPQRSTUV, vscaleOPQRSTUV);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vfpacc01234567 = _mm256_min_ps(vfpacc01234567, voutput_max_less_zero_point);
      vfpacc89ABCDEF = _mm256_min_ps(vfpacc89ABCDEF, voutput_max_less_zero_point);
      vfpaccGHIJKLMN = _mm256_min_ps(vfpaccGHIJKLMN, voutput_max_less_zero_point);
      vfpaccOPQRSTUV = _mm256_min_ps(vfpaccOPQRSTUV, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vfpacc01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vfpacc89ABCDEF);
      vaccGHIJKLMN = _mm256_cvtps_epi32(vfpaccGHIJKLMN);
      vaccOPQRSTUV = _mm256_cvtps_epi32(vfpaccOPQRSTUV);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      const __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);
      const __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(vaccGHIJKLMN, vaccOPQRSTUV), voutput_zero_point);

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
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
        __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((uintptr_t) w + 8 * sizeof(int32_t)));


        const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
        const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        const __m256i vprod0x0123456789ABCDEF = _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
        const __m128i vprod0x89ABCDEF = _mm256_extracti128_si256(vprod0x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod0x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod0x89ABCDEF));

        const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
        const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        const __m256i vprod1x0123456789ABCDEF = _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
        const __m128i vprod1x89ABCDEF = _mm256_extracti128_si256(vprod1x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod1x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod1x89ABCDEF));

        const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
        const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        const __m256i vprod2x0123456789ABCDEF = _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
        const __m128i vprod2x89ABCDEF = _mm256_extracti128_si256(vprod2x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod2x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod2x89ABCDEF));

        const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
        const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        const __m256i vprod3x0123456789ABCDEF = _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
        const __m128i vprod3x89ABCDEF = _mm256_extracti128_si256(vprod3x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod3x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod3x89ABCDEF));

        const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
        const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        const __m256i vprod4x0123456789ABCDEF = _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
        const __m128i vprod4x89ABCDEF = _mm256_extracti128_si256(vprod4x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod4x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod4x89ABCDEF));

        const __m256i vi5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
        const __m256i vk5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        const __m256i vprod5x0123456789ABCDEF = _mm256_mullo_epi16(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF);
        const __m128i vprod5x89ABCDEF = _mm256_extracti128_si256(vprod5x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod5x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod5x89ABCDEF));

        const __m256i vi6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
        const __m256i vk6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        const __m256i vprod6x0123456789ABCDEF = _mm256_mullo_epi16(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF);
        const __m128i vprod6x89ABCDEF = _mm256_extracti128_si256(vprod6x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod6x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod6x89ABCDEF));

        const __m256i vi7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
        const __m256i vk7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        const __m256i vprod7x0123456789ABCDEF = _mm256_mullo_epi16(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF);
        const __m128i vprod7x89ABCDEF = _mm256_extracti128_si256(vprod7x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod7x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod7x89ABCDEF));

        const __m256i vi8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));
        const __m256i vk8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        const __m256i vprod8x0123456789ABCDEF = _mm256_mullo_epi16(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF);
        const __m128i vprod8x89ABCDEF = _mm256_extracti128_si256(vprod8x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod8x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod8x89ABCDEF));

        const __m256i vi9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i9));
        const __m256i vk9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 288)));
        i9 += 16;

        const __m256i vprod9x0123456789ABCDEF = _mm256_mullo_epi16(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF);
        const __m128i vprod9x89ABCDEF = _mm256_extracti128_si256(vprod9x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod9x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod9x89ABCDEF));

        const __m256i vi10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i10));
        const __m256i vk10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 320)));
        i10 += 16;

        const __m256i vprod10x0123456789ABCDEF = _mm256_mullo_epi16(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF);
        const __m128i vprod10x89ABCDEF = _mm256_extracti128_si256(vprod10x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod10x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod10x89ABCDEF));

        const __m256i vi11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i11));
        const __m256i vk11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 352)));
        i11 += 16;

        const __m256i vprod11x0123456789ABCDEF = _mm256_mullo_epi16(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF);
        const __m128i vprod11x89ABCDEF = _mm256_extracti128_si256(vprod11x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod11x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod11x89ABCDEF));

        const __m256i vi12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i12));
        const __m256i vk12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 384)));
        i12 += 16;

        const __m256i vprod12x0123456789ABCDEF = _mm256_mullo_epi16(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF);
        const __m128i vprod12x89ABCDEF = _mm256_extracti128_si256(vprod12x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod12x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod12x89ABCDEF));

        const __m256i vi13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i13));
        const __m256i vk13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 416)));
        i13 += 16;

        const __m256i vprod13x0123456789ABCDEF = _mm256_mullo_epi16(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF);
        const __m128i vprod13x89ABCDEF = _mm256_extracti128_si256(vprod13x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod13x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod13x89ABCDEF));

        const __m256i vi14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i14));
        const __m256i vk14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 448)));
        i14 += 16;

        const __m256i vprod14x0123456789ABCDEF = _mm256_mullo_epi16(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF);
        const __m128i vprod14x89ABCDEF = _mm256_extracti128_si256(vprod14x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod14x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod14x89ABCDEF));

        const __m256i vi15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i15));
        const __m256i vk15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 480)));
        i15 += 16;

        const __m256i vprod15x0123456789ABCDEF = _mm256_mullo_epi16(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF);
        const __m128i vprod15x89ABCDEF = _mm256_extracti128_si256(vprod15x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod15x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod15x89ABCDEF));

        const __m256i vi16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i16));
        const __m256i vk16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 512)));
        i16 += 16;

        const __m256i vprod16x0123456789ABCDEF = _mm256_mullo_epi16(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF);
        const __m128i vprod16x89ABCDEF = _mm256_extracti128_si256(vprod16x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod16x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod16x89ABCDEF));

        const __m256i vi17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i17));
        const __m256i vk17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 544)));
        i17 += 16;

        const __m256i vprod17x0123456789ABCDEF = _mm256_mullo_epi16(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF);
        const __m128i vprod17x89ABCDEF = _mm256_extracti128_si256(vprod17x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod17x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod17x89ABCDEF));

        const __m256i vi18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i18));
        const __m256i vk18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 576)));
        i18 += 16;

        const __m256i vprod18x0123456789ABCDEF = _mm256_mullo_epi16(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF);
        const __m128i vprod18x89ABCDEF = _mm256_extracti128_si256(vprod18x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod18x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod18x89ABCDEF));

        const __m256i vi19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i19));
        const __m256i vk19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 608)));
        i19 += 16;

        const __m256i vprod19x0123456789ABCDEF = _mm256_mullo_epi16(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF);
        const __m128i vprod19x89ABCDEF = _mm256_extracti128_si256(vprod19x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod19x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod19x89ABCDEF));

        const __m256i vi20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i20));
        const __m256i vk20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 640)));
        i20 += 16;

        const __m256i vprod20x0123456789ABCDEF = _mm256_mullo_epi16(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF);
        const __m128i vprod20x89ABCDEF = _mm256_extracti128_si256(vprod20x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod20x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod20x89ABCDEF));

        const __m256i vi21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i21));
        const __m256i vk21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 672)));
        i21 += 16;

        const __m256i vprod21x0123456789ABCDEF = _mm256_mullo_epi16(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF);
        const __m128i vprod21x89ABCDEF = _mm256_extracti128_si256(vprod21x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod21x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod21x89ABCDEF));

        const __m256i vi22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i22));
        const __m256i vk22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 704)));
        i22 += 16;

        const __m256i vprod22x0123456789ABCDEF = _mm256_mullo_epi16(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF);
        const __m128i vprod22x89ABCDEF = _mm256_extracti128_si256(vprod22x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod22x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod22x89ABCDEF));

        const __m256i vi23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i23));
        const __m256i vk23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 736)));
        i23 += 16;

        const __m256i vprod23x0123456789ABCDEF = _mm256_mullo_epi16(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF);
        const __m128i vprod23x89ABCDEF = _mm256_extracti128_si256(vprod23x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod23x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod23x89ABCDEF));

        const __m256i vi24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i24));
        const __m256i vk24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 768)));
        i24 += 16;

        const __m256i vprod24x0123456789ABCDEF = _mm256_mullo_epi16(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF);
        const __m128i vprod24x89ABCDEF = _mm256_extracti128_si256(vprod24x0123456789ABCDEF, 1);
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod24x0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(vprod24x89ABCDEF));

        k += 16;

        __m256 vfpacc01234567 = _mm256_cvtepi32_ps(vacc01234567);
        __m256 vfpacc89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

        const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t)));
        const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t) + 8 * sizeof(float)));
        vfpacc01234567 = _mm256_mul_ps(vfpacc01234567, vscale01234567);
        vfpacc89ABCDEF = _mm256_mul_ps(vfpacc89ABCDEF, vscale89ABCDEF);

        const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
        vfpacc01234567 = _mm256_min_ps(vfpacc01234567, voutput_max_less_zero_point);
        vfpacc89ABCDEF = _mm256_min_ps(vfpacc89ABCDEF, voutput_max_less_zero_point);

        vacc01234567 = _mm256_cvtps_epi32(vfpacc01234567);
        vacc89ABCDEF = _mm256_cvtps_epi32(vfpacc89ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);
        __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc89ABCDEF), _mm256_extracti128_si256(vacc89ABCDEF, 1)), voutput_zero_point);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);

        __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          if (c & 8) {
            _mm_storel_epi64((__m128i*) output, vout0123456789ABCDEF);
            vout0123456789ABCDEF = _mm_unpackhi_epi64(vout0123456789ABCDEF, vout0123456789ABCDEF);
            output += 8;
          }
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456789ABCDEF));
            vout0123456789ABCDEF = _mm_srli_epi64(vout0123456789ABCDEF, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456789ABCDEF, 0));
            vout0123456789ABCDEF = _mm_srli_epi32(vout0123456789ABCDEF, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456789ABCDEF, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
