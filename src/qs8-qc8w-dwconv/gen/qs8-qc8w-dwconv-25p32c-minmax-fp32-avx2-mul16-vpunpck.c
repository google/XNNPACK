// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul16_vpunpck(
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

      __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
      __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);
      __m256i vaccGHIJOPQR = _mm256_inserti128_si256(vaccGHIJKLMN, _mm256_castsi256_si128(vaccOPQRSTUV), 1);
      __m256i vaccKLMNSTUV = _mm256_permute2x128_si256(vaccGHIJKLMN, vaccOPQRSTUV, 0x31);


      const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
      const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m256i vk0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      const __m256i vprod0x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
      const __m256i vprod0x0123456789ABCDEFhi = _mm256_srai_epi16(vprod0x0123456789ABCDEFlo, 15);
      const __m256i vprod0xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV);
      const __m256i vprod0xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod0xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod0xGHIJKLMNOPQRSTUVlo, vprod0xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod0xGHIJKLMNOPQRSTUVlo, vprod0xGHIJKLMNOPQRSTUVhi));

      const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
      const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m256i vk1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      const __m256i vprod1x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
      const __m256i vprod1x0123456789ABCDEFhi = _mm256_srai_epi16(vprod1x0123456789ABCDEFlo, 15);
      const __m256i vprod1xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV);
      const __m256i vprod1xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod1xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod1xGHIJKLMNOPQRSTUVlo, vprod1xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod1xGHIJKLMNOPQRSTUVlo, vprod1xGHIJKLMNOPQRSTUVhi));

      const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
      const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m256i vi2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m256i vk2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      const __m256i vprod2x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
      const __m256i vprod2x0123456789ABCDEFhi = _mm256_srai_epi16(vprod2x0123456789ABCDEFlo, 15);
      const __m256i vprod2xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV);
      const __m256i vprod2xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod2xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod2xGHIJKLMNOPQRSTUVlo, vprod2xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod2xGHIJKLMNOPQRSTUVlo, vprod2xGHIJKLMNOPQRSTUVhi));

      const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
      const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m256i vi3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m256i vk3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      const __m256i vprod3x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
      const __m256i vprod3x0123456789ABCDEFhi = _mm256_srai_epi16(vprod3x0123456789ABCDEFlo, 15);
      const __m256i vprod3xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV);
      const __m256i vprod3xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod3xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod3xGHIJKLMNOPQRSTUVlo, vprod3xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod3xGHIJKLMNOPQRSTUVlo, vprod3xGHIJKLMNOPQRSTUVhi));

      const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
      const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m256i vi4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m256i vk4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
      const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);
      const __m256i vprod4xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV);
      const __m256i vprod4xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod4xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod4xGHIJKLMNOPQRSTUVlo, vprod4xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod4xGHIJKLMNOPQRSTUVlo, vprod4xGHIJKLMNOPQRSTUVhi));

      const __m256i vi5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
      const __m256i vk5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m256i vi5xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m256i vk5xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      const __m256i vprod5x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF);
      const __m256i vprod5x0123456789ABCDEFhi = _mm256_srai_epi16(vprod5x0123456789ABCDEFlo, 15);
      const __m256i vprod5xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV);
      const __m256i vprod5xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod5xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod5x0123456789ABCDEFlo, vprod5x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod5x0123456789ABCDEFlo, vprod5x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod5xGHIJKLMNOPQRSTUVlo, vprod5xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod5xGHIJKLMNOPQRSTUVlo, vprod5xGHIJKLMNOPQRSTUVhi));

      const __m256i vi6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
      const __m256i vk6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m256i vi6xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m256i vk6xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      const __m256i vprod6x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF);
      const __m256i vprod6x0123456789ABCDEFhi = _mm256_srai_epi16(vprod6x0123456789ABCDEFlo, 15);
      const __m256i vprod6xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV);
      const __m256i vprod6xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod6xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod6x0123456789ABCDEFlo, vprod6x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod6x0123456789ABCDEFlo, vprod6x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod6xGHIJKLMNOPQRSTUVlo, vprod6xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod6xGHIJKLMNOPQRSTUVlo, vprod6xGHIJKLMNOPQRSTUVhi));

      const __m256i vi7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
      const __m256i vk7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m256i vi7xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m256i vk7xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      const __m256i vprod7x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF);
      const __m256i vprod7x0123456789ABCDEFhi = _mm256_srai_epi16(vprod7x0123456789ABCDEFlo, 15);
      const __m256i vprod7xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV);
      const __m256i vprod7xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod7xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod7x0123456789ABCDEFlo, vprod7x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod7x0123456789ABCDEFlo, vprod7x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod7xGHIJKLMNOPQRSTUVlo, vprod7xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod7xGHIJKLMNOPQRSTUVlo, vprod7xGHIJKLMNOPQRSTUVhi));

      const __m256i vi8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));
      const __m256i vk8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m256i vi8xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m256i vk8xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      const __m256i vprod8x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF);
      const __m256i vprod8x0123456789ABCDEFhi = _mm256_srai_epi16(vprod8x0123456789ABCDEFlo, 15);
      const __m256i vprod8xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV);
      const __m256i vprod8xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod8xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod8x0123456789ABCDEFlo, vprod8x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod8x0123456789ABCDEFlo, vprod8x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod8xGHIJKLMNOPQRSTUVlo, vprod8xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod8xGHIJKLMNOPQRSTUVlo, vprod8xGHIJKLMNOPQRSTUVhi));

      const __m256i vi9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i9));
      const __m256i vk9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m256i vi9xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m256i vk9xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      i9 += 32;

      const __m256i vprod9x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF);
      const __m256i vprod9x0123456789ABCDEFhi = _mm256_srai_epi16(vprod9x0123456789ABCDEFlo, 15);
      const __m256i vprod9xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV);
      const __m256i vprod9xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod9xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod9x0123456789ABCDEFlo, vprod9x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod9x0123456789ABCDEFlo, vprod9x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod9xGHIJKLMNOPQRSTUVlo, vprod9xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod9xGHIJKLMNOPQRSTUVlo, vprod9xGHIJKLMNOPQRSTUVhi));

      const __m256i vi10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i10));
      const __m256i vk10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m256i vi10xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m256i vk10xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      i10 += 32;

      const __m256i vprod10x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF);
      const __m256i vprod10x0123456789ABCDEFhi = _mm256_srai_epi16(vprod10x0123456789ABCDEFlo, 15);
      const __m256i vprod10xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV);
      const __m256i vprod10xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod10xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod10x0123456789ABCDEFlo, vprod10x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod10x0123456789ABCDEFlo, vprod10x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod10xGHIJKLMNOPQRSTUVlo, vprod10xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod10xGHIJKLMNOPQRSTUVlo, vprod10xGHIJKLMNOPQRSTUVhi));

      const __m256i vi11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i11));
      const __m256i vk11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m256i vi11xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m256i vk11xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      i11 += 32;

      const __m256i vprod11x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF);
      const __m256i vprod11x0123456789ABCDEFhi = _mm256_srai_epi16(vprod11x0123456789ABCDEFlo, 15);
      const __m256i vprod11xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV);
      const __m256i vprod11xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod11xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod11x0123456789ABCDEFlo, vprod11x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod11x0123456789ABCDEFlo, vprod11x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod11xGHIJKLMNOPQRSTUVlo, vprod11xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod11xGHIJKLMNOPQRSTUVlo, vprod11xGHIJKLMNOPQRSTUVhi));

      const __m256i vi12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i12));
      const __m256i vk12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m256i vi12xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m256i vk12xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(int8_t))));
      i12 += 32;

      const __m256i vprod12x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF);
      const __m256i vprod12x0123456789ABCDEFhi = _mm256_srai_epi16(vprod12x0123456789ABCDEFlo, 15);
      const __m256i vprod12xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV);
      const __m256i vprod12xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod12xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod12x0123456789ABCDEFlo, vprod12x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod12x0123456789ABCDEFlo, vprod12x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod12xGHIJKLMNOPQRSTUVlo, vprod12xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod12xGHIJKLMNOPQRSTUVlo, vprod12xGHIJKLMNOPQRSTUVhi));

      const __m256i vi13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i13));
      const __m256i vk13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t))));
      const __m256i vi13xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m256i vk13xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(int8_t))));
      i13 += 32;

      const __m256i vprod13x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF);
      const __m256i vprod13x0123456789ABCDEFhi = _mm256_srai_epi16(vprod13x0123456789ABCDEFlo, 15);
      const __m256i vprod13xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV);
      const __m256i vprod13xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod13xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod13x0123456789ABCDEFlo, vprod13x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod13x0123456789ABCDEFlo, vprod13x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod13xGHIJKLMNOPQRSTUVlo, vprod13xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod13xGHIJKLMNOPQRSTUVlo, vprod13xGHIJKLMNOPQRSTUVhi));

      const __m256i vi14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i14));
      const __m256i vk14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t))));
      const __m256i vi14xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m256i vk14xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(int8_t))));
      i14 += 32;

      const __m256i vprod14x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF);
      const __m256i vprod14x0123456789ABCDEFhi = _mm256_srai_epi16(vprod14x0123456789ABCDEFlo, 15);
      const __m256i vprod14xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV);
      const __m256i vprod14xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod14xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod14x0123456789ABCDEFlo, vprod14x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod14x0123456789ABCDEFlo, vprod14x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod14xGHIJKLMNOPQRSTUVlo, vprod14xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod14xGHIJKLMNOPQRSTUVlo, vprod14xGHIJKLMNOPQRSTUVhi));

      const __m256i vi15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i15));
      const __m256i vk15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t))));
      const __m256i vi15xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m256i vk15xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(int8_t))));
      i15 += 32;

      const __m256i vprod15x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF);
      const __m256i vprod15x0123456789ABCDEFhi = _mm256_srai_epi16(vprod15x0123456789ABCDEFlo, 15);
      const __m256i vprod15xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV);
      const __m256i vprod15xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod15xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod15x0123456789ABCDEFlo, vprod15x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod15x0123456789ABCDEFlo, vprod15x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod15xGHIJKLMNOPQRSTUVlo, vprod15xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod15xGHIJKLMNOPQRSTUVlo, vprod15xGHIJKLMNOPQRSTUVhi));

      const __m256i vi16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i16));
      const __m256i vk16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t))));
      const __m256i vi16xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m256i vk16xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(int8_t))));
      i16 += 32;

      const __m256i vprod16x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF);
      const __m256i vprod16x0123456789ABCDEFhi = _mm256_srai_epi16(vprod16x0123456789ABCDEFlo, 15);
      const __m256i vprod16xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV);
      const __m256i vprod16xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod16xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod16x0123456789ABCDEFlo, vprod16x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod16x0123456789ABCDEFlo, vprod16x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod16xGHIJKLMNOPQRSTUVlo, vprod16xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod16xGHIJKLMNOPQRSTUVlo, vprod16xGHIJKLMNOPQRSTUVhi));

      const __m256i vi17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i17));
      const __m256i vk17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t))));
      const __m256i vi17xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m256i vk17xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(int8_t))));
      i17 += 32;

      const __m256i vprod17x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF);
      const __m256i vprod17x0123456789ABCDEFhi = _mm256_srai_epi16(vprod17x0123456789ABCDEFlo, 15);
      const __m256i vprod17xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV);
      const __m256i vprod17xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod17xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod17x0123456789ABCDEFlo, vprod17x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod17x0123456789ABCDEFlo, vprod17x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod17xGHIJKLMNOPQRSTUVlo, vprod17xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod17xGHIJKLMNOPQRSTUVlo, vprod17xGHIJKLMNOPQRSTUVhi));

      const __m256i vi18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i18));
      const __m256i vk18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t))));
      const __m256i vi18xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m256i vk18xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(int8_t))));
      i18 += 32;

      const __m256i vprod18x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF);
      const __m256i vprod18x0123456789ABCDEFhi = _mm256_srai_epi16(vprod18x0123456789ABCDEFlo, 15);
      const __m256i vprod18xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV);
      const __m256i vprod18xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod18xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod18x0123456789ABCDEFlo, vprod18x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod18x0123456789ABCDEFlo, vprod18x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod18xGHIJKLMNOPQRSTUVlo, vprod18xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod18xGHIJKLMNOPQRSTUVlo, vprod18xGHIJKLMNOPQRSTUVhi));

      const __m256i vi19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i19));
      const __m256i vk19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t))));
      const __m256i vi19xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m256i vk19xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(int8_t))));
      i19 += 32;

      const __m256i vprod19x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF);
      const __m256i vprod19x0123456789ABCDEFhi = _mm256_srai_epi16(vprod19x0123456789ABCDEFlo, 15);
      const __m256i vprod19xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV);
      const __m256i vprod19xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod19xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod19x0123456789ABCDEFlo, vprod19x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod19x0123456789ABCDEFlo, vprod19x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod19xGHIJKLMNOPQRSTUVlo, vprod19xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod19xGHIJKLMNOPQRSTUVlo, vprod19xGHIJKLMNOPQRSTUVhi));

      const __m256i vi20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i20));
      const __m256i vk20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t))));
      const __m256i vi20xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m256i vk20xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(int8_t))));
      i20 += 32;

      const __m256i vprod20x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF);
      const __m256i vprod20x0123456789ABCDEFhi = _mm256_srai_epi16(vprod20x0123456789ABCDEFlo, 15);
      const __m256i vprod20xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV);
      const __m256i vprod20xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod20xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod20x0123456789ABCDEFlo, vprod20x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod20x0123456789ABCDEFlo, vprod20x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod20xGHIJKLMNOPQRSTUVlo, vprod20xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod20xGHIJKLMNOPQRSTUVlo, vprod20xGHIJKLMNOPQRSTUVhi));

      const __m256i vi21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i21));
      const __m256i vk21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t))));
      const __m256i vi21xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m256i vk21xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(int8_t))));
      i21 += 32;

      const __m256i vprod21x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF);
      const __m256i vprod21x0123456789ABCDEFhi = _mm256_srai_epi16(vprod21x0123456789ABCDEFlo, 15);
      const __m256i vprod21xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV);
      const __m256i vprod21xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod21xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod21x0123456789ABCDEFlo, vprod21x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod21x0123456789ABCDEFlo, vprod21x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod21xGHIJKLMNOPQRSTUVlo, vprod21xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod21xGHIJKLMNOPQRSTUVlo, vprod21xGHIJKLMNOPQRSTUVhi));

      const __m256i vi22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i22));
      const __m256i vk22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t))));
      const __m256i vi22xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m256i vk22xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(int8_t))));
      i22 += 32;

      const __m256i vprod22x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF);
      const __m256i vprod22x0123456789ABCDEFhi = _mm256_srai_epi16(vprod22x0123456789ABCDEFlo, 15);
      const __m256i vprod22xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV);
      const __m256i vprod22xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod22xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod22x0123456789ABCDEFlo, vprod22x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod22x0123456789ABCDEFlo, vprod22x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod22xGHIJKLMNOPQRSTUVlo, vprod22xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod22xGHIJKLMNOPQRSTUVlo, vprod22xGHIJKLMNOPQRSTUVhi));

      const __m256i vi23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i23));
      const __m256i vk23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t))));
      const __m256i vi23xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m256i vk23xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(int8_t))));
      i23 += 32;

      const __m256i vprod23x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF);
      const __m256i vprod23x0123456789ABCDEFhi = _mm256_srai_epi16(vprod23x0123456789ABCDEFlo, 15);
      const __m256i vprod23xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV);
      const __m256i vprod23xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod23xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod23x0123456789ABCDEFlo, vprod23x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod23x0123456789ABCDEFlo, vprod23x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod23xGHIJKLMNOPQRSTUVlo, vprod23xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod23xGHIJKLMNOPQRSTUVlo, vprod23xGHIJKLMNOPQRSTUVhi));

      const __m256i vi24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i24));
      const __m256i vk24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t))));
      const __m256i vi24xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m256i vk24xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(int8_t))));
      i24 += 32;

      const __m256i vprod24x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF);
      const __m256i vprod24x0123456789ABCDEFhi = _mm256_srai_epi16(vprod24x0123456789ABCDEFlo, 15);
      const __m256i vprod24xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV);
      const __m256i vprod24xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod24xGHIJKLMNOPQRSTUVlo, 15);

      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod24x0123456789ABCDEFlo, vprod24x0123456789ABCDEFhi));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod24x0123456789ABCDEFlo, vprod24x0123456789ABCDEFhi));
      vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod24xGHIJKLMNOPQRSTUVlo, vprod24xGHIJKLMNOPQRSTUVhi));
      vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod24xGHIJKLMNOPQRSTUVlo, vprod24xGHIJKLMNOPQRSTUVhi));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

      vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
      vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);
      vaccGHIJKLMN = _mm256_inserti128_si256(vaccGHIJOPQR, _mm256_castsi256_si128(vaccKLMNSTUV), 1);
      vaccOPQRSTUV = _mm256_permute2x128_si256(vaccGHIJOPQR, vaccKLMNSTUV, 0x31);

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

        __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
        __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);


        const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
        const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        const __m256i vprod0x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
        const __m256i vprod0x0123456789ABCDEFhi = _mm256_srai_epi16(vprod0x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));

        const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
        const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        const __m256i vprod1x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
        const __m256i vprod1x0123456789ABCDEFhi = _mm256_srai_epi16(vprod1x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));

        const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
        const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        const __m256i vprod2x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
        const __m256i vprod2x0123456789ABCDEFhi = _mm256_srai_epi16(vprod2x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));

        const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
        const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        const __m256i vprod3x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
        const __m256i vprod3x0123456789ABCDEFhi = _mm256_srai_epi16(vprod3x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));

        const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
        const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
        const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));

        const __m256i vi5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
        const __m256i vk5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        const __m256i vprod5x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF);
        const __m256i vprod5x0123456789ABCDEFhi = _mm256_srai_epi16(vprod5x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod5x0123456789ABCDEFlo, vprod5x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod5x0123456789ABCDEFlo, vprod5x0123456789ABCDEFhi));

        const __m256i vi6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
        const __m256i vk6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        const __m256i vprod6x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF);
        const __m256i vprod6x0123456789ABCDEFhi = _mm256_srai_epi16(vprod6x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod6x0123456789ABCDEFlo, vprod6x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod6x0123456789ABCDEFlo, vprod6x0123456789ABCDEFhi));

        const __m256i vi7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
        const __m256i vk7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        const __m256i vprod7x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF);
        const __m256i vprod7x0123456789ABCDEFhi = _mm256_srai_epi16(vprod7x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod7x0123456789ABCDEFlo, vprod7x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod7x0123456789ABCDEFlo, vprod7x0123456789ABCDEFhi));

        const __m256i vi8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));
        const __m256i vk8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        const __m256i vprod8x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF);
        const __m256i vprod8x0123456789ABCDEFhi = _mm256_srai_epi16(vprod8x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod8x0123456789ABCDEFlo, vprod8x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod8x0123456789ABCDEFlo, vprod8x0123456789ABCDEFhi));

        const __m256i vi9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i9));
        const __m256i vk9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 288)));
        i9 += 16;

        const __m256i vprod9x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF);
        const __m256i vprod9x0123456789ABCDEFhi = _mm256_srai_epi16(vprod9x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod9x0123456789ABCDEFlo, vprod9x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod9x0123456789ABCDEFlo, vprod9x0123456789ABCDEFhi));

        const __m256i vi10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i10));
        const __m256i vk10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 320)));
        i10 += 16;

        const __m256i vprod10x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF);
        const __m256i vprod10x0123456789ABCDEFhi = _mm256_srai_epi16(vprod10x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod10x0123456789ABCDEFlo, vprod10x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod10x0123456789ABCDEFlo, vprod10x0123456789ABCDEFhi));

        const __m256i vi11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i11));
        const __m256i vk11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 352)));
        i11 += 16;

        const __m256i vprod11x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF);
        const __m256i vprod11x0123456789ABCDEFhi = _mm256_srai_epi16(vprod11x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod11x0123456789ABCDEFlo, vprod11x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod11x0123456789ABCDEFlo, vprod11x0123456789ABCDEFhi));

        const __m256i vi12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i12));
        const __m256i vk12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 384)));
        i12 += 16;

        const __m256i vprod12x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF);
        const __m256i vprod12x0123456789ABCDEFhi = _mm256_srai_epi16(vprod12x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod12x0123456789ABCDEFlo, vprod12x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod12x0123456789ABCDEFlo, vprod12x0123456789ABCDEFhi));

        const __m256i vi13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i13));
        const __m256i vk13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 416)));
        i13 += 16;

        const __m256i vprod13x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF);
        const __m256i vprod13x0123456789ABCDEFhi = _mm256_srai_epi16(vprod13x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod13x0123456789ABCDEFlo, vprod13x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod13x0123456789ABCDEFlo, vprod13x0123456789ABCDEFhi));

        const __m256i vi14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i14));
        const __m256i vk14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 448)));
        i14 += 16;

        const __m256i vprod14x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF);
        const __m256i vprod14x0123456789ABCDEFhi = _mm256_srai_epi16(vprod14x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod14x0123456789ABCDEFlo, vprod14x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod14x0123456789ABCDEFlo, vprod14x0123456789ABCDEFhi));

        const __m256i vi15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i15));
        const __m256i vk15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 480)));
        i15 += 16;

        const __m256i vprod15x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF);
        const __m256i vprod15x0123456789ABCDEFhi = _mm256_srai_epi16(vprod15x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod15x0123456789ABCDEFlo, vprod15x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod15x0123456789ABCDEFlo, vprod15x0123456789ABCDEFhi));

        const __m256i vi16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i16));
        const __m256i vk16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 512)));
        i16 += 16;

        const __m256i vprod16x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF);
        const __m256i vprod16x0123456789ABCDEFhi = _mm256_srai_epi16(vprod16x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod16x0123456789ABCDEFlo, vprod16x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod16x0123456789ABCDEFlo, vprod16x0123456789ABCDEFhi));

        const __m256i vi17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i17));
        const __m256i vk17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 544)));
        i17 += 16;

        const __m256i vprod17x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF);
        const __m256i vprod17x0123456789ABCDEFhi = _mm256_srai_epi16(vprod17x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod17x0123456789ABCDEFlo, vprod17x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod17x0123456789ABCDEFlo, vprod17x0123456789ABCDEFhi));

        const __m256i vi18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i18));
        const __m256i vk18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 576)));
        i18 += 16;

        const __m256i vprod18x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF);
        const __m256i vprod18x0123456789ABCDEFhi = _mm256_srai_epi16(vprod18x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod18x0123456789ABCDEFlo, vprod18x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod18x0123456789ABCDEFlo, vprod18x0123456789ABCDEFhi));

        const __m256i vi19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i19));
        const __m256i vk19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 608)));
        i19 += 16;

        const __m256i vprod19x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF);
        const __m256i vprod19x0123456789ABCDEFhi = _mm256_srai_epi16(vprod19x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod19x0123456789ABCDEFlo, vprod19x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod19x0123456789ABCDEFlo, vprod19x0123456789ABCDEFhi));

        const __m256i vi20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i20));
        const __m256i vk20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 640)));
        i20 += 16;

        const __m256i vprod20x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF);
        const __m256i vprod20x0123456789ABCDEFhi = _mm256_srai_epi16(vprod20x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod20x0123456789ABCDEFlo, vprod20x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod20x0123456789ABCDEFlo, vprod20x0123456789ABCDEFhi));

        const __m256i vi21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i21));
        const __m256i vk21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 672)));
        i21 += 16;

        const __m256i vprod21x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF);
        const __m256i vprod21x0123456789ABCDEFhi = _mm256_srai_epi16(vprod21x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod21x0123456789ABCDEFlo, vprod21x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod21x0123456789ABCDEFlo, vprod21x0123456789ABCDEFhi));

        const __m256i vi22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i22));
        const __m256i vk22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 704)));
        i22 += 16;

        const __m256i vprod22x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF);
        const __m256i vprod22x0123456789ABCDEFhi = _mm256_srai_epi16(vprod22x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod22x0123456789ABCDEFlo, vprod22x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod22x0123456789ABCDEFlo, vprod22x0123456789ABCDEFhi));

        const __m256i vi23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i23));
        const __m256i vk23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 736)));
        i23 += 16;

        const __m256i vprod23x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF);
        const __m256i vprod23x0123456789ABCDEFhi = _mm256_srai_epi16(vprod23x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod23x0123456789ABCDEFlo, vprod23x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod23x0123456789ABCDEFlo, vprod23x0123456789ABCDEFhi));

        const __m256i vi24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i24));
        const __m256i vk24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (k + 768)));
        i24 += 16;

        const __m256i vprod24x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF);
        const __m256i vprod24x0123456789ABCDEFhi = _mm256_srai_epi16(vprod24x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod24x0123456789ABCDEFlo, vprod24x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod24x0123456789ABCDEFlo, vprod24x0123456789ABCDEFhi));

        vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
        vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);

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
