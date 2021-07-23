// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-sse-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <tmmintrin.h>

#include <xnnpack/dwconv.h>


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up24x9__ssse3_mul16(
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


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
      const __m128i vk0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i0 += 24;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);
      const __m128i vxi0x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x89ABCDEF, vi0x89ABCDEF), 8);
      const __m128i vxk0x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x89ABCDEF, vk0x89ABCDEF), 8);
      const __m128i vxi0xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi0xGHIJKLMN, vi0xGHIJKLMN), 8);
      const __m128i vxk0xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk0xGHIJKLMN, vk0xGHIJKLMN), 8);

      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x89ABCDEFlo = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);
      const __m128i vprod0x89ABCDEFhi = _mm_mulhi_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);
      const __m128i vprod0xGHIJKLMNlo = _mm_mullo_epi16(vxi0xGHIJKLMN, vxk0xGHIJKLMN);
      const __m128i vprod0xGHIJKLMNhi = _mm_mulhi_epi16(vxi0xGHIJKLMN, vxk0xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod0x89ABCDEFlo, vprod0x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod0x89ABCDEFlo, vprod0x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod0xGHIJKLMNlo, vprod0xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod0xGHIJKLMNlo, vprod0xGHIJKLMNhi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
      const __m128i vk1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i1 += 24;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);
      const __m128i vxi1x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x89ABCDEF, vi1x89ABCDEF), 8);
      const __m128i vxk1x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x89ABCDEF, vk1x89ABCDEF), 8);
      const __m128i vxi1xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi1xGHIJKLMN, vi1xGHIJKLMN), 8);
      const __m128i vxk1xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk1xGHIJKLMN, vk1xGHIJKLMN), 8);

      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x89ABCDEFlo = _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);
      const __m128i vprod1x89ABCDEFhi = _mm_mulhi_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);
      const __m128i vprod1xGHIJKLMNlo = _mm_mullo_epi16(vxi1xGHIJKLMN, vxk1xGHIJKLMN);
      const __m128i vprod1xGHIJKLMNhi = _mm_mulhi_epi16(vxi1xGHIJKLMN, vxk1xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod1x89ABCDEFlo, vprod1x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod1x89ABCDEFlo, vprod1x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod1xGHIJKLMNlo, vprod1xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod1xGHIJKLMNlo, vprod1xGHIJKLMNhi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
      const __m128i vk2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i2 += 24;

      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);
      const __m128i vxi2x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x89ABCDEF, vi2x89ABCDEF), 8);
      const __m128i vxk2x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x89ABCDEF, vk2x89ABCDEF), 8);
      const __m128i vxi2xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi2xGHIJKLMN, vi2xGHIJKLMN), 8);
      const __m128i vxk2xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk2xGHIJKLMN, vk2xGHIJKLMN), 8);

      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x89ABCDEFlo = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);
      const __m128i vprod2x89ABCDEFhi = _mm_mulhi_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);
      const __m128i vprod2xGHIJKLMNlo = _mm_mullo_epi16(vxi2xGHIJKLMN, vxk2xGHIJKLMN);
      const __m128i vprod2xGHIJKLMNhi = _mm_mulhi_epi16(vxi2xGHIJKLMN, vxk2xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod2x89ABCDEFlo, vprod2x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod2x89ABCDEFlo, vprod2x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod2xGHIJKLMNlo, vprod2xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod2xGHIJKLMNlo, vprod2xGHIJKLMNhi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
      const __m128i vk3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      i3 += 24;

      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);
      const __m128i vxi3x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x89ABCDEF, vi3x89ABCDEF), 8);
      const __m128i vxk3x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x89ABCDEF, vk3x89ABCDEF), 8);
      const __m128i vxi3xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi3xGHIJKLMN, vi3xGHIJKLMN), 8);
      const __m128i vxk3xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk3xGHIJKLMN, vk3xGHIJKLMN), 8);

      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x89ABCDEFlo = _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);
      const __m128i vprod3x89ABCDEFhi = _mm_mulhi_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);
      const __m128i vprod3xGHIJKLMNlo = _mm_mullo_epi16(vxi3xGHIJKLMN, vxk3xGHIJKLMN);
      const __m128i vprod3xGHIJKLMNhi = _mm_mulhi_epi16(vxi3xGHIJKLMN, vxk3xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod3x89ABCDEFlo, vprod3x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod3x89ABCDEFlo, vprod3x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod3xGHIJKLMNlo, vprod3xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod3xGHIJKLMNlo, vprod3xGHIJKLMNhi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
      const __m128i vk4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      i4 += 24;

      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);
      const __m128i vxi4x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x89ABCDEF, vi4x89ABCDEF), 8);
      const __m128i vxk4x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x89ABCDEF, vk4x89ABCDEF), 8);
      const __m128i vxi4xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi4xGHIJKLMN, vi4xGHIJKLMN), 8);
      const __m128i vxk4xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk4xGHIJKLMN, vk4xGHIJKLMN), 8);

      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x89ABCDEFlo = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);
      const __m128i vprod4x89ABCDEFhi = _mm_mulhi_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);
      const __m128i vprod4xGHIJKLMNlo = _mm_mullo_epi16(vxi4xGHIJKLMN, vxk4xGHIJKLMN);
      const __m128i vprod4xGHIJKLMNhi = _mm_mulhi_epi16(vxi4xGHIJKLMN, vxk4xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod4x89ABCDEFlo, vprod4x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod4x89ABCDEFlo, vprod4x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod4xGHIJKLMNlo, vprod4xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod4xGHIJKLMNlo, vprod4xGHIJKLMNhi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
      const __m128i vk5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      i5 += 24;

      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);
      const __m128i vxi5x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x89ABCDEF, vi5x89ABCDEF), 8);
      const __m128i vxk5x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x89ABCDEF, vk5x89ABCDEF), 8);
      const __m128i vxi5xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi5xGHIJKLMN, vi5xGHIJKLMN), 8);
      const __m128i vxk5xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk5xGHIJKLMN, vk5xGHIJKLMN), 8);

      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x89ABCDEFlo = _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);
      const __m128i vprod5x89ABCDEFhi = _mm_mulhi_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);
      const __m128i vprod5xGHIJKLMNlo = _mm_mullo_epi16(vxi5xGHIJKLMN, vxk5xGHIJKLMN);
      const __m128i vprod5xGHIJKLMNhi = _mm_mulhi_epi16(vxi5xGHIJKLMN, vxk5xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod5x89ABCDEFlo, vprod5x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod5x89ABCDEFlo, vprod5x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod5xGHIJKLMNlo, vprod5xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod5xGHIJKLMNlo, vprod5xGHIJKLMNhi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
      const __m128i vk6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      i6 += 24;

      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
      const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);
      const __m128i vxi6x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x89ABCDEF, vi6x89ABCDEF), 8);
      const __m128i vxk6x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x89ABCDEF, vk6x89ABCDEF), 8);
      const __m128i vxi6xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi6xGHIJKLMN, vi6xGHIJKLMN), 8);
      const __m128i vxk6xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk6xGHIJKLMN, vk6xGHIJKLMN), 8);

      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x89ABCDEFlo = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);
      const __m128i vprod6x89ABCDEFhi = _mm_mulhi_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);
      const __m128i vprod6xGHIJKLMNlo = _mm_mullo_epi16(vxi6xGHIJKLMN, vxk6xGHIJKLMN);
      const __m128i vprod6xGHIJKLMNhi = _mm_mulhi_epi16(vxi6xGHIJKLMN, vxk6xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod6x89ABCDEFlo, vprod6x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod6x89ABCDEFlo, vprod6x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod6xGHIJKLMNlo, vprod6xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod6xGHIJKLMNlo, vprod6xGHIJKLMNhi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vi7xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i7 + 16));
      const __m128i vk7xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      i7 += 24;

      const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
      const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);
      const __m128i vxi7x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x89ABCDEF, vi7x89ABCDEF), 8);
      const __m128i vxk7x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x89ABCDEF, vk7x89ABCDEF), 8);
      const __m128i vxi7xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi7xGHIJKLMN, vi7xGHIJKLMN), 8);
      const __m128i vxk7xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk7xGHIJKLMN, vk7xGHIJKLMN), 8);

      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x89ABCDEFlo = _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);
      const __m128i vprod7x89ABCDEFhi = _mm_mulhi_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);
      const __m128i vprod7xGHIJKLMNlo = _mm_mullo_epi16(vxi7xGHIJKLMN, vxk7xGHIJKLMN);
      const __m128i vprod7xGHIJKLMNhi = _mm_mulhi_epi16(vxi7xGHIJKLMN, vxk7xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod7x89ABCDEFlo, vprod7x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod7x89ABCDEFlo, vprod7x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod7xGHIJKLMNlo, vprod7xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod7xGHIJKLMNlo, vprod7xGHIJKLMNhi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const __m128i vi8xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i8 + 16));
      const __m128i vk8xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      i8 += 24;

      const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
      const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);
      const __m128i vxi8x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x89ABCDEF, vi8x89ABCDEF), 8);
      const __m128i vxk8x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x89ABCDEF, vk8x89ABCDEF), 8);
      const __m128i vxi8xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi8xGHIJKLMN, vi8xGHIJKLMN), 8);
      const __m128i vxk8xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vk8xGHIJKLMN, vk8xGHIJKLMN), 8);

      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x89ABCDEFlo = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);
      const __m128i vprod8x89ABCDEFhi = _mm_mulhi_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);
      const __m128i vprod8xGHIJKLMNlo = _mm_mullo_epi16(vxi8xGHIJKLMN, vxk8xGHIJKLMN);
      const __m128i vprod8xGHIJKLMNhi = _mm_mulhi_epi16(vxi8xGHIJKLMN, vxk8xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod8x89ABCDEFlo, vprod8x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod8x89ABCDEFlo, vprod8x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vprod8xGHIJKLMNlo, vprod8xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vprod8xGHIJKLMNlo, vprod8xGHIJKLMNhi));

      w = (const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 216 * sizeof(int8_t));

      const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.multiplier);
      const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.rounding);

      const __m128i vnmask0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123);
      const __m128i vnmask4567 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567);
      const __m128i vnmask89AB = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc89AB);
      const __m128i vnmaskCDEF = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccCDEF);
      const __m128i vnmaskGHIJ = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccGHIJ);
      const __m128i vnmaskKLMN = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccKLMN);

      const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
      const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);
      const __m128i vabsacc89AB = _mm_abs_epi32(vacc89AB);
      const __m128i vabsaccCDEF = _mm_abs_epi32(vaccCDEF);
      const __m128i vabsaccGHIJ = _mm_abs_epi32(vaccGHIJ);
      const __m128i vabsaccKLMN = _mm_abs_epi32(vaccKLMN);

      const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
      const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
      const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
      const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);
      const __m128i vabsacc9B = _mm_shuffle_epi32(vabsacc89AB, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprod8A = _mm_mul_epu32(vabsacc89AB, vmultiplier);
      const __m128i vabsprod9B = _mm_mul_epu32(vabsacc9B, vmultiplier);
      const __m128i vabsaccDF = _mm_shuffle_epi32(vabsaccCDEF, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprodCE = _mm_mul_epu32(vabsaccCDEF, vmultiplier);
      const __m128i vabsprodDF = _mm_mul_epu32(vabsaccDF, vmultiplier);
      const __m128i vabsaccHJ = _mm_shuffle_epi32(vabsaccGHIJ, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprodGI = _mm_mul_epu32(vabsaccGHIJ, vmultiplier);
      const __m128i vabsprodHJ = _mm_mul_epu32(vabsaccHJ, vmultiplier);
      const __m128i vabsaccLN = _mm_shuffle_epi32(vabsaccKLMN, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprodKM = _mm_mul_epu32(vabsaccKLMN, vmultiplier);
      const __m128i vabsprodLN = _mm_mul_epu32(vabsaccLN, vmultiplier);

      const __m128i vnmask02 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask13 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmask46 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask57 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmask8A = _mm_shuffle_epi32(vnmask89AB, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask9B = _mm_shuffle_epi32(vnmask89AB, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmaskCE = _mm_shuffle_epi32(vnmaskCDEF, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmaskDF = _mm_shuffle_epi32(vnmaskCDEF, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmaskGI = _mm_shuffle_epi32(vnmaskGHIJ, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmaskHJ = _mm_shuffle_epi32(vnmaskGHIJ, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmaskKM = _mm_shuffle_epi32(vnmaskKLMN, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmaskLN = _mm_shuffle_epi32(vnmaskKLMN, _MM_SHUFFLE(3, 3, 1, 1));

      const __m128i vprod02 = _mm_sub_epi64(_mm_xor_si128(vabsprod02, vnmask02), vnmask02);
      const __m128i vprod13 = _mm_sub_epi64(_mm_xor_si128(vabsprod13, vnmask13), vnmask13);
      const __m128i vprod46 = _mm_sub_epi64(_mm_xor_si128(vabsprod46, vnmask46), vnmask46);
      const __m128i vprod57 = _mm_sub_epi64(_mm_xor_si128(vabsprod57, vnmask57), vnmask57);
      const __m128i vprod8A = _mm_sub_epi64(_mm_xor_si128(vabsprod8A, vnmask8A), vnmask8A);
      const __m128i vprod9B = _mm_sub_epi64(_mm_xor_si128(vabsprod9B, vnmask9B), vnmask9B);
      const __m128i vprodCE = _mm_sub_epi64(_mm_xor_si128(vabsprodCE, vnmaskCE), vnmaskCE);
      const __m128i vprodDF = _mm_sub_epi64(_mm_xor_si128(vabsprodDF, vnmaskDF), vnmaskDF);
      const __m128i vprodGI = _mm_sub_epi64(_mm_xor_si128(vabsprodGI, vnmaskGI), vnmaskGI);
      const __m128i vprodHJ = _mm_sub_epi64(_mm_xor_si128(vabsprodHJ, vnmaskHJ), vnmaskHJ);
      const __m128i vprodKM = _mm_sub_epi64(_mm_xor_si128(vabsprodKM, vnmaskKM), vnmaskKM);
      const __m128i vprodLN = _mm_sub_epi64(_mm_xor_si128(vabsprodLN, vnmaskLN), vnmaskLN);

      const __m128i vq31prod02 = _mm_srli_epi64(_mm_add_epi64(vprod02, vrounding), 31);
      const __m128i vq31prod13 = _mm_srli_epi64(_mm_add_epi64(vprod13, vrounding), 31);
      const __m128i vq31prod46 = _mm_srli_epi64(_mm_add_epi64(vprod46, vrounding), 31);
      const __m128i vq31prod57 = _mm_srli_epi64(_mm_add_epi64(vprod57, vrounding), 31);
      const __m128i vq31prod8A = _mm_srli_epi64(_mm_add_epi64(vprod8A, vrounding), 31);
      const __m128i vq31prod9B = _mm_srli_epi64(_mm_add_epi64(vprod9B, vrounding), 31);
      const __m128i vq31prodCE = _mm_srli_epi64(_mm_add_epi64(vprodCE, vrounding), 31);
      const __m128i vq31prodDF = _mm_srli_epi64(_mm_add_epi64(vprodDF, vrounding), 31);
      const __m128i vq31prodGI = _mm_srli_epi64(_mm_add_epi64(vprodGI, vrounding), 31);
      const __m128i vq31prodHJ = _mm_srli_epi64(_mm_add_epi64(vprodHJ, vrounding), 31);
      const __m128i vq31prodKM = _mm_srli_epi64(_mm_add_epi64(vprodKM, vrounding), 31);
      const __m128i vq31prodLN = _mm_srli_epi64(_mm_add_epi64(vprodLN, vrounding), 31);

      const __m128i vq31prod0213 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod02), _mm_castsi128_ps(vq31prod13), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prod4657 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod46), _mm_castsi128_ps(vq31prod57), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prod8A9B = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod8A), _mm_castsi128_ps(vq31prod9B), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prodCEDF = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prodCE), _mm_castsi128_ps(vq31prodDF), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prodGIHJ = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prodGI), _mm_castsi128_ps(vq31prodHJ), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prodKMLN = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prodKM), _mm_castsi128_ps(vq31prodLN), _MM_SHUFFLE(2, 0, 2, 0)));

      const __m128i vq31prod0123 = _mm_shuffle_epi32(vq31prod0213, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prod4567 = _mm_shuffle_epi32(vq31prod4657, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prod89AB = _mm_shuffle_epi32(vq31prod8A9B, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prodCDEF = _mm_shuffle_epi32(vq31prodCEDF, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prodGHIJ = _mm_shuffle_epi32(vq31prodGIHJ, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prodKLMN = _mm_shuffle_epi32(vq31prodKMLN, _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_mask);
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

      const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_threshold);
      const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse2.shift);
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

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);
      __m128i voutGHIJKLMN = _mm_adds_epi16(_mm_packs_epi32(vaccGHIJ, vaccKLMN), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout89ABCDEF = _mm_max_epi16(vout89ABCDEF, voutput_min);
      voutGHIJKLMN = _mm_max_epi16(voutGHIJKLMN, voutput_min);

      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_max);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);
      vout89ABCDEF = _mm_min_epi16(vout89ABCDEF, voutput_max);
      voutGHIJKLMN = _mm_min_epi16(voutGHIJKLMN, voutput_max);

      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
      __m128i voutGHIJKLMNGHIJKLMN = _mm_packs_epi16(voutGHIJKLMN, voutGHIJKLMN);


      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      _mm_storel_epi64((__m128i*) (output + 16), voutGHIJKLMNGHIJKLMN);
      output += 24;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 24);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        i0 += 8;

        const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
        const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 24));
        i1 += 8;

        const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
        const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        i2 += 8;

        const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
        const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 72));
        i3 += 8;

        const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
        const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        i4 += 8;

        const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
        const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 120));
        i5 += 8;

        const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
        const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        i6 += 8;

        const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
        const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 168));
        i7 += 8;

        const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
        const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        i8 += 8;

        const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
        const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

        k += 8;

        const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.multiplier);
        const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.rounding);

        const __m128i vnmask0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123);
        const __m128i vnmask4567 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567);

        const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
        const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

        const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
        const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
        const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
        const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

        const __m128i vnmask02 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(2, 2, 0, 0));
        const __m128i vnmask13 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i vnmask46 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(2, 2, 0, 0));
        const __m128i vnmask57 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vprod02 = _mm_sub_epi64(_mm_xor_si128(vabsprod02, vnmask02), vnmask02);
        const __m128i vprod13 = _mm_sub_epi64(_mm_xor_si128(vabsprod13, vnmask13), vnmask13);
        const __m128i vprod46 = _mm_sub_epi64(_mm_xor_si128(vabsprod46, vnmask46), vnmask46);
        const __m128i vprod57 = _mm_sub_epi64(_mm_xor_si128(vabsprod57, vnmask57), vnmask57);

        const __m128i vq31prod02 = _mm_srli_epi64(_mm_add_epi64(vprod02, vrounding), 31);
        const __m128i vq31prod13 = _mm_srli_epi64(_mm_add_epi64(vprod13, vrounding), 31);
        const __m128i vq31prod46 = _mm_srli_epi64(_mm_add_epi64(vprod46, vrounding), 31);
        const __m128i vq31prod57 = _mm_srli_epi64(_mm_add_epi64(vprod57, vrounding), 31);

        const __m128i vq31prod0213 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod02), _mm_castsi128_ps(vq31prod13), _MM_SHUFFLE(2, 0, 2, 0)));
        const __m128i vq31prod4657 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod46), _mm_castsi128_ps(vq31prod57), _MM_SHUFFLE(2, 0, 2, 0)));

        const __m128i vq31prod0123 = _mm_shuffle_epi32(vq31prod0213, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vq31prod4567 = _mm_shuffle_epi32(vq31prod4657, _MM_SHUFFLE(3, 1, 2, 0));

        const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_mask);
        const __m128i vrem0123 =
          _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));
        const __m128i vrem4567 =
          _mm_add_epi32(_mm_and_si128(vq31prod4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod4567));

        const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_threshold);
        const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse2.shift);
        vacc0123 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
        vacc4567 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_min));
        vout01234567 = _mm_min_epi16(vout01234567, _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_max));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
