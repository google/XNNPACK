// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-sse-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/dwconv.h>


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up24x9__avx_mul16(
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
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
      const __m128i vxi0xGHIJKLMN = _mm_cvtepi8_epi16(vi0xGHIJKLMN);
      const __m128i vk0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk0xGHIJKLMN = _mm_cvtepi8_epi16(vk0xGHIJKLMN);
      i0 += 24;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);
      __m128i vprodGHIJKLMN = _mm_mullo_epi16(vxi0xGHIJKLMN, vxk0xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
      const __m128i vxi1xGHIJKLMN = _mm_cvtepi8_epi16(vi1xGHIJKLMN);
      const __m128i vk1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk1xGHIJKLMN = _mm_cvtepi8_epi16(vk1xGHIJKLMN);
      i1 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi1xGHIJKLMN, vxk1xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
      const __m128i vxi2xGHIJKLMN = _mm_cvtepi8_epi16(vi2xGHIJKLMN);
      const __m128i vk2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk2xGHIJKLMN = _mm_cvtepi8_epi16(vk2xGHIJKLMN);
      i2 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi2xGHIJKLMN, vxk2xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
      const __m128i vxi3xGHIJKLMN = _mm_cvtepi8_epi16(vi3xGHIJKLMN);
      const __m128i vk3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk3xGHIJKLMN = _mm_cvtepi8_epi16(vk3xGHIJKLMN);
      i3 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi3xGHIJKLMN, vxk3xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
      const __m128i vxi4xGHIJKLMN = _mm_cvtepi8_epi16(vi4xGHIJKLMN);
      const __m128i vk4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk4xGHIJKLMN = _mm_cvtepi8_epi16(vk4xGHIJKLMN);
      i4 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi4xGHIJKLMN, vxk4xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
      const __m128i vxi5xGHIJKLMN = _mm_cvtepi8_epi16(vi5xGHIJKLMN);
      const __m128i vk5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk5xGHIJKLMN = _mm_cvtepi8_epi16(vk5xGHIJKLMN);
      i5 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi5xGHIJKLMN, vxk5xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
      const __m128i vxi6xGHIJKLMN = _mm_cvtepi8_epi16(vi6xGHIJKLMN);
      const __m128i vk6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vxk6xGHIJKLMN = _mm_cvtepi8_epi16(vk6xGHIJKLMN);
      i6 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi6xGHIJKLMN, vxk6xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      const __m128i vi7xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i7 + 16));
      const __m128i vxi7xGHIJKLMN = _mm_cvtepi8_epi16(vi7xGHIJKLMN);
      const __m128i vk7xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      const __m128i vxk7xGHIJKLMN = _mm_cvtepi8_epi16(vk7xGHIJKLMN);
      i7 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi7xGHIJKLMN, vxk7xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      const __m128i vi8xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i8 + 16));
      const __m128i vxi8xGHIJKLMN = _mm_cvtepi8_epi16(vi8xGHIJKLMN);
      const __m128i vk8xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      const __m128i vxk8xGHIJKLMN = _mm_cvtepi8_epi16(vk8xGHIJKLMN);
      i8 += 24;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);
      vprodGHIJKLMN = _mm_mullo_epi16(vxi8xGHIJKLMN, vxk8xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vprodGHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_srai_epi32(_mm_unpackhi_epi16(vprodGHIJKLMN, vprodGHIJKLMN), 16));

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


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
      __m128i voutGHIJKLMNGHIJKLMN = _mm_packs_epi16(voutGHIJKLMN, voutGHIJKLMN);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      voutGHIJKLMNGHIJKLMN = _mm_max_epi8(voutGHIJKLMNGHIJKLMN, voutput_min);

      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_max);
      vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);
      voutGHIJKLMNGHIJKLMN = _mm_min_epi8(voutGHIJKLMNGHIJKLMN, voutput_max);

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
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 24));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 72));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 120));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 168));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.multiplier);
        const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.rounding);

        const __m128i vacc13 = _mm_shuffle_epi32(vacc0123, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i vacc57 = _mm_shuffle_epi32(vacc4567, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vprod02 = _mm_add_epi64(_mm_mul_epi32(vacc0123, vmultiplier), vrounding);
        const __m128i vprod46 = _mm_add_epi64(_mm_mul_epi32(vacc4567, vmultiplier), vrounding);

        const __m128i vprod13 = _mm_add_epi64(_mm_mul_epi32(vacc13, vmultiplier), vrounding);
        const __m128i vprod57 = _mm_add_epi64(_mm_mul_epi32(vacc57, vmultiplier), vrounding);

        const __m128i vq31prod02 = _mm_srli_epi64(vprod02, 31);
        const __m128i vq31prod13 = _mm_add_epi64(vprod13, vprod13);
        const __m128i vq31prod46 = _mm_srli_epi64(vprod46, 31);
        const __m128i vq31prod57 = _mm_add_epi64(vprod57, vprod57);

        const __m128i vq31prod0123 = _mm_blend_epi16(vq31prod02, vq31prod13, 0xCC);
        const __m128i vq31prod4567 = _mm_blend_epi16(vq31prod46, vq31prod57, 0xCC);

        const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_mask);
        const __m128i vrem0123 =
          _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));
        const __m128i vrem4567 =
          _mm_add_epi32(_mm_and_si128(vq31prod4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod4567));

        const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_threshold);
        const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse4.shift);
        vacc0123 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
        vacc4567 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_min));
        vout0123456701234567 = _mm_min_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_max));

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
