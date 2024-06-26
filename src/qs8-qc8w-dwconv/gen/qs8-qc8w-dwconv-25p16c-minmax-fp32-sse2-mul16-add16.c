// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-sse-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__sse2_mul16_add16(
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
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i0 += 16;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);
      const __m128i vxi0x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x89ABCDEF, vi0x89ABCDEF), 8);
      const __m128i vxk0x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x89ABCDEF, vk0x89ABCDEF), 8);

      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      i1 += 16;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);
      const __m128i vxi1x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x89ABCDEF, vi1x89ABCDEF), 8);
      const __m128i vxk1x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x89ABCDEF, vk1x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF));

      const __m128i vsignprod1x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod1x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod1x01234567));
      const __m128i vsignprod1x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod1x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod1x89ABCDEF));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i2 += 16;

      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);
      const __m128i vxi2x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x89ABCDEF, vi2x89ABCDEF), 8);
      const __m128i vxk2x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x89ABCDEF, vk2x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      i3 += 16;

      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);
      const __m128i vxi3x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x89ABCDEF, vi3x89ABCDEF), 8);
      const __m128i vxk3x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x89ABCDEF, vk3x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF));

      const __m128i vsignprod3x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod3x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod3x01234567));
      const __m128i vsignprod3x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod3x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod3x89ABCDEF));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      i4 += 16;

      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);
      const __m128i vxi4x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x89ABCDEF, vi4x89ABCDEF), 8);
      const __m128i vxk4x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x89ABCDEF, vk4x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      i5 += 16;

      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);
      const __m128i vxi5x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x89ABCDEF, vi5x89ABCDEF), 8);
      const __m128i vxk5x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x89ABCDEF, vk5x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF));

      const __m128i vsignprod5x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod5x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod5x01234567));
      const __m128i vsignprod5x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod5x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod5x89ABCDEF));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      i6 += 16;

      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
      const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);
      const __m128i vxi6x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x89ABCDEF, vi6x89ABCDEF), 8);
      const __m128i vxk6x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x89ABCDEF, vk6x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      i7 += 16;

      const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
      const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);
      const __m128i vxi7x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x89ABCDEF, vi7x89ABCDEF), 8);
      const __m128i vxk7x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x89ABCDEF, vk7x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF));

      const __m128i vsignprod7x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod7x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod7x01234567));
      const __m128i vsignprod7x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod7x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod7x89ABCDEF));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      i8 += 16;

      const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
      const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);
      const __m128i vxi8x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x89ABCDEF, vi8x89ABCDEF), 8);
      const __m128i vxk8x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x89ABCDEF, vk8x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);


      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vi9x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i9 + 8));
      const __m128i vk9x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      i9 += 16;

      const __m128i vxi9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi9x01234567, vi9x01234567), 8);
      const __m128i vxk9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk9x01234567, vk9x01234567), 8);
      const __m128i vxi9x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi9x89ABCDEF, vi9x89ABCDEF), 8);
      const __m128i vxk9x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk9x89ABCDEF, vk9x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF));

      const __m128i vsignprod9x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod9x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod9x01234567));
      const __m128i vsignprod9x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod9x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod9x89ABCDEF));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vi10x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i10 + 8));
      const __m128i vk10x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      i10 += 16;

      const __m128i vxi10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi10x01234567, vi10x01234567), 8);
      const __m128i vxk10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk10x01234567, vk10x01234567), 8);
      const __m128i vxi10x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi10x89ABCDEF, vi10x89ABCDEF), 8);
      const __m128i vxk10x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk10x89ABCDEF, vk10x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);


      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vi11x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i11 + 8));
      const __m128i vk11x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      i11 += 16;

      const __m128i vxi11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi11x01234567, vi11x01234567), 8);
      const __m128i vxk11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk11x01234567, vk11x01234567), 8);
      const __m128i vxi11x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi11x89ABCDEF, vi11x89ABCDEF), 8);
      const __m128i vxk11x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk11x89ABCDEF, vk11x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF));

      const __m128i vsignprod11x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod11x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod11x01234567));
      const __m128i vsignprod11x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod11x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod11x89ABCDEF));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vi12x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i12 + 8));
      const __m128i vk12x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      i12 += 16;

      const __m128i vxi12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi12x01234567, vi12x01234567), 8);
      const __m128i vxk12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk12x01234567, vk12x01234567), 8);
      const __m128i vxi12x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi12x89ABCDEF, vi12x89ABCDEF), 8);
      const __m128i vxk12x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk12x89ABCDEF, vk12x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);


      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      const __m128i vi13x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i13 + 8));
      const __m128i vk13x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(int8_t)));
      i13 += 16;

      const __m128i vxi13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi13x01234567, vi13x01234567), 8);
      const __m128i vxk13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk13x01234567, vk13x01234567), 8);
      const __m128i vxi13x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi13x89ABCDEF, vi13x89ABCDEF), 8);
      const __m128i vxk13x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk13x89ABCDEF, vk13x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF));

      const __m128i vsignprod13x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod13x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod13x01234567));
      const __m128i vsignprod13x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod13x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod13x89ABCDEF));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t)));
      const __m128i vi14x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i14 + 8));
      const __m128i vk14x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(int8_t)));
      i14 += 16;

      const __m128i vxi14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi14x01234567, vi14x01234567), 8);
      const __m128i vxk14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk14x01234567, vk14x01234567), 8);
      const __m128i vxi14x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi14x89ABCDEF, vi14x89ABCDEF), 8);
      const __m128i vxk14x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk14x89ABCDEF, vk14x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);


      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t)));
      const __m128i vi15x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i15 + 8));
      const __m128i vk15x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(int8_t)));
      i15 += 16;

      const __m128i vxi15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi15x01234567, vi15x01234567), 8);
      const __m128i vxk15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk15x01234567, vk15x01234567), 8);
      const __m128i vxi15x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi15x89ABCDEF, vi15x89ABCDEF), 8);
      const __m128i vxk15x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk15x89ABCDEF, vk15x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF));

      const __m128i vsignprod15x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod15x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod15x01234567));
      const __m128i vsignprod15x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod15x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod15x89ABCDEF));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t)));
      const __m128i vi16x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i16 + 8));
      const __m128i vk16x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(int8_t)));
      i16 += 16;

      const __m128i vxi16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi16x01234567, vi16x01234567), 8);
      const __m128i vxk16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk16x01234567, vk16x01234567), 8);
      const __m128i vxi16x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi16x89ABCDEF, vi16x89ABCDEF), 8);
      const __m128i vxk16x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk16x89ABCDEF, vk16x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);


      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t)));
      const __m128i vi17x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i17 + 8));
      const __m128i vk17x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(int8_t)));
      i17 += 16;

      const __m128i vxi17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi17x01234567, vi17x01234567), 8);
      const __m128i vxk17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk17x01234567, vk17x01234567), 8);
      const __m128i vxi17x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi17x89ABCDEF, vi17x89ABCDEF), 8);
      const __m128i vxk17x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk17x89ABCDEF, vk17x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF));

      const __m128i vsignprod17x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod17x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod17x01234567));
      const __m128i vsignprod17x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod17x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod17x89ABCDEF));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t)));
      const __m128i vi18x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i18 + 8));
      const __m128i vk18x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(int8_t)));
      i18 += 16;

      const __m128i vxi18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi18x01234567, vi18x01234567), 8);
      const __m128i vxk18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk18x01234567, vk18x01234567), 8);
      const __m128i vxi18x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi18x89ABCDEF, vi18x89ABCDEF), 8);
      const __m128i vxk18x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk18x89ABCDEF, vk18x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);


      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t)));
      const __m128i vi19x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i19 + 8));
      const __m128i vk19x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(int8_t)));
      i19 += 16;

      const __m128i vxi19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi19x01234567, vi19x01234567), 8);
      const __m128i vxk19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk19x01234567, vk19x01234567), 8);
      const __m128i vxi19x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi19x89ABCDEF, vi19x89ABCDEF), 8);
      const __m128i vxk19x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk19x89ABCDEF, vk19x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF));

      const __m128i vsignprod19x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod19x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod19x01234567));
      const __m128i vsignprod19x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod19x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod19x89ABCDEF));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t)));
      const __m128i vi20x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i20 + 8));
      const __m128i vk20x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(int8_t)));
      i20 += 16;

      const __m128i vxi20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi20x01234567, vi20x01234567), 8);
      const __m128i vxk20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk20x01234567, vk20x01234567), 8);
      const __m128i vxi20x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi20x89ABCDEF, vi20x89ABCDEF), 8);
      const __m128i vxk20x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk20x89ABCDEF, vk20x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);


      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t)));
      const __m128i vi21x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i21 + 8));
      const __m128i vk21x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(int8_t)));
      i21 += 16;

      const __m128i vxi21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi21x01234567, vi21x01234567), 8);
      const __m128i vxk21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk21x01234567, vk21x01234567), 8);
      const __m128i vxi21x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi21x89ABCDEF, vi21x89ABCDEF), 8);
      const __m128i vxk21x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk21x89ABCDEF, vk21x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF));

      const __m128i vsignprod21x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod21x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod21x01234567));
      const __m128i vsignprod21x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod21x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod21x89ABCDEF));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t)));
      const __m128i vi22x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i22 + 8));
      const __m128i vk22x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(int8_t)));
      i22 += 16;

      const __m128i vxi22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi22x01234567, vi22x01234567), 8);
      const __m128i vxk22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk22x01234567, vk22x01234567), 8);
      const __m128i vxi22x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi22x89ABCDEF, vi22x89ABCDEF), 8);
      const __m128i vxk22x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk22x89ABCDEF, vk22x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);


      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t)));
      const __m128i vi23x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i23 + 8));
      const __m128i vk23x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(int8_t)));
      i23 += 16;

      const __m128i vxi23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi23x01234567, vi23x01234567), 8);
      const __m128i vxk23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk23x01234567, vk23x01234567), 8);
      const __m128i vxi23x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi23x89ABCDEF, vi23x89ABCDEF), 8);
      const __m128i vxk23x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk23x89ABCDEF, vk23x89ABCDEF), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF));

      const __m128i vsignprod23x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod23x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod23x01234567));
      const __m128i vsignprod23x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod23x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod23x89ABCDEF));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t)));
      const __m128i vi24x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i24 + 8));
      const __m128i vk24x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(int8_t)));
      i24 += 16;

      const __m128i vxi24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi24x01234567, vi24x01234567), 8);
      const __m128i vxk24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk24x01234567, vk24x01234567), 8);
      const __m128i vxi24x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi24x89ABCDEF, vi24x89ABCDEF), 8);
      const __m128i vxk24x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vk24x89ABCDEF, vk24x89ABCDEF), 8);

      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);

      const __m128i vsignprod24x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod24x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod24x01234567));
      const __m128i vsignprod24x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod89ABCDEF);
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod89ABCDEF, vsignprod24x89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod89ABCDEF, vsignprod24x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      const __m128 vscale89AB = _mm_loadu_ps((const float*) w + 8);
      const __m128 vscaleCDEF = _mm_loadu_ps((const float*) w + 12);
      w = (const void*) ((const float*) w + 16);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale89AB);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscaleCDEF);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout89ABCDEF = _mm_max_epi16(vout89ABCDEF, voutput_min);

      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);


      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        i0 += 8;

        const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
        const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        i1 += 8;

        const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
        const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        const __m128i vsignprod1x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod1x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod1x01234567));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        i2 += 8;

        const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
        const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        i3 += 8;

        const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
        const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        const __m128i vsignprod3x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod3x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod3x01234567));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        i4 += 8;

        const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
        const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        i5 += 8;

        const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
        const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        const __m128i vsignprod5x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod5x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod5x01234567));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        i6 += 8;

        const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
        const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        i7 += 8;

        const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
        const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        const __m128i vsignprod7x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod7x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod7x01234567));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        i8 += 8;

        const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
        const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        i9 += 8;

        const __m128i vxi9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi9x01234567, vi9x01234567), 8);
        const __m128i vxk9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk9x01234567, vk9x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));

        const __m128i vsignprod9x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod9x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod9x01234567));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) (k + 160));
        i10 += 8;

        const __m128i vxi10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi10x01234567, vi10x01234567), 8);
        const __m128i vxk10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk10x01234567, vk10x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) (k + 176));
        i11 += 8;

        const __m128i vxi11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi11x01234567, vi11x01234567), 8);
        const __m128i vxk11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk11x01234567, vk11x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));

        const __m128i vsignprod11x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod11x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod11x01234567));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        i12 += 8;

        const __m128i vxi12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi12x01234567, vi12x01234567), 8);
        const __m128i vxk12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk12x01234567, vk12x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) (k + 208));
        i13 += 8;

        const __m128i vxi13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi13x01234567, vi13x01234567), 8);
        const __m128i vxk13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk13x01234567, vk13x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));

        const __m128i vsignprod13x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod13x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod13x01234567));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) (k + 224));
        i14 += 8;

        const __m128i vxi14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi14x01234567, vi14x01234567), 8);
        const __m128i vxk14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk14x01234567, vk14x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) (k + 240));
        i15 += 8;

        const __m128i vxi15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi15x01234567, vi15x01234567), 8);
        const __m128i vxk15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk15x01234567, vk15x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));

        const __m128i vsignprod15x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod15x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod15x01234567));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) (k + 256));
        i16 += 8;

        const __m128i vxi16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi16x01234567, vi16x01234567), 8);
        const __m128i vxk16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk16x01234567, vk16x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) (k + 272));
        i17 += 8;

        const __m128i vxi17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi17x01234567, vi17x01234567), 8);
        const __m128i vxk17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk17x01234567, vk17x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));

        const __m128i vsignprod17x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod17x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod17x01234567));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) (k + 288));
        i18 += 8;

        const __m128i vxi18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi18x01234567, vi18x01234567), 8);
        const __m128i vxk18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk18x01234567, vk18x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) (k + 304));
        i19 += 8;

        const __m128i vxi19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi19x01234567, vi19x01234567), 8);
        const __m128i vxk19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk19x01234567, vk19x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));

        const __m128i vsignprod19x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod19x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod19x01234567));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) (k + 320));
        i20 += 8;

        const __m128i vxi20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi20x01234567, vi20x01234567), 8);
        const __m128i vxk20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk20x01234567, vk20x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) (k + 336));
        i21 += 8;

        const __m128i vxi21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi21x01234567, vi21x01234567), 8);
        const __m128i vxk21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk21x01234567, vk21x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));

        const __m128i vsignprod21x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod21x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod21x01234567));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) (k + 352));
        i22 += 8;

        const __m128i vxi22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi22x01234567, vi22x01234567), 8);
        const __m128i vxk22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk22x01234567, vk22x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) (k + 368));
        i23 += 8;

        const __m128i vxi23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi23x01234567, vi23x01234567), 8);
        const __m128i vxk23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk23x01234567, vk23x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));

        const __m128i vsignprod23x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod23x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod23x01234567));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) (k + 384));
        i24 += 8;

        const __m128i vxi24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi24x01234567, vi24x01234567), 8);
        const __m128i vxk24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk24x01234567, vk24x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        const __m128i vsignprod24x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod24x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod24x01234567));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


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
