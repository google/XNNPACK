// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif

#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/vadd.h>


void xnn_qc8_dwconv_minmax_fp32_ukernel_25p16c__xop_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);


      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
      const __m128i vi9x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i9 + 8));
      const __m128i vxi9x89ABCDEF = _mm_cvtepi8_epi16(vi9x89ABCDEF);
      const __m128i vk9x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vxk9x89ABCDEF = _mm_cvtepi8_epi16(vk9x89ABCDEF);
      i9 += 16;


      vprod01234567 = _mm_macc_epi16(vxi9x01234567, vxk9x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
      const __m128i vi10x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i10 + 8));
      const __m128i vxi10x89ABCDEF = _mm_cvtepi8_epi16(vi10x89ABCDEF);
      const __m128i vk10x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vxk10x89ABCDEF = _mm_cvtepi8_epi16(vk10x89ABCDEF);
      i10 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);


      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
      const __m128i vi11x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i11 + 8));
      const __m128i vxi11x89ABCDEF = _mm_cvtepi8_epi16(vi11x89ABCDEF);
      const __m128i vk11x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      const __m128i vxk11x89ABCDEF = _mm_cvtepi8_epi16(vk11x89ABCDEF);
      i11 += 16;


      vprod01234567 = _mm_macc_epi16(vxi11x01234567, vxk11x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
      const __m128i vi12x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i12 + 8));
      const __m128i vxi12x89ABCDEF = _mm_cvtepi8_epi16(vi12x89ABCDEF);
      const __m128i vk12x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const __m128i vxk12x89ABCDEF = _mm_cvtepi8_epi16(vk12x89ABCDEF);
      i12 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);


      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
      const __m128i vi13x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i13 + 8));
      const __m128i vxi13x89ABCDEF = _mm_cvtepi8_epi16(vi13x89ABCDEF);
      const __m128i vk13x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(int8_t)));
      const __m128i vxk13x89ABCDEF = _mm_cvtepi8_epi16(vk13x89ABCDEF);
      i13 += 16;


      vprod01234567 = _mm_macc_epi16(vxi13x01234567, vxk13x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t)));
      const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
      const __m128i vi14x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i14 + 8));
      const __m128i vxi14x89ABCDEF = _mm_cvtepi8_epi16(vi14x89ABCDEF);
      const __m128i vk14x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(int8_t)));
      const __m128i vxk14x89ABCDEF = _mm_cvtepi8_epi16(vk14x89ABCDEF);
      i14 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);


      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t)));
      const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
      const __m128i vi15x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i15 + 8));
      const __m128i vxi15x89ABCDEF = _mm_cvtepi8_epi16(vi15x89ABCDEF);
      const __m128i vk15x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(int8_t)));
      const __m128i vxk15x89ABCDEF = _mm_cvtepi8_epi16(vk15x89ABCDEF);
      i15 += 16;


      vprod01234567 = _mm_macc_epi16(vxi15x01234567, vxk15x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t)));
      const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
      const __m128i vi16x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i16 + 8));
      const __m128i vxi16x89ABCDEF = _mm_cvtepi8_epi16(vi16x89ABCDEF);
      const __m128i vk16x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(int8_t)));
      const __m128i vxk16x89ABCDEF = _mm_cvtepi8_epi16(vk16x89ABCDEF);
      i16 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);


      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t)));
      const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
      const __m128i vi17x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i17 + 8));
      const __m128i vxi17x89ABCDEF = _mm_cvtepi8_epi16(vi17x89ABCDEF);
      const __m128i vk17x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(int8_t)));
      const __m128i vxk17x89ABCDEF = _mm_cvtepi8_epi16(vk17x89ABCDEF);
      i17 += 16;


      vprod01234567 = _mm_macc_epi16(vxi17x01234567, vxk17x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t)));
      const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
      const __m128i vi18x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i18 + 8));
      const __m128i vxi18x89ABCDEF = _mm_cvtepi8_epi16(vi18x89ABCDEF);
      const __m128i vk18x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(int8_t)));
      const __m128i vxk18x89ABCDEF = _mm_cvtepi8_epi16(vk18x89ABCDEF);
      i18 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);


      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t)));
      const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
      const __m128i vi19x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i19 + 8));
      const __m128i vxi19x89ABCDEF = _mm_cvtepi8_epi16(vi19x89ABCDEF);
      const __m128i vk19x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(int8_t)));
      const __m128i vxk19x89ABCDEF = _mm_cvtepi8_epi16(vk19x89ABCDEF);
      i19 += 16;


      vprod01234567 = _mm_macc_epi16(vxi19x01234567, vxk19x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t)));
      const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
      const __m128i vi20x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i20 + 8));
      const __m128i vxi20x89ABCDEF = _mm_cvtepi8_epi16(vi20x89ABCDEF);
      const __m128i vk20x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(int8_t)));
      const __m128i vxk20x89ABCDEF = _mm_cvtepi8_epi16(vk20x89ABCDEF);
      i20 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);


      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t)));
      const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
      const __m128i vi21x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i21 + 8));
      const __m128i vxi21x89ABCDEF = _mm_cvtepi8_epi16(vi21x89ABCDEF);
      const __m128i vk21x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(int8_t)));
      const __m128i vxk21x89ABCDEF = _mm_cvtepi8_epi16(vk21x89ABCDEF);
      i21 += 16;


      vprod01234567 = _mm_macc_epi16(vxi21x01234567, vxk21x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t)));
      const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
      const __m128i vi22x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i22 + 8));
      const __m128i vxi22x89ABCDEF = _mm_cvtepi8_epi16(vi22x89ABCDEF);
      const __m128i vk22x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(int8_t)));
      const __m128i vxk22x89ABCDEF = _mm_cvtepi8_epi16(vk22x89ABCDEF);
      i22 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);


      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t)));
      const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
      const __m128i vi23x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i23 + 8));
      const __m128i vxi23x89ABCDEF = _mm_cvtepi8_epi16(vi23x89ABCDEF);
      const __m128i vk23x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(int8_t)));
      const __m128i vxk23x89ABCDEF = _mm_cvtepi8_epi16(vk23x89ABCDEF);
      i23 += 16;


      vprod01234567 = _mm_macc_epi16(vxi23x01234567, vxk23x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t)));
      const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
      const __m128i vi24x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i24 + 8));
      const __m128i vxi24x89ABCDEF = _mm_cvtepi8_epi16(vi24x89ABCDEF);
      const __m128i vk24x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(int8_t)));
      const __m128i vxk24x89ABCDEF = _mm_cvtepi8_epi16(vk24x89ABCDEF);
      i24 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

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

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
        i9 += 8;


        vprod01234567 = _mm_macc_epi16(vxi9x01234567, vxk9x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) (k + 160));
        const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
        i10 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) (k + 176));
        const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
        i11 += 8;


        vprod01234567 = _mm_macc_epi16(vxi11x01234567, vxk11x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
        i12 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) (k + 208));
        const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
        i13 += 8;


        vprod01234567 = _mm_macc_epi16(vxi13x01234567, vxk13x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) (k + 224));
        const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
        i14 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) (k + 240));
        const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
        i15 += 8;


        vprod01234567 = _mm_macc_epi16(vxi15x01234567, vxk15x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) (k + 256));
        const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
        i16 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) (k + 272));
        const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
        i17 += 8;


        vprod01234567 = _mm_macc_epi16(vxi17x01234567, vxk17x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) (k + 288));
        const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
        i18 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) (k + 304));
        const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
        i19 += 8;


        vprod01234567 = _mm_macc_epi16(vxi19x01234567, vxk19x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) (k + 320));
        const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
        i20 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) (k + 336));
        const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
        i21 += 8;


        vprod01234567 = _mm_macc_epi16(vxi21x01234567, vxk21x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) (k + 352));
        const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
        i22 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) (k + 368));
        const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
        i23 += 8;


        vprod01234567 = _mm_macc_epi16(vxi23x01234567, vxk23x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) (k + 384));
        const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
        i24 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

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

void xnn_qc8_dwconv_minmax_fp32_ukernel_3p16c__xop_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t));

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

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

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

void xnn_qc8_dwconv_minmax_fp32_ukernel_9p16c__xop_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t));

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

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

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

void xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
      a1 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
      vacc1x0 = _mm_maddd_epi16(vxa1, vxb0, vacc1x0);
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
      vacc1x1 = _mm_maddd_epi16(vxa1, vxb1, vacc1x1);
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
      vacc1x2 = _mm_maddd_epi16(vxa1, vxb2, vacc1x2);
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);
      vacc1x3 = _mm_maddd_epi16(vxa1, vxb3, vacc1x3);

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
        vacc1x0 = _mm_maddd_epi16(vxa1, vxb0, vacc1x0);
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
        vacc1x1 = _mm_maddd_epi16(vxa1, vxb1, vacc1x1);
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
        vacc1x2 = _mm_maddd_epi16(vxa1, vxb2, vacc1x2);
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);
        vacc1x3 = _mm_maddd_epi16(vxa1, vxb3, vacc1x3);

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__xop_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);


      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
      const __m128i vi9x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i9 + 8));
      const __m128i vxi9x89ABCDEF = _mm_cvtepi8_epi16(vi9x89ABCDEF);
      const __m128i vk9x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vxk9x89ABCDEF = _mm_cvtepi8_epi16(vk9x89ABCDEF);
      i9 += 16;


      vprod01234567 = _mm_macc_epi16(vxi9x01234567, vxk9x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
      const __m128i vi10x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i10 + 8));
      const __m128i vxi10x89ABCDEF = _mm_cvtepi8_epi16(vi10x89ABCDEF);
      const __m128i vk10x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vxk10x89ABCDEF = _mm_cvtepi8_epi16(vk10x89ABCDEF);
      i10 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);


      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
      const __m128i vi11x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i11 + 8));
      const __m128i vxi11x89ABCDEF = _mm_cvtepi8_epi16(vi11x89ABCDEF);
      const __m128i vk11x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      const __m128i vxk11x89ABCDEF = _mm_cvtepi8_epi16(vk11x89ABCDEF);
      i11 += 16;


      vprod01234567 = _mm_macc_epi16(vxi11x01234567, vxk11x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
      const __m128i vi12x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i12 + 8));
      const __m128i vxi12x89ABCDEF = _mm_cvtepi8_epi16(vi12x89ABCDEF);
      const __m128i vk12x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const __m128i vxk12x89ABCDEF = _mm_cvtepi8_epi16(vk12x89ABCDEF);
      i12 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);


      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
      const __m128i vi13x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i13 + 8));
      const __m128i vxi13x89ABCDEF = _mm_cvtepi8_epi16(vi13x89ABCDEF);
      const __m128i vk13x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(int8_t)));
      const __m128i vxk13x89ABCDEF = _mm_cvtepi8_epi16(vk13x89ABCDEF);
      i13 += 16;


      vprod01234567 = _mm_macc_epi16(vxi13x01234567, vxk13x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t)));
      const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
      const __m128i vi14x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i14 + 8));
      const __m128i vxi14x89ABCDEF = _mm_cvtepi8_epi16(vi14x89ABCDEF);
      const __m128i vk14x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(int8_t)));
      const __m128i vxk14x89ABCDEF = _mm_cvtepi8_epi16(vk14x89ABCDEF);
      i14 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);


      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t)));
      const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
      const __m128i vi15x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i15 + 8));
      const __m128i vxi15x89ABCDEF = _mm_cvtepi8_epi16(vi15x89ABCDEF);
      const __m128i vk15x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(int8_t)));
      const __m128i vxk15x89ABCDEF = _mm_cvtepi8_epi16(vk15x89ABCDEF);
      i15 += 16;


      vprod01234567 = _mm_macc_epi16(vxi15x01234567, vxk15x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t)));
      const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
      const __m128i vi16x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i16 + 8));
      const __m128i vxi16x89ABCDEF = _mm_cvtepi8_epi16(vi16x89ABCDEF);
      const __m128i vk16x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(int8_t)));
      const __m128i vxk16x89ABCDEF = _mm_cvtepi8_epi16(vk16x89ABCDEF);
      i16 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);


      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t)));
      const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
      const __m128i vi17x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i17 + 8));
      const __m128i vxi17x89ABCDEF = _mm_cvtepi8_epi16(vi17x89ABCDEF);
      const __m128i vk17x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(int8_t)));
      const __m128i vxk17x89ABCDEF = _mm_cvtepi8_epi16(vk17x89ABCDEF);
      i17 += 16;


      vprod01234567 = _mm_macc_epi16(vxi17x01234567, vxk17x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t)));
      const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
      const __m128i vi18x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i18 + 8));
      const __m128i vxi18x89ABCDEF = _mm_cvtepi8_epi16(vi18x89ABCDEF);
      const __m128i vk18x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(int8_t)));
      const __m128i vxk18x89ABCDEF = _mm_cvtepi8_epi16(vk18x89ABCDEF);
      i18 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);


      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t)));
      const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
      const __m128i vi19x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i19 + 8));
      const __m128i vxi19x89ABCDEF = _mm_cvtepi8_epi16(vi19x89ABCDEF);
      const __m128i vk19x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(int8_t)));
      const __m128i vxk19x89ABCDEF = _mm_cvtepi8_epi16(vk19x89ABCDEF);
      i19 += 16;


      vprod01234567 = _mm_macc_epi16(vxi19x01234567, vxk19x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t)));
      const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
      const __m128i vi20x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i20 + 8));
      const __m128i vxi20x89ABCDEF = _mm_cvtepi8_epi16(vi20x89ABCDEF);
      const __m128i vk20x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(int8_t)));
      const __m128i vxk20x89ABCDEF = _mm_cvtepi8_epi16(vk20x89ABCDEF);
      i20 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);


      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t)));
      const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
      const __m128i vi21x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i21 + 8));
      const __m128i vxi21x89ABCDEF = _mm_cvtepi8_epi16(vi21x89ABCDEF);
      const __m128i vk21x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(int8_t)));
      const __m128i vxk21x89ABCDEF = _mm_cvtepi8_epi16(vk21x89ABCDEF);
      i21 += 16;


      vprod01234567 = _mm_macc_epi16(vxi21x01234567, vxk21x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t)));
      const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
      const __m128i vi22x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i22 + 8));
      const __m128i vxi22x89ABCDEF = _mm_cvtepi8_epi16(vi22x89ABCDEF);
      const __m128i vk22x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(int8_t)));
      const __m128i vxk22x89ABCDEF = _mm_cvtepi8_epi16(vk22x89ABCDEF);
      i22 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);


      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t)));
      const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
      const __m128i vi23x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i23 + 8));
      const __m128i vxi23x89ABCDEF = _mm_cvtepi8_epi16(vi23x89ABCDEF);
      const __m128i vk23x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(int8_t)));
      const __m128i vxk23x89ABCDEF = _mm_cvtepi8_epi16(vk23x89ABCDEF);
      i23 += 16;


      vprod01234567 = _mm_macc_epi16(vxi23x01234567, vxk23x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t)));
      const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
      const __m128i vi24x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i24 + 8));
      const __m128i vxi24x89ABCDEF = _mm_cvtepi8_epi16(vi24x89ABCDEF);
      const __m128i vk24x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(int8_t)));
      const __m128i vxk24x89ABCDEF = _mm_cvtepi8_epi16(vk24x89ABCDEF);
      i24 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
        i9 += 8;


        vprod01234567 = _mm_macc_epi16(vxi9x01234567, vxk9x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) (k + 160));
        const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
        i10 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) (k + 176));
        const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
        i11 += 8;


        vprod01234567 = _mm_macc_epi16(vxi11x01234567, vxk11x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
        i12 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) (k + 208));
        const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
        i13 += 8;


        vprod01234567 = _mm_macc_epi16(vxi13x01234567, vxk13x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) (k + 224));
        const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
        i14 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) (k + 240));
        const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
        i15 += 8;


        vprod01234567 = _mm_macc_epi16(vxi15x01234567, vxk15x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) (k + 256));
        const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
        i16 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) (k + 272));
        const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
        i17 += 8;


        vprod01234567 = _mm_macc_epi16(vxi17x01234567, vxk17x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) (k + 288));
        const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
        i18 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) (k + 304));
        const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
        i19 += 8;


        vprod01234567 = _mm_macc_epi16(vxi19x01234567, vxk19x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) (k + 320));
        const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
        i20 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) (k + 336));
        const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
        i21 += 8;


        vprod01234567 = _mm_macc_epi16(vxi21x01234567, vxk21x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) (k + 352));
        const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
        i22 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) (k + 368));
        const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
        i23 += 8;


        vprod01234567 = _mm_macc_epi16(vxi23x01234567, vxk23x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) (k + 384));
        const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
        i24 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

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

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__xop_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);
      vprod89ABCDEF = _mm_macc_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF, vprod89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_macc_epi16(vxi1x01234567, vxk1x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_macc_epi16(vxi3x01234567, vxk3x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_macc_epi16(vxi5x01234567, vxk5x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_macc_epi16(vxi7x01234567, vxk7x01234567, vprod01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

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

void xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
      a1 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
      vacc1x0 = _mm_maddd_epi16(vxa1, vxb0, vacc1x0);
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
      vacc1x1 = _mm_maddd_epi16(vxa1, vxb1, vacc1x1);
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
      vacc1x2 = _mm_maddd_epi16(vxa1, vxb2, vacc1x2);
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);
      vacc1x3 = _mm_maddd_epi16(vxa1, vxb3, vacc1x3);

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
        vacc1x0 = _mm_maddd_epi16(vxa1, vxb0, vacc1x0);
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
        vacc1x1 = _mm_maddd_epi16(vxa1, vxb1, vacc1x1);
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
        vacc1x2 = _mm_maddd_epi16(vxa1, vxb2, vacc1x2);
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);
        vacc1x3 = _mm_maddd_epi16(vxa1, vxb3, vacc1x3);

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__xop_mul32_ld32_x8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse4_mul32.bias);
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4_mul32.a_multiplier);
  const __m128i vb_multiplier = _mm_load_si128((const __m128i*) params->sse4_mul32.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->sse4_mul32.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4_mul32.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4_mul32.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4_mul32.output_max);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const __m128i va0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
    const __m128i vb0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b)));
    const __m128i va4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
    const __m128i vb4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 4)));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
    __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

    vacc0123 = _mm_macc_epi32(vb0123, vb_multiplier, vacc0123);
    vacc4567 = _mm_macc_epi32(vb4567, vb_multiplier, vacc4567);

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
      const __m128i vb0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b)));
      const __m128i va4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
      const __m128i vb4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 4)));

      __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
      __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

      vacc0123 = _mm_macc_epi32(vb0123, vb_multiplier, vacc0123);
      vacc4567 = _mm_macc_epi32(vb4567, vb_multiplier, vacc4567);

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4_mul32.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->sse4_mul32.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4_mul32.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4_mul32.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4_mul32.output_max);

  __m128i vbias = _mm_cvtsi32_si128(params->sse4_mul32.b_multiplier[0] * (int32_t) *input_b);
  vbias = _mm_shuffle_epi32(vbias, _MM_SHUFFLE(0, 0, 0, 0));
  vbias = _mm_add_epi32(vbias, _mm_load_si128((const __m128i*) params->sse4_mul32.bias));
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const __m128i va0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
    const __m128i va4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
    __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
      const __m128i va4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));

      __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
      __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__xop_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128i vk_zero_point = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*) params->fp32_sse2.kernel_zero_point));
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0)));
      const __m128i vk0x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi0x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 4)));
      const __m128i vk0x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 4 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi0x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 8)));
      const __m128i vk0x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi0xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 12)));
      const __m128i vk0xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 12 * sizeof(uint8_t))))), vk_zero_point);
      i0 += 16;

      vacc0123 = _mm_macc_epi32(vi0x0123, vk0x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi0x4567, vk0x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi0x89AB, vk0x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi0xCDEF, vk0xCDEF, vaccCDEF);

      const __m128i vi1x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1)));
      const __m128i vk1x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi1x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 4)));
      const __m128i vk1x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 20 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi1x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 8)));
      const __m128i vk1x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi1xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 12)));
      const __m128i vk1xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 28 * sizeof(uint8_t))))), vk_zero_point);
      i1 += 16;

      vacc0123 = _mm_macc_epi32(vi1x0123, vk1x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi1x4567, vk1x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi1x89AB, vk1x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi1xCDEF, vk1xCDEF, vaccCDEF);

      const __m128i vi2x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2)));
      const __m128i vk2x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi2x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 4)));
      const __m128i vk2x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 36 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi2x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 8)));
      const __m128i vk2x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi2xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 12)));
      const __m128i vk2xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 44 * sizeof(uint8_t))))), vk_zero_point);
      i2 += 16;

      vacc0123 = _mm_macc_epi32(vi2x0123, vk2x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi2x4567, vk2x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi2x89AB, vk2x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi2xCDEF, vk2xCDEF, vaccCDEF);

      const __m128i vi3x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3)));
      const __m128i vk3x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi3x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 4)));
      const __m128i vk3x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 52 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi3x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 8)));
      const __m128i vk3x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi3xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 12)));
      const __m128i vk3xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 60 * sizeof(uint8_t))))), vk_zero_point);
      i3 += 16;

      vacc0123 = _mm_macc_epi32(vi3x0123, vk3x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi3x4567, vk3x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi3x89AB, vk3x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi3xCDEF, vk3xCDEF, vaccCDEF);

      const __m128i vi4x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4)));
      const __m128i vk4x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi4x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 4)));
      const __m128i vk4x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 68 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi4x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 8)));
      const __m128i vk4x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi4xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 12)));
      const __m128i vk4xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 76 * sizeof(uint8_t))))), vk_zero_point);
      i4 += 16;

      vacc0123 = _mm_macc_epi32(vi4x0123, vk4x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi4x4567, vk4x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi4x89AB, vk4x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi4xCDEF, vk4xCDEF, vaccCDEF);

      const __m128i vi5x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5)));
      const __m128i vk5x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi5x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 4)));
      const __m128i vk5x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 84 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi5x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 8)));
      const __m128i vk5x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi5xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 12)));
      const __m128i vk5xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 92 * sizeof(uint8_t))))), vk_zero_point);
      i5 += 16;

      vacc0123 = _mm_macc_epi32(vi5x0123, vk5x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi5x4567, vk5x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi5x89AB, vk5x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi5xCDEF, vk5xCDEF, vaccCDEF);

      const __m128i vi6x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6)));
      const __m128i vk6x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi6x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 4)));
      const __m128i vk6x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 100 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi6x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 8)));
      const __m128i vk6x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi6xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 12)));
      const __m128i vk6xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 108 * sizeof(uint8_t))))), vk_zero_point);
      i6 += 16;

      vacc0123 = _mm_macc_epi32(vi6x0123, vk6x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi6x4567, vk6x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi6x89AB, vk6x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi6xCDEF, vk6xCDEF, vaccCDEF);

      const __m128i vi7x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7)));
      const __m128i vk7x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi7x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 4)));
      const __m128i vk7x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 116 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi7x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 8)));
      const __m128i vk7x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi7xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 12)));
      const __m128i vk7xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 124 * sizeof(uint8_t))))), vk_zero_point);
      i7 += 16;

      vacc0123 = _mm_macc_epi32(vi7x0123, vk7x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi7x4567, vk7x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi7x89AB, vk7x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi7xCDEF, vk7xCDEF, vaccCDEF);

      const __m128i vi8x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8)));
      const __m128i vk8x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi8x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 4)));
      const __m128i vk8x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 132 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi8x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 8)));
      const __m128i vk8x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi8xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 12)));
      const __m128i vk8xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 140 * sizeof(uint8_t))))), vk_zero_point);
      i8 += 16;

      vacc0123 = _mm_macc_epi32(vi8x0123, vk8x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi8x4567, vk8x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi8x89AB, vk8x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi8xCDEF, vk8xCDEF, vaccCDEF);

      const __m128i vi9x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i9)));
      const __m128i vk9x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi9x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i9 + 4)));
      const __m128i vk9x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 148 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi9x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i9 + 8)));
      const __m128i vk9x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi9xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i9 + 12)));
      const __m128i vk9xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 156 * sizeof(uint8_t))))), vk_zero_point);
      i9 += 16;

      vacc0123 = _mm_macc_epi32(vi9x0123, vk9x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi9x4567, vk9x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi9x89AB, vk9x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi9xCDEF, vk9xCDEF, vaccCDEF);

      const __m128i vi10x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i10)));
      const __m128i vk10x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi10x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i10 + 4)));
      const __m128i vk10x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 164 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi10x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i10 + 8)));
      const __m128i vk10x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi10xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i10 + 12)));
      const __m128i vk10xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 172 * sizeof(uint8_t))))), vk_zero_point);
      i10 += 16;

      vacc0123 = _mm_macc_epi32(vi10x0123, vk10x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi10x4567, vk10x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi10x89AB, vk10x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi10xCDEF, vk10xCDEF, vaccCDEF);

      const __m128i vi11x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i11)));
      const __m128i vk11x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi11x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i11 + 4)));
      const __m128i vk11x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 180 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi11x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i11 + 8)));
      const __m128i vk11x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi11xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i11 + 12)));
      const __m128i vk11xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 188 * sizeof(uint8_t))))), vk_zero_point);
      i11 += 16;

      vacc0123 = _mm_macc_epi32(vi11x0123, vk11x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi11x4567, vk11x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi11x89AB, vk11x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi11xCDEF, vk11xCDEF, vaccCDEF);

      const __m128i vi12x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i12)));
      const __m128i vk12x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi12x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i12 + 4)));
      const __m128i vk12x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 196 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi12x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i12 + 8)));
      const __m128i vk12x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi12xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i12 + 12)));
      const __m128i vk12xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 204 * sizeof(uint8_t))))), vk_zero_point);
      i12 += 16;

      vacc0123 = _mm_macc_epi32(vi12x0123, vk12x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi12x4567, vk12x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi12x89AB, vk12x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi12xCDEF, vk12xCDEF, vaccCDEF);

      const __m128i vi13x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i13)));
      const __m128i vk13x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi13x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i13 + 4)));
      const __m128i vk13x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 212 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi13x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i13 + 8)));
      const __m128i vk13x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi13xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i13 + 12)));
      const __m128i vk13xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 220 * sizeof(uint8_t))))), vk_zero_point);
      i13 += 16;

      vacc0123 = _mm_macc_epi32(vi13x0123, vk13x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi13x4567, vk13x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi13x89AB, vk13x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi13xCDEF, vk13xCDEF, vaccCDEF);

      const __m128i vi14x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i14)));
      const __m128i vk14x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi14x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i14 + 4)));
      const __m128i vk14x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 228 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi14x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i14 + 8)));
      const __m128i vk14x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi14xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i14 + 12)));
      const __m128i vk14xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 236 * sizeof(uint8_t))))), vk_zero_point);
      i14 += 16;

      vacc0123 = _mm_macc_epi32(vi14x0123, vk14x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi14x4567, vk14x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi14x89AB, vk14x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi14xCDEF, vk14xCDEF, vaccCDEF);

      const __m128i vi15x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i15)));
      const __m128i vk15x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi15x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i15 + 4)));
      const __m128i vk15x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 244 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi15x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i15 + 8)));
      const __m128i vk15x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi15xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i15 + 12)));
      const __m128i vk15xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 252 * sizeof(uint8_t))))), vk_zero_point);
      i15 += 16;

      vacc0123 = _mm_macc_epi32(vi15x0123, vk15x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi15x4567, vk15x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi15x89AB, vk15x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi15xCDEF, vk15xCDEF, vaccCDEF);

      const __m128i vi16x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i16)));
      const __m128i vk16x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi16x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i16 + 4)));
      const __m128i vk16x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 260 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi16x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i16 + 8)));
      const __m128i vk16x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi16xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i16 + 12)));
      const __m128i vk16xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 268 * sizeof(uint8_t))))), vk_zero_point);
      i16 += 16;

      vacc0123 = _mm_macc_epi32(vi16x0123, vk16x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi16x4567, vk16x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi16x89AB, vk16x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi16xCDEF, vk16xCDEF, vaccCDEF);

      const __m128i vi17x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i17)));
      const __m128i vk17x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi17x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i17 + 4)));
      const __m128i vk17x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 276 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi17x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i17 + 8)));
      const __m128i vk17x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi17xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i17 + 12)));
      const __m128i vk17xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 284 * sizeof(uint8_t))))), vk_zero_point);
      i17 += 16;

      vacc0123 = _mm_macc_epi32(vi17x0123, vk17x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi17x4567, vk17x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi17x89AB, vk17x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi17xCDEF, vk17xCDEF, vaccCDEF);

      const __m128i vi18x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i18)));
      const __m128i vk18x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi18x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i18 + 4)));
      const __m128i vk18x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 292 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi18x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i18 + 8)));
      const __m128i vk18x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi18xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i18 + 12)));
      const __m128i vk18xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 300 * sizeof(uint8_t))))), vk_zero_point);
      i18 += 16;

      vacc0123 = _mm_macc_epi32(vi18x0123, vk18x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi18x4567, vk18x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi18x89AB, vk18x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi18xCDEF, vk18xCDEF, vaccCDEF);

      const __m128i vi19x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i19)));
      const __m128i vk19x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi19x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i19 + 4)));
      const __m128i vk19x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 308 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi19x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i19 + 8)));
      const __m128i vk19x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi19xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i19 + 12)));
      const __m128i vk19xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 316 * sizeof(uint8_t))))), vk_zero_point);
      i19 += 16;

      vacc0123 = _mm_macc_epi32(vi19x0123, vk19x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi19x4567, vk19x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi19x89AB, vk19x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi19xCDEF, vk19xCDEF, vaccCDEF);

      const __m128i vi20x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i20)));
      const __m128i vk20x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi20x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i20 + 4)));
      const __m128i vk20x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 324 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi20x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i20 + 8)));
      const __m128i vk20x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi20xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i20 + 12)));
      const __m128i vk20xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 332 * sizeof(uint8_t))))), vk_zero_point);
      i20 += 16;

      vacc0123 = _mm_macc_epi32(vi20x0123, vk20x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi20x4567, vk20x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi20x89AB, vk20x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi20xCDEF, vk20xCDEF, vaccCDEF);

      const __m128i vi21x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i21)));
      const __m128i vk21x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi21x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i21 + 4)));
      const __m128i vk21x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 340 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi21x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i21 + 8)));
      const __m128i vk21x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi21xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i21 + 12)));
      const __m128i vk21xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 348 * sizeof(uint8_t))))), vk_zero_point);
      i21 += 16;

      vacc0123 = _mm_macc_epi32(vi21x0123, vk21x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi21x4567, vk21x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi21x89AB, vk21x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi21xCDEF, vk21xCDEF, vaccCDEF);

      const __m128i vi22x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i22)));
      const __m128i vk22x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi22x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i22 + 4)));
      const __m128i vk22x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 356 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi22x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i22 + 8)));
      const __m128i vk22x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi22xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i22 + 12)));
      const __m128i vk22xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 364 * sizeof(uint8_t))))), vk_zero_point);
      i22 += 16;

      vacc0123 = _mm_macc_epi32(vi22x0123, vk22x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi22x4567, vk22x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi22x89AB, vk22x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi22xCDEF, vk22xCDEF, vaccCDEF);

      const __m128i vi23x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i23)));
      const __m128i vk23x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi23x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i23 + 4)));
      const __m128i vk23x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 372 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi23x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i23 + 8)));
      const __m128i vk23x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi23xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i23 + 12)));
      const __m128i vk23xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 380 * sizeof(uint8_t))))), vk_zero_point);
      i23 += 16;

      vacc0123 = _mm_macc_epi32(vi23x0123, vk23x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi23x4567, vk23x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi23x89AB, vk23x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi23xCDEF, vk23xCDEF, vaccCDEF);

      const __m128i vi24x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i24)));
      const __m128i vk24x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi24x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i24 + 4)));
      const __m128i vk24x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 388 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi24x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i24 + 8)));
      const __m128i vk24x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi24xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i24 + 12)));
      const __m128i vk24xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 396 * sizeof(uint8_t))))), vk_zero_point);
      i24 += 16;

      vacc0123 = _mm_macc_epi32(vi24x0123, vk24x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi24x4567, vk24x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi24x89AB, vk24x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi24xCDEF, vk24xCDEF, vaccCDEF);

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(uint8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

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
      __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);

        const __m128i vi0x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0)));
        const __m128i vk0x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) k))), vk_zero_point);
        i0 += 4;

        vacc0123 = _mm_macc_epi32(vi0x0123, vk0x0123, vacc0123);
        const __m128i vi1x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1)));
        const __m128i vk1x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 16)))), vk_zero_point);
        i1 += 4;

        vacc0123 = _mm_macc_epi32(vi1x0123, vk1x0123, vacc0123);
        const __m128i vi2x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2)));
        const __m128i vk2x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 32)))), vk_zero_point);
        i2 += 4;

        vacc0123 = _mm_macc_epi32(vi2x0123, vk2x0123, vacc0123);
        const __m128i vi3x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3)));
        const __m128i vk3x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 48)))), vk_zero_point);
        i3 += 4;

        vacc0123 = _mm_macc_epi32(vi3x0123, vk3x0123, vacc0123);
        const __m128i vi4x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4)));
        const __m128i vk4x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 64)))), vk_zero_point);
        i4 += 4;

        vacc0123 = _mm_macc_epi32(vi4x0123, vk4x0123, vacc0123);
        const __m128i vi5x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5)));
        const __m128i vk5x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 80)))), vk_zero_point);
        i5 += 4;

        vacc0123 = _mm_macc_epi32(vi5x0123, vk5x0123, vacc0123);
        const __m128i vi6x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6)));
        const __m128i vk6x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 96)))), vk_zero_point);
        i6 += 4;

        vacc0123 = _mm_macc_epi32(vi6x0123, vk6x0123, vacc0123);
        const __m128i vi7x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7)));
        const __m128i vk7x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 112)))), vk_zero_point);
        i7 += 4;

        vacc0123 = _mm_macc_epi32(vi7x0123, vk7x0123, vacc0123);
        const __m128i vi8x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8)));
        const __m128i vk8x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 128)))), vk_zero_point);
        i8 += 4;

        vacc0123 = _mm_macc_epi32(vi8x0123, vk8x0123, vacc0123);
        const __m128i vi9x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i9)));
        const __m128i vk9x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 144)))), vk_zero_point);
        i9 += 4;

        vacc0123 = _mm_macc_epi32(vi9x0123, vk9x0123, vacc0123);
        const __m128i vi10x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i10)));
        const __m128i vk10x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 160)))), vk_zero_point);
        i10 += 4;

        vacc0123 = _mm_macc_epi32(vi10x0123, vk10x0123, vacc0123);
        const __m128i vi11x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i11)));
        const __m128i vk11x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 176)))), vk_zero_point);
        i11 += 4;

        vacc0123 = _mm_macc_epi32(vi11x0123, vk11x0123, vacc0123);
        const __m128i vi12x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i12)));
        const __m128i vk12x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 192)))), vk_zero_point);
        i12 += 4;

        vacc0123 = _mm_macc_epi32(vi12x0123, vk12x0123, vacc0123);
        const __m128i vi13x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i13)));
        const __m128i vk13x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 208)))), vk_zero_point);
        i13 += 4;

        vacc0123 = _mm_macc_epi32(vi13x0123, vk13x0123, vacc0123);
        const __m128i vi14x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i14)));
        const __m128i vk14x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 224)))), vk_zero_point);
        i14 += 4;

        vacc0123 = _mm_macc_epi32(vi14x0123, vk14x0123, vacc0123);
        const __m128i vi15x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i15)));
        const __m128i vk15x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 240)))), vk_zero_point);
        i15 += 4;

        vacc0123 = _mm_macc_epi32(vi15x0123, vk15x0123, vacc0123);
        const __m128i vi16x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i16)));
        const __m128i vk16x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 256)))), vk_zero_point);
        i16 += 4;

        vacc0123 = _mm_macc_epi32(vi16x0123, vk16x0123, vacc0123);
        const __m128i vi17x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i17)));
        const __m128i vk17x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 272)))), vk_zero_point);
        i17 += 4;

        vacc0123 = _mm_macc_epi32(vi17x0123, vk17x0123, vacc0123);
        const __m128i vi18x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i18)));
        const __m128i vk18x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 288)))), vk_zero_point);
        i18 += 4;

        vacc0123 = _mm_macc_epi32(vi18x0123, vk18x0123, vacc0123);
        const __m128i vi19x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i19)));
        const __m128i vk19x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 304)))), vk_zero_point);
        i19 += 4;

        vacc0123 = _mm_macc_epi32(vi19x0123, vk19x0123, vacc0123);
        const __m128i vi20x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i20)));
        const __m128i vk20x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 320)))), vk_zero_point);
        i20 += 4;

        vacc0123 = _mm_macc_epi32(vi20x0123, vk20x0123, vacc0123);
        const __m128i vi21x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i21)));
        const __m128i vk21x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 336)))), vk_zero_point);
        i21 += 4;

        vacc0123 = _mm_macc_epi32(vi21x0123, vk21x0123, vacc0123);
        const __m128i vi22x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i22)));
        const __m128i vk22x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 352)))), vk_zero_point);
        i22 += 4;

        vacc0123 = _mm_macc_epi32(vi22x0123, vk22x0123, vacc0123);
        const __m128i vi23x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i23)));
        const __m128i vk23x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 368)))), vk_zero_point);
        i23 += 4;

        vacc0123 = _mm_macc_epi32(vi23x0123, vk23x0123, vacc0123);
        const __m128i vi24x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i24)));
        const __m128i vk24x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 384)))), vk_zero_point);
        i24 += 4;

        vacc0123 = _mm_macc_epi32(vi24x0123, vk24x0123, vacc0123);

        k += 4;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        vscaled0123 = _mm_mul_ps(vscaled0123, _mm_load_ps(params->fp32_sse2.scale));
        vscaled0123 = _mm_min_ps(vscaled0123, _mm_load_ps(params->fp32_sse2.output_max_less_zero_point));
        vacc0123 = _mm_cvtps_epi32(vscaled0123);

        w = (const void*) ((const int32_t*) w + 4);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc0123), voutput_zero_point);

        vout0123 = _mm_packus_epi16(vout0123, vout0123);
        vout0123 = _mm_max_epu8(vout0123, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if XNN_LIKELY(c >= 4) {
          _mm_storeu_si32(output, vout0123);
          output += 4;
          c -= 4;
        } else {
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123, 0));
            vout0123 = _mm_srli_epi32(vout0123, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (uint8_t) _mm_extract_epi8(vout0123, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__xop_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128i vk_zero_point = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*) params->fp32_sse2.kernel_zero_point));
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0)));
      const __m128i vk0x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi0x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 4)));
      const __m128i vk0x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 4 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi0x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 8)));
      const __m128i vk0x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi0xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 12)));
      const __m128i vk0xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 12 * sizeof(uint8_t))))), vk_zero_point);
      i0 += 16;

      vacc0123 = _mm_macc_epi32(vi0x0123, vk0x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi0x4567, vk0x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi0x89AB, vk0x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi0xCDEF, vk0xCDEF, vaccCDEF);

      const __m128i vi1x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1)));
      const __m128i vk1x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi1x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 4)));
      const __m128i vk1x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 20 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi1x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 8)));
      const __m128i vk1x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi1xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 12)));
      const __m128i vk1xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 28 * sizeof(uint8_t))))), vk_zero_point);
      i1 += 16;

      vacc0123 = _mm_macc_epi32(vi1x0123, vk1x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi1x4567, vk1x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi1x89AB, vk1x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi1xCDEF, vk1xCDEF, vaccCDEF);

      const __m128i vi2x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2)));
      const __m128i vk2x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi2x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 4)));
      const __m128i vk2x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 36 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi2x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 8)));
      const __m128i vk2x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi2xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 12)));
      const __m128i vk2xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 44 * sizeof(uint8_t))))), vk_zero_point);
      i2 += 16;

      vacc0123 = _mm_macc_epi32(vi2x0123, vk2x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi2x4567, vk2x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi2x89AB, vk2x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi2xCDEF, vk2xCDEF, vaccCDEF);

      const __m128i vi3x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3)));
      const __m128i vk3x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi3x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 4)));
      const __m128i vk3x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 52 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi3x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 8)));
      const __m128i vk3x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi3xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 12)));
      const __m128i vk3xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 60 * sizeof(uint8_t))))), vk_zero_point);
      i3 += 16;

      vacc0123 = _mm_macc_epi32(vi3x0123, vk3x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi3x4567, vk3x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi3x89AB, vk3x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi3xCDEF, vk3xCDEF, vaccCDEF);

      const __m128i vi4x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4)));
      const __m128i vk4x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi4x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 4)));
      const __m128i vk4x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 68 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi4x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 8)));
      const __m128i vk4x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi4xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 12)));
      const __m128i vk4xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 76 * sizeof(uint8_t))))), vk_zero_point);
      i4 += 16;

      vacc0123 = _mm_macc_epi32(vi4x0123, vk4x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi4x4567, vk4x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi4x89AB, vk4x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi4xCDEF, vk4xCDEF, vaccCDEF);

      const __m128i vi5x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5)));
      const __m128i vk5x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi5x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 4)));
      const __m128i vk5x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 84 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi5x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 8)));
      const __m128i vk5x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi5xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 12)));
      const __m128i vk5xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 92 * sizeof(uint8_t))))), vk_zero_point);
      i5 += 16;

      vacc0123 = _mm_macc_epi32(vi5x0123, vk5x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi5x4567, vk5x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi5x89AB, vk5x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi5xCDEF, vk5xCDEF, vaccCDEF);

      const __m128i vi6x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6)));
      const __m128i vk6x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi6x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 4)));
      const __m128i vk6x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 100 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi6x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 8)));
      const __m128i vk6x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi6xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 12)));
      const __m128i vk6xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 108 * sizeof(uint8_t))))), vk_zero_point);
      i6 += 16;

      vacc0123 = _mm_macc_epi32(vi6x0123, vk6x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi6x4567, vk6x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi6x89AB, vk6x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi6xCDEF, vk6xCDEF, vaccCDEF);

      const __m128i vi7x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7)));
      const __m128i vk7x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi7x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 4)));
      const __m128i vk7x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 116 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi7x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 8)));
      const __m128i vk7x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi7xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 12)));
      const __m128i vk7xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 124 * sizeof(uint8_t))))), vk_zero_point);
      i7 += 16;

      vacc0123 = _mm_macc_epi32(vi7x0123, vk7x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi7x4567, vk7x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi7x89AB, vk7x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi7xCDEF, vk7xCDEF, vaccCDEF);

      const __m128i vi8x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8)));
      const __m128i vk8x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi8x4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 4)));
      const __m128i vk8x4567 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 132 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi8x89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 8)));
      const __m128i vk8x89AB = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(uint8_t))))), vk_zero_point);
      const __m128i vi8xCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 12)));
      const __m128i vk8xCDEF = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 140 * sizeof(uint8_t))))), vk_zero_point);
      i8 += 16;

      vacc0123 = _mm_macc_epi32(vi8x0123, vk8x0123, vacc0123);
      vacc4567 = _mm_macc_epi32(vi8x4567, vk8x4567, vacc4567);
      vacc89AB = _mm_macc_epi32(vi8x89AB, vk8x89AB, vacc89AB);
      vaccCDEF = _mm_macc_epi32(vi8xCDEF, vk8xCDEF, vaccCDEF);

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(uint8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

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
      __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);

        const __m128i vi0x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0)));
        const __m128i vk0x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) k))), vk_zero_point);
        i0 += 4;

        vacc0123 = _mm_macc_epi32(vi0x0123, vk0x0123, vacc0123);
        const __m128i vi1x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1)));
        const __m128i vk1x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 16)))), vk_zero_point);
        i1 += 4;

        vacc0123 = _mm_macc_epi32(vi1x0123, vk1x0123, vacc0123);
        const __m128i vi2x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2)));
        const __m128i vk2x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 32)))), vk_zero_point);
        i2 += 4;

        vacc0123 = _mm_macc_epi32(vi2x0123, vk2x0123, vacc0123);
        const __m128i vi3x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3)));
        const __m128i vk3x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 48)))), vk_zero_point);
        i3 += 4;

        vacc0123 = _mm_macc_epi32(vi3x0123, vk3x0123, vacc0123);
        const __m128i vi4x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4)));
        const __m128i vk4x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 64)))), vk_zero_point);
        i4 += 4;

        vacc0123 = _mm_macc_epi32(vi4x0123, vk4x0123, vacc0123);
        const __m128i vi5x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5)));
        const __m128i vk5x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 80)))), vk_zero_point);
        i5 += 4;

        vacc0123 = _mm_macc_epi32(vi5x0123, vk5x0123, vacc0123);
        const __m128i vi6x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6)));
        const __m128i vk6x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 96)))), vk_zero_point);
        i6 += 4;

        vacc0123 = _mm_macc_epi32(vi6x0123, vk6x0123, vacc0123);
        const __m128i vi7x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7)));
        const __m128i vk7x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 112)))), vk_zero_point);
        i7 += 4;

        vacc0123 = _mm_macc_epi32(vi7x0123, vk7x0123, vacc0123);
        const __m128i vi8x0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8)));
        const __m128i vk8x0123 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 128)))), vk_zero_point);
        i8 += 4;

        vacc0123 = _mm_macc_epi32(vi8x0123, vk8x0123, vacc0123);

        k += 4;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        vscaled0123 = _mm_mul_ps(vscaled0123, _mm_load_ps(params->fp32_sse2.scale));
        vscaled0123 = _mm_min_ps(vscaled0123, _mm_load_ps(params->fp32_sse2.output_max_less_zero_point));
        vacc0123 = _mm_cvtps_epi32(vscaled0123);

        w = (const void*) ((const int32_t*) w + 4);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc0123), voutput_zero_point);

        vout0123 = _mm_packus_epi16(vout0123, vout0123);
        vout0123 = _mm_max_epu8(vout0123, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if XNN_LIKELY(c >= 4) {
          _mm_storeu_si32(output, vout0123);
          output += 4;
          c -= 4;
        } else {
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123, 0));
            vout0123 = _mm_srli_epi32(vout0123, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (uint8_t) _mm_extract_epi8(vout0123, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t k = 0;
    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

      vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
      const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

      vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
      const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

      vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
      const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

      vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);

      w = (const void*) ((const uint8_t*) w + 32);
      k += 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = 0;
    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepu8_epi16(va1);
      a1 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

      vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
      vacc1x0 = _mm_maddd_epi16(vxa1, vxb0, vacc1x0);
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
      const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

      vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
      vacc1x1 = _mm_maddd_epi16(vxa1, vxb1, vacc1x1);
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
      const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

      vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
      vacc1x2 = _mm_maddd_epi16(vxa1, vxb2, vacc1x2);
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
      const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

      vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);
      vacc1x3 = _mm_maddd_epi16(vxa1, vxb3, vacc1x3);

      w = (const void*) ((const uint8_t*) w + 32);
      k += 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
        *c1 = (uint8_t) _mm_extract_epi8(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

        vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
        const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

        vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

        vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
        const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

        vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = 0;
      const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepu8_epi16(va1);
        a1 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

        vacc0x0 = _mm_maddd_epi16(vxa0, vxb0, vacc0x0);
        vacc1x0 = _mm_maddd_epi16(vxa1, vxb0, vacc1x0);
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
        const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

        vacc0x1 = _mm_maddd_epi16(vxa0, vxb1, vacc0x1);
        vacc1x1 = _mm_maddd_epi16(vxa1, vxb1, vacc1x1);
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

        vacc0x2 = _mm_maddd_epi16(vxa0, vxb2, vacc0x2);
        vacc1x2 = _mm_maddd_epi16(vxa1, vxb2, vacc1x2);
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
        const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

        vacc0x3 = _mm_maddd_epi16(vxa0, vxb3, vacc0x3);
        vacc1x3 = _mm_maddd_epi16(vxa1, vxb3, vacc1x3);

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c1 = (uint8_t) _mm_extract_epi8(vout, 4);
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__xop_mul32_ld32_x8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse4.bias);
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4.a_multiplier);
  const __m128i vb_multiplier = _mm_load_si128((const __m128i*) params->sse4.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->sse4.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4.output_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const __m128i va0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
    const __m128i vb0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b)));
    const __m128i va4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
    const __m128i vb4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 4)));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
    __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

    vacc0123 = _mm_macc_epi32(vb0123, vb_multiplier, vacc0123);
    vacc4567 = _mm_macc_epi32(vb4567, vb_multiplier, vacc4567);

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
      const __m128i vb0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b)));
      const __m128i va4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
      const __m128i vb4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_b + 4)));

      __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
      __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

      vacc0123 = _mm_macc_epi32(vb0123, vb_multiplier, vacc0123);
      vacc4567 = _mm_macc_epi32(vb4567, vb_multiplier, vacc4567);

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_vaddc_minmax_ukernel__xop_mul32_ld32_x8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->sse4.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4.output_max);

  __m128i vbias = _mm_cvtsi32_si128(params->sse4.b_multiplier[0] * (int32_t) *input_b);
  vbias = _mm_shuffle_epi32(vbias, _MM_SHUFFLE(0, 0, 0, 0));
  vbias = _mm_add_epi32(vbias, _mm_load_si128((const __m128i*) params->sse4.bias));
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const __m128i va0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
    const __m128i va4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
    __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a)));
      const __m128i va4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input_a + 4)));

      __m128i vacc0123 = _mm_macc_epi32(va0123, va_multiplier, vbias);
      __m128i vacc4567 = _mm_macc_epi32(va4567, va_multiplier, vbias);

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}
