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

#include <xnnpack/dwconv.h>
#include <xnnpack/unaligned.h>


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul16_add16_vpunpck(
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
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((uintptr_t) w + 8 * sizeof(int32_t)));

      __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
      __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);


      const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
      const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      i0 += 16;

      __m256i vacc0123456789ABCDEF = _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);


      const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
      const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i1 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

      __m256i vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
      const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      i2 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);


      const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
      const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i3 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
      const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      i4 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);


      const __m256i vi5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
      const __m256i vk5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i5 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
      const __m256i vk6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      i6 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF);


      const __m256i vi7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
      const __m256i vk7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i7 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));
      const __m256i vk8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      i8 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF);


      const __m256i vi9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i9));
      const __m256i vk9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i9 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i10));
      const __m256i vk10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      i10 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF);


      const __m256i vi11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i11));
      const __m256i vk11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i11 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i12));
      const __m256i vk12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      i12 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF);


      const __m256i vi13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i13));
      const __m256i vk13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i13 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i14));
      const __m256i vk14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      i14 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF);


      const __m256i vi15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i15));
      const __m256i vk15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i15 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i16));
      const __m256i vk16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      i16 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF);


      const __m256i vi17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i17));
      const __m256i vk17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i17 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i18));
      const __m256i vk18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      i18 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF);


      const __m256i vi19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i19));
      const __m256i vk19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      i19 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i20));
      const __m256i vk20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      i20 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF);


      const __m256i vi21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i21));
      const __m256i vk21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      i21 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i22));
      const __m256i vk22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      i22 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF);


      const __m256i vi23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i23));
      const __m256i vk23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      i23 += 16;

      vacc0123456789ABCDEF = _mm256_add_epi16(vacc0123456789ABCDEF, _mm256_mullo_epi16(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      const __m256i vi24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i24));
      const __m256i vk24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      i24 += 16;

      vacc0123456789ABCDEF = _mm256_mullo_epi16(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF);

      vsignacc0123456789ABCDEF = _mm256_srai_epi16(vacc0123456789ABCDEF, 15);
      vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));
      vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vacc0123456789ABCDEF, vsignacc0123456789ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t));

      vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
      vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);

      __m256 vfpacc01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vfpacc89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w);
      const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 16 * sizeof(float));
      vfpacc01234567 = _mm256_mul_ps(vfpacc01234567, vscale01234567);
      vfpacc89ABCDEF = _mm256_mul_ps(vfpacc89ABCDEF, vscale89ABCDEF);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vfpacc01234567 = _mm256_min_ps(vfpacc01234567, voutput_max_less_zero_point);
      vfpacc89ABCDEF = _mm256_min_ps(vfpacc89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vfpacc01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vfpacc89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      const __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
        __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((uintptr_t) w + 8 * sizeof(int32_t)));

        __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
        __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);


        const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
        const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t))));

        const __m256i vprod0x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
        const __m256i vprod0x0123456789ABCDEFhi = _mm256_srai_epi16(vprod0x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));

        const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
        const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));

        const __m256i vprod1x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
        const __m256i vprod1x0123456789ABCDEFhi = _mm256_srai_epi16(vprod1x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));

        const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
        const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));

        const __m256i vprod2x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
        const __m256i vprod2x0123456789ABCDEFhi = _mm256_srai_epi16(vprod2x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));

        const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
        const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t))));

        const __m256i vprod3x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
        const __m256i vprod3x0123456789ABCDEFhi = _mm256_srai_epi16(vprod3x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));

        const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
        const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t))));

        const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
        const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));

        const __m256i vi5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
        const __m256i vk5x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t))));

        const __m256i vprod5x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF);
        const __m256i vprod5x0123456789ABCDEFhi = _mm256_srai_epi16(vprod5x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod5x0123456789ABCDEFlo, vprod5x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod5x0123456789ABCDEFlo, vprod5x0123456789ABCDEFhi));

        const __m256i vi6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
        const __m256i vk6x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t))));

        const __m256i vprod6x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF);
        const __m256i vprod6x0123456789ABCDEFhi = _mm256_srai_epi16(vprod6x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod6x0123456789ABCDEFlo, vprod6x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod6x0123456789ABCDEFlo, vprod6x0123456789ABCDEFhi));

        const __m256i vi7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
        const __m256i vk7x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t))));

        const __m256i vprod7x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF);
        const __m256i vprod7x0123456789ABCDEFhi = _mm256_srai_epi16(vprod7x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod7x0123456789ABCDEFlo, vprod7x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod7x0123456789ABCDEFlo, vprod7x0123456789ABCDEFhi));

        const __m256i vi8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));
        const __m256i vk8x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t))));

        const __m256i vprod8x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF);
        const __m256i vprod8x0123456789ABCDEFhi = _mm256_srai_epi16(vprod8x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod8x0123456789ABCDEFlo, vprod8x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod8x0123456789ABCDEFlo, vprod8x0123456789ABCDEFhi));

        const __m256i vi9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i9));
        const __m256i vk9x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t))));

        const __m256i vprod9x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF);
        const __m256i vprod9x0123456789ABCDEFhi = _mm256_srai_epi16(vprod9x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod9x0123456789ABCDEFlo, vprod9x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod9x0123456789ABCDEFlo, vprod9x0123456789ABCDEFhi));

        const __m256i vi10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i10));
        const __m256i vk10x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t))));

        const __m256i vprod10x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF);
        const __m256i vprod10x0123456789ABCDEFhi = _mm256_srai_epi16(vprod10x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod10x0123456789ABCDEFlo, vprod10x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod10x0123456789ABCDEFlo, vprod10x0123456789ABCDEFhi));

        const __m256i vi11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i11));
        const __m256i vk11x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t))));

        const __m256i vprod11x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF);
        const __m256i vprod11x0123456789ABCDEFhi = _mm256_srai_epi16(vprod11x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod11x0123456789ABCDEFlo, vprod11x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod11x0123456789ABCDEFlo, vprod11x0123456789ABCDEFhi));

        const __m256i vi12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i12));
        const __m256i vk12x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t))));

        const __m256i vprod12x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF);
        const __m256i vprod12x0123456789ABCDEFhi = _mm256_srai_epi16(vprod12x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod12x0123456789ABCDEFlo, vprod12x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod12x0123456789ABCDEFlo, vprod12x0123456789ABCDEFhi));

        const __m256i vi13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i13));
        const __m256i vk13x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t))));

        const __m256i vprod13x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF);
        const __m256i vprod13x0123456789ABCDEFhi = _mm256_srai_epi16(vprod13x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod13x0123456789ABCDEFlo, vprod13x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod13x0123456789ABCDEFlo, vprod13x0123456789ABCDEFhi));

        const __m256i vi14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i14));
        const __m256i vk14x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t))));

        const __m256i vprod14x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF);
        const __m256i vprod14x0123456789ABCDEFhi = _mm256_srai_epi16(vprod14x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod14x0123456789ABCDEFlo, vprod14x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod14x0123456789ABCDEFlo, vprod14x0123456789ABCDEFhi));

        const __m256i vi15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i15));
        const __m256i vk15x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t))));

        const __m256i vprod15x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF);
        const __m256i vprod15x0123456789ABCDEFhi = _mm256_srai_epi16(vprod15x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod15x0123456789ABCDEFlo, vprod15x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod15x0123456789ABCDEFlo, vprod15x0123456789ABCDEFhi));

        const __m256i vi16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i16));
        const __m256i vk16x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t))));

        const __m256i vprod16x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF);
        const __m256i vprod16x0123456789ABCDEFhi = _mm256_srai_epi16(vprod16x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod16x0123456789ABCDEFlo, vprod16x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod16x0123456789ABCDEFlo, vprod16x0123456789ABCDEFhi));

        const __m256i vi17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i17));
        const __m256i vk17x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t))));

        const __m256i vprod17x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF);
        const __m256i vprod17x0123456789ABCDEFhi = _mm256_srai_epi16(vprod17x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod17x0123456789ABCDEFlo, vprod17x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod17x0123456789ABCDEFlo, vprod17x0123456789ABCDEFhi));

        const __m256i vi18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i18));
        const __m256i vk18x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t))));

        const __m256i vprod18x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF);
        const __m256i vprod18x0123456789ABCDEFhi = _mm256_srai_epi16(vprod18x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod18x0123456789ABCDEFlo, vprod18x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod18x0123456789ABCDEFlo, vprod18x0123456789ABCDEFhi));

        const __m256i vi19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i19));
        const __m256i vk19x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t))));

        const __m256i vprod19x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF);
        const __m256i vprod19x0123456789ABCDEFhi = _mm256_srai_epi16(vprod19x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod19x0123456789ABCDEFlo, vprod19x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod19x0123456789ABCDEFlo, vprod19x0123456789ABCDEFhi));

        const __m256i vi20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i20));
        const __m256i vk20x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t))));

        const __m256i vprod20x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF);
        const __m256i vprod20x0123456789ABCDEFhi = _mm256_srai_epi16(vprod20x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod20x0123456789ABCDEFlo, vprod20x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod20x0123456789ABCDEFlo, vprod20x0123456789ABCDEFhi));

        const __m256i vi21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i21));
        const __m256i vk21x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t))));

        const __m256i vprod21x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF);
        const __m256i vprod21x0123456789ABCDEFhi = _mm256_srai_epi16(vprod21x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod21x0123456789ABCDEFlo, vprod21x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod21x0123456789ABCDEFlo, vprod21x0123456789ABCDEFhi));

        const __m256i vi22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i22));
        const __m256i vk22x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t))));

        const __m256i vprod22x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF);
        const __m256i vprod22x0123456789ABCDEFhi = _mm256_srai_epi16(vprod22x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod22x0123456789ABCDEFlo, vprod22x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod22x0123456789ABCDEFlo, vprod22x0123456789ABCDEFhi));

        const __m256i vi23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i23));
        const __m256i vk23x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t))));

        const __m256i vprod23x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF);
        const __m256i vprod23x0123456789ABCDEFhi = _mm256_srai_epi16(vprod23x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod23x0123456789ABCDEFlo, vprod23x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod23x0123456789ABCDEFlo, vprod23x0123456789ABCDEFhi));

        const __m256i vi24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i24));
        const __m256i vk24x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t))));

        const __m256i vprod24x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF);
        const __m256i vprod24x0123456789ABCDEFhi = _mm256_srai_epi16(vprod24x0123456789ABCDEFlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod24x0123456789ABCDEFlo, vprod24x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod24x0123456789ABCDEFlo, vprod24x0123456789ABCDEFhi));

        vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
        vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);


        __m256 vfpacc01234567 = _mm256_cvtepi32_ps(vacc01234567);
        __m256 vfpacc89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

        const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t)));
        const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t) + 8 * sizeof(float)));
        vfpacc01234567 = _mm256_mul_ps(vfpacc01234567, vscale01234567);
        vfpacc89ABCDEF = _mm256_mul_ps(vfpacc89ABCDEF, vscale89ABCDEF);

        const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
        vfpacc01234567 = _mm256_min_ps(vfpacc01234567, voutput_max_less_zero_point);
        vfpacc89ABCDEF = _mm256_min_ps(vfpacc89ABCDEF, voutput_max_less_zero_point);

        vacc01234567 = _mm256_cvtps_epi32(vfpacc01234567);
        vacc89ABCDEF = _mm256_cvtps_epi32(vfpacc89ABCDEF);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);
        __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc89ABCDEF), _mm256_extracti128_si256(vacc89ABCDEF, 1)), voutput_zero_point);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);

        __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

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
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
