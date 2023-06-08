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

#include <xnnpack/dwconv.h>
#include <xnnpack/unaligned.h>


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__avx2_mul32(
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
    for (; c >= 8; c -= 8) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


      const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      i0 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

      const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      i1 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

      const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i2 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

      const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      i3 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

      const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      i4 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

      const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      i5 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

      const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i6 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

      const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      i7 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

      const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      i8 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

      const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
      const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t))));
      i9 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));

      const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
      const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i10 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));

      const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
      const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t))));
      i11 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));

      const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
      const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      i12 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));

      const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
      const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t))));
      i13 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));

      const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
      const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i14 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));

      const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
      const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t))));
      i15 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));

      const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
      const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      i16 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));

      const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
      const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t))));
      i17 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));

      const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
      const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i18 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));

      const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
      const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t))));
      i19 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));

      const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
      const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      i20 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));

      const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
      const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t))));
      i21 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));

      const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
      const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i22 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));

      const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
      const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t))));
      i23 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));

      const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
      const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      i24 += 8;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);

      const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w);
      w = (const void*) ((const float*) w + 8);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
        const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));

        const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
        const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));

        const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
        const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));

        const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
        const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));

        const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
        const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));

        const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
        const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));

        const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
        const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));

        const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
        const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));

        const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
        const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));

        const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
        const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));

        const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
        const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));

        const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
        const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));

        const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
        const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));

        const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
        const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));

        const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
        const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));

        const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
        const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t))));

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));


        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t)));
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

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
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
