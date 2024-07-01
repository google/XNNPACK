// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c16s16r__avx2_mul16_vpunpck(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  do {
    const void* w = weights;

    // First pass to process 5 inputs.
    {
      int32_t* b = buffer;

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
      input += 5;

      size_t c = round_up_po2(channels, 16);

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

        w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t));

        vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
        vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);
        vaccGHIJKLMN = _mm256_inserti128_si256(vaccGHIJOPQR, _mm256_castsi256_si128(vaccKLMNSTUV), 1);
        vaccOPQRSTUV = _mm256_permute2x128_si256(vaccGHIJOPQR, vaccKLMNSTUV, 0x31);

        _mm256_storeu_si256((__m256i*) b, vacc01234567);
        _mm256_storeu_si256((__m256i*) (b + 8), vacc89ABCDEF);
        _mm256_storeu_si256((__m256i*) (b + 16), vaccGHIJKLMN);
        _mm256_storeu_si256((__m256i*) (b + 24), vaccOPQRSTUV);
        b += 32;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
          __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((uintptr_t) w + 8 * sizeof(int32_t)));

          __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
          __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);


          const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
          const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t))));
          i0 += 16;

          const __m256i vprod0x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
          const __m256i vprod0x0123456789ABCDEFhi = _mm256_srai_epi16(vprod0x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));

          const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
          const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));
          i1 += 16;

          const __m256i vprod1x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
          const __m256i vprod1x0123456789ABCDEFhi = _mm256_srai_epi16(vprod1x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));

          const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
          const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));
          i2 += 16;

          const __m256i vprod2x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
          const __m256i vprod2x0123456789ABCDEFhi = _mm256_srai_epi16(vprod2x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));

          const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
          const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t))));
          i3 += 16;

          const __m256i vprod3x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
          const __m256i vprod3x0123456789ABCDEFhi = _mm256_srai_epi16(vprod3x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));

          const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
          const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t))));
          i4 += 16;

          const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
          const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));

          vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
          vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);

          w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t));

          _mm256_storeu_si256((__m256i*) b, vacc01234567);
          _mm256_storeu_si256((__m256i*) (b + 8), vacc89ABCDEF);
          b += 16;
          c -= 16;
        } while (c != 0);

      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
        int32_t* b = buffer;

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
        input += 5;

        size_t c = round_up_po2(channels, 16);

        for (; c >= 32; c -= 32) {
          __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);
          __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) (b + 8));
          __m256i vaccGHIJKLMN = _mm256_loadu_si256((const __m256i*) (b + 16));
          __m256i vaccOPQRSTUV = _mm256_loadu_si256((const __m256i*) (b + 24));

          __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
          __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);
          __m256i vaccGHIJOPQR = _mm256_inserti128_si256(vaccGHIJKLMN, _mm256_castsi256_si128(vaccOPQRSTUV), 1);
          __m256i vaccKLMNSTUV = _mm256_permute2x128_si256(vaccGHIJKLMN, vaccOPQRSTUV, 0x31);


          const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
          const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 0 * sizeof(int8_t))));
          const __m256i vi0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i0 + 16)));
          const __m256i vk0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int8_t))));
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
          const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int8_t))));
          const __m256i vi1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i1 + 16)));
          const __m256i vk1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(int8_t))));
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
          const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(int8_t))));
          const __m256i vi2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i2 + 16)));
          const __m256i vk2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 80 * sizeof(int8_t))));
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
          const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 96 * sizeof(int8_t))));
          const __m256i vi3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i3 + 16)));
          const __m256i vk3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 112 * sizeof(int8_t))));
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
          const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 128 * sizeof(int8_t))));
          const __m256i vi4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i4 + 16)));
          const __m256i vk4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 144 * sizeof(int8_t))));
          i4 += 32;

          const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
          const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);
          const __m256i vprod4xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV);
          const __m256i vprod4xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod4xGHIJKLMNOPQRSTUVlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
          vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod4xGHIJKLMNOPQRSTUVlo, vprod4xGHIJKLMNOPQRSTUVhi));
          vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod4xGHIJKLMNOPQRSTUVlo, vprod4xGHIJKLMNOPQRSTUVhi));

          w = (const void*) ((uintptr_t) w + 160 * sizeof(int8_t));

          vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
          vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);
          vaccGHIJKLMN = _mm256_inserti128_si256(vaccGHIJOPQR, _mm256_castsi256_si128(vaccKLMNSTUV), 1);
          vaccOPQRSTUV = _mm256_permute2x128_si256(vaccGHIJOPQR, vaccKLMNSTUV, 0x31);

          _mm256_storeu_si256((__m256i*) b, vacc01234567);
          _mm256_storeu_si256((__m256i*) (b + 8), vacc89ABCDEF);
          _mm256_storeu_si256((__m256i*) (b + 16), vaccGHIJKLMN);
          _mm256_storeu_si256((__m256i*) (b + 24), vaccOPQRSTUV);
          b += 32;
        }

        if XNN_UNLIKELY(c != 0) {
          do {
            __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);
            __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) (b + 8));

            __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
            __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);


            const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
            const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w)));
            i0 += 16;
            const __m256i vprod0x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
            const __m256i vprod0x0123456789ABCDEFhi = _mm256_srai_epi16(vprod0x0123456789ABCDEFlo, 15);

            vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));
            vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));

            const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
            const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int8_t))));
            i1 += 16;
            const __m256i vprod1x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
            const __m256i vprod1x0123456789ABCDEFhi = _mm256_srai_epi16(vprod1x0123456789ABCDEFlo, 15);

            vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));
            vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));

            const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
            const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int8_t))));
            i2 += 16;
            const __m256i vprod2x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
            const __m256i vprod2x0123456789ABCDEFhi = _mm256_srai_epi16(vprod2x0123456789ABCDEFlo, 15);

            vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));
            vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));

            const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
            const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(int8_t))));
            i3 += 16;
            const __m256i vprod3x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
            const __m256i vprod3x0123456789ABCDEFhi = _mm256_srai_epi16(vprod3x0123456789ABCDEFlo, 15);

            vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));
            vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));

            const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
            const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(int8_t))));
            i4 += 16;
            const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
            const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);

            vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
            vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));

            vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
            vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);

            w = (const void*) ((uintptr_t) w + 80 * sizeof(int8_t));

            _mm256_storeu_si256((__m256i*) b, vacc01234567);
            _mm256_storeu_si256((__m256i*) (b + 8), vacc89ABCDEF);
            b += 16;
            c -= 16;
          } while (c != 0);
        }
    }


    // Last pass to process up to 5 inputs.
    {
      const int32_t* b = buffer;

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

      size_t c = channels;

      for (; c >= 32; c -= 32) {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);
        __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) (b + 8));
        __m256i vaccGHIJKLMN = _mm256_loadu_si256((const __m256i*) (b + 16));
        __m256i vaccOPQRSTUV = _mm256_loadu_si256((const __m256i*) (b + 24));
        b += 32;

        __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
        __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);
        __m256i vaccGHIJOPQR = _mm256_inserti128_si256(vaccGHIJKLMN, _mm256_castsi256_si128(vaccOPQRSTUV), 1);
        __m256i vaccKLMNSTUV = _mm256_permute2x128_si256(vaccGHIJKLMN, vaccOPQRSTUV, 0x31);


        const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
        const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 0 * sizeof(int8_t))));
        const __m256i vi0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const __m256i vk0xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int8_t))));
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
        const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int8_t))));
        const __m256i vi1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const __m256i vk1xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(int8_t))));
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
        const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(int8_t))));
        const __m256i vi2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const __m256i vk2xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 80 * sizeof(int8_t))));
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
        const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 96 * sizeof(int8_t))));
        const __m256i vi3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const __m256i vk3xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 112 * sizeof(int8_t))));
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
        const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 128 * sizeof(int8_t))));
        const __m256i vi4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const __m256i vk4xGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 144 * sizeof(int8_t))));
        i4 += 32;

        const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
        const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);
        const __m256i vprod4xGHIJKLMNOPQRSTUVlo =  _mm256_mullo_epi16(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV);
        const __m256i vprod4xGHIJKLMNOPQRSTUVhi = _mm256_srai_epi16(vprod4xGHIJKLMNOPQRSTUVlo, 15);

        vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
        vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
        vaccGHIJOPQR = _mm256_add_epi32(vaccGHIJOPQR, _mm256_unpacklo_epi16(vprod4xGHIJKLMNOPQRSTUVlo, vprod4xGHIJKLMNOPQRSTUVhi));
        vaccKLMNSTUV = _mm256_add_epi32(vaccKLMNSTUV, _mm256_unpackhi_epi16(vprod4xGHIJKLMNOPQRSTUVlo, vprod4xGHIJKLMNOPQRSTUVhi));

        vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
        vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);
        vaccGHIJKLMN = _mm256_inserti128_si256(vaccGHIJOPQR, _mm256_castsi256_si128(vaccKLMNSTUV), 1);
        vaccOPQRSTUV = _mm256_permute2x128_si256(vaccGHIJOPQR, vaccKLMNSTUV, 0x31);

        w = (const void*) ((uintptr_t) w + 160 * sizeof(int8_t));

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
        do {
          __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);
          __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) (b + 8));
          b += 16;

          __m256i vacc012389AB = _mm256_inserti128_si256(vacc01234567, _mm256_castsi256_si128(vacc89ABCDEF), 1);
          __m256i vacc4567CDEF = _mm256_permute2x128_si256(vacc01234567, vacc89ABCDEF, 0x31);


          const __m256i vi0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
          const __m256i vk0x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w)));
          i0 += 16;

          const __m256i vprod0x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF);
          const __m256i vprod0x0123456789ABCDEFhi = _mm256_srai_epi16(vprod0x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod0x0123456789ABCDEFlo, vprod0x0123456789ABCDEFhi));

          const __m256i vi1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
          const __m256i vk1x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int8_t))));
          i1 += 16;

          const __m256i vprod1x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF);
          const __m256i vprod1x0123456789ABCDEFhi = _mm256_srai_epi16(vprod1x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod1x0123456789ABCDEFlo, vprod1x0123456789ABCDEFhi));

          const __m256i vi2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
          const __m256i vk2x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int8_t))));
          i2 += 16;

          const __m256i vprod2x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF);
          const __m256i vprod2x0123456789ABCDEFhi = _mm256_srai_epi16(vprod2x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod2x0123456789ABCDEFlo, vprod2x0123456789ABCDEFhi));

          const __m256i vi3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
          const __m256i vk3x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(int8_t))));
          i3 += 16;

          const __m256i vprod3x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF);
          const __m256i vprod3x0123456789ABCDEFhi = _mm256_srai_epi16(vprod3x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod3x0123456789ABCDEFlo, vprod3x0123456789ABCDEFhi));

          const __m256i vi4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
          const __m256i vk4x0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(int8_t))));
          i4 += 16;

          const __m256i vprod4x0123456789ABCDEFlo =  _mm256_mullo_epi16(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF);
          const __m256i vprod4x0123456789ABCDEFhi = _mm256_srai_epi16(vprod4x0123456789ABCDEFlo, 15);

          vacc012389AB = _mm256_add_epi32(vacc012389AB, _mm256_unpacklo_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));
          vacc4567CDEF = _mm256_add_epi32(vacc4567CDEF, _mm256_unpackhi_epi16(vprod4x0123456789ABCDEFlo, vprod4x0123456789ABCDEFhi));

          vacc01234567 = _mm256_inserti128_si256(vacc012389AB, _mm256_castsi256_si128(vacc4567CDEF), 1);
          vacc89ABCDEF = _mm256_permute2x128_si256(vacc012389AB, vacc4567CDEF, 0x31);

          __m256 vfpacc01234567 = _mm256_cvtepi32_ps(vacc01234567);
          __m256 vfpacc89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

          const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 80 * sizeof(int8_t)));
          const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) ((uintptr_t) w + 80 * sizeof(int8_t) + 8 * sizeof(float) ));
          vfpacc01234567 = _mm256_mul_ps(vfpacc01234567, vscale01234567);
          vfpacc89ABCDEF = _mm256_mul_ps(vfpacc89ABCDEF, vscale89ABCDEF);

          const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
          vfpacc01234567 = _mm256_min_ps(vfpacc01234567, voutput_max_less_zero_point);
          vfpacc89ABCDEF = _mm256_min_ps(vfpacc89ABCDEF, voutput_max_less_zero_point);

          vacc01234567 = _mm256_cvtps_epi32(vfpacc01234567);
          vacc89ABCDEF = _mm256_cvtps_epi32(vfpacc89ABCDEF);

          w = (void*) ((uintptr_t) w + 80 * sizeof(int8_t) + 16 * sizeof(float));

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
    }


    input = (const int8_t**) ((uintptr_t) input + input_stride);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
