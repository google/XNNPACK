// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-avx2-mul32.c.in
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


void xnn_qu8_dwconv_minmax_fp32_ukernel_5f5m5l32c8s8r__avx2_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const __m256i vk_zero_point = _mm256_cvtepu16_epi32(_mm_load_si128((const __m128i*) params->fp32_avx2.kernel_zero_point));

  do {
    const void* w = weights;

    // First pass to process 5 inputs.
    {
      int32_t* b = buffer;
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
      input += 5;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 32; c -= 32) {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
        __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));
        __m256i vaccGHIJKLMN = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 16));
        __m256i vaccOPQRSTUV = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 24));


        const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
        const __m256i vk0x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 8 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 16)));
        const __m256i vk0xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 24)));
        const __m256i vk0xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 24 * sizeof(uint8_t)))), vk_zero_point);
        i0 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi0xGHIJKLMN, vk0xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi0xOPQRSTUV, vk0xOPQRSTUV));

        const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
        const __m256i vk1x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 40 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 16)));
        const __m256i vk1xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 24)));
        const __m256i vk1xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 56 * sizeof(uint8_t)))), vk_zero_point);
        i1 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi1xGHIJKLMN, vk1xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi1xOPQRSTUV, vk1xOPQRSTUV));

        const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
        const __m256i vk2x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 72 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 16)));
        const __m256i vk2xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 24)));
        const __m256i vk2xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 88 * sizeof(uint8_t)))), vk_zero_point);
        i2 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi2xGHIJKLMN, vk2xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi2xOPQRSTUV, vk2xOPQRSTUV));

        const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
        const __m256i vk3x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 104 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 16)));
        const __m256i vk3xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 24)));
        const __m256i vk3xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 120 * sizeof(uint8_t)))), vk_zero_point);
        i3 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi3xGHIJKLMN, vk3xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi3xOPQRSTUV, vk3xOPQRSTUV));

        const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
        const __m256i vk4x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 136 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 16)));
        const __m256i vk4xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 24)));
        const __m256i vk4xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 152 * sizeof(uint8_t)))), vk_zero_point);
        i4 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi4xGHIJKLMN, vk4xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi4xOPQRSTUV, vk4xOPQRSTUV));

        w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(uint8_t));

        _mm256_storeu_si256((__m256i*) b, vacc01234567);
        _mm256_storeu_si256((__m256i*) (b + 8), vacc89ABCDEF);
        _mm256_storeu_si256((__m256i*) (b + 16), vaccGHIJKLMN);
        _mm256_storeu_si256((__m256i*) (b + 24), vaccOPQRSTUV);
        b += 32;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);

          const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
          const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t)))), vk_zero_point);
          i0 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

          const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
          const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)))), vk_zero_point);
          i1 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

          const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
          const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
          i2 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

          const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
          const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)))), vk_zero_point);
          i3 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

          const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
          const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
          i4 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

          w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t));

          _mm256_storeu_si256((__m256i*) b, vacc01234567);
          b += 8;
          c -= 8;
        } while (c != 0);
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      int32_t* b = buffer;
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
      input += 5;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 32; c -= 32) {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);
        __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) (b + 8));
        __m256i vaccGHIJKLMN = _mm256_loadu_si256((const __m256i*) (b + 16));
        __m256i vaccOPQRSTUV = _mm256_loadu_si256((const __m256i*) (b + 24));


        const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
        const __m256i vk0x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 16)));
        const __m256i vk0xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 24)));
        const __m256i vk0xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(uint8_t)))), vk_zero_point);
        i0 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi0xGHIJKLMN, vk0xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi0xOPQRSTUV, vk0xOPQRSTUV));

        const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
        const __m256i vk1x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 40 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 16)));
        const __m256i vk1xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 48 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 24)));
        const __m256i vk1xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 56 * sizeof(uint8_t)))), vk_zero_point);
        i1 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi1xGHIJKLMN, vk1xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi1xOPQRSTUV, vk1xOPQRSTUV));

        const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 64 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
        const __m256i vk2x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 72 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 16)));
        const __m256i vk2xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 80 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 24)));
        const __m256i vk2xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 88 * sizeof(uint8_t)))), vk_zero_point);
        i2 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi2xGHIJKLMN, vk2xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi2xOPQRSTUV, vk2xOPQRSTUV));

        const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 96 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
        const __m256i vk3x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 104 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 16)));
        const __m256i vk3xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 112 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 24)));
        const __m256i vk3xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 120 * sizeof(uint8_t)))), vk_zero_point);
        i3 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi3xGHIJKLMN, vk3xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi3xOPQRSTUV, vk3xOPQRSTUV));

        const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 128 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
        const __m256i vk4x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 136 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 16)));
        const __m256i vk4xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 144 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 24)));
        const __m256i vk4xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 152 * sizeof(uint8_t)))), vk_zero_point);
        i4 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi4xGHIJKLMN, vk4xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi4xOPQRSTUV, vk4xOPQRSTUV));

        w = (const void*) ((uintptr_t) w + 160 * sizeof(uint8_t));

        _mm256_storeu_si256((__m256i*) b, vacc01234567);
        _mm256_storeu_si256((__m256i*) (b + 8), vacc89ABCDEF);
        _mm256_storeu_si256((__m256i*) (b + 16), vaccGHIJKLMN);
        _mm256_storeu_si256((__m256i*) (b + 24), vaccOPQRSTUV);
        b += 32;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);

          const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
          const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w))), vk_zero_point);
          i0 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

          const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
          const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(uint8_t)))), vk_zero_point);
          i1 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

          const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
          const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
          i2 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

          const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
          const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(uint8_t)))), vk_zero_point);
          i3 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

          const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
          const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
          i4 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

          w = (const void*) ((uintptr_t) w + 40 * sizeof(uint8_t));

          _mm256_storeu_si256((__m256i*) b, vacc01234567);
          b += 8;
          c -= 8;
        } while (c != 0);
      }
    }

    // Last pass to process up to 5 inputs.
    {
      const int32_t* b = buffer;
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

      size_t c = channels;

      for (; c >= 32; c -= 32) {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);
        __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) (b + 8));
        __m256i vaccGHIJKLMN = _mm256_loadu_si256((const __m256i*) (b + 16));
        __m256i vaccOPQRSTUV = _mm256_loadu_si256((const __m256i*) (b + 24));
        b += 32;


        const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
        const __m256i vk0x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 16)));
        const __m256i vk0xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi0xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 24)));
        const __m256i vk0xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(uint8_t)))), vk_zero_point);
        i0 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi0xGHIJKLMN, vk0xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi0xOPQRSTUV, vk0xOPQRSTUV));

        const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
        const __m256i vk1x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 40 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 16)));
        const __m256i vk1xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 48 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi1xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 24)));
        const __m256i vk1xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 56 * sizeof(uint8_t)))), vk_zero_point);
        i1 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi1xGHIJKLMN, vk1xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi1xOPQRSTUV, vk1xOPQRSTUV));

        const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 64 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
        const __m256i vk2x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 72 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 16)));
        const __m256i vk2xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 80 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi2xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 24)));
        const __m256i vk2xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 88 * sizeof(uint8_t)))), vk_zero_point);
        i2 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi2xGHIJKLMN, vk2xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi2xOPQRSTUV, vk2xOPQRSTUV));

        const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 96 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
        const __m256i vk3x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 104 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 16)));
        const __m256i vk3xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 112 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi3xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 24)));
        const __m256i vk3xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 120 * sizeof(uint8_t)))), vk_zero_point);
        i3 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi3xGHIJKLMN, vk3xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi3xOPQRSTUV, vk3xOPQRSTUV));

        const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 128 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
        const __m256i vk4x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 136 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4xGHIJKLMN = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 16)));
        const __m256i vk4xGHIJKLMN = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 144 * sizeof(uint8_t)))), vk_zero_point);
        const __m256i vi4xOPQRSTUV = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 24)));
        const __m256i vk4xOPQRSTUV = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 152 * sizeof(uint8_t)))), vk_zero_point);
        i4 += 32;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));
        vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_mullo_epi32(vi4xGHIJKLMN, vk4xGHIJKLMN));
        vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_mullo_epi32(vi4xOPQRSTUV, vk4xOPQRSTUV));

        w = (const void*) ((uintptr_t) w + 160 * sizeof(uint8_t));

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);
        __m256 vscaledGHIJKLMN = _mm256_cvtepi32_ps(vaccGHIJKLMN);
        __m256 vscaledOPQRSTUV = _mm256_cvtepi32_ps(vaccOPQRSTUV);

        const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale);
        vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale);
        vscaledGHIJKLMN = _mm256_mul_ps(vscaledGHIJKLMN, vscale);
        vscaledOPQRSTUV = _mm256_mul_ps(vscaledOPQRSTUV, vscale);

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

        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
        __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV), _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1)), _MM_SHUFFLE(3, 1, 2, 0));

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);
        voutGHIJKLMNOPQRSTUV = _mm_max_epu8(voutGHIJKLMNOPQRSTUV, voutput_min);

        _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
        _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
        output += 32;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) b);
          b += 8;

          const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
          const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
          i0 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

          const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
          const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(uint8_t)))), vk_zero_point);
          i1 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

          const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
          const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
          i2 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

          const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
          const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(uint8_t)))), vk_zero_point);
          i3 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

          const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
          const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
          i4 += 8;

          vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

          __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
          vscaled01234567 = _mm256_mul_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.scale));
          vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
          vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

          w = (void*) ((uintptr_t) w + 40 * sizeof(uint8_t));

          const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
          __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

          __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

          const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
          vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

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
              *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
              output += 1;
            }
            c = 0;
          }
        } while (c != 0);
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
