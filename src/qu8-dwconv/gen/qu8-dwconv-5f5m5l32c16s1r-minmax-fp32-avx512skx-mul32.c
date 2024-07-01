// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-avx512skx-mul32.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_qu8_dwconv_minmax_fp32_ukernel_5f5m5l32c16s1r__avx512skx_mul32(
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

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi16((int16_t) params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_broadcastb_epi8(_mm_load_si128((const __m128i*) params->fp32_avx512.output_min));
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  const __m512i vk_zero_point = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*) params->fp32_avx512.kernel_zero_point));

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

      size_t c = channels;

      for (; c >= 32; c -= 32) {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
        __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((const int32_t*) w + 16));

        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
        i0 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
        i1 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
        i2 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
        i3 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(uint8_t)))), vk_zero_point);
        i4 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

        w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(uint8_t));

        _mm512_storeu_si512((__m512i*) b, vacc0123456789ABCDEF);
        _mm512_storeu_si512((__m512i*) (b + 16), vaccGHIJKLMNOPQRSTUV);
        b += 32;
      }

      if XNN_UNLIKELY(c != 0) {
        for (; c >= 16; c -= 16) {
          __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);

          const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
          const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
          i0 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

          const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
          const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
          i1 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

          const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
          const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
          i2 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

          const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
          const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
          i3 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

          const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
          const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
          i4 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

          w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t));

          _mm512_storeu_si512((__m512i*) b, vacc0123456789ABCDEF);
          b += 16;
        }
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);

        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t));

        _mm512_storeu_si512((__m512i*) b, vacc0123456789ABCDEF);
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

      size_t c = channels;

      for (; c >= 32; c -= 32) {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(b);
        __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((const int32_t*) b + 16));

        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
        i0 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(uint8_t)))), vk_zero_point);
        i1 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 80 * sizeof(uint8_t)))), vk_zero_point);
        i2 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 96 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 112 * sizeof(uint8_t)))), vk_zero_point);
        i3 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 128 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 144 * sizeof(uint8_t)))), vk_zero_point);
        i4 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

        w = (const void*) ((uintptr_t) w + 160 * sizeof(uint8_t));

        _mm512_storeu_si512((__m512i*) b, vacc0123456789ABCDEF);
        _mm512_storeu_si512((__m512i*) (b + 16), vaccGHIJKLMNOPQRSTUV);
        b += 32;
      }

      if XNN_UNLIKELY(c != 0) {
        for (; c >= 16; c -= 16) {
          __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(b);

          const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
          const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
          i0 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

          const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
          const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
          i1 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

          const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
          const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
          i2 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

          const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
          const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(uint8_t)))), vk_zero_point);
          i3 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

          const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
          const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(uint8_t)))), vk_zero_point);
          i4 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

          w = (const void*) ((uintptr_t) w + 80 * sizeof(uint8_t));

          _mm512_storeu_si512((__m512i*) b, vacc0123456789ABCDEF);
          b += 16;
        }
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(b);

        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(uint8_t)))), vk_zero_point);
        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        w = (const void*) ((uintptr_t) w + 80 * sizeof(uint8_t));

        _mm512_storeu_si512((__m512i*) b, vacc0123456789ABCDEF);
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
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(b);
        __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((const int32_t*) b + 16));
        b += 32;

        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
        i0 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(uint8_t)))), vk_zero_point);
        i1 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 80 * sizeof(uint8_t)))), vk_zero_point);
        i2 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 96 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 112 * sizeof(uint8_t)))), vk_zero_point);
        i3 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 128 * sizeof(uint8_t)))), vk_zero_point);
        const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 144 * sizeof(uint8_t)))), vk_zero_point);
        i4 += 32;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
        vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

        w = (const void*) ((uintptr_t) w + 160 * sizeof(uint8_t));

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
        vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

        __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
        __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
        const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
        const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packus_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
        __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
        const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
        const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
        __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packus_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

        vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epu8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
        voutGHIJKLMNOPQRSTUV = _mm_max_epu8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

        _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
        output += 32;
      }

      if XNN_UNLIKELY(c != 0) {
        // Prepare mask for valid 8-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
        do {
          __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(b);
          b += 16;

          const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
          const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 0 * sizeof(uint8_t)))), vk_zero_point);
          i0 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

          const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
          const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(uint8_t)))), vk_zero_point);
          i1 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

          const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
          const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(uint8_t)))), vk_zero_point);
          i2 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

          const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
          const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 48 * sizeof(uint8_t)))), vk_zero_point);
          i3 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

          const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
          const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((uintptr_t) w + 64 * sizeof(uint8_t)))), vk_zero_point);
          i4 += 16;

          vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

          __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
          vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
          vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
          vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

          w = (void*) ((uintptr_t) w + 80 * sizeof(uint8_t));

          __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

          const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
          const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
          __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
          vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));


          if XNN_LIKELY(c >= 16) {
            _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
            output += 16;
            c -= 16;
          } else {
            _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
            output = (uint8_t*) ((uintptr_t) output + c);
            c = 0;
          }
        } while (c != 0);
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
