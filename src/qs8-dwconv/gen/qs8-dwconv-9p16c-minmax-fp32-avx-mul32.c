// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-sse-mul32.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul32(
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


      const __m128i vi0x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0)));
      const __m128i vk0x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)))));
      const __m128i vi0x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 4)));
      const __m128i vk0x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 4 * sizeof(int8_t)))));
      const __m128i vi0x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 8)));
      const __m128i vk0x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)))));
      const __m128i vi0xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0 + 12)));
      const __m128i vk0xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 12 * sizeof(int8_t)))));
      i0 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi0x0123, vk0x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi0x4567, vk0x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi0x89AB, vk0x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi0xCDEF, vk0xCDEF));

      const __m128i vi1x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1)));
      const __m128i vk1x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)))));
      const __m128i vi1x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 4)));
      const __m128i vk1x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 20 * sizeof(int8_t)))));
      const __m128i vi1x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 8)));
      const __m128i vk1x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)))));
      const __m128i vi1xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1 + 12)));
      const __m128i vk1xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 28 * sizeof(int8_t)))));
      i1 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi1x0123, vk1x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi1x4567, vk1x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi1x89AB, vk1x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi1xCDEF, vk1xCDEF));

      const __m128i vi2x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2)));
      const __m128i vk2x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)))));
      const __m128i vi2x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 4)));
      const __m128i vk2x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 36 * sizeof(int8_t)))));
      const __m128i vi2x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 8)));
      const __m128i vk2x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)))));
      const __m128i vi2xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2 + 12)));
      const __m128i vk2xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 44 * sizeof(int8_t)))));
      i2 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi2x0123, vk2x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi2x4567, vk2x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi2x89AB, vk2x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi2xCDEF, vk2xCDEF));

      const __m128i vi3x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3)));
      const __m128i vk3x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)))));
      const __m128i vi3x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 4)));
      const __m128i vk3x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 52 * sizeof(int8_t)))));
      const __m128i vi3x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 8)));
      const __m128i vk3x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)))));
      const __m128i vi3xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3 + 12)));
      const __m128i vk3xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 60 * sizeof(int8_t)))));
      i3 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi3x0123, vk3x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi3x4567, vk3x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi3x89AB, vk3x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi3xCDEF, vk3xCDEF));

      const __m128i vi4x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4)));
      const __m128i vk4x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)))));
      const __m128i vi4x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 4)));
      const __m128i vk4x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 68 * sizeof(int8_t)))));
      const __m128i vi4x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 8)));
      const __m128i vk4x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)))));
      const __m128i vi4xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4 + 12)));
      const __m128i vk4xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 76 * sizeof(int8_t)))));
      i4 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi4x0123, vk4x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi4x4567, vk4x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi4x89AB, vk4x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi4xCDEF, vk4xCDEF));

      const __m128i vi5x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5)));
      const __m128i vk5x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)))));
      const __m128i vi5x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 4)));
      const __m128i vk5x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 84 * sizeof(int8_t)))));
      const __m128i vi5x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 8)));
      const __m128i vk5x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)))));
      const __m128i vi5xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5 + 12)));
      const __m128i vk5xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 92 * sizeof(int8_t)))));
      i5 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi5x0123, vk5x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi5x4567, vk5x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi5x89AB, vk5x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi5xCDEF, vk5xCDEF));

      const __m128i vi6x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6)));
      const __m128i vk6x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)))));
      const __m128i vi6x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 4)));
      const __m128i vk6x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 100 * sizeof(int8_t)))));
      const __m128i vi6x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 8)));
      const __m128i vk6x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)))));
      const __m128i vi6xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6 + 12)));
      const __m128i vk6xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 108 * sizeof(int8_t)))));
      i6 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi6x0123, vk6x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi6x4567, vk6x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi6x89AB, vk6x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi6xCDEF, vk6xCDEF));

      const __m128i vi7x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7)));
      const __m128i vk7x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)))));
      const __m128i vi7x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 4)));
      const __m128i vk7x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 116 * sizeof(int8_t)))));
      const __m128i vi7x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 8)));
      const __m128i vk7x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)))));
      const __m128i vi7xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7 + 12)));
      const __m128i vk7xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 124 * sizeof(int8_t)))));
      i7 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi7x0123, vk7x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi7x4567, vk7x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi7x89AB, vk7x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi7xCDEF, vk7xCDEF));

      const __m128i vi8x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8)));
      const __m128i vk8x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)))));
      const __m128i vi8x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 4)));
      const __m128i vk8x4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 132 * sizeof(int8_t)))));
      const __m128i vi8x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 8)));
      const __m128i vk8x89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)))));
      const __m128i vi8xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8 + 12)));
      const __m128i vk8xCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) ((uintptr_t) w + 16 * sizeof(int32_t) + 140 * sizeof(int8_t)))));
      i8 += 16;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi8x0123, vk8x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi8x4567, vk8x4567));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_mullo_epi32(vi8x89AB, vk8x89AB));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_mullo_epi32(vi8xCDEF, vk8xCDEF));

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

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);

        const __m128i vi0x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i0)));
        const __m128i vk0x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) k)));
        i0 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi0x0123, vk0x0123));
        const __m128i vi1x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i1)));
        const __m128i vk1x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 16))));
        i1 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi1x0123, vk1x0123));
        const __m128i vi2x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i2)));
        const __m128i vk2x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 32))));
        i2 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi2x0123, vk2x0123));
        const __m128i vi3x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i3)));
        const __m128i vk3x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 48))));
        i3 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi3x0123, vk3x0123));
        const __m128i vi4x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i4)));
        const __m128i vk4x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 64))));
        i4 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi4x0123, vk4x0123));
        const __m128i vi5x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i5)));
        const __m128i vk5x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 80))));
        i5 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi5x0123, vk5x0123));
        const __m128i vi6x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i6)));
        const __m128i vk6x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 96))));
        i6 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi6x0123, vk6x0123));
        const __m128i vi7x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i7)));
        const __m128i vk7x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 112))));
        i7 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi7x0123, vk7x0123));
        const __m128i vi8x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(i8)));
        const __m128i vk8x0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*((const int*) (k + 128))));
        i8 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi8x0123, vk8x0123));

        k += 4;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        vscaled0123 = _mm_mul_ps(vscaled0123, _mm_load_ps(params->fp32_sse4.scale));
        vscaled0123 = _mm_min_ps(vscaled0123, _mm_load_ps(params->fp32_sse4.output_max_less_zero_point));
        vacc0123 = _mm_cvtps_epi32(vscaled0123);

        w = (const void*) ((const int32_t*) w + 4);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc0123), voutput_zero_point);

        vout0123 = _mm_packs_epi16(vout0123, vout0123);
        vout0123 = _mm_max_epi8(vout0123, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

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
            *output = (int8_t) _mm_extract_epi8(vout0123, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
