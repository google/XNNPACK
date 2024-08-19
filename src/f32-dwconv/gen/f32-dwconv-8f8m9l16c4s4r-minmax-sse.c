// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xmmintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 8);

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 8 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      input += 8;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        __m128 vacc0123p0 = _mm_load_ps(w);
        __m128 vacc4567p0 = _mm_load_ps(w + 4);
        __m128 vacc89ABp0 = _mm_load_ps(w + 8);
        __m128 vaccCDEFp0 = _mm_load_ps(w + 12);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        const __m128 vi0x89AB = _mm_loadu_ps(i0 + 8);
        const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 12);
        i0 += 16;

        const __m128 vk0x0123 = _mm_load_ps(w + 16);
        const __m128 vk0x4567 = _mm_load_ps(w + 20);
        const __m128 vk0x89AB = _mm_load_ps(w + 24);
        const __m128 vk0xCDEF = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi0xCDEF, vk0xCDEF));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        const __m128 vi1x89AB = _mm_loadu_ps(i1 + 8);
        const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 12);
        i1 += 16;

        const __m128 vk1x0123 = _mm_load_ps(w + 32);
        const __m128 vk1x4567 = _mm_load_ps(w + 36);
        const __m128 vk1x89AB = _mm_load_ps(w + 40);
        const __m128 vk1xCDEF = _mm_load_ps(w + 44);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi1xCDEF, vk1xCDEF));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
        const __m128 vi2x89AB = _mm_loadu_ps(i2 + 8);
        const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 12);
        i2 += 16;

        const __m128 vk2x0123 = _mm_load_ps(w + 48);
        const __m128 vk2x4567 = _mm_load_ps(w + 52);
        const __m128 vk2x89AB = _mm_load_ps(w + 56);
        const __m128 vk2xCDEF = _mm_load_ps(w + 60);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi2x89AB, vk2x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi2xCDEF, vk2xCDEF));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
        const __m128 vi3x89AB = _mm_loadu_ps(i3 + 8);
        const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 12);
        i3 += 16;

        const __m128 vk3x0123 = _mm_load_ps(w + 64);
        const __m128 vk3x4567 = _mm_load_ps(w + 68);
        const __m128 vk3x89AB = _mm_load_ps(w + 72);
        const __m128 vk3xCDEF = _mm_load_ps(w + 76);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
        const __m128 vi4x89AB = _mm_loadu_ps(i4 + 8);
        const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 12);
        i4 += 16;

        const __m128 vk4x0123 = _mm_load_ps(w + 80);
        const __m128 vk4x4567 = _mm_load_ps(w + 84);
        const __m128 vk4x89AB = _mm_load_ps(w + 88);
        const __m128 vk4xCDEF = _mm_load_ps(w + 92);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi4x89AB, vk4x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi4xCDEF, vk4xCDEF));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
        const __m128 vi5x89AB = _mm_loadu_ps(i5 + 8);
        const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 12);
        i5 += 16;

        const __m128 vk5x0123 = _mm_load_ps(w + 96);
        const __m128 vk5x4567 = _mm_load_ps(w + 100);
        const __m128 vk5x89AB = _mm_load_ps(w + 104);
        const __m128 vk5xCDEF = _mm_load_ps(w + 108);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi5x89AB, vk5x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi5xCDEF, vk5xCDEF));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
        const __m128 vi6x89AB = _mm_loadu_ps(i6 + 8);
        const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 12);
        i6 += 16;

        const __m128 vk6x0123 = _mm_load_ps(w + 112);
        const __m128 vk6x4567 = _mm_load_ps(w + 116);
        const __m128 vk6x89AB = _mm_load_ps(w + 120);
        const __m128 vk6xCDEF = _mm_load_ps(w + 124);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi6x89AB, vk6x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi6xCDEF, vk6xCDEF));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
        const __m128 vi7x89AB = _mm_loadu_ps(i7 + 8);
        const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 12);
        i7 += 16;

        const __m128 vk7x0123 = _mm_load_ps(w + 128);
        const __m128 vk7x4567 = _mm_load_ps(w + 132);
        const __m128 vk7x89AB = _mm_load_ps(w + 136);
        const __m128 vk7xCDEF = _mm_load_ps(w + 140);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi7x89AB, vk7x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi7xCDEF, vk7xCDEF));

        w += 144;


        _mm_store_ps(b, vacc0123p0);
        _mm_store_ps(b + 4, vacc4567p0);
        _mm_store_ps(b + 8, vacc89ABp0);
        _mm_store_ps(b + 12, vaccCDEFp0);
        b += 16;
      }

      for (; c != 0; c -= 4) {
        __m128 vacc0123p0 = _mm_load_ps(w);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        const __m128 vk0x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        const __m128 vk1x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        const __m128 vk2x0123 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        const __m128 vk3x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        const __m128 vk4x0123 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        i5 += 4;

        const __m128 vk5x0123 = _mm_load_ps(w + 24);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        i6 += 4;

        const __m128 vk6x0123 = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        i7 += 4;

        const __m128 vk7x0123 = _mm_load_ps(w + 32);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        w += 36;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Middle pass to process 8 inputs in each iteration.
    for (size_t ks = kernel_size - 8; ks > 9; ks -= 8) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      input += 8;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        __m128 vacc4567p0 = _mm_load_ps(b + 4);
        __m128 vacc89ABp0 = _mm_load_ps(b + 8);
        __m128 vaccCDEFp0 = _mm_load_ps(b + 12);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        const __m128 vi0x89AB = _mm_loadu_ps(i0 + 8);
        const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 12);
        i0 += 16;

        const __m128 vk0x0123 = _mm_load_ps(w);
        const __m128 vk0x4567 = _mm_load_ps(w + 4);
        const __m128 vk0x89AB = _mm_load_ps(w + 8);
        const __m128 vk0xCDEF = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi0xCDEF, vk0xCDEF));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        const __m128 vi1x89AB = _mm_loadu_ps(i1 + 8);
        const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 12);
        i1 += 16;

        const __m128 vk1x0123 = _mm_load_ps(w + 16);
        const __m128 vk1x4567 = _mm_load_ps(w + 20);
        const __m128 vk1x89AB = _mm_load_ps(w + 24);
        const __m128 vk1xCDEF = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi1xCDEF, vk1xCDEF));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
        const __m128 vi2x89AB = _mm_loadu_ps(i2 + 8);
        const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 12);
        i2 += 16;

        const __m128 vk2x0123 = _mm_load_ps(w + 32);
        const __m128 vk2x4567 = _mm_load_ps(w + 36);
        const __m128 vk2x89AB = _mm_load_ps(w + 40);
        const __m128 vk2xCDEF = _mm_load_ps(w + 44);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi2x89AB, vk2x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi2xCDEF, vk2xCDEF));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
        const __m128 vi3x89AB = _mm_loadu_ps(i3 + 8);
        const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 12);
        i3 += 16;

        const __m128 vk3x0123 = _mm_load_ps(w + 48);
        const __m128 vk3x4567 = _mm_load_ps(w + 52);
        const __m128 vk3x89AB = _mm_load_ps(w + 56);
        const __m128 vk3xCDEF = _mm_load_ps(w + 60);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
        const __m128 vi4x89AB = _mm_loadu_ps(i4 + 8);
        const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 12);
        i4 += 16;

        const __m128 vk4x0123 = _mm_load_ps(w + 64);
        const __m128 vk4x4567 = _mm_load_ps(w + 68);
        const __m128 vk4x89AB = _mm_load_ps(w + 72);
        const __m128 vk4xCDEF = _mm_load_ps(w + 76);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi4x89AB, vk4x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi4xCDEF, vk4xCDEF));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
        const __m128 vi5x89AB = _mm_loadu_ps(i5 + 8);
        const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 12);
        i5 += 16;

        const __m128 vk5x0123 = _mm_load_ps(w + 80);
        const __m128 vk5x4567 = _mm_load_ps(w + 84);
        const __m128 vk5x89AB = _mm_load_ps(w + 88);
        const __m128 vk5xCDEF = _mm_load_ps(w + 92);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi5x89AB, vk5x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi5xCDEF, vk5xCDEF));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
        const __m128 vi6x89AB = _mm_loadu_ps(i6 + 8);
        const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 12);
        i6 += 16;

        const __m128 vk6x0123 = _mm_load_ps(w + 96);
        const __m128 vk6x4567 = _mm_load_ps(w + 100);
        const __m128 vk6x89AB = _mm_load_ps(w + 104);
        const __m128 vk6xCDEF = _mm_load_ps(w + 108);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi6x89AB, vk6x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi6xCDEF, vk6xCDEF));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
        const __m128 vi7x89AB = _mm_loadu_ps(i7 + 8);
        const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 12);
        i7 += 16;

        const __m128 vk7x0123 = _mm_load_ps(w + 112);
        const __m128 vk7x4567 = _mm_load_ps(w + 116);
        const __m128 vk7x89AB = _mm_load_ps(w + 120);
        const __m128 vk7xCDEF = _mm_load_ps(w + 124);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi7x89AB, vk7x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi7xCDEF, vk7xCDEF));

        w += 128;


        _mm_store_ps(b, vacc0123p0);
        _mm_store_ps(b + 4, vacc4567p0);
        _mm_store_ps(b + 8, vacc89ABp0);
        _mm_store_ps(b + 12, vaccCDEFp0);
        b += 16;
      }

      for (; c != 0; c -= 4) {
        __m128 vacc0123p0 = _mm_load_ps(b);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        const __m128 vk0x0123 = _mm_load_ps(w);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        const __m128 vk1x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        const __m128 vk2x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        const __m128 vk3x0123 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        const __m128 vk4x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        i5 += 4;

        const __m128 vk5x0123 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        i6 += 4;

        const __m128 vk6x0123 = _mm_load_ps(w + 24);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        i7 += 4;

        const __m128 vk7x0123 = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        w += 32;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Last pass to process up to 9 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = input[8];
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        __m128 vacc4567p0 = _mm_load_ps(b + 4);
        __m128 vacc89ABp0 = _mm_load_ps(b + 8);
        __m128 vaccCDEFp0 = _mm_load_ps(b + 12);
        b += 16;


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        const __m128 vi0x89AB = _mm_loadu_ps(i0 + 8);
        const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 12);
        i0 += 16;

        __m128 vk0x0123 = _mm_load_ps(w);
        __m128 vk0x4567 = _mm_load_ps(w + 4);
        __m128 vk0x89AB = _mm_load_ps(w + 8);
        __m128 vk0xCDEF = _mm_load_ps(w + 12);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi0xCDEF, vk0xCDEF));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        const __m128 vi1x89AB = _mm_loadu_ps(i1 + 8);
        const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 12);
        i1 += 16;

        __m128 vk1x0123 = _mm_load_ps(w + 16);
        __m128 vk1x4567 = _mm_load_ps(w + 20);
        __m128 vk1x89AB = _mm_load_ps(w + 24);
        __m128 vk1xCDEF = _mm_load_ps(w + 28);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi1xCDEF, vk1xCDEF));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
        const __m128 vi2x89AB = _mm_loadu_ps(i2 + 8);
        const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 12);
        i2 += 16;

        __m128 vk2x0123 = _mm_load_ps(w + 32);
        __m128 vk2x4567 = _mm_load_ps(w + 36);
        __m128 vk2x89AB = _mm_load_ps(w + 40);
        __m128 vk2xCDEF = _mm_load_ps(w + 44);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi2x89AB, vk2x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi2xCDEF, vk2xCDEF));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
        const __m128 vi3x89AB = _mm_loadu_ps(i3 + 8);
        const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 12);
        i3 += 16;

        __m128 vk3x0123 = _mm_load_ps(w + 48);
        __m128 vk3x4567 = _mm_load_ps(w + 52);
        __m128 vk3x89AB = _mm_load_ps(w + 56);
        __m128 vk3xCDEF = _mm_load_ps(w + 60);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
        const __m128 vi4x89AB = _mm_loadu_ps(i4 + 8);
        const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 12);
        i4 += 16;

        __m128 vk4x0123 = _mm_load_ps(w + 64);
        __m128 vk4x4567 = _mm_load_ps(w + 68);
        __m128 vk4x89AB = _mm_load_ps(w + 72);
        __m128 vk4xCDEF = _mm_load_ps(w + 76);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi4x89AB, vk4x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi4xCDEF, vk4xCDEF));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
        const __m128 vi5x89AB = _mm_loadu_ps(i5 + 8);
        const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 12);
        i5 += 16;

        __m128 vk5x0123 = _mm_load_ps(w + 80);
        __m128 vk5x4567 = _mm_load_ps(w + 84);
        __m128 vk5x89AB = _mm_load_ps(w + 88);
        __m128 vk5xCDEF = _mm_load_ps(w + 92);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi5x89AB, vk5x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi5xCDEF, vk5xCDEF));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
        const __m128 vi6x89AB = _mm_loadu_ps(i6 + 8);
        const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 12);
        i6 += 16;

        __m128 vk6x0123 = _mm_load_ps(w + 96);
        __m128 vk6x4567 = _mm_load_ps(w + 100);
        __m128 vk6x89AB = _mm_load_ps(w + 104);
        __m128 vk6xCDEF = _mm_load_ps(w + 108);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi6x89AB, vk6x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi6xCDEF, vk6xCDEF));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
        const __m128 vi7x89AB = _mm_loadu_ps(i7 + 8);
        const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 12);
        i7 += 16;

        __m128 vk7x0123 = _mm_load_ps(w + 112);
        __m128 vk7x4567 = _mm_load_ps(w + 116);
        __m128 vk7x89AB = _mm_load_ps(w + 120);
        __m128 vk7xCDEF = _mm_load_ps(w + 124);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi7x89AB, vk7x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi7xCDEF, vk7xCDEF));

        const __m128 vi8x0123 = _mm_loadu_ps(i8);
        const __m128 vi8x4567 = _mm_loadu_ps(i8 + 4);
        const __m128 vi8x89AB = _mm_loadu_ps(i8 + 8);
        const __m128 vi8xCDEF = _mm_loadu_ps(i8 + 12);
        i8 += 16;

        __m128 vk8x0123 = _mm_load_ps(w + 128);
        __m128 vk8x4567 = _mm_load_ps(w + 132);
        __m128 vk8x89AB = _mm_load_ps(w + 136);
        __m128 vk8xCDEF = _mm_load_ps(w + 140);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi8x4567, vk8x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi8x89AB, vk8x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi8xCDEF, vk8xCDEF));

        w += 144;


        __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
        __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);
        __m128 vacc89AB = _mm_max_ps(vacc89ABp0, vmin);
        __m128 vaccCDEF = _mm_max_ps(vaccCDEFp0, vmin);

        vacc0123 = _mm_min_ps(vacc0123, vmax);
        vacc4567 = _mm_min_ps(vacc4567, vmax);
        vacc89AB = _mm_min_ps(vacc89AB, vmax);
        vaccCDEF = _mm_min_ps(vaccCDEF, vmax);

        _mm_storeu_ps(output, vacc0123);
        _mm_storeu_ps(output + 4, vacc4567);
        _mm_storeu_ps(output + 8, vacc89AB);
        _mm_storeu_ps(output + 12, vaccCDEF);
        output += 16;
      }


      for (; c >= 4; c -= 4) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        b += 4;


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        __m128 vk0x0123 = _mm_load_ps(w);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        __m128 vk1x0123 = _mm_load_ps(w + 4);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        __m128 vk2x0123 = _mm_load_ps(w + 8);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        __m128 vk3x0123 = _mm_load_ps(w + 12);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        __m128 vk4x0123 = _mm_load_ps(w + 16);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        i5 += 4;

        __m128 vk5x0123 = _mm_load_ps(w + 20);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        i6 += 4;

        __m128 vk6x0123 = _mm_load_ps(w + 24);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        i7 += 4;

        __m128 vk7x0123 = _mm_load_ps(w + 28);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        const __m128 vi8x0123 = _mm_loadu_ps(i8);
        i8 += 4;

        __m128 vk8x0123 = _mm_load_ps(w + 32);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));

        w += 36;



        __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);

        vacc0123 = _mm_min_ps(vacc0123, vmax);

        _mm_storeu_ps(output, vacc0123);
        output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        __m128 vacc0123p0 = _mm_load_ps(b);

        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        __m128 vk0x0123 = _mm_load_ps(w);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        __m128 vk1x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        __m128 vk2x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        __m128 vk3x0123 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        __m128 vk4x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        __m128 vk5x0123 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        __m128 vk6x0123 = _mm_load_ps(w + 24);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        __m128 vk7x0123 = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        const __m128 vi8x0123 = _mm_loadu_ps(i8);
        __m128 vk8x0123 = _mm_load_ps(w + 32);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));


        __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
        vacc0123 = _mm_min_ps(vacc0123, vmax);

        if (c & 2) {
          _mm_storel_pi((__m64*) output, vacc0123);
          vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vacc0123);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
