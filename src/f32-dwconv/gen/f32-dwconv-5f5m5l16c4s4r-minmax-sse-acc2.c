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


void xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2(
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
  assert(kernel_size > 5);

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 5 inputs.
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
      input += 5;

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
        __m128 vacc0123p1 = _mm_mul_ps(vi1x0123, vk1x0123);
        __m128 vacc4567p1 = _mm_mul_ps(vi1x4567, vk1x4567);
        __m128 vacc89ABp1 = _mm_mul_ps(vi1x89AB, vk1x89AB);
        __m128 vaccCDEFp1 = _mm_mul_ps(vi1xCDEF, vk1xCDEF);

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
        vacc0123p1 = _mm_add_ps(vacc0123p1, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p1 = _mm_add_ps(vacc4567p1, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp1 = _mm_add_ps(vacc89ABp1, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp1 = _mm_add_ps(vaccCDEFp1, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

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

        w += 96;

        // Add up all accumulators to vacc0123456789ABCDEFp0
        vacc0123p0 = _mm_add_ps(vacc0123p0, vacc0123p1);
        vacc4567p0 = _mm_add_ps(vacc4567p0, vacc4567p1);
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, vacc89ABp1);
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, vaccCDEFp1);

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
        __m128 vacc0123p1 = _mm_mul_ps(vi1x0123, vk1x0123);

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        const __m128 vk2x0123 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        const __m128 vk3x0123 = _mm_load_ps(w + 16);
        vacc0123p1 = _mm_add_ps(vacc0123p1, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        const __m128 vk4x0123 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        w += 24;

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = _mm_add_ps(vacc0123p0, vacc0123p1);

        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
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
      input += 5;

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
        __m128 vacc0123p1 = _mm_mul_ps(vi1x0123, vk1x0123);
        __m128 vacc4567p1 = _mm_mul_ps(vi1x4567, vk1x4567);
        __m128 vacc89ABp1 = _mm_mul_ps(vi1x89AB, vk1x89AB);
        __m128 vaccCDEFp1 = _mm_mul_ps(vi1xCDEF, vk1xCDEF);

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
        vacc0123p1 = _mm_add_ps(vacc0123p1, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p1 = _mm_add_ps(vacc4567p1, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp1 = _mm_add_ps(vacc89ABp1, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp1 = _mm_add_ps(vaccCDEFp1, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

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

        w += 80;

        // Add up all accumulators to vacc0123456789ABCDEFp0
        vacc0123p0 = _mm_add_ps(vacc0123p0, vacc0123p1);
        vacc4567p0 = _mm_add_ps(vacc4567p0, vacc4567p1);
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, vacc89ABp1);
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, vaccCDEFp1);

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
        __m128 vacc0123p1 = _mm_mul_ps(vi1x0123, vk1x0123);

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        const __m128 vk2x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        const __m128 vk3x0123 = _mm_load_ps(w + 12);
        vacc0123p1 = _mm_add_ps(vacc0123p1, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        const __m128 vk4x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        w += 20;

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = _mm_add_ps(vacc0123p0, vacc0123p1);

        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Last pass to process up to 5 inputs.
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

        __m128 vacc0123p1 = _mm_mul_ps(vi1x0123, vk1x0123);
        __m128 vacc4567p1 = _mm_mul_ps(vi1x4567, vk1x4567);
        __m128 vacc89ABp1 = _mm_mul_ps(vi1x89AB, vk1x89AB);
        __m128 vaccCDEFp1 = _mm_mul_ps(vi1xCDEF, vk1xCDEF);

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

        vacc0123p1 = _mm_add_ps(vacc0123p1, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p1 = _mm_add_ps(vacc4567p1, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp1 = _mm_add_ps(vacc89ABp1, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp1 = _mm_add_ps(vaccCDEFp1, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

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

        w += 80;

        // Add up all accumulators to vacc0123456789ABCDEFp0
        vacc0123p0 = _mm_add_ps(vacc0123p0, vacc0123p1);
        vacc4567p0 = _mm_add_ps(vacc4567p0, vacc4567p1);
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, vacc89ABp1);
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, vaccCDEFp1);

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

        __m128 vacc0123p1 = _mm_mul_ps(vi1x0123, vk1x0123);

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        __m128 vk2x0123 = _mm_load_ps(w + 8);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        __m128 vk3x0123 = _mm_load_ps(w + 12);

        vacc0123p1 = _mm_add_ps(vacc0123p1, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        __m128 vk4x0123 = _mm_load_ps(w + 16);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        w += 20;


        // Add up all accumulators to vacc0123p0
        vacc0123p0 = _mm_add_ps(vacc0123p0, vacc0123p1);

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
        __m128 vacc0123p1 = _mm_mul_ps(vi1x0123, vk1x0123);

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        __m128 vk2x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        __m128 vk3x0123 = _mm_load_ps(w + 12);
        vacc0123p1 = _mm_add_ps(vacc0123p1, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        __m128 vk4x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        // Add up all accumulators to vacc0123456789ABCDEFp0
        vacc0123p0 = _mm_add_ps(vacc0123p0, vacc0123p1);

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
