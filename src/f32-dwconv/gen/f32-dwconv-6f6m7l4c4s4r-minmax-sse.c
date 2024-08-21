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


void xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse(
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
  assert(kernel_size > 6);

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 6 inputs.
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
      input += 6;

      // Process c channels and write to buffer.
      size_t c = 0;
      for (; c < channels; c += 4) {
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

        w += 28;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Middle pass to process 6 inputs in each iteration.
    for (size_t ks = kernel_size - 6; ks > 7; ks -= 6) {
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
      input += 6;

      size_t c = 0;
      for (; c < channels; c += 4) {
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

        w += 24;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Last pass to process up to 7 inputs.
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

      size_t c = channels;


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

        w += 28;



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
