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

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse(
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
  assert(kernel_size > 2);

  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);
  do {
    const float* w = weights;

    // First pass to process 2 inputs.
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
      input += 2;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 8; c -= 8) {
        __m128 vacc0123p0 = _mm_load_ps(w);
        __m128 vacc4567p0 = _mm_load_ps(w + 4);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        i0 += 8;

        const __m128 vk0x0123 = _mm_load_ps(w + 8);
        const __m128 vk0x4567 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        i1 += 8;

        const __m128 vk1x0123 = _mm_load_ps(w + 16);
        const __m128 vk1x4567 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

        w += 24;


        _mm_store_ps(b, vacc0123p0);
        _mm_store_ps(b + 4, vacc4567p0);
        b += 8;
      }

      if (c != 0) {
        __m128 vacc0123p0 = _mm_load_ps(w);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        const __m128 vk0x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        const __m128 vk1x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        w += 12;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Middle pass to process 2 inputs in each iteration.
    for (size_t ks = kernel_size - 2; ks > 2; ks -= 2) {
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
      input += 2;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 8; c -= 8) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        __m128 vacc4567p0 = _mm_load_ps(b + 4);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        i0 += 8;

        const __m128 vk0x0123 = _mm_load_ps(w);
        const __m128 vk0x4567 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        i1 += 8;

        const __m128 vk1x0123 = _mm_load_ps(w + 8);
        const __m128 vk1x4567 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

        w += 16;


        _mm_store_ps(b, vacc0123p0);
        _mm_store_ps(b + 4, vacc4567p0);
        b += 8;
      }

      if (c != 0) {
        __m128 vacc0123p0 = _mm_load_ps(b);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        const __m128 vk0x0123 = _mm_load_ps(w);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        const __m128 vk1x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        w += 8;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Last pass to process up to 2 inputs.
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

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        __m128 vacc4567p0 = _mm_load_ps(b + 4);
        b += 8;


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        i0 += 8;

        __m128 vk0x0123 = _mm_load_ps(w);
        __m128 vk0x4567 = _mm_load_ps(w + 4);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        i1 += 8;

        __m128 vk1x0123 = _mm_load_ps(w + 8);
        __m128 vk1x4567 = _mm_load_ps(w + 12);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

        w += 16;


        __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
        __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);

        vacc0123 = _mm_min_ps(vacc0123, vmax);
        vacc4567 = _mm_min_ps(vacc4567, vmax);

        _mm_storeu_ps(output, vacc0123);
        _mm_storeu_ps(output + 4, vacc4567);
        output += 8;
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

        w += 8;



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
