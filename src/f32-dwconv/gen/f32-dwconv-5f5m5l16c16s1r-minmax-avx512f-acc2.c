// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-avx512.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
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
      size_t c = channels;

      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(w);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        const __m512 vk0x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        const __m512 vk2x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        const __m512 vk3x0 = _mm512_load_ps(w + 64);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        const __m512 vk4x0 = _mm512_load_ps(w + 80);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 96;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(w);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);

        const __m512 vk2x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);

        const __m512 vk3x0 = _mm512_load_ps(w + 64);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);

        const __m512 vk4x0 = _mm512_load_ps(w + 80);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 96;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
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

      size_t c = channels;

      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(b);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        const __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        const __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        const __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        const __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        const __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(b);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);

        const __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);

        const __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);

        const __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
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
        __m512 vaccp0 = _mm512_load_ps(b);
        b += 16;


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        __m512 vk0x0 = _mm512_load_ps(w);

        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        __m512 vk1x0 = _mm512_load_ps(w + 16);

        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        __m512 vk2x0 = _mm512_load_ps(w + 32);

        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        __m512 vk3x0 = _mm512_load_ps(w + 48);

        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        __m512 vk4x0 = _mm512_load_ps(w + 64);

        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;


        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        __m512 vacc = _mm512_max_ps(vmin, vaccp0);

        vacc = _mm512_min_ps(vmax, vacc);

        _mm512_storeu_ps(output, vacc);
        output += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        __m512 vaccp0 = _mm512_load_ps(b);
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));

        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);
        __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);
        __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);
        __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);
        __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);
        __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        __m512 vacc = _mm512_max_ps(vmin, vaccp0);
        vacc = _mm512_min_ps(vmax, vacc);

        _mm512_mask_storeu_ps(output, vmask, vacc);
        output += c;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
