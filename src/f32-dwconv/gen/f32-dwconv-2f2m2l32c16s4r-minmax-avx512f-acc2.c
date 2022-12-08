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

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_minmax_ukernel_2f2m2l32c16s4r__avx512f_acc2(
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

  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
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
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(w);
        __m512 vacc1p0 = _mm512_load_ps(w + 16);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        const __m512 vk0x0 = _mm512_load_ps(w + 32);
        const __m512 vk0x1 = _mm512_load_ps(w + 48);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        const __m512 vk1x0 = _mm512_load_ps(w + 64);
        const __m512 vk1x1 = _mm512_load_ps(w + 80);
        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);
        __m512 vacc1p1 = _mm512_mul_ps(vi1x1, vk1x1);

        w += 96;

        // Add up all accumulators to vacc0p0
        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);
        vacc1p0 = _mm512_add_ps(vacc1p0, vacc1p1);

        _mm512_store_ps(b, vacc0p0);
        _mm512_store_ps(b + 16, vacc1p0);
        b += 32;
      }

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

        w += 48;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(w);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        w += 48;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
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
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(b);
        __m512 vacc1p0 = _mm512_load_ps(b + 16);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        const __m512 vk0x0 = _mm512_load_ps(w);
        const __m512 vk0x1 = _mm512_load_ps(w + 16);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        const __m512 vk1x1 = _mm512_load_ps(w + 48);
        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);
        __m512 vacc1p1 = _mm512_mul_ps(vi1x1, vk1x1);

        w += 64;

        // Add up all accumulators to vacc0p0
        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);
        vacc1p0 = _mm512_add_ps(vacc1p0, vacc1p1);

        _mm512_store_ps(b, vacc0p0);
        _mm512_store_ps(b + 16, vacc1p0);
        b += 32;
      }

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

        w += 32;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(b);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        w += 32;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
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
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(b);
        __m512 vacc1p0 = _mm512_load_ps(b + 16);
        b += 32;


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        __m512 vk0x0 = _mm512_load_ps(w);
        __m512 vk0x1 = _mm512_load_ps(w + 16);

        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vk1x1 = _mm512_load_ps(w + 48);

        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);
        __m512 vacc1p1 = _mm512_mul_ps(vi1x1, vk1x1);

        w += 64;

        // Add up all accumulators to vacc0p0
        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);
        vacc1p0 = _mm512_add_ps(vacc1p0, vacc1p1);

        __m512 vacc0 = _mm512_max_ps(vacc0p0, vmin);
        __m512 vacc1 = _mm512_max_ps(vacc1p0, vmin);

        vacc0 = _mm512_min_ps(vacc0, vmax);
        vacc1 = _mm512_min_ps(vacc1, vmax);

        _mm512_storeu_ps(output, vacc0);
        _mm512_storeu_ps(output + 16, vacc1);
        output += 32;
      }


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

        w += 32;


        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        __m512 vacc = _mm512_max_ps(vaccp0, vmin);

        vacc = _mm512_min_ps(vacc, vmax);

        _mm512_storeu_ps(output, vacc);
        output += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        __m512 vaccp0 = _mm512_load_ps(b);
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);
        __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);
        __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        __m512 vacc = _mm512_max_ps(vaccp0, vmin);
        vacc = _mm512_min_ps(vacc, vmax);

        _mm512_mask_storeu_ps(output, vmask, vacc);
        output += c;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
