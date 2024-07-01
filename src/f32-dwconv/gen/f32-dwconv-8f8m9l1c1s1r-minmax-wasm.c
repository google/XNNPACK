// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm(
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
  assert(kernel_size > 8);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
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
      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = w[0];

        const float vi0 = *i0++;
        const float vk0 = w[1];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[2];
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        const float vi2 = *i2++;
        const float vk2 = w[3];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[4];
        vacc0p0 = math_muladd_f32(vi3, vk3, vacc0p0);
        const float vi4 = *i4++;
        const float vk4 = w[5];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);
        const float vi5 = *i5++;
        const float vk5 = w[6];
        vacc0p0 = math_muladd_f32(vi5, vk5, vacc0p0);
        const float vi6 = *i6++;
        const float vk6 = w[7];
        vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);
        const float vi7 = *i7++;
        const float vk7 = w[8];
        vacc0p0 = math_muladd_f32(vi7, vk7, vacc0p0);

        w += 9;


        *b++ = vacc0p0;
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

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[3];
        vacc0p0 = math_muladd_f32(vi3, vk3, vacc0p0);
        const float vi4 = *i4++;
        const float vk4 = w[4];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);
        const float vi5 = *i5++;
        const float vk5 = w[5];
        vacc0p0 = math_muladd_f32(vi5, vk5, vacc0p0);
        const float vi6 = *i6++;
        const float vk6 = w[6];
        vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);
        const float vi7 = *i7++;
        const float vk7 = w[7];
        vacc0p0 = math_muladd_f32(vi7, vk7, vacc0p0);


        w += 8;
        *b++ = vacc0p0;
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

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b++;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[3];
        vacc0p0 = math_muladd_f32(vi3, vk3, vacc0p0);
        const float vi4 = *i4++;
        const float vk4 = w[4];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);
        const float vi5 = *i5++;
        const float vk5 = w[5];
        vacc0p0 = math_muladd_f32(vi5, vk5, vacc0p0);
        const float vi6 = *i6++;
        const float vk6 = w[6];
        vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);
        const float vi7 = *i7++;
        const float vk7 = w[7];
        vacc0p0 = math_muladd_f32(vi7, vk7, vacc0p0);
        const float vi8 = *i8++;
        const float vk8 = w[8];
        vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

        w += 9;


        float vacc0 = __builtin_wasm_max_f32(vacc0p0, vmin);
        vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
        *output++ = vacc0;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
