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


void xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2(
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
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 6);

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
      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = w[0];

        const float vi0 = *i0++;
        const float vk0 = w[1];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[2];
        float vacc0p1 = vi1 * vk1;
        const float vi2 = *i2++;
        const float vk2 = w[3];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[4];
        vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);
        const float vi4 = *i4++;
        const float vk4 = w[5];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);
        const float vi5 = *i5++;
        const float vk5 = w[6];
        vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

        w += 7;

        // Add up all accumulators to vacc0p0
        vacc0p0 = vacc0p0 + vacc0p1;

        *b++ = vacc0p0;
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

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        float vacc0p1 = vi1 * vk1;
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[3];
        vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);
        const float vi4 = *i4++;
        const float vk4 = w[4];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);
        const float vi5 = *i5++;
        const float vk5 = w[5];
        vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

        // Add up all accumulators to vacc0p0
        vacc0p0 = vacc0p0 + vacc0p1;

        w += 6;
        *b++ = vacc0p0;
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

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b++;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        float vacc0p1 = vi1 * vk1;
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[3];
        vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);
        const float vi4 = *i4++;
        const float vk4 = w[4];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);
        const float vi5 = *i5++;
        const float vk5 = w[5];
        vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);
        const float vi6 = *i6++;
        const float vk6 = w[6];
        vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

        w += 7;

        // Add up all accumulators to vacc0p0
        vacc0p0 = vacc0p0 + vacc0p1;

        *output++ = vacc0p0;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
